import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from genMoPlan.models.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)

class TemporalUnet(nn.Module):
    """Temporal UNet model for diffusion.
    
    A U-Net architecture specifically designed for temporal data, with optional attention mechanisms.
    
    Args:
        prediction_length (int): Length of the prediction sequence.
        input_dim (int): Dimension of the input state space.
        output_dim (int): Dimension of the output state space. Has to be the same as the input dimension.
        query_dim (int): Dimension of the state space of the query.
        base_channels (int, optional): Base number of channels for hidden layers. Defaults to 32.
        channel_multipliers (tuple, optional): Multipliers for the base channels to determine dimensions of hidden layers.
            Defaults to (1, 2, 4, 8).
        conv_kernel_size (int, optional): Kernel size of the convolutional layers. Defaults to 5.
        attention (bool, optional): Whether to use attention in the model. Defaults to False.
    """
    # Input and output dimensions are the same
    def __init__(
        self,
        prediction_length,
        input_dim,
        output_dim,
        query_dim=0,
        # UNet specific parameters
        base_hidden_dim=32,
        hidden_dim_mult=(1, 2, 4, 8),
        conv_kernel_size=5,
        attention=False,
        time_embed_dim=None,
        **kwargs,
    ):
        super().__init__()

        assert input_dim == output_dim, "Input and output dimensions must be the same"

        num_channels = [input_dim, *map(lambda m: base_hidden_dim * m, hidden_dim_mult)]
        in_out = list(zip(num_channels[:-1], num_channels[1:]))
        if time_embed_dim is None:
            time_embed_dim = base_hidden_dim

        print(f"[ models/unet ] Channel dimensions: {in_out}, time_embed_dim: {time_embed_dim}")

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_hidden_dim),
            nn.Linear(base_hidden_dim, base_hidden_dim * 4),
            nn.Mish(),
            nn.Linear(base_hidden_dim * 4, base_hidden_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out, time_embed_dim=time_embed_dim, kernel_size=conv_kernel_size
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out, time_embed_dim=time_embed_dim, kernel_size=conv_kernel_size
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                prediction_length = prediction_length // 2

        mid_dim = num_channels[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, time_embed_dim=time_embed_dim, kernel_size=conv_kernel_size
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, time_embed_dim=time_embed_dim, kernel_size=conv_kernel_size
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, time_embed_dim=time_embed_dim, kernel_size=conv_kernel_size
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, time_embed_dim=time_embed_dim, kernel_size=conv_kernel_size
                        ),
                        (
                            Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                            if attention
                            else nn.Identity()
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                prediction_length = prediction_length * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(base_hidden_dim, base_hidden_dim, kernel_size=conv_kernel_size),
            nn.Conv1d(base_hidden_dim, input_dim, 1),
        )

    def forward(self, x, query, time):
        """Forward pass through the Temporal UNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, prediction_length, input_dim].
            query (torch.Tensor): Query tensor.
            time (torch.Tensor): Time tensor. Usually the timestep of the noising/denoising process.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, prediction_length, output_dim].
        """

        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            h_val = h.pop()
            x = torch.cat((x, h_val), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, time_embed_dim, kernel_size):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x prediction_length ]
        t : [ batch_size x time_embed_dim ]
        returns:
        out : [ batch_size x out_channels x prediction_length ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

