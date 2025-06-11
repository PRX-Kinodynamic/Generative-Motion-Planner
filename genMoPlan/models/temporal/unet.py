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
from genMoPlan.models.temporal.base import TemporalModel


class GlobalQueryProcessor(nn.Module):
    """Process global query sequences into conditioning embeddings.
    
    Follows similar pattern to time processing but handles temporal sequences.
    """
    
    def __init__(self, global_query_dim, global_query_length, global_query_embed_dim):
        super().__init__()
        self.global_query_dim = global_query_dim
        self.global_query_length = global_query_length
        self.global_query_embed_dim = global_query_embed_dim
        
        if global_query_length > 1:
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(global_query_dim, global_query_embed_dim, kernel_size=3, padding=1),
                nn.Mish(),
                nn.Conv1d(global_query_embed_dim, global_query_embed_dim, kernel_size=3, padding=1),
                nn.Mish(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
        else:
            # Simple linear projection for single timestep
            self.temporal_encoder = nn.Linear(global_query_dim, global_query_embed_dim)
        
        # Final conditioning MLP - similar to time_mlp structure
        self.query_mlp = nn.Sequential(
            nn.Linear(global_query_embed_dim, global_query_embed_dim * 4),
            nn.Mish(),
            nn.Linear(global_query_embed_dim * 4, global_query_embed_dim),
        )
    
    def forward(self, global_query):
        """
        Args:
            query (torch.Tensor): Query tensor of shape [batch, query_length, query_dim]
            
        Returns:
            torch.Tensor: Query embedding of shape [batch, query_embed_dim]
        """
        if self.global_query_length > 1:
            # 1D convolution approach for temporal patterns
            global_query_transposed = global_query.transpose(1, 2)  # [batch, global_query_dim, global_query_length]
            global_query_encoded = self.temporal_encoder(global_query_transposed)
        else:
            # Single timestep processing
            global_query_encoded = self.temporal_encoder(global_query.squeeze(1))
        
        # Apply conditioning MLP
        global_query_embedding = self.query_mlp(global_query_encoded)
        
        return global_query_embedding


class LocalQueryProcessor(nn.Module):
    """Process local queries that vary per timestep."""
    
    def __init__(self, local_query_dim, local_query_embed_dim):
        super().__init__()
        self.local_query_dim = local_query_dim
        self.local_query_embed_dim = local_query_embed_dim
        
        # Process each timestep's query independently
        self.local_mlp = nn.Sequential(
            nn.Linear(local_query_dim, local_query_embed_dim * 2),
            nn.Mish(),
            nn.Linear(local_query_embed_dim * 2, local_query_embed_dim),
        )
    
    def forward(self, local_query):
        """
        Args:
            local_query: [batch, prediction_length, local_query_dim]
        Returns:
            [batch, prediction_length, local_query_embed_dim]
        """
        batch, seq_len, query_dim = local_query.shape
        # Process each timestep independently
        local_query_flat = local_query.view(-1, query_dim)
        local_embed_flat = self.local_mlp(local_query_flat)
        local_embed = local_embed_flat.view(batch, seq_len, self.local_query_embed_dim)
        return local_embed


class TemporalUnet(TemporalModel):
    """Temporal UNet model for diffusion.
    
    A U-Net architecture specifically designed for temporal data, with optional attention mechanisms
    and query conditioning support.
    
    Args:
        prediction_length (int): Length of the prediction sequence.
        input_dim (int): Dimension of the input state space.
        output_dim (int): Dimension of the output state space.
        global_query_dim (int): Dimension of the state space of the global query.
        global_query_length (int): Length of the global query sequence.
        local_query_dim (int): Dimension of the state space of the local query.
        base_channels (int, optional): Base number of channels for hidden layers. Defaults to 32.
        channel_multipliers (tuple, optional): Multipliers for the base channels to determine dimensions of hidden layers.
            Defaults to (1, 2, 4, 8).
        conv_kernel_size (int, optional): Kernel size of the convolutional layers. Defaults to 5.
        attention (bool, optional): Whether to use attention in the model. Defaults to False.
        global_query_embed_dim (int, optional): Dimension of global query embeddings. Defaults to base_hidden_dim.
        local_query_embed_dim (int, optional): Dimension of local query embeddings. Defaults to base_hidden_dim.
    """
    def __init__(
        self,
        prediction_length,
        input_dim,
        output_dim,
        global_query_dim=0,
        global_query_length=0,
        local_query_dim=0,
        # UNet specific parameters
        base_hidden_dim=32,
        hidden_dim_mult=(1, 2, 4, 8),
        conv_kernel_size=5,
        attention=False,
        time_embed_dim=None,
        global_query_embed_dim=None,
        local_query_embed_dim=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(
            prediction_length,
            input_dim,
            output_dim,
            global_query_dim,
            global_query_length,
            local_query_dim,
        )

        num_channels = [input_dim, *map(lambda m: base_hidden_dim * m, hidden_dim_mult)]
        in_out = list(zip(num_channels[:-1], num_channels[1:]))
        if time_embed_dim is None:
            time_embed_dim = base_hidden_dim
        if global_query_embed_dim is None:
            global_query_embed_dim = base_hidden_dim
        if local_query_embed_dim is None:
            local_query_embed_dim = base_hidden_dim

        if verbose:
            print(f"[ models/unet ] Channel dimensions: {in_out}, time_embed_dim: {time_embed_dim}")
            if global_query_dim > 0:
                print(f"[ models/unet ] Global query conditioning enabled: global_query_dim={global_query_dim}, global_query_embed_dim={global_query_embed_dim}")
            if local_query_dim > 0:
                print(f"[ models/unet ] Local query conditioning enabled: local_query_dim={local_query_dim}, local_query_embed_dim={local_query_embed_dim}")

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        self.use_global_query = global_query_dim > 0
        if self.use_global_query:
            self.global_query_processor = GlobalQueryProcessor(
                global_query_dim, global_query_length, global_query_embed_dim
            )

        self.use_local_query = local_query_dim > 0
        if self.use_local_query:
            self.local_query_processor = LocalQueryProcessor(
                local_query_dim, local_query_embed_dim
            )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ResidualTemporalBlock(
                        dim_in, dim_out, 
                        time_embed_dim=time_embed_dim,
                        global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
                        local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
                        kernel_size=conv_kernel_size
                    ),
                    ResidualTemporalBlock(
                        dim_out, dim_out, 
                        time_embed_dim=time_embed_dim,
                        global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
                        local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
                        kernel_size=conv_kernel_size
                    ),
                    (
                        Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                        if attention
                        else nn.Identity()
                    ),
                    Downsample1d(dim_out) if not is_last else nn.Identity(),
                ])
            )

            if not is_last:
                prediction_length = prediction_length // 2

        mid_dim = num_channels[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, 
            time_embed_dim=time_embed_dim,
            global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
            local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
            kernel_size=conv_kernel_size
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, 
            time_embed_dim=time_embed_dim,
            global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
            local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
            kernel_size=conv_kernel_size
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ResidualTemporalBlock(
                        dim_out * 2, dim_in, 
                        time_embed_dim=time_embed_dim,
                        global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
                        local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
                        kernel_size=conv_kernel_size
                    ),
                    ResidualTemporalBlock(
                        dim_in, dim_in, 
                        time_embed_dim=time_embed_dim,
                        global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
                        local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
                        kernel_size=conv_kernel_size
                    ),
                    (
                        Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                        if attention
                        else nn.Identity()
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity(),
                ])
            )

            if not is_last:
                prediction_length = prediction_length * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(base_hidden_dim, base_hidden_dim, kernel_size=conv_kernel_size),
            nn.Conv1d(base_hidden_dim, output_dim, 1),
        )

    def forward(self, x, global_query=None, local_query=None, time=None):
        """Forward pass through the Temporal UNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, prediction_length, input_dim].
            global_query (torch.Tensor): Global query tensor of shape [batch, global_query_length, global_query_dim] or None.
            local_query (torch.Tensor): Local query tensor of shape [batch, prediction_length, local_query_dim] or None.
            time (torch.Tensor): Time tensor. Usually the timestep of the noising/denoising process.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, prediction_length, output_dim].
        """
        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        
        # Process global query
        q_global = None
        if self.use_global_query and global_query is not None:
            q_global = self.global_query_processor(global_query)

        # Process local query
        q_local = None
        if self.use_local_query and local_query is not None:
            q_local = self.local_query_processor(local_query)
            q_local = einops.rearrange(q_local, "b h t -> b t h")

        h = []
        local_queries = []

        # Encoder
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t, q_global, q_local)
            x = resnet2(x, t, q_global, q_local)
            x = attn(x)
            h.append(x)
            
            if q_local is not None:
                local_queries.append(q_local)
            
            x = downsample(x)
            if q_local is not None:
                if hasattr(downsample, '__class__') and 'Identity' not in str(downsample.__class__):
                    q_local = downsample(q_local)

        # Middle blocks
        x = self.mid_block1(x, t, q_global, q_local)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, q_global, q_local)

        # Decoder
        for resnet, resnet2, attn, upsample in self.ups:
            h_val = h.pop()
            x = torch.cat((x, h_val), dim=1)
            
            if q_local is not None and local_queries:
                q_local = local_queries.pop()
            
            x = resnet(x, t, q_global, q_local)
            x = resnet2(x, t, q_global, q_local)
            x = attn(x)
            
            x = upsample(x)
            if q_local is not None:
                if hasattr(upsample, '__class__') and 'Identity' not in str(upsample.__class__):
                    q_local = upsample(q_local)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        out_channels,
        time_embed_dim,
        global_query_embed_dim=0,
        local_query_embed_dim=0,
        kernel_size=5,
    ):
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

        # Global query conditioning
        self.use_global_query = global_query_embed_dim > 0
        if self.use_global_query:
            self.global_query_gamma = nn.Sequential(
                nn.Mish(),
                nn.Linear(global_query_embed_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
            self.global_query_beta = nn.Sequential(
                nn.Mish(),
                nn.Linear(global_query_embed_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )

        # Local query conditioning
        self.use_local_query = local_query_embed_dim > 0
        if self.use_local_query:
            self.local_query_gamma = nn.Conv1d(local_query_embed_dim, out_channels, 1)
            self.local_query_beta = nn.Conv1d(local_query_embed_dim, out_channels, 1)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t, q_global=None, q_local=None):
        """
        Args:
            x: [batch_size x inp_channels x prediction_length]
            t: [batch_size x time_embed_dim] 
            q_global: [batch_size x global_query_embed_dim] - global query
            q_local: [batch_size x local_query_embed_dim x prediction_length] - local queries
            
        Returns:
            out: [batch_size x out_channels x prediction_length]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        
        # Apply global FiLM conditioning
        if self.use_global_query and q_global is not None:
            gamma_global = self.global_query_gamma(q_global)
            beta_global = self.global_query_beta(q_global)
            out = out * (1 + gamma_global) + beta_global
        
        # Apply local FiLM conditioning (per-timestep)
        if self.use_local_query and q_local is not None:
            gamma_local = self.local_query_gamma(q_local)
            beta_local = self.local_query_beta(q_local)
            out = out * (1 + gamma_local) + beta_local
        
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

