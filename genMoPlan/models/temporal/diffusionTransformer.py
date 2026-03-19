import torch
import torch.nn as nn
import einops

from genMoPlan.models.helpers import SinusoidalPosEmb
from genMoPlan.models.temporal.base import TemporalModel
from genMoPlan.utils.constants import MASK_ON, MASK_OFF


class QueryEncoder(nn.Module):
    """Encode global query tokens via MLP projection + optional self-attention.

    For global_query_length=1, only the MLP projection is used (self-attention
    on a single token is a no-op). For longer sequences, self-attention layers
    contextualize query tokens before they're used as K/V in cross-attention.
    """

    def __init__(self, global_query_dim, hidden_dim, global_query_length,
                 num_heads=4, dropout=0.1, num_encoder_layers=1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(global_query_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.positional_encoding = nn.Parameter(
            torch.randn(1, global_query_length, hidden_dim) * 0.02
        )
        # Self-attention encoder only for multi-token queries
        self.use_self_attention = global_query_length > 1
        if self.use_self_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_encoder_layers,
            )

    def forward(self, global_query):
        # global_query: [batch, query_len, global_query_dim]
        q = self.projection(global_query)  # [batch, query_len, hidden_dim]
        q = q + self.positional_encoding
        if self.use_self_attention:
            q = self.encoder(q)
        return q  # [batch, query_len, hidden_dim]


class LocalQueryProcessor(nn.Module):
    """Process local queries that vary per timestep."""

    def __init__(self, local_query_dim, local_query_embed_dim):
        super().__init__()
        self.local_query_dim = local_query_dim
        self.local_query_embed_dim = local_query_embed_dim

        self.local_mlp = nn.Sequential(
            nn.Linear(local_query_dim, local_query_embed_dim * 2),
            nn.Mish(),
            nn.Linear(local_query_embed_dim * 2, local_query_embed_dim),
        )

    def forward(self, local_query: torch.Tensor) -> torch.Tensor:
        b, t, d = local_query.shape
        flat = local_query.reshape(-1, d)
        emb = self.local_mlp(flat)
        return emb.view(b, t, self.local_query_embed_dim)


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with AdaLN-Zero modulation, cross-attention, and LayerScale."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        time_embed_dim: int,
        global_query_dim: int = 0,
        local_query_embed_dim: int = 0,
        dropout: float = 0.1,
        use_windowed_attention: bool = False,
        attention_window_size: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_global_query = global_query_dim > 0
        self.use_local_query = local_query_embed_dim > 0
        self.use_windowed_attention = use_windowed_attention
        self.attention_window_size = attention_window_size

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # AdaLN-Zero modulations (separate for attention and feedforward)
        self.time_gamma_attn = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_beta_attn = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_gamma_ff = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
        self.time_beta_ff = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))

        # Cross-attention for global query (Q=horizon tokens, KV=projected query tokens)
        if self.use_global_query:
            self.norm_cross = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.time_gamma_cross = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
            self.time_beta_cross = nn.Sequential(nn.Mish(), nn.Linear(time_embed_dim, hidden_dim))
            self.alpha_cross = nn.Parameter(torch.tensor(1e-2))

        if self.use_local_query:
            self.local_gamma_attn = nn.Linear(local_query_embed_dim, hidden_dim)
            self.local_beta_attn = nn.Linear(local_query_embed_dim, hidden_dim)
            self.local_gamma_ff = nn.Linear(local_query_embed_dim, hidden_dim)
            self.local_beta_ff = nn.Linear(local_query_embed_dim, hidden_dim)

        # LayerScale parameters for residuals
        self.alpha_attn = nn.Parameter(torch.tensor(1e-2))
        self.alpha_ff = nn.Parameter(torch.tensor(1e-2))

        self.dropout = nn.Dropout(dropout)

    def _zero_init_modulation(self):
        """Zero-initialize modulation projection layers (AdaLN-Zero)."""
        zero_linears = []
        zero_linears.append(self.time_gamma_attn[-1]); zero_linears.append(self.time_beta_attn[-1])
        zero_linears.append(self.time_gamma_ff[-1]); zero_linears.append(self.time_beta_ff[-1])
        if self.use_global_query:
            zero_linears.append(self.time_gamma_cross[-1]); zero_linears.append(self.time_beta_cross[-1])
        if self.use_local_query:
            zero_linears.append(self.local_gamma_attn); zero_linears.append(self.local_beta_attn)
            zero_linears.append(self.local_gamma_ff); zero_linears.append(self.local_beta_ff)
        for lin in zero_linears:
            nn.init.zeros_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        q_global: torch.Tensor = None,
        q_local: torch.Tensor = None,
    ) -> torch.Tensor:
        # Sub-block 1: Self-attention with AdaLN-Zero and LayerScale
        y1 = self.norm1(x)

        b, s, _ = y1.shape

        gamma_attn = self.time_gamma_attn(t).unsqueeze(1).expand(b, s, -1)
        beta_attn = self.time_beta_attn(t).unsqueeze(1).expand(b, s, -1)

        if self.use_local_query and q_local is not None:
            gamma_attn = gamma_attn + self.local_gamma_attn(q_local)
            beta_attn = beta_attn + self.local_beta_attn(q_local)

        y1 = y1 * (1 + gamma_attn) + beta_attn
        attn_mask = None
        if self.use_windowed_attention and self.attention_window_size > 0:
            seq_len = s
            if self.attention_window_size < seq_len - 1:
                device = y1.device
                idx = torch.arange(seq_len, device=device)
                dist = (idx[None, :] - idx[:, None]).abs()
                mask_bool = dist > self.attention_window_size
                attn_mask = torch.zeros((seq_len, seq_len), device=device, dtype=y1.dtype)
                attn_mask.masked_fill_(mask_bool, float("-inf"))
        attn_out = self.self_attn(y1, y1, y1, attn_mask=attn_mask)[0]
        x = x + self.alpha_attn * self.dropout(attn_out)

        # Sub-block 1.5: Cross-attention with global query (Q=x, KV=q_global)
        if self.use_global_query and q_global is not None:
            y_cross = self.norm_cross(x)
            gamma_cross = self.time_gamma_cross(t).unsqueeze(1).expand(b, s, -1)
            beta_cross = self.time_beta_cross(t).unsqueeze(1).expand(b, s, -1)
            y_cross = y_cross * (1 + gamma_cross) + beta_cross
            cross_out = self.cross_attn(y_cross, q_global, q_global)[0]
            x = x + self.alpha_cross * self.dropout(cross_out)

        # Sub-block 2: Feedforward with AdaLN-Zero and LayerScale
        y2 = self.norm2(x)

        gamma_ff = self.time_gamma_ff(t).unsqueeze(1).expand(b, s, -1)
        beta_ff = self.time_beta_ff(t).unsqueeze(1).expand(b, s, -1)
        if self.use_local_query and q_local is not None:
            gamma_ff = gamma_ff + self.local_gamma_ff(q_local)
            beta_ff = beta_ff + self.local_beta_ff(q_local)

        y2 = y2 * (1 + gamma_ff) + beta_ff
        ff_out = self.ff(y2)
        x = x + self.alpha_ff * ff_out
        return x


class TemporalDiffusionTransformer(TemporalModel):
    """Transformer for temporal diffusion with FiLM conditioning.
    """

    def __init__(
        self,
        prediction_length,
        input_dim,
        output_dim,
        global_query_dim=0,
        global_query_length=0,
        local_query_dim=0,
        # Transformer specific parameters
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=None,
        dropout=0.1,
        time_embed_dim=None,
        local_query_embed_dim=None,
        query_encoder_layers=1,
        use_windowed_attention=False,
        attention_window_size=0,
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

        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        if time_embed_dim is None:
            time_embed_dim = hidden_dim
        if local_query_embed_dim is None:
            local_query_embed_dim = hidden_dim

        if verbose:
            print(
                f"[ models/diffusionTransformer ] Hidden dim: {hidden_dim}, layers: {num_layers}, heads: {num_heads}"
            )
            print(
                f"[ models/diffusionTransformer ] Feedforward dim: {feedforward_dim}, time_embed_dim: {time_embed_dim}"
            )
            if use_windowed_attention:
                print(
                    f"[ models/diffusionTransformer ] Windowed attention enabled: window={attention_window_size}"
                )
            if global_query_dim > 0:
                print(
                    f"[ models/diffusionTransformer ] Global query cross-attention enabled: global_query_dim={global_query_dim}, query_encoder_layers={query_encoder_layers}"
                )
            if local_query_dim > 0:
                print(
                    f"[ models/diffusionTransformer ] Local query conditioning enabled: local_query_dim={local_query_dim}, local_query_embed_dim={local_query_embed_dim}"
                )

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1))
        self.positional_encoding = nn.Parameter(
            torch.randn(1, prediction_length, hidden_dim) * 0.02
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        self.use_windowed_attention = use_windowed_attention
        self.attention_window_size = attention_window_size

        self.use_global_query = global_query_dim > 0
        if self.use_global_query:
            self.query_encoder = QueryEncoder(
                global_query_dim=global_query_dim,
                hidden_dim=hidden_dim,
                global_query_length=global_query_length,
                num_heads=num_heads,
                dropout=dropout,
                num_encoder_layers=query_encoder_layers,
            )

        self.use_local_query = local_query_dim > 0
        if self.use_local_query:
            self.local_query_processor = LocalQueryProcessor(
                local_query_dim, local_query_embed_dim
            )

        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    time_embed_dim=time_embed_dim,
                    global_query_dim=(
                        hidden_dim if self.use_global_query else 0
                    ),
                    local_query_embed_dim=(
                        local_query_embed_dim if self.use_local_query else 0
                    ),
                    dropout=dropout,
                    use_windowed_attention=self.use_windowed_attention,
                    attention_window_size=self.attention_window_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Apply AdaLN-Zero zero-initialization after global init
        for layer in self.layers:
            if hasattr(layer, "_zero_init_modulation"):
                layer._zero_init_modulation()

    def forward(self, x, global_query=None, local_query=None, time=None, mask=None):
        """Forward pass through the Diffusion Transformer.

        Args:
            x (torch.Tensor): [batch, prediction_length, input_dim]
            global_query (torch.Tensor): [batch, global_query_length, global_query_dim] or None
            local_query (torch.Tensor): [batch, prediction_length, local_query_dim] or None
            time (torch.Tensor): [batch] or [batch, ...] timestep indices
            mask (torch.Tensor): [batch, prediction_length] binary mask in {0,1}.
                Mask convention:
                - MASK_OFF (1.0): Position is UNMASKED (valid/present) - keeps original x value
                - MASK_ON (0.0): Position is MASKED (missing/padded) - replaced with learned mask_token
                Formula: x = mask * x + (1 - mask) * mask_token

        Returns:
            torch.Tensor: Output shape [batch, prediction_length, output_dim].
        """

        if getattr(self, "expect_mask", False) and mask is None:
            raise ValueError("Mask expected but not provided")

        t = self.time_mlp(time)

        q_global = None
        if self.use_global_query and global_query is not None:
            q_global = self.query_encoder(global_query)  # [batch, query_len, hidden_dim]

        q_local = None
        if self.use_local_query and local_query is not None:
            q_local = self.local_query_processor(local_query)

        if mask is not None:
            if mask.dim() != 2 or mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask must have shape [batch, prediction_length]={x.shape[:2]}, got {tuple(mask.shape)}"
                )
            mask = mask.to(dtype=x.dtype, device=x.device)
            mask_unsq = mask.unsqueeze(-1)
            masked_fill_value = self.mask_token.to(dtype=x.dtype, device=x.device).view(1, 1, 1)
            # Apply mask: MASK_OFF (1) keeps x, MASK_ON (0) uses mask_token
            x = mask_unsq * x + (1.0 - mask_unsq) * masked_fill_value

        x = self.input_projection(x)
        x = x + self.positional_encoding

        for layer in self.layers:
            x = layer(x, t, q_global, q_local)

        x = self.output_projection(x)
        return x
