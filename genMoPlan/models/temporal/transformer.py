import torch
import torch.nn as nn
import math
import einops
from typing import Optional

from genMoPlan.models.helpers import SinusoidalPosEmb
from genMoPlan.models.temporal.base import TemporalModel


class GlobalQueryProcessor(nn.Module):
    """Process global query sequences into conditioning embeddings for transformer."""
    
    def __init__(self, global_query_dim, global_query_length, global_query_embed_dim):
        super().__init__()
        self.global_query_dim = global_query_dim
        self.global_query_length = global_query_length
        self.global_query_embed_dim = global_query_embed_dim
        
        if global_query_length > 1:
            # Use transformer encoder for temporal patterns
            self.input_projection = nn.Linear(global_query_dim, global_query_embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=global_query_embed_dim,
                nhead=8,
                dim_feedforward=global_query_embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            # Simple linear projection for single timestep
            self.temporal_encoder = nn.Linear(global_query_dim, global_query_embed_dim)
        
        # Final conditioning MLP
        self.query_mlp = nn.Sequential(
            nn.Linear(global_query_embed_dim, global_query_embed_dim * 4),
            nn.GELU(),
            nn.Linear(global_query_embed_dim * 4, global_query_embed_dim),
        )
    
    def forward(self, global_query):
        """
        Args:
            global_query (torch.Tensor): Query tensor of shape [batch, query_length, query_dim]
            
        Returns:
            torch.Tensor: Query embedding of shape [batch, query_embed_dim]
        """
        if self.global_query_length > 1:
            # Project to embedding dimension
            global_query_projected = self.input_projection(global_query)
            # Apply transformer encoder
            global_query_encoded = self.temporal_encoder(global_query_projected)
            # Pool across sequence length
            global_query_pooled = self.pooling(global_query_encoded.transpose(1, 2)).squeeze(-1)
        else:
            # Single timestep processing
            global_query_pooled = self.temporal_encoder(global_query.squeeze(1))
        
        # Apply conditioning MLP
        global_query_embedding = self.query_mlp(global_query_pooled)
        
        return global_query_embedding


class LocalQueryProcessor(nn.Module):
    """Process local queries that vary per timestep using transformer."""
    
    def __init__(self, local_query_dim, local_query_embed_dim):
        super().__init__()
        self.local_query_dim = local_query_dim
        self.local_query_embed_dim = local_query_embed_dim
        
        # Process each timestep's query independently
        self.local_mlp = nn.Sequential(
            nn.Linear(local_query_dim, local_query_embed_dim * 2),
            nn.GELU(),
            nn.Linear(local_query_embed_dim * 2, local_query_embed_dim),
        )
    
    def forward(self, local_query):
        """
        Args:
            local_query: [batch, prediction_length, local_query_dim]
        Returns:
            [batch, prediction_length, local_query_embed_dim]
        """
        return self.local_mlp(local_query)


class TimeConditionedTransformerBlock(nn.Module):
    """Transformer block with time and query conditioning using cross-attention and FiLM."""
    
    def __init__(
        self,
        hidden_dim,
        num_heads,
        feedforward_dim,
        time_embed_dim,
        global_query_embed_dim=0,
        local_query_embed_dim=0,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_global_query = global_query_embed_dim > 0
        self.use_local_query = local_query_embed_dim > 0
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Time conditioning via FiLM
        self.time_gamma = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        self.time_beta = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        
        # Global query conditioning via FiLM
        if self.use_global_query:
            self.global_query_gamma = nn.Sequential(
                nn.GELU(),
                nn.Linear(global_query_embed_dim, hidden_dim)
            )
            self.global_query_beta = nn.Sequential(
                nn.GELU(),
                nn.Linear(global_query_embed_dim, hidden_dim)
            )
        
        # Local query conditioning via cross-attention
        if self.use_local_query:
            self.local_query_cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.local_query_projection = nn.Linear(local_query_embed_dim, hidden_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        if self.use_local_query:
            self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t, q_global=None, q_local=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            t: [batch, time_embed_dim]
            q_global: [batch, global_query_embed_dim]
            q_local: [batch, seq_len, local_query_embed_dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Apply time conditioning via FiLM
        gamma_time = self.time_gamma(t).unsqueeze(1)  # [batch, 1, hidden_dim]
        beta_time = self.time_beta(t).unsqueeze(1)    # [batch, 1, hidden_dim]
        x = x * (1 + gamma_time) + beta_time
        
        # Apply global query conditioning via FiLM
        if self.use_global_query and q_global is not None:
            gamma_global = self.global_query_gamma(q_global).unsqueeze(1)  # [batch, 1, hidden_dim]
            beta_global = self.global_query_beta(q_global).unsqueeze(1)    # [batch, 1, hidden_dim]
            x = x * (1 + gamma_global) + beta_global
        
        # Apply local query conditioning via cross-attention
        if self.use_local_query and q_local is not None:
            q_local_proj = self.local_query_projection(q_local)  # [batch, seq_len, hidden_dim]
            cross_attn_out, _ = self.local_query_cross_attn(x, q_local_proj, q_local_proj)
            x = x + self.dropout(cross_attn_out)
            x = self.norm3(x)
        
        # Feedforward with residual connection
        ff_out = self.feedforward(x)
        x = x + ff_out
        x = self.norm2(x)
        
        return x


class TemporalTransformer(TemporalModel):
    """Temporal Transformer model for diffusion and flow matching.
    
    A transformer architecture specifically designed for temporal data, with time conditioning
    and optional query conditioning support.
    
    Args:
        prediction_length (int): Length of the prediction sequence.
        input_dim (int): Dimension of the input state space.
        output_dim (int): Dimension of the output state space.
        global_query_dim (int): Dimension of the state space of the global query.
        global_query_length (int): Length of the global query sequence.
        local_query_dim (int): Dimension of the state space of the local query.
        hidden_dim (int, optional): Hidden dimension of transformer. Defaults to 256.
        num_layers (int, optional): Number of transformer layers. Defaults to 6.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        feedforward_dim (int, optional): Dimension of feedforward network. Defaults to 4*hidden_dim.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        time_embed_dim (int, optional): Dimension of time embeddings. Defaults to hidden_dim.
        global_query_embed_dim (int, optional): Dimension of global query embeddings. Defaults to hidden_dim.
        local_query_embed_dim (int, optional): Dimension of local query embeddings. Defaults to hidden_dim.
        use_positional_encoding (bool, optional): Whether to use learnable positional encoding. Defaults to True.
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
        global_query_embed_dim=None,
        local_query_embed_dim=None,
        use_positional_encoding=True,
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
        if global_query_embed_dim is None:
            global_query_embed_dim = hidden_dim
        if local_query_embed_dim is None:
            local_query_embed_dim = hidden_dim
        
        if verbose:
            print(f"[ models/transformer ] Hidden dim: {hidden_dim}, layers: {num_layers}, heads: {num_heads}")
            print(f"[ models/transformer ] Feedforward dim: {feedforward_dim}, time_embed_dim: {time_embed_dim}")
            if global_query_dim > 0:
                print(f"[ models/transformer ] Global query conditioning enabled: global_query_dim={global_query_dim}, global_query_embed_dim={global_query_embed_dim}")
            if local_query_dim > 0:
                print(f"[ models/transformer ] Local query conditioning enabled: local_query_dim={local_query_dim}, local_query_embed_dim={local_query_embed_dim}")
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(torch.randn(1, prediction_length, hidden_dim) * 0.02)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        
        # Query processors
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
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TimeConditionedTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                time_embed_dim=time_embed_dim,
                global_query_embed_dim=global_query_embed_dim if self.use_global_query else 0,
                local_query_embed_dim=local_query_embed_dim if self.use_local_query else 0,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following transformer best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, global_query=None, local_query=None, time=None):
        """Forward pass through the Temporal Transformer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, prediction_length, input_dim].
            global_query (torch.Tensor): Global query tensor of shape [batch, global_query_length, global_query_dim] or None.
            local_query (torch.Tensor): Local query tensor of shape [batch, prediction_length, local_query_dim] or None.
            time (torch.Tensor): Time tensor. Usually the timestep of the noising/denoising process.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, prediction_length, output_dim].
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = x + self.positional_encoding
        
        # Process time embedding
        t = self.time_mlp(time)  # [batch, time_embed_dim]
        
        # Process global query
        q_global = None
        if self.use_global_query and global_query is not None:
            q_global = self.global_query_processor(global_query)
        
        # Process local query
        q_local = None
        if self.use_local_query and local_query is not None:
            q_local = self.local_query_processor(local_query)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, t, q_global, q_local)
        
        # Project to output dimension
        x = self.output_projection(x)  # [batch, seq_len, output_dim]
        
        return x
