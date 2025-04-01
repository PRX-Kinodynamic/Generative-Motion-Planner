import torch
import torch.nn as nn
import math

from genMoPlan.models.helpers import SinusoidalPosEmb

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        prediction_length,
        input_dim,
        output_dim,
        query_dim=0,
        query_length=0,
        # Transformer specific parameters
        hidden_dim=128,
        depth=4,
        heads=4,
        hidden_dim_mult=4,
        dropout=0.05,
        time_embed_dim=None,
        use_relative_pos=False,
        recency_decay_rate=0.0,
        **kwargs,
    ):
        super().__init__()

        if time_embed_dim is None:
            time_embed_dim = hidden_dim
        
        self.use_relative_pos = use_relative_pos
        
        print(f"[ models/transformer ] Initializing TemporalTransformer with hidden_dim: {hidden_dim}, depth: {depth}, heads: {heads}, time_embed_dim: {time_embed_dim}")
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Only register absolute positional embedding if using absolute encoding
        if not self.use_relative_pos:
            self.register_buffer(
                "pos_embedding", 
                self._create_sinusoidal_positional_embedding(prediction_length, hidden_dim)
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                prediction_length,
                time_embed_dim,
                heads=heads,
                hidden_dim_mult=hidden_dim_mult,
                dropout=dropout,
                use_relative_pos=self.use_relative_pos,
                recency_decay_rate=recency_decay_rate
            )
            for _ in range(depth)
        ])
        
        # Optional query embedding
        self.has_query = query_dim is not None and query_dim > 0
        if self.has_query:
            self.query_proj = nn.Sequential(
                nn.Linear(query_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize model
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for better convergence
        """
        for p in self.parameters():
            if p.dim() > 1:  # Skip LayerNorm and biases
                nn.init.xavier_uniform_(p)
    
    def _create_sinusoidal_positional_embedding(self, length, dim):
        """
        Create sinusoidal positional embeddings for absolute transformer positions
        """
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pos_embedding = torch.zeros(length, dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding
        
    def forward(self, x, query, time):
        """
        x: [batch_size, prediction_length, input_dim]
        query: conditional information [batch_size, query_dim] or None
        time: [batch_size] time embedding
        """
        time_embed = self.time_mlp(time)  # [batch_size, time_embed_dim]
        
        x_embed = self.input_proj(x)  # [batch_size, prediction_length, hidden_dim]
        
        if not self.use_relative_pos:
            x_embed = x_embed + self.pos_embedding.unsqueeze(0)  # [batch_size, prediction_length, hidden_dim]
        
        if self.has_query and query is not None:
            query_emb = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, dim]
            x_embed = x_embed + query_emb
        
        for block in self.blocks:
            x_embed = block(x_embed, time_embed)
        
        x_pred = self.output_proj(x_embed)  # [batch_size, prediction_length, output_dim]
        
        return x_pred


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, prediction_length, time_embed_dim, heads, hidden_dim_mult, dropout, use_relative_pos, recency_decay_rate):
        super().__init__()
        self.use_relative_pos = use_relative_pos

        self.norm1 = nn.LayerNorm(hidden_dim)
        
        
        if use_relative_pos:
            self.attn = RelativeMultiheadAttention(hidden_dim, heads, prediction_length, dropout, recency_decay_rate)
        else:
            self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        ff_dim = hidden_dim * hidden_dim_mult
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, t):
        """
        x: [batch_size, prediction_length, hidden_dim]
        t: [batch_size, time_embed_dim]
        """
        # Time embedding
        time_emb = self.time_proj(t).unsqueeze(1)  # [batch_size, 1, dim]
        
        # Apply time embedding
        x_time = x + time_emb
        
        # Self-attention with residual
        x_norm = self.norm1(x_time)
        
        if self.use_relative_pos:
            attn_out = self.attn(x_norm)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            
        x = x + attn_out
        
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class RelativeMultiheadAttention(nn.Module):
    """
    Multi-head Attention with Relative Positional Encoding.
    
    This implementation adds relative positional information directly into the attention
    calculation rather than adding positional encodings to the input embeddings.
    
    The relative positional encoding is implemented as a learnable bias matrix where:
    - The matrix has size (2 * max_seq_len - 1, num_heads)
    - Each position in the matrix represents the bias to apply for a specific relative distance
    - For a relative distance d, we add a bias value to the attention score
    - The center of the matrix (max_seq_len-1) represents a relative distance of 0
    - Positions to the left represent negative distances, to the right positive distances
    
    Key advantages:
    1. Learns position-specific attention patterns rather than using fixed encodings
    2. Captures token relationships based on their relative positions rather than absolute
    3. Can be more effective for tasks where relative ordering is more important than absolute positioning
    
    This implementation:
    - Creates a learnable bias table for all possible relative positions
    - Pre-computes relative position indices for efficient lookups during forward pass
    - Integrates the relative position bias directly into the attention score calculation
    """
    def __init__(self, dim, num_heads, max_seq_len, dropout, recency_decay_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.recency_decay_rate = recency_decay_rate
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Create relative position bias table
        self.max_seq_len = max_seq_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_seq_len - 1, num_heads))
        
        # Initialize the relative position bias
        nn.init.xavier_uniform_(self.rel_pos_bias)
        
        # Generate relative position indices for future lookups
        pos_indices = torch.arange(max_seq_len, device=self.rel_pos_bias.device)
        rel_indices = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0) + max_seq_len - 1
        self.register_buffer("rel_indices", rel_indices)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Calculate QKV projections
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [batch, num_heads, seq_len, head_dim]
        
        # Compute attention matrix with scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]
        
        # Compute relative distances (absolute difference between positions)
        pos_indices = torch.arange(seq_len, device=x.device)
        rel_dist = torch.abs(pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0))  # shape: [seq_len, seq_len]

        # Compute the decay factor for each relative position
        decay = torch.exp(-self.recency_decay_rate * rel_dist)  # shape: [seq_len, seq_len]
        
        # Get the learned relative bias and permute it to shape [num_heads, seq_len, seq_len]
        rel_pos_bias = self.rel_pos_bias[self.rel_indices[:seq_len, :seq_len]]  # [seq_len, seq_len, num_heads]
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]

        # Apply the decay factor; unsqueeze to match dimensions: [1, 1, seq_len, seq_len]
        decay = decay.unsqueeze(0).unsqueeze(0)

        # Add the decayed bias to the attention scores
        attn = attn + (decay * rel_pos_bias).unsqueeze(0)
        
        # Apply softmax and dropout
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.proj(out)
        
        return out

