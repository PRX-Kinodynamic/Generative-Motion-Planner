# DiT (Diffusion Transformer) Architecture

Architecture of `TemporalDiffusionTransformer` as used with the `dit_test` variation on cartpole flow matching.

## Concrete Parameters (cartpole + dit_test)

| Parameter | Value | Source |
|---|---|---|
| prediction_length | 32 | history(1) + horizon(31) |
| input_dim | 5 | state_dim(4) + 1 angle sin/cos |
| output_dim | 4 | velocity field in original state space |
| hidden_dim | 128 | dit_test override |
| num_layers | 4 | dit_test override |
| num_heads | 4 | dit_test override (head_dim=32) |
| feedforward_dim | 512 | default: hidden_dim * 4 |
| time_embed_dim | 128 | default: hidden_dim |
| dropout | 0.01 | dit_test override |
| positional_encoding | always on | learned absolute PE |

## Architecture Flow

```
                                    INPUTS
                                    ------

  x [batch, 32, 5]                              t [batch]
  (manifold-embedded state)                      (flow time in [0,1])
        |                                             |
        |                                             v
        |                                   +--------------------+
        |                                   | SinusoidalPosEmb   |
        |                                   | scalar --> 128D    |
        |                                   +--------------------+
        |                                             |
        |                                             v
        |                                   +--------------------+
        |                                   | Linear(128 --> 512)|
        |                                   | Mish()             |
        |                                   | Linear(512 --> 128)|
        |                                   +--------------------+
        |                                             |
        |                                        t_emb [batch, 128]
        |                                             |
        v                                             |
+-------------------+                                 |
| Input Projection  |                                 |
| Linear(5 --> 128) |                                 |
+-------------------+                                 |
        |                                             |
        v                                             |
+---------------------------+                         |
| + positional_encoding     |                         |
| [1, 32, 128] (learned)   |                         |
| Always applied.           |                         |
+---------------------------+                         |
        |                                             |
        v                                             |
  x [batch, 32, 128]                                 |
        |                                             |
        +=============================================+
        |                                             |
        |   +=========================================|========+
        |   ||         TRANSFORMER BLOCK  (x4)        |       ||
        |   ||                                        |       ||
        |   ||   +--- SUB-BLOCK A: ATTENTION ---------|--+    ||
        |   ||   |                                    |  |    ||
        |   ||   |    residual = x                    |  |    ||
        |   ||   |        |                           |  |    ||
        |   ||   |        v                           |  |    ||
        |   ||   |    LayerNorm(128)                  |  |    ||
        |   ||   |        |                           |  |    ||
        |   ||   |        v                           |  |    ||
        |   ||   |    +-- AdaLN-Zero -------+         |  |    ||
        |   ||   |    |                     |         |  |    ||
        |   ||   |    | gamma = Mish -->    |<--------+  |    ||
        |   ||   |    |   Linear(128-->128) |  t_emb     |    ||
        |   ||   |    | beta  = Mish -->    |            |    ||
        |   ||   |    |   Linear(128-->128) |            |    ||
        |   ||   |    | (zero-initialized)  |            |    ||
        |   ||   |    |                     |            |    ||
        |   ||   |    | y = norm(x)         |            |    ||
        |   ||   |    |   * (1 + gamma(t))  |            |    ||
        |   ||   |    |   + beta(t)         |            |    ||
        |   ||   |    +---------------------+            |    ||
        |   ||   |        |                              |    ||
        |   ||   |        v                              |    ||
        |   ||   |    +-------------------------+        |    ||
        |   ||   |    | MultiheadAttention      |        |    ||
        |   ||   |    |   embed=128, heads=4    |        |    ||
        |   ||   |    |   head_dim=32           |        |    ||
        |   ||   |    |   dropout=0.01          |        |    ||
        |   ||   |    |                         |        |    ||
        |   ||   |    | TOGGLE:                 |        |    ||
        |   ||   |    | use_windowed_attention  |        |    ||
        |   ||   |    | If True + window=W:     |        |    ||
        |   ||   |    |   |i-j| > W --> -inf    |        |    ||
        |   ||   |    | (currently: full attn)  |        |    ||
        |   ||   |    +-------------------------+        |    ||
        |   ||   |        |                              |    ||
        |   ||   |        v                              |    ||
        |   ||   |    Dropout(0.01)                      |    ||
        |   ||   |        |                              |    ||
        |   ||   |        v                              |    ||
        |   ||   |    +-- LayerScale --------+           |    ||
        |   ||   |    | alpha_attn           |           |    ||
        |   ||   |    | (init=0.01, learned) |           |    ||
        |   ||   |    +---------------------+            |    ||
        |   ||   |        |                              |    ||
        |   ||   |    x = residual + alpha_attn * out    |    ||
        |   ||   |                                       |    ||
        |   ||   +---------------------------------------+    ||
        |   ||       |                                        ||
        |   ||   +--- SUB-BLOCK B: FEEDFORWARD -------+       ||
        |   ||   |                                    |       ||
        |   ||   |    residual = x                    |       ||
        |   ||   |        |                           |       ||
        |   ||   |        v                           |       ||
        |   ||   |    LayerNorm(128)                  |       ||
        |   ||   |        |                           |       ||
        |   ||   |        v                           |       ||
        |   ||   |    +-- AdaLN-Zero -------+         |       ||
        |   ||   |    |                     |         |       ||
        |   ||   |    | gamma = Mish -->    |<--------+       ||
        |   ||   |    |   Linear(128-->128) |  t_emb          ||
        |   ||   |    | beta  = Mish -->    |                 ||
        |   ||   |    |   Linear(128-->128) |                 ||
        |   ||   |    | (zero-initialized)  |                 ||
        |   ||   |    |                     |                 ||
        |   ||   |    | y = norm(x)         |                 ||
        |   ||   |    |   * (1 + gamma(t))  |                 ||
        |   ||   |    |   + beta(t)         |                 ||
        |   ||   |    +---------------------+                 ||
        |   ||   |        |                                   ||
        |   ||   |        v                                   ||
        |   ||   |    Linear(128 --> 512)                     ||
        |   ||   |    GELU()                                  ||
        |   ||   |    Dropout(0.01)                           ||
        |   ||   |    Linear(512 --> 128)                     ||
        |   ||   |    Dropout(0.01)                           ||
        |   ||   |        |                                   ||
        |   ||   |        v                                   ||
        |   ||   |    +-- LayerScale --------+                ||
        |   ||   |    | alpha_ff             |                ||
        |   ||   |    | (init=0.01, learned) |                ||
        |   ||   |    +---------------------+                 ||
        |   ||   |        |                                   ||
        |   ||   |    x = residual + alpha_ff * out           ||
        |   ||   |                                            ||
        |   ||   +--------------------------------------------+|
        |   ||                                                 ||
        |   +=================================================+|
        |   | (repeated 4 times, t_emb shared across blocks)   |
        |   +==================================================+
        |
        v
+-------------------+
| Output Projection |
| LayerNorm(128)    |
| Linear(128 --> 4) |
+-------------------+
        |
        v

  OUTPUT: v [batch, 32, 4]
  (velocity field)
```

## Architecture Toggles

### `use_windowed_attention` + `attention_window_size` (currently: False)

Controls the attention receptive field in `MultiheadAttention`.

- **False** (current): Full attention. Every token attends to all 32 positions. Attention cost is O(seq_len^2).
- **True** with `attention_window_size=W`: An additive mask sets attention scores to `-inf` for positions where `|i - j| > W`. Each token only attends to its local neighborhood of 2W+1 positions. Useful for longer sequences where full attention is expensive or where local context is sufficient.

## Design Notes

### AdaLN-Zero Initialization
All gamma/beta projection layers are **zero-initialized** (weights and biases set to 0). At initialization:
- `gamma(t) = 0` for all t, so `(1 + gamma) = 1` (identity scale)
- `beta(t) = 0` for all t (no shift)

This means every transformer block starts as an approximate identity function.

### LayerScale Initialization
Both `alpha_attn` and `alpha_ff` are initialized to 0.01. Combined with AdaLN-Zero, the residual contributions are near-zero at init, making the full network close to a no-op. This stabilizes training and allows deeper networks to train without divergence.

### Time Conditioning
The time embedding is a **global** signal — the same 128D vector modulates every block and every position identically. Position-specific behavior comes only from the learned positional encoding and the attention pattern.
