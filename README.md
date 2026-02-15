# LLM sandbox

## Positional Encoding Methods

Implemented positional encoding variants:
- **LEARNED** - Standard learned positional embeddings (GPT-2 style)
- **SINUSOIDAL** - Fixed sinusoidal positional encoding (Transformer original)
- **ROPE** - Rotary Position Embedding (RoFormer)
- **NOPE** - No positional encoding
- **DROPE** - Dynamic RoPE: train with RoPE, then switch to NoPE at a specified step

### Using DRoPE

DRoPE allows you to start training with RoPE for better positional awareness, then switch to NoPE:

```python
from gpt2.model import GPTConfig, GPT, PositionalEncoding

config = GPTConfig(
    pe=PositionalEncoding.DROPE,
    drope_switch_step=100  # Switch from RoPE to NoPE at step 100
)
model = GPT(config)

# During training, update the step
for step in range(max_steps):
    model.set_step(step)
    # ... training loop ...
```

## TODO
- Add perplexity on the train dataset
- Train with NoPE
- Train with RoPE ✓
- Train with sinusoidal
- Train with RoPE then NoPE (DRoPE) ✓
