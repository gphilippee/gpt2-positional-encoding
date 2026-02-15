# LLM sandbox

## Positional Encoding Methods

Implemented positional encoding variants:
- **LEARNED** - Standard learned positional embeddings (GPT-2 style)
- **SINUSOIDAL** - Fixed sinusoidal positional encoding (Transformer original)
- **ROPE** - Rotary Position Embedding (RoFormer)
- **NOPE** - No positional encoding

## TODO
- Add perplexity on the train dataset
- Train with NoPE
- Train with RoPE ✓
- Train with sinusoidal
- Train with RoPE then NoPE (DRoPE)
