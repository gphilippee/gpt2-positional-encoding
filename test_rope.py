"""
Quick test script to verify RoPE implementation works.
"""
import torch
from gpt2.model import GPTConfig, GPT, PositionalEncoding

def test_rope():
    print("Testing RoPE implementation...")
    
    # Create a small model with RoPE
    config = GPTConfig(
        block_size=128,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        pe=PositionalEncoding.ROPE
    )
    
    model = GPT(config)
    model.eval()
    
    # Create a dummy input
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(idx)
    
    print(f"Input shape: {idx.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: ({batch_size}, 1, {config.vocab_size})")
    
    assert logits.shape == (batch_size, 1, config.vocab_size), "Output shape mismatch!"
    
    # Test generation
    generated = model.generate(idx[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    print(f"Expected: ({1}, {10 + 20})")
    
    assert generated.shape == (1, 30), "Generation shape mismatch!"
    
    print("✓ RoPE implementation test passed!")
    return True

if __name__ == "__main__":
    test_rope()
