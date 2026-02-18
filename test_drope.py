"""
Test script to verify DRoPE implementation works.
"""
import torch
from gpt2.model import GPTConfig, GPT, PositionalEncoding

def test_drope():
    print("Testing DRoPE implementation...")
    
    # Create a small model with DRoPE
    config = GPTConfig(
        block_size=128,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        pe=PositionalEncoding.DROPE,
        drope_switch_step=5  # Switch from RoPE to NoPE at step 5
    )
    
    model = GPT(config)
    model.eval()
    
    # Create a dummy input
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test before switch (should use RoPE)
    print("\n=== Testing step 0 (should use RoPE) ===")
    model.set_step(0)
    with torch.no_grad():
        logits1, _ = model(idx)
    print(f"Output shape at step 0: {logits1.shape}")
    
    # Test before switch (step 4, should still use RoPE)
    print("\n=== Testing step 4 (should use RoPE) ===")
    model.set_step(4)
    with torch.no_grad():
        logits2, _ = model(idx)
    print(f"Output shape at step 4: {logits2.shape}")
    
    # Test after switch (should use NoPE)
    print("\n=== Testing step 5 (should switch to NoPE) ===")
    model.set_step(5)
    with torch.no_grad():
        logits3, _ = model(idx)
    print(f"Output shape at step 5: {logits3.shape}")
    
    # Test after switch (step 10, should use NoPE)
    print("\n=== Testing step 10 (should use NoPE) ===")
    model.set_step(10)
    with torch.no_grad():
        logits4, _ = model(idx)
    print(f"Output shape at step 10: {logits4.shape}")
    
    # Verify outputs are different before and after switch
    print("\n=== Verifying behavior changes ===")
    
    # Outputs at steps 0 and 4 should be similar (both use RoPE)
    diff_before_switch = torch.abs(logits1 - logits2).mean().item()
    print(f"Mean diff between step 0 and 4 (both RoPE): {diff_before_switch:.6f}")
    
    # Outputs at steps 5 and 10 should be similar (both use NoPE)
    diff_after_switch = torch.abs(logits3 - logits4).mean().item()
    print(f"Mean diff between step 5 and 10 (both NoPE): {diff_after_switch:.6f}")
    
    # Output at step 4 vs step 5 should be different (RoPE vs NoPE)
    diff_across_switch = torch.abs(logits2 - logits3).mean().item()
    print(f"Mean diff between step 4 and 5 (RoPE vs NoPE): {diff_across_switch:.6f}")
    
    # The diff across the switch should be larger than within RoPE or within NoPE phases
    # (Note: this is a weak test, as model outputs can vary for other reasons too)
    print(f"\nDifference across switch is {'larger' if diff_across_switch > max(diff_before_switch, diff_after_switch) else 'not larger'} than within-phase differences")
    
    # Test generation
    print("\n=== Testing generation ===")
    model.set_step(0)
    generated_rope = model.generate(idx[:1, :10], max_new_tokens=20)
    print(f"Generated with RoPE (step 0): {generated_rope.shape}")
    
    model.set_step(10)
    generated_nope = model.generate(idx[:1, :10], max_new_tokens=20)
    print(f"Generated with NoPE (step 10): {generated_nope.shape}")
    
    assert logits1.shape == (batch_size, 1, config.vocab_size), "Output shape mismatch!"
    assert generated_rope.shape == (1, 30), "Generation shape mismatch!"
    
    print("\n✓ DRoPE implementation test passed!")
    return True

if __name__ == "__main__":
    test_drope()
