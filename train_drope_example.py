"""
Example training script demonstrating DRoPE usage.

This shows how to train a model that starts with RoPE and switches to NoPE
at a specified training step.
"""
import os
import torch
import time
from gpt2.data import get_data, get_batch
from gpt2.model import GPTConfig, GPT, PositionalEncoding

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 21
eval_interval = 5
learning_rate = 3e-4
device = 'cpu'  # Change to 'cuda' or 'mps' if available
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# DRoPE configuration: switch from RoPE to NoPE at step 10
drope_switch_step = 10

# ------------
torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
if __name__ == "__main__":
    # Note: This assumes you have input.txt in the current directory
    # If not, this will fail. Replace with your own data loading.
    # train_data, val_data, vocab_size, encode, decode = get_data('input.txt')
    
    # For demonstration purposes, we'll use dummy values
    print("NOTE: This is a demonstration script. Replace data loading with your actual dataset.")
    vocab_size = 65
    train_data = torch.randint(0, vocab_size, (10000,))
    val_data = torch.randint(0, vocab_size, (2000,))

    # model init with DRoPE
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        pe=PositionalEncoding.DROPE,
        drope_switch_step=drope_switch_step,
    )
    model = GPT(config)
    model.to(device)
    
    print(f"\nTraining with DRoPE:")
    print(f"  - Using RoPE for steps 0-{drope_switch_step-1}")
    print(f"  - Switching to NoPE at step {drope_switch_step}\n")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for iter in range(max_iters):
        # Update the model's current step for DRoPE
        model.set_step(iter)
        
        # Indicate when we switch from RoPE to NoPE
        if iter == drope_switch_step:
            print(f">>> Switching from RoPE to NoPE at step {iter}")

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            pe_status = "RoPE" if iter < drope_switch_step else "NoPE"
            print(f"step {iter} ({pe_status}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print(f"\nTraining took {end_time - start_time:.2f} seconds")

    # save the model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter_num': max_iters - 1,
    }
    print(f"Saving checkpoint to ckpt_drope.pt")
    torch.save(checkpoint, 'ckpt_drope.pt')
