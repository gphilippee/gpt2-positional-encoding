
import os
import torch
import time
from gpt2.data import get_data, get_batch
from gpt2.model import GPTConfig, GPT

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 21
eval_interval = 5
learning_rate = 3e-4
device = 'mps'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

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
    train_data, val_data, vocab_size, encode, decode = get_data('input.txt')

    # model init
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(config)
    model.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # save the model
            model_path = f"ckpt_{iter}.pt"
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'iter_num': iter,
            }
            print(f"saving checkpoint to {model_path}")
            torch.save(checkpoint, model_path)

        # sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds")

    # save the model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter_num': iter,
    }
    print(f"saving checkpoint to ckpt.pt")
    torch.save(checkpoint, 'ckpt.pt')
