
import os
import torch

def get_data(file_path='input.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Mapping from characters to integers and vice versa
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode

def get_batch(data, block_size, batch_size, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
