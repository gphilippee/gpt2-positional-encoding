
import os
import torch
from gpt2.data import get_data
from gpt2.model import GPTConfig, GPT

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
num_samples = 1 # number of samples to draw
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# data
train_data, val_data, vocab_size, encode, decode = get_data('input.txt')

# load model
model_path = 'ckpt_20.pt'
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # usually needed if training with DistributedDataParallel
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print(f"Checkpoint found at {model_path}")
else:
    print("No checkpoint found at ckpt.pt, using default config and random weights (init model)")
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)

model.eval()
model.to(device)

val_data = val_data.to(device)

# run perplexity
with torch.no_grad():
    for k in range(num_samples):
        perplexity = model.perplexity(val_data[None, :])
        print('---------------')
        print(perplexity)
        print('---------------')
