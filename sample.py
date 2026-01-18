
import os
import torch
from gpt2.data import get_data
from gpt2.model import GPTConfig, GPT

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to 0 probability
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# load model
if os.path.exists('ckpt.pt'):
    checkpoint = torch.load('ckpt.pt', map_location=device, weights_only=False)
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
else:
    print("No checkpoint found at ckpt.pt, using default config and random weights (init model)")
    train_data, val_data, vocab_size, encode, decode = get_data('input.txt')
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)

model.eval()
model.to(device)

# data
train_data, val_data, vocab_size, encode, decode = get_data('input.txt')

# generation
start_ids = encode('\n')
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print('---------------')
        print(decode(y[0].tolist()))
        print('---------------')
