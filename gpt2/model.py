
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from enum import Enum
import math

class PositionalEncoding(Enum):
    LEARNED = "learned"
    ROPE = "rope"
    NOPE = "nope"
    SINUSOIDAL = "sinusoidal"

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65 # default for tiny shakespeare
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit faster and better
    pe: PositionalEncoding = PositionalEncoding.NOPE

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, config: GPTConfig):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, block_size, n_embd)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is torch.arange(0, t) (t)
        # output is (1, t, n_embd)
        return self.pe[:, :len(x), :]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    
    Applies rotary embeddings to query and key tensors.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len):
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (B, nh, T, hs)
            k: Key tensor of shape (B, nh, T, hs)
            seq_len: Sequence length
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim)
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        # Initialize RoPE if using rotary positional embeddings
        if config.pe == PositionalEncoding.ROPE:
            head_dim = config.n_embd // config.n_head
            self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len=config.block_size)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply RoPE if using rotary positional embeddings
        if self.config.pe == PositionalEncoding.ROPE:
            q, k = self.rope(q, k, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )

        match config.pe:
            case PositionalEncoding.LEARNED:
                modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
            case PositionalEncoding.SINUSOIDAL:
                modules['wpe'] = SinusoidalPositionalEncoding(config)
            case PositionalEncoding.ROPE | PositionalEncoding.NOPE:
                pass

        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, full: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        if self.config.pe in (PositionalEncoding.ROPE, PositionalEncoding.NOPE):
            emb = tok_emb
        else:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd) or (1, t, n_embd)
            emb = tok_emb + pos_emb
        
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif full:
            logits = self.lm_head(x)
            # not targets, loss to None
            loss = None
        else:
            # inference-only optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def perplexity(self, idx: torch.Tensor) -> float:
        sum_nll = 0
        total = 0
        for start in range(0, idx.size(1)-1, self.config.block_size):
            end = min(idx.size(-1)-1, start+self.config.block_size)
            targets = idx[:, start+1:end+1]
            idx_cond = idx[:, start:end]
            logits, _ = self(idx_cond, full=True)
            probs = F.softmax(logits, dim=-1)
            target_probs = probs.gather(-1, targets.unsqueeze(-1))
            target_log_probs = target_probs.squeeze().log()
            nll = -target_log_probs
            sum_nll += nll.sum().item()
            total += nll.size(0)
        perplexity = math.exp(sum_nll / total)
        return perplexity

