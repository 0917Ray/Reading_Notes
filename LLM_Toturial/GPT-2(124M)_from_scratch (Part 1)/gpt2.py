from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------- #
@dataclass # GPT-2 config
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 50257 # GPT-2's vocab size: 50,000 BPE tokens + 256 bytes tokens + 1 <|EOS|> token
    n_layer:    int = 12 # number of transformer blocks
    n_head:     int = 12
    n_embed:    int = 768
# ---------------------------------------------------------------------- #

class CausalSelfAttention(nn.Module): # a more efficient implementation of self-attention using matrix multiplication and masking
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0, "embedding dimension must be divisible by number of heads"
        # conbine the projection of query, key, value into a single linear layer for efficiency
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x) # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embed, dim=2) # (B, T, C) each
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, nh, T, T)
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side (B, T, C)
        y = self.c_proj(y) # (B, T, C)
        return y


class MLP(nn.Module): # FeedForward part

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed*4),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.n_embed*4, config.n_embed),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config) # multi-head attention
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config) # feed-forward

    # use pre-LayerNorm architecture, which is more stable than post-LayerNorm
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # residual connection with layer norm
        x = x + self.mlp(self.ln2(x))  # residual connection with layer norm
        return x

class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # skeleton of GPT-2:
        self.Transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # token embedding table
            wpe = nn.Embedding(config.block_size, config.n_embed), # position embedding table
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # n_layer * transformer blocks
            ln_f = nn.LayerNorm(config.n_embed), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # language modeling head, project to vocab_size

    def forward(self, idx, target=None):
        # idx is of shape (B, T) where B is batch size and T is sequence length
        B, T = idx.size()
        assert T<=self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # calfully choose device for position embedding to avoid device mismatch error
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.Transformer.wpe(pos) # (T, n_embed)
        tok_emb = self.Transformer.wte(idx) # (B, T, n_embed)
        x = tok_emb + pos_emb # (B, T, n_embed)

        for block in self.Transformer.h:
            x = block(x) # (B, T, n_embed)
        x = self.Transformer.ln_f(x) # (B, T, n_embed), final layer norm
        logits = self.lm_head(x) # (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T) # or target = target.reshape(-1)

            loss = F.cross_entropy(logits, target)
        return logits, loss
    
# ---------------------------------------------------------------------- #
# output test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

num_return_sequences = 3
max_length = 20

model = GPT2(GPTconfig())
model.eval()
model.to(device) # ***||***

import tiktoken
encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode("Halo, I am a language model,")
tokens = torch.tensor(tokens, dtype=torch.long, device=device)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, sequence_length)
x = tokens.to(device) # ***||***

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x) # (num_return_sequences, sequence_length, vocab_size)
        logits = logits[:, -1, :] # (num_return_sequences, vocab_size), get the last time step's logits
        probs = F.softmax(logits, dim=-1) # (num_return_sequences, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1) # (num_return_sequences, 1)
        x = torch.cat((x, next_token), dim=1) # append the new token to the sequence
print("Generated sequences:")
for i in range(num_return_sequences):
    generated_tokens = x[i].tolist()
    generated_text = encoder.decode(generated_tokens)
    print(f"{i+1}: {generated_text}")


