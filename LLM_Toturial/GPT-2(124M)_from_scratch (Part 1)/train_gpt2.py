import requests, time, os
import torch
import torch.nn as nn
from torch.nn import functional as F
from utilits import plot_loss, COLORS
import tiktoken
from dataclasses import dataclass

batch_size = 64
block_size = 256
learning_rate = 1e-4
max_iters = 1000 # max_iters = 5000
eval_interval = 100 # eval_interval = 500
eval_iters = 200 # eval_iters = 200
# ---------------------------------------------------------------------- #
@dataclass # GPT-2 config
class GPTconfig:
    block_size: int = 1024 # 1024
    vocab_size: int = 50257 # GPT-2's vocab size: 50,000 BPE tokens + 256 bytes tokens + 1 <|EOS|> token
    n_layer:    int = 12 # 12 # number of transformer blocks
    n_head:     int = 12 # 12 # number of attention heads
    n_embed:    int = 768 # 768 # embedding dimension
# ---------------------------------------------------------------------- #

torch.manual_seed(1337) # set random seed for reproducibility

# download dataset  (tiny shakespeare dataset)
if '__file__' in globals(): # only works if run as a python script
    base_dir = os.path.dirname(__file__) # for python script
else:
    base_dir = os.getcwd() # for jupyter notebook
input_file_path = os.path.join(base_dir, 'data/input.txt') # build local path
if not os.path.exists(input_file_path):
    text_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(text_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode(text)
n = int(0.9*len(tokens))
train_data = torch.tensor(tokens[:n], dtype=torch.long) # first 90% for training
val_data = torch.tensor(tokens[n:], dtype=torch.long)   # last 10% for validation

print(f"train_data has {len(train_data)} tokens, val_data has {len(val_data)} tokens")

# data loading function
def get_batch(data_source='train'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if data_source == 'train' else val_data
    idx = torch.randint(len(data)-block_size, (batch_size,)) # randomly pick batch_size starting points for the chunks
    x = torch.stack([data[i:i+block_size] for i in idx]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]) # (batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x, y

# estimate loss on train and val sets, to avoid noisy estimates
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for data_source in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_source)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[data_source] = losses.mean()
    model.train()
    return out

class CausalSelfAttention(nn.Module): # a more efficient implementation of self-attention using matrix multiplication and masking
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0, "embedding dimension must be divisible by number of heads"
        # conbine the projection of query, key, value into a single linear layer for efficiency
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # scale factor for initialization of c_proj weights, to prevent early saturation of softmax
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
        self.c_fc = nn.Linear(config.n_embed, config.n_embed*4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

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

        # weight tying: wte and lm_head
        self.Transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights) # initialize weights on all modules and their submodules

    def _init_weights(self, module): # initialize weights with normal distribution, and biases with zeros
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # scale down std for deeper models to prevent early saturation of softmax
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    current_device = torch.cuda.current_device() # device index
    device_name = torch.cuda.get_device_name(current_device) # device name
    total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9 # total memory in GB
else:
    device_name = "CPU"
    total_memory = 0
print(f"Using device: {device}")
model = GPT2(GPTconfig()).to(device)

# print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f'The GPT2 model has {num_params:,} parameters')
num_params_M = num_params / 1e6

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_list = []
start_time = time.time()
for iter in range(max_iters):
    # every eval_interval steps, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
end_time = time.time()
print(f'\nTraining completed in {end_time - start_time:.2f} seconds on {device}')
print(f'Final training loss: {loss.item():.4f}\n')
save_path = os.path.join(base_dir, 'outputs', f'NanoGPT({num_params_M:.1f}M)_loss_curve_{max_iters}iterations.pdf')
plot_loss(loss_list, save_path=save_path, figsize=(18, 6), color = COLORS()['green'], 
          title=f'Training Loss Curve of NanoGPT({num_params_M:.1f}M) over {max_iters} Iterations', 
          xlab='Iteration', ylab='Loss', 
          info_text=f'Device: {device_name}, Total Time: {end_time - start_time:.2f} s, Final Loss: {loss.item():.4f}', 
          show_plot=True)

# ---------------------------------------------------------------------- #
# output test
num_return_sequences = 3
max_length = 20

model = GPT2(GPTconfig())
model.eval()
model.to(device) # ***||***

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