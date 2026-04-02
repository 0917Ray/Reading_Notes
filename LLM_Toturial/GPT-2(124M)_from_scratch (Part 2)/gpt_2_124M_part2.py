import os, time, requests
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
from utilits import plot_loss, plot_2_loss, COLORS

# --- Configuration ---
batch_size = 32
block_size = 1024
learning_rate = 1e-4
max_iters = 30
eval_interval = 1 # Evaluate every eval_interval iterations
eval_iters = 1 # Number of validation batches to evaluate at each interval
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 50304 # origianlly 50257, 50304 for nice number
    n_layer:    int = 12
    n_head:     int = 12
    n_embed:    int = 768

# --- Data Preparation ---
base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
data_path = os.path.join(base_dir, 'data/input.txt')

if not os.path.exists(data_path): # Download the dataset if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    text_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(data_path, 'w') as f:
        f.write(requests.get(text_url).text)

with open(data_path, 'r', encoding='utf-8') as f: # read the text file
    text = f.read()

encoder = tiktoken.get_encoding("gpt2") # Converts text to tokens
all_tokens = encoder.encode(text)
n = int(0.9 * len(all_tokens))
train_data = torch.tensor(all_tokens[:n], dtype=torch.long)
val_data = torch.tensor(all_tokens[n:], dtype=torch.long)
# print(f"{'training data:':<20} {len(train_data)} {'tokens':>10}, {('validation data:'):<20} {len(val_data)} {'tokens':>10}")

def get_batch(data_source='train'): # Data loading function
    data = train_data if data_source == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- Model Definition ---
class FlashAttention(nn.Module): # Multi-head self-attention using Flash Attention
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = True # Custom initialization for the output projection layer
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size() # (batch size, sequence length, embedding dimension)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2) # (B, T, 3*C) -> (B, T, C) each
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash Attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module): # Feedforward network with GELU activation
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module): # Transformer block consisting of self-attention and feedforward layers with residual connections
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed) # Pre-LayerNorm
        self.attn = FlashAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed) # Pre-LayerNorm
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # Residual connection
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT2(nn.Module): # The main GPT-2 model class
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embed), # Positional Embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer blocks
            ln_f = nn.LayerNorm(config.n_embed), # Final LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # Language Modeling Head (output logits)
        self.transformer.wte.weight = self.lm_head.weight # Weight Tying
        self.apply(self._init_weights) # Custom weight initialization

    def _init_weights(self, module): # Custom weight initialization:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos) # Token + Positional Embeddings
        for block in self.transformer.h: x = block(x) # Pass through Transformer blocks
        x = self.transformer.ln_f(x) # Final LayerNorm
        logits = self.lm_head(x) # Output logits
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    # --- Training Preparation ---
    torch.set_float32_matmul_precision('high') # Enable mixed precision for faster training on compatible hardware
    model = GPT2(GPTconfig()).to(device)
    if device == 'cuda':
        model = torch.compile(model)
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params:.1f}M params | Device: {device_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    train_losses, val_losses = [], []
    val_iters_idx = [] # To track the iteration numbers at which validation losses are recorded
    start_time = time.time()

    for i in range(max_iters):
        t0 = time.time()

        # evaluate the model on validation set at regular intervals
        if i % eval_interval == 0 or i == max_iters - 1:
            model.eval()
            with torch.no_grad():
                v_losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    Xv, Yv = get_batch('val')
                    _, v_loss = model(Xv, Yv)
                    v_losses[k] = v_loss.item()
                current_val_loss = v_losses.mean().item()
                val_losses.append(current_val_loss)
                val_iters_idx.append(i)
            model.train()

        xb, yb = get_batch('train')
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # Mixed precision for faster training
            logits, loss = model(xb, yb)
        train_losses.append(loss.item())
        
        optimizer.zero_grad(set_to_none=True) # Clear gradients
        loss.backward() # Backpropagation
        optimizer.step() # Update model parameters
        
        dt = (time.time() - t0) * 1000  # Convert to milliseconds
        
        if i % eval_interval == 0 or i == max_iters - 1:
            throughput = (batch_size * block_size) / dt * 1000  # Tokens per second
            print(f"Step {i:5d} | "
            f"Loss: {loss.item():8.4f} | "
            f"Val Loss: {val_losses[-1]:8.4f} | "
            f"Time: {dt:8.2f}ms | "
            f"Throughput: {throughput:8.1f} tok/s")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Finished. Total time: {total_duration:.2f}s")

    # --- Plotting Loss Curves ---
    os.makedirs(os.path.join(base_dir, 'final_outputs'), exist_ok=True) # Ensure output directory exists
    meta_info = f"Device: {device_name}, Time: {total_duration:.1f}s, Train Loss: {loss.item():.4f}, Val Loss: {val_losses[-1]:.4f}"

    # (i) Training Loss Curve
    path_train = os.path.join(base_dir, 'final_outputs', f'NanoGPT_{num_params:.1f}M_{max_iters}iterations_{batch_size}batch_{block_size}block_train_only.pdf')
    plot_loss(train_losses, save_path=path_train, figsize=(18, 6), color=COLORS()['green'], 
              title=f"NanoGPT({num_params:.1f}M) Training Loss Curve ({max_iters} iterations)", info_text=meta_info)

    # (ii) Training vs Validation Loss Curve
    path_compare = os.path.join(base_dir, 'final_outputs', f'NanoGPT_{num_params:.1f}M_{max_iters}iterations_{batch_size}batch_{block_size}block_compare.pdf')
    plot_2_loss(train_losses, val_losses, val_iters_idx, save_path=path_compare,  figsize=(18, 6),
                 color1=COLORS()['green'], color2=COLORS()['red'],
                 title=f"NanoGPT({num_params:.1f}M) Training vs Validation Loss ({max_iters} iterations)", info_text=meta_info)

    # --- Save the trained model ---
    os.makedirs(os.path.join(base_dir, 'saved_models'), exist_ok=True) # Ensure output directory exists
    model_save_path = os.path.join(base_dir, 'saved_models', f'NanoGPT_{num_params:.1f}M_{max_iters}iterations.pth')
    torch.save(model.state_dict(), model_save_path)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': GPTconfig,
    }
    torch.save(checkpoint, model_save_path)
    print(f"\n[INFO] Model saved to {model_save_path}")
