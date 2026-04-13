# For single GPU: python gpt_2_124M_final.py
# For multi-GPU (e.g., 4 GPUs): 
# $ cd "/home/fcr/0_Learn/LLM_Learn/GPT-2(124M)_from_scratch"
# $ torchrun --standalone --nproc_per_node=4 gpt2_train.py

# Note: --standalone is used for single-node multi-GPU training.

# For debugging:
# import sys; sys.exit(0) # Debugging

# Run on tumx:
# $ tmux new -s xxx
# $ torchrun --standalone --nproc_per_node=4 gpt2_train.py
# $ tmux ls
# $ tmux attach -t xxx
# =================== =================== =================== =================== #
import os, time, requests
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
import inspect
from utilits import plot_loss, plot_2_loss, COLORS

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import json
# =================== =================== =================== =================== #
# --- Configuration ---
batch_size = 16
block_size = 1024
max_iters = 10
eval_interval = 1 # Evaluate every eval_interval iterations
eval_iters = 2 # Number of validation batches to evaluate at each interval
device, device_name = 'cuda' if torch.cuda.is_available() else 'cpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
betas = (0.9, 0.95)
lr_max = 6e-4
lr_min = lr_max * 0.1
warm_steps = 715
max_steps = 19073
@dataclass
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 50304 # origianlly 50257, 50304 for nice number
    n_layer:    int = 12
    n_head:     int = 12
    n_embed:    int = 768
# =================== =================== =================== =================== #
# --- Data Preparation ---
base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
def load_tokens(filename):
    """
    Utility function to load pre-tokenized data from a .npy file and convert it to a PyTorch tensor.
    """
    numpytokens = np.fromfile(filename, dtype=np.uint16) # Pickle allowed for loading arrays of objects (e.g., arrays of different lengths)
    # pytorchtokens = torch.tensor(numpytokens, dtype=torch.long)
    return torch.from_numpy(numpytokens.astype(np.int64))

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val'], "split must be 'train' or 'val'"
        
        # get the shared data tensor for this process
        data_root = os.path.join(base_dir, "data/FineWeb-edu-10BT")
        shards = os.listdir(data_root) # List all files in the data directory
        shards = sorted([s for s in shards if split in s]) # Sorted filter for .npy files
        shards = [os.path.join(data_root, s) for s in shards] # Full paths to the .npy files
        self.shards = shards # Store the list of shard paths
        assert len(shards) > 0, f"[WARNING] No data shards found for split '{split}' in {data_root}"
        if master_process:
            print(f"[MASTER INFO] Found {len(shards)} data shards for split '{split}':")
            
        self.current_shard_idx = 0
        self.tokens = load_tokens(shards[self.current_shard_idx]) # Load the first shard
        self.current_position = self.B * self.T * self.process_rank # Start position for this process's first batch

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard_idx])
            self.current_position = self.B * self.T * self.process_rank

        return x.to(device), y.to(device)

    def reset(self):
        self.current_shard_idx = 0
        self.tokens = load_tokens(self.shards[self.current_shard_idx])
        self.current_position = self.B * self.T * self.process_rank

def get_lr(step):
    if step < warm_steps:
        return lr_max * (step+1) / warm_steps
    if step > max_steps:
        return lr_min
    decay_ratio = (step - warm_steps) / (max_steps - warm_steps)
    assert 0 <= decay_ratio <= 1, "Decay ratio should be between 0 and 1"
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + torch.cos(torch.tensor(decay_ratio * math.pi)))

import json

@torch.no_grad()
def evaluate_hellaswag(model, device, enc, file_path, num_samples=2000):
    model.eval()
    num_correct = 0
    num_total = 0
    if not os.path.exists(file_path):
        return 0.0
        
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples: break
            example = json.loads(line)
            
            # 渲染数据
            ctx_tokens = enc.encode(example["ctx"])
            tok_rows = [ctx_tokens + enc.encode(" " + end) for end in example["endings"]]
            mask_rows = [[0]*len(ctx_tokens) + [1]*len(enc.encode(" " + end)) for end in example["endings"]]
            
            max_len = max(len(row) for row in tok_rows)
            tokens = torch.zeros((4, max_len), dtype=torch.long, device=device)
            mask = torch.zeros((4, max_len), dtype=torch.long, device=device)
            for j, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
                tokens[j, :len(tok_row)] = torch.tensor(tok_row)
                mask[j, :len(mask_row)] = torch.tensor(mask_row)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(tokens)
            
            # 计算 Acc_norm
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tokens = tokens[:, 1:].contiguous()
            shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                          shift_tokens.view(-1), reduction='none').view(4, -1)
            shift_mask = mask[:, 1:].contiguous()
            avg_loss = (shift_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
            
            if avg_loss.argmin().item() == example["label"]:
                num_correct += 1
            num_total += 1
            
    model.train()
    return num_correct / num_total
# =================== =================== =================== =================== #
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {name: param for name, param in self.named_parameters()}
        param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}
        decay_params = [p for name, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for name, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        if master_process:
            print(f"[MASTER INFO] Decayed parameter tensors: {len(decay_params)} with {num_decay/1e6:.1f}M parameters")
            print(f"[MASTER INFO] Non-decayed parameter tensors: {len(no_decay_params)} with {num_no_decay/1e6:.1f}M parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # Check if fused AdamW is available in the current PyTorch version
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"[MASTER INFO] Using fused AdamW optimizer: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=[0.9, 0.95], eps=1e-8, fused=use_fused)
        return optimizer

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
# =================== =================== =================== =================== #
# --- Training Loop ---
if __name__ == "__main__":
    # --- Distributed Data Parallel (DDP) ---
    ddp = int(os.environ.get('RANK', -1)) != -1 # Check if we're in a distributed environment
    if ddp:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        init_process_group(backend='nccl') # Initialize the process group for ddp
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        if master_process:
            print(f"[MASTER INFO] Distributed training initialized with {ddp_world_size} processes")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Single process training on device: {device}")

    # --- Gradient Accumulation ---
    total_batch_size = 524288 # 2**19, ~0.5M tokens
    assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0, "Total batch size must be divisible by (batch_size * block_size * ddp_world_size)"
    acc_steps = total_batch_size // (batch_size * block_size * ddp_world_size)
    if master_process:
        print(f"[MASTER INFO] Batch size: {batch_size} | Block size: {block_size} | Gradient Accumulation Steps: {acc_steps} | Paralleled GPUs: {ddp_world_size}")
        print(f"[MASTER INFO] Total batch size: {batch_size} * {block_size} * {ddp_world_size} * {acc_steps} = {total_batch_size} tokens")

    # --- Training Preparation ---
    torch.manual_seed(1337+ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337+ddp_rank)

    torch.set_float32_matmul_precision('high') # Enable mixed precision for faster training on compatible hardware
    model = GPT2(GPTconfig()).to(device)
    model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    if master_process:
        print(f"[MASTER INFO] Model Size: {num_params:.1f}M params")
    

    # Data Loaders for training and validation
    train_loader = DataLoader(batch_size, block_size, ddp_rank, ddp_world_size, split='train')
    val_loader = DataLoader(batch_size, block_size, ddp_rank, ddp_world_size, split='val')

    # optimizer = torch.optim.AdamW(model.parameters(), betas=betas, eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=lr_max, device=device)

    # --- Training Loop ---
    train_losses, val_losses = [], []
    val_iters_idx = [] # To track the iteration numbers at which validation losses are recorded
    lrs = [] # To track learning rates
    hswag_accs = []
    hswag_steps = []
    encoder = tiktoken.get_encoding("gpt2")
    hswag_file = os.path.join(base_dir, "data/hellaswag/hellaswag_validation.jsonl")
    start_time = time.time()

    for step in range(max_iters):
        t0 = time.time()

        # evaluate the model on validation set at regular intervals
        if step % eval_interval == 0 or step == max_iters - 1:
            model.eval()
            val_loader.reset()  # Reset the validation loader to start from the beginning of the validation data
            with torch.no_grad():
                val_loss_accum = torch.zeros(1, device=device)
                for k in range(eval_iters):
                    Xv, Yv = val_loader.next_batch()
                    Xv, Yv = Xv.to(device), Yv.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        _, v_loss = model(Xv, Yv)
                    val_loss_accum += v_loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    current_val_loss = val_loss_accum / eval_iters
                    val_losses.append(current_val_loss.item())
                    val_iters_idx.append(step)
                    hswag_acc = evaluate_hellaswag(raw_model, device, encoder, hswag_file, num_samples=200)
                    hswag_accs.append(hswag_acc)
                    hswag_steps.append(step)
            model.train()

        optimizer.zero_grad(set_to_none=True) # Clear gradients
        train_loss_accum = torch.zeros(1, device=device)
        for micro_step in range(acc_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == acc_steps - 1) # Only sync gradients on the last micro step
                
            xb, yb = train_loader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16): # Mixed precision for faster training
                logits, loss = model(xb, yb)
            train_loss_accum += loss.detach() # Accumulate loss for logging (detach to avoid backprop through the entire history)
            (loss / acc_steps).backward() # Backpropagation
        if ddp:
            dist.all_reduce(train_loss_accum, op=dist.ReduceOp.SUM)
        if master_process:
            train_loss_accum /= (acc_steps * ddp_world_size)
            train_losses.append(train_loss_accum.item())

        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        lr = get_lr(step)
        lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step() # Update model parameters
        
        dt = (time.time() - t0) * 1000  # Convert to milliseconds
        
        if step % eval_interval == 0 or step == max_iters - 1:
            throughput = (batch_size * block_size) / dt * 1000  # Tokens per second
            if master_process:
                print(f"Step {step:5d} | "
                f"Loss: {train_loss_accum.item():8.4f} | "
                f"Val Loss: {val_losses[-1]:8.4f} | "
                f"Time: {dt:8.2f}ms | "
                f"Throughput: {throughput:8.1f} tok/s | "
                f"LR: {lr:.2e} | "
                f"HellaSwag Acc: {hswag_acc:.4f}")

    end_time = time.time()
    total_duration = end_time - start_time
    if master_process:
        print(f"[MASTER INFO] Finished. Total time: {total_duration:.2f}s")

    # --- Plotting Loss Curves ---
    if master_process:
        os.makedirs(os.path.join(base_dir, 'final_outputs'), exist_ok=True) # Ensure output directory exists
        meta_info = f"Device: {device_name}, Time: {total_duration:.1f}s, Train Loss: {loss.item():.4f}, Val Loss: {val_losses[-1]:.4f}"

        # (1) Training Loss Curve
        path_train = os.path.join(base_dir, 'final_outputs', f'GPT2_{num_params:.1f}M_{max_iters}iterations_{batch_size}batch_{block_size}block_train_only.pdf')
        plot_loss(train_losses, save_path=path_train, figsize=(18, 6), color=COLORS()['green'], 
                title=f"GPT2({num_params:.1f}M) Training Loss Curve ({max_iters} iterations on FineWeb-10BT)", info_text=meta_info)

        # (2) Training vs Validation Loss Curve
        path_compare = os.path.join(base_dir, 'final_outputs', f'GPT2_{num_params:.1f}M_{max_iters}iterations_{batch_size}batch_{block_size}block_compare.pdf')
        plot_2_loss(train_losses, val_losses, val_iters_idx, save_path=path_compare,  figsize=(18, 6),
                    color1=COLORS()['green'], color2=COLORS()['red'],
                    title=f"GPT2({num_params:.1f}M) Training vs Validation Loss ({max_iters} iterations on FineWeb-10BT)", info_text=meta_info)

        # (3) HellaSwag Acc Curve
        path_hellaswag = os.path.join(base_dir, 'final_outputs', f'GPT2_{num_params:.1f}M_{max_iters}iterations_{batch_size}batch_{block_size}block_hellaswag.pdf')
        plot_loss(hswag_accs, save_path=path_hellaswag, figsize=(18, 6), color=COLORS()['green'], marker = True,
                title=f"GPT2({num_params:.1f}M) HellaSwag Accuracy Curve ({max_iters} iterations on FineWeb-10BT)", info_text=meta_info)

        # (4) Learning Rate Schedule
        path_lr = os.path.join(base_dir, 'final_outputs', f'GPT2_{num_params:.1f}M_{max_iters}iterations_{batch_size}batch_{block_size}block_lr_schedule.pdf')
        plot_loss(lrs, save_path=path_lr, figsize=(18, 6), color=COLORS()['blue'], title=f"GPT2({num_params:.1f}M) Learning Rate Schedule ({max_iters} iterations on FineWeb-10BT)", xlabel="Iteration", ylabel="Learning Rate", info_text=meta_info)

        # --- Save the trained model ---
        if master_process:
            os.makedirs(os.path.join(base_dir, 'saved_models'), exist_ok=True) # Ensure output directory exists
            model_save_path = os.path.join(base_dir, 'saved_models', f'GPT2_{num_params:.1f}M_{max_iters}iterations.pth')
            torch.save(model.state_dict(), model_save_path)

            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'config': raw_model.config,
            }
            torch.save(checkpoint, model_save_path)
            print(f"\n[MASTER INFO] Model saved to {model_save_path}")

        if ddp:
            destroy_process_group() # Clean up the process group after training