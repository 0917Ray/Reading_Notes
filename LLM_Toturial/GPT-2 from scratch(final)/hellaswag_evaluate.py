import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os
import json
from tqdm import tqdm
from gpt2_train import GPT2, GPTconfig

# --- 1. 配置与路径 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/home/fcr/0_Learn/LLM_Learn/GPT-2(124M)_from_scratch/saved_models/GPT2_124.5M_20000iterations.pth'
hellaswag_path = '/home/fcr/0_Learn/LLM_Learn/GPT-2(124M)_from_scratch/data/hellaswag/hellaswag_validation.jsonl'

# --- 2. 加载模型 ---
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Can not find model: {model_path}")

torch.serialization.add_safe_globals([GPTconfig])
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
config = checkpoint['config']
model = GPT2(config)

state_dict = checkpoint['model_state_dict']
# 核心：处理 torch.compile 产生的 _orig_mod. 以及 DDP 的 module. 前缀
state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[INFO] Model loaded from {model_path}. Starting HellaSwag evaluation...")

# --- 3. HellaSwag 数据渲染逻辑 ---
encoder = tiktoken.get_encoding("gpt2")

def render_example(example):
    """
    将 HellaSwag 样本渲染为 4 个候选项的 tokens 和 mask
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]
    
    ctx_tokens = encoder.encode(ctx)
    tok_rows, mask_rows = [], []
    
    for end in endings:
        # GPT-2 续写通常需要在开头加空格
        end_tokens = encoder.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        
    # Padding 到当前 4 个选项中的最大长度
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
        
    return tokens.to(device), mask.to(device), label

# --- 4. 评估循环 ---
num_correct = 0
num_total = 0

if not os.path.exists(hellaswag_path):
    raise FileNotFoundError(f"HellaSwag data not found at: {hellaswag_path}")

# 读取所有行进行评测
with open(hellaswag_path, "r") as f:
    lines = f.readlines()

for i, line in enumerate(tqdm(lines, desc="Evaluating HellaSwag")):
    example = json.loads(line)
    tokens, mask, label = render_example(example)
    
    with torch.no_grad():
        # 这里类似推断，一次传入 (4, T) 的批次
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(tokens) # (4, T, V)
        
        # --- 计算逻辑 ---
        # 1. Shift logits 和 tokens 对应自回归预测位置
        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = tokens[:, 1:].contiguous()
        
        # 2. 计算平铺后的 CrossEntropy (不进行 reduction)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
        shift_losses = shift_losses.view(4, -1)
        
        # 3. 使用 mask 只提取“回答部分”的 Loss
        shift_mask = mask[:, 1:].contiguous()
        masked_loss = shift_losses * shift_mask
        
        # 4. 计算每个选项的平均负对数似然 (Acc_norm)
        sum_loss = masked_loss.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        
        # 5. 选取 Loss 最小（概率最大）的作为预测
        pred = avg_loss.argmin().item()
        
        if pred == label:
            num_correct += 1
        num_total += 1

# --- 5. 打印最终结果 ---
acc = num_correct / num_total
print("\n" + "="*50)
print(f"HELLASWAG RESULTS")
print(f"Model: {os.path.basename(model_path)}")
print(f"Total Samples: {num_total}")
print(f"Accuracy (norm): {acc:.4f}")
print("="*50)