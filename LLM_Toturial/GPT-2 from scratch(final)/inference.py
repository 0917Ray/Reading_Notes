import torch
import tiktoken
import os
from gpt2_train import GPT2, GPTconfig

# --- 1. 配置与路径 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/home/fcr/0_Learn/LLM_Learn/GPT-2(124M)_from_scratch/saved_models/GPT2_124.5M_20000iterations.pth'

# --- 推理参数 ---
num_samples = 3      # 一次生成几条句子 (k)
max_gen_tokens = 50  # 每条句子的长度
temperature = 1.0    # 1.0 = 原样; >1.0 更混乱; <1.0 更保守
top_k = 50           # Top-K 采样
prompt = "Hello, I am a language, "

# --- 2. 加载模型 ---
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Can not find model: {model_path}")

torch.serialization.add_safe_globals([GPTconfig])
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
config = checkpoint['config']
# config = checkpoint['config'](vocab_size=50257)
model = GPT2(config)

state_dict = checkpoint['model_state_dict']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[INFO] Model loaded. Generating {num_samples} samples...")

# --- 3. 执行并行生成 ---
encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode(prompt)
# 将输入 tokens 复制 num_samples 份，形成批处理 (B, T)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

with torch.no_grad():
    for _ in range(max_gen_tokens):
        # 裁剪上下文
        x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
        
        logits, _ = model(x_cond)
        # 只取最后一个时间步 (B, T, V) -> (B, V)
        logits = logits[:, -1, :] / temperature
        
        # Top-K 过滤
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 计算概率并采样
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # 拼接结果
        x = torch.cat((x, next_token), dim=1)

# --- 4. 打印结果 ---
print("\n" + "="*50)
print(f"PROMPT: {prompt}")
print("="*50)

for i in range(num_samples):
    generated_text = encoder.decode(x[i].tolist())
    print(f"\n[SAMPLE {i+1}]:")
    print(generated_text)
    print("-" * 30)