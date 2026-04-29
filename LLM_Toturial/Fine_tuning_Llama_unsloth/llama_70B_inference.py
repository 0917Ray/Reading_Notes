# Environment setup
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only use GPU 2 for training
os.environ["WANDB_DISABLED"] = "true"     # Disable Weights & Biases logging for cleaner output

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported
import time
from datetime import datetime
import json

model_path = "/home/fcr/LLM Learn/Post-training/fine-tune_Llama-3.3-70B_unsloth/outputs_20260429_135712/checkpoint-60"
if os.path.exists(model_path):
    print("[INFO] File exists!", os.listdir(model_path))
else:
    print("[WARNING] File does not exist. Check the path:", model_path)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # 指向你保存的文件夹
    max_seq_length = 2048,   # 与训练时保持一致
    load_in_4bit = True,     # 既然是 70B，建议继续用 4bit 加载以节省显存
)

# Prompt for inference
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Inference
FastLanguageModel.for_inference(model) # Switch to inference mode
# instruction = "Continue the Fibonacci sequence."
# input_content = "1, 1, 2, 3, 5, 8"
instruction = "Describe the structure of an atom."
input_content = ""
prompt = alpaca_prompt.format(instruction, input_content, "")
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
print(f"{'='*30} PROMPT {'='*30}")
print(prompt)

# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             "Continue the Fibonacci sequence.",
#             "1, 1, 2, 3, 5, 8",
#             "",
#         )
#     ],
#     return_tensors="pt",
# ).to("cuda")


outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True,
                        repetition_penalty=1.2, # Penalize repetition to encourage more diverse outputs
                        temperature=0.7,           # Control randomness in generation (lower is more deterministic)
                        )
decoded_output = tokenizer.batch_decode(outputs)
response = decoded_output[0].split("### Response:")[1]
print(f"{'='*30} RESPONSE {'='*30}")
print(response)

# Streaming Inference
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256,
#                     repetition_penalty=1.2, # Penalize repetition to encourage more diverse outputs
#                     temperature=0.7,           # Control randomness in generation (lower is more deterministic)
#                     )