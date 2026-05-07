import os
import json
import csv
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

# ==========================================
# 1. Basic Configurations & Model Loading
# ==========================================
@dataclass # Define a dataclass for structured generation results
class GenerationResult:
    id: int
    type: str
    instruction: str
    input: str
    response: str


max_seq_length = 2048
dtype = None
load_in_4bit = True

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
model_path = f"{current_dir}/outputs_20260430_103122/checkpoint-500"

if os.path.exists(model_path):
    print("[INFO] File exists!", os.listdir(model_path))
else:
    print("[WARNING] File does not exist. Check the path:", model_path)


print(f"[INFO] Loading model from: {model_path} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Switch to inference mode
FastLanguageModel.for_inference(model)

# ==========================================
# 2. Prepare Prompt Template (Must Match Training Format)
# ==========================================
def save_rich_results(results: List[GenerationResult], output_dir: Path):
    """Saves the generation results in multiple formats: JSON, CSV, and a Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. save as JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=4)

    # 2. save as CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type", "instruction", "input", "response"])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # 3. save as Markdown report
    md_path = output_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Model Inference Test Report\n\n")
        f.write(f"- **Test Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Model Path**: `{model_path}`\n\n")
        f.write("--- \n\n")
        for r in results:
            f.write(f"### Case {r.id}: {r.type}\n")
            f.write(f"**[Instruction]**: {r.instruction}\n\n")
            if r.input: f.write(f"**[Input]**: {r.input}\n\n")
            f.write(f"**[Response]**:\n\n{r.response}\n\n")
            f.write("---\n\n")

    print(f"\n[INFO] Report generated in directory: {output_dir}")
    print(f"  - Markdown: {md_path.name}")
    print(f"  - CSV: {csv_path.name}")
    print(f"  - JSON: {json_path.name}")

# ==========================================
# 3. Prepare Prompt Template (Must Match Training Format)
# ==========================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# ==========================================
# 4. Build Test Cases (k=6)
# ==========================================
test_instances = [
    {
        "type": "English Simple Question",
        "instruction": "What is the capital of France?",
        "input": ""
    },
    {
        "type": "Chinese Simple Question",
        "instruction": "请用通俗易懂的一句话解释什么是大语言模型.",
        "input": ""
    },
    {
        "type": "English Multiple Choice Question",
        "instruction": "Which of the following animals is a mammal?",
        "input": "A) Shark\nB) Dolphin\nC) Crocodile\nD) Turtle"
    },
    {
        "type": "Chinese Multiple Choice Question",
        "instruction": "以下哪个中国省份的简称是‘粤’？",
        "input": "A) 浙江省\nB) 广东省\nC) 福建省\nD) 海南省"
    },
    {
        "type": "Translation and Task Understanding",
        "instruction": "Translate the following sentence into French.",
        "input": "I am currently learning how to fine-tune a large language model using LoRA."
    },
    {
        "type": "Mathematical Reasoning",
        "instruction": "Continue the Fibonacci sequence.",
        "input": "1, 1, 2, 3, 5, 8"
    }
]

# ==========================================
# 5. Execute Inference Tests
# ==========================================
all_results = []
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"{model_path}/inference_results_{run_id}")
streamer = TextStreamer(tokenizer, skip_prompt=True) # Steaming output without showing the prompt

for i, item in enumerate(test_instances, 1):
    print(f"\n" + "="*50)
    print(f"Case {i}: {item['type']}")
    
    formatted_prompt = alpaca_prompt.format(item["instruction"], item["input"], "")
    inputs = tokenizer([formatted_prompt], return_tensors="pt", clean_up_tokenization_spaces=False).to("cuda")

    output_tokens = model.generate(
        **inputs, 
        streamer=streamer, 
        max_new_tokens=256, 
        max_length=None, 
        pad_token_id=tokenizer.eos_token_id
    )

    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    if "### Response:" in full_text:
        response_text = full_text.split("### Response:")[1].strip()
    else:
        response_text = full_text.strip()

    res = GenerationResult(
        id=i,
        type=item["type"],
        instruction=item["instruction"],
        input=item["input"],
        response=response_text
    )
    all_results.append(res)
# ==========================================
# 6. Save Results in Multiple Formats (JSON, CSV, Markdown)
# ==========================================
save_rich_results(all_results, output_dir)