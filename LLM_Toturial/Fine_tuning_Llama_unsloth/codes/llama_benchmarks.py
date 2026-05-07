import os
import subprocess
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ==========================================
# 1. Basic Configurations
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
lora_model_path = f"{current_dir}/outputs_20260430_103122/checkpoint-500"
base_model_path = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S") # dir with timestamp to avoid overwriting previous results
output_dir = f"{lora_model_path}/benchmark_results_{run_id}"

def run_benchmarks(peft_path, base_path, output_dir):
    print(f"[INFO] Base Model: {base_path}")
    if not os.path.isdir(peft_path):
        raise FileNotFoundError(f"PEFT Model not found: {peft_path}")
    else:
        print(f"[INFO] LoRA Weights: {peft_path}")

    os.makedirs(output_dir, exist_ok=True) # make sure the output directory exists, if not it will be created
    print(f"[INFO] Test results will be saved in: {output_dir}")

    # ==========================================
    # 2. Build Test Cases
    # ==========================================
    # Some popular evaluation tasks for LLMs
    # - arc_easy: scientific reasoning
    # - hellaswag: context continuation and common sense reasoning (English)
    # - mmlu: massive multitask language understanding (English, slow to run, can be removed as needed)
    # - ceval-valid: C-Eval Chinese evaluation set (Chinese in validation set)
    tasks = "hellaswag,mmlu"

    # Build HuggingFace model arguments, combining base + peft
    model_args = f"pretrained={base_path},peft={peft_path},tokenizer={base_path}"

    cmd = [
        "lm-eval", "run",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--device", "cuda:0", # Because CUDA_VISIBLE_DEVICES="3" is specified externally, cuda:0 maps to GPU 3
        "--batch_size", "auto:4",
        "--output_path", output_dir,
        # "--limit", "100" # Limit to 100 samples per task for quick benchmarking
    ]

    # print(f"\n[INFO] Executing command:\n{' '.join(cmd)}\n")
    print(f"\n[INFO] Running benchmark: {tasks}\n")

    # ==========================================
    # 3. Run the Benchmark Command
    # ==========================================
    try:
        # use subprocess to run the command and capture output
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] All benchmarks completed successfully!")
        print(f"[SUCCESS] Detailed metrics and JSON results have been saved to folder: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Benchmark execution failed!")
        print(f"[ERROR] Please check if lm-eval tool is correctly installed (pip install \"lm_eval[hf]\"). Error message: {e}")
    except FileNotFoundError:
        print("\n[ERROR] Command not found! Please ensure that the 'lm_eval[hf]' command is available in your PATH. You may need to install the lm_eval[hf] package using 'pip install \"lm_eval[hf]\"'.")

if __name__ == "__main__":
    run_benchmarks(lora_model_path, base_model_path, output_dir)