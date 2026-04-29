# Step 1: create environment
# $ conda create -n llama python=3.10 -y
# $ conda activate llama

# Step 2: install PyTorch
# for CUDA 12.8:
# $ pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# for CUDA 13.0:
# $ pip3 install torch torchvision

# Step 3: install Unsloth
# $ pip install unsloth

# Step 4: install other dependencies
# $ pip install -r requirements.txt

# 1. Environment Preparation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Only use GPU 3 for training
os.environ["WANDB_DISABLED"] = "true"     # Disable Weights & Biases logging for cleaner output

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import is_bfloat16_supported
from utilities import check_environment, plot_lr, plot_training_loss, plot_training_val_loss, COLORS
import time
from datetime import datetime
import json

# check_environment() # Check environment before training

# 2. Model Loading
# 2.1 Basic Configurations
max_seq_length = 2048      # Supports RoPE Scaling internally
dtype          = None      # None for auto detection (Float16/BFloat16)
load_in_4bit   = True      # Use 4-bit quantization to reduce VRAM usage

# 2.2 Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", # Start with 3B for practice
    # model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", # Switch to 70B for actual fine-tuning
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# 3. LoRA Adapter
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,               # Rank: controls the number of trainable parameters
    target_modules = [    # The parts of the model we want to fine-tune
        "q_proj", "k_proj", "v_proj", "o_proj", # for attention layers
        "gate_proj", "up_proj", "down_proj",    # for feed-forward layers
    ],
    lora_alpha = 16,      # Scaling factor for LoRA updates
    lora_dropout = 0,     # For performance, recommended to 0
    bias = "none",        # For performance, set to none
    use_gradient_checkpointing = "unsloth", # Extremely important: enabling this can significantly save GPU memory
    random_state = 3407,
    use_rslora = False,   # Rank-stable LoRA,
    loftq_config = None,  # Quantization initialization configuration
)

# 4. Dataset Preparation
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# 4.1 Quick Data Inspection
# raw_example = dataset[1]

# print(raw_example, "\n")
# print(f"Example type: {type(raw_example)}", "\n")

# for key, value in raw_example.items():
#     print(f"{key:<15}: {value}")


# 4.2 Prompt Formatting
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = [alpaca_prompt.format(instruction, input, output) + EOS_TOKEN for instruction, input, output in zip(instructions, inputs, outputs)]
    return {"text": texts}

dataset = dataset.map(formatting_prompts, batched = True)
train_val = dataset.train_test_split(test_size=0.03, seed=3407)
train_dataset = train_val["train"]
eval_dataset = train_val["test"]
print(f"[INFO] Train dataset size: {len(train_dataset)}")
print(f"[INFO] Eval dataset size: {len(eval_dataset)}")

# print(f"Type: {type(dataset[1])}", "\n")
# print(dataset[1]["text"])

# 5. Training
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_output_dir = f"{current_dir}/outputs_{run_id}"
max_steps = 20

if not os.path.exists(unique_output_dir):
    os.makedirs(unique_output_dir)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2, # Number of CPU cores used for data loading
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4, # Number of parallel data per GPU
        gradient_accumulation_steps = 8, # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        warmup_steps = 5,
        max_steps = max_steps, # Total number of training steps
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(), # Use FP16 if BFloat16 is not supported
        bf16 = is_bfloat16_supported(),
        logging_steps = 1, # Log training loss every 1 step
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear", # or "cosine" etc.
        seed = 3407,
        eval_strategy = "steps",
        # evaluation_strategy="steps", # for older versions of transformers
        eval_steps=10, # Evaluate every 20 steps
        save_strategy="steps",
        save_steps=20, # Save checkpoint every 20 steps
        output_dir = unique_output_dir,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False, # because we want to minimize the loss
        # report_to = "wandb", # Upload the loss curve to the cloud
    ),
)

timer_start = time.time()
trainer_stats = trainer.train()
timer_end = time.time()
training_time = timer_end - timer_start
print(f"[INFO] Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")


# Training Metrics
gpu_stats = torch.cuda.get_device_properties(0)
print(f"[INFO] GPU = {gpu_stats.name}. Max memory = {round(gpu_stats.total_memory / 1024 ** 3, 3)} GB.")
print(f"[INFO] {round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)} GB of memory reserved.")

# 6. Save Model

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# 7. Plotting
COLORS = COLORS()
json_path = f"{unique_output_dir}/checkpoint-{max_steps}/trainer_state.json"

with open(json_path, "r", encoding="utf-8") as f:
    state_data = json.load(f)

history = state_data["log_history"]

train_steps = [entry["step"] for entry in history if "loss" in entry]
train_losses = [entry["loss"] for entry in history if "loss" in entry]

learning_rates = [entry["learning_rate"] for entry in history if "learning_rate" in entry]

eval_steps = [entry["step"] for entry in history if "eval_loss" in entry]
eval_losses = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

plot_lr(learning_rates, color=COLORS["purple"], save_path=f"{unique_output_dir}/figures/learning_rate_schedule.pdf")
plot_training_loss(train_losses, train_steps, color=COLORS["green"], save_path=f"{unique_output_dir}/figures/training_loss_curves.pdf")
plot_training_val_loss(train_losses, val_loss_list=eval_losses, train_steps=train_steps, val_steps=eval_steps, train_color=COLORS["green"], val_color=COLORS["purple"], save_path=f"{unique_output_dir}/figures/train_val_loss_curves.pdf")
print(f"[INFO] Figures have been saved to {unique_output_dir}/figures/")