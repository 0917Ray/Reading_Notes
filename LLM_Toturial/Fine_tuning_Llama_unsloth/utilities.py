import torch
import sys
import os
import psutil
import shutil
import requests
from unsloth import is_bfloat16_supported
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import numpy as np

def check_environment():
    """
    Standardized Environment Diagnostic Tool
    Focuses on: System, Disk, GPU, Connectivity, and Libraries.
    """
    line_width = 70
    print("-" * line_width)
    print("LLM TRAINING ENVIRONMENT DIAGNOSTIC REPORT")
    print("-" * line_width)

    # 1. System Information
    print("[SYSTEM]")
    print(f"{'Python Version':<30}: {sys.version.split()[0]}")
    print(f"{'System RAM':<30}: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # 2. Disk Space
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"{'Disk Free Space':<30}: {free_gb:.2f} GB")
    if free_gb < 50:
        print("[WARNING] Less than 50GB free space. Consider clearing disk for large models.")

    # 3. GPU Hardware
    print("\n[GPU HARDWARE]")
    device_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{'Device Name':<30}: {props.name}")
        print(f"{'Total VRAM':<30}: {props.total_memory / (1024**3):.2f} GB")
        print(f"{'Number of GPUs':<30}: {device_count}")
        print(f"{'Compute Capability':<30}: {props.major}.{props.minor}")
        
    else:
        print("[WARNING] No NVIDIA GPU detected.")

    # 4. CUDA Software Stack
    print("\n[CUDA SOFTWARE]")
    if torch.cuda.is_available():
        print(f"{'PyTorch CUDA Version':<30}: {torch.version.cuda}")
        print(f"{'Current VRAM Allocated':<30}: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"{'Native BF16 Support':<30}: {'Yes' if is_bfloat16_supported() else 'No (Will fallback to FP16)'}")
    else:
        print("[WARNING] No NVIDIA GPU detected.")

    # 5. Connectivity & Access
    print("\n[CONNECTIVITY]")
    # Hugging Face Connection
    try:
        requests.get("https://huggingface.co", timeout=5)
        print(f"{'HF Connection':<30}: OK")
    except Exception:
        print(f"{'HF Connection':<30}: FAILED (Check proxy/VPN)")

    # 6. HF Token check
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"{'HF Token':<30}: DETECTED")
    else:
        print(f"{'HF Token':<30}: [WARNING] MISSING (Check environment variables)")

    # 7. Core Libraries
    print("\n[LIBRARIES]")
    libs = ["unsloth", "transformers", "bitsandbytes", "xformers", "datasets", "trl", "peft"]
    for lib in libs:
        try:
            version = __import__(lib).__version__
            print(f"{lib:<30}: {version}")
        except ImportError:
            print(f"{lib:<30}: NOT INSTALLED")

    print("-" * line_width)
    print("END OF REPORT")
    print("-" * line_width + "\n")


# ===================== Plot ====================== #
def COLORS():
    return {
        "blue":   "#516480",  # blue
        "green": "#41745F",   # WHU green
        "purple": "#AE4C67",  # CityU purple
        "yellow":"#FFDD11", # yellow
        "red": "#D17477", # red
    }

def plot_lr(learning_rate_list, color="blue", save_path=None, figsize=(12, 5)):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    steps = np.arange(1, len(learning_rate_list) + 1)
    plt.figure(figsize=figsize)
    plt.plot(steps, learning_rate_list, color=color, label='Learning Rate', linewidth=2)

    ax = plt.gca()
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.tick_params(axis="both", which="both",
        top=True, right=True, bottom=True, left=True,
        labeltop=False, labelright=False,
        direction="in")
    
    # main ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # minor ticks
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ticks = list(ax.get_xticks())
    if 1 not in ticks:
        ticks.append(1)
    if steps[-1] not in ticks:
        ticks.append(steps[-1])
    ticks = sorted(set(int(t) for t in ticks if t >= steps[0]))
    ax.set_xticks(ticks)

    # grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.18)

    ax.set_xlim(left=steps[0], right=steps[-1]) # left=1, right=max step
    ax.margins(x=0)

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, transparent=True)

def plot_training_loss(loss_list, loss_step_list=None, color="blue", save_path=None, figsize=(12, 5)):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if loss_step_list is None:
        loss_step_list = list(range(1, len(loss_list) + 1))
    plt.figure(figsize=figsize)
    plt.plot(loss_step_list, loss_list, color=color, label='Training Loss', linewidth=2)

    ax = plt.gca()
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.tick_params(axis="both", which="both",
        top=True, right=True, bottom=True, left=True,
        labeltop=False, labelright=False,
        direction="in")
    
    # main ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # minor ticks
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ticks = list(ax.get_xticks())
    if 1 not in ticks:
        ticks.append(1)
    if loss_step_list[-1] not in ticks:
        ticks.append(loss_step_list[-1])
    ticks = sorted(set(int(t) for t in ticks if t >= loss_step_list[0]))
    ax.set_xticks(ticks)

    # grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.18)

    ax.set_xlim(left=loss_step_list[0], right=loss_step_list[-1]) # left=1, right=max step
    ax.margins(x=0)

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, transparent=True)

def plot_training_val_loss(train_loss_list, val_loss_list, train_steps, val_steps, train_color="green", val_color="red", save_path=None, figsize=(12, 5)):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    
    plt.figure(figsize=figsize)
    plt.plot(train_steps, train_loss_list, color=train_color, label='Training Loss', linewidth=2)
    plt.plot(val_steps, val_loss_list, color=val_color, label='Validation Loss', linewidth=2, linestyle='-' , marker='o', markersize=4)

    ax = plt.gca()
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.tick_params(axis="both", which="both",
        top=True, right=True, bottom=True, left=True,
        labeltop=False, labelright=False,
        direction="in")
    
    # main ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # minor ticks
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ticks = list(ax.get_xticks())
    if 1 not in ticks:
        ticks.append(1)
    if train_steps[-1] not in ticks:
        ticks.append(train_steps[-1])
    ticks = sorted(set(int(t) for t in ticks if t >= train_steps[0]))
    ax.set_xticks(ticks)

    # grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.18)

    ax.set_xlim(left=train_steps[0], right=train_steps[-1]) # left=1, right=max step
    ax.margins(x=0)

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, transparent=True)



if __name__ == "__main__":
    check_environment()