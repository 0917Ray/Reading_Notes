import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import numpy as np

def print_progress(step, total_steps, loss=None, extra=""):
    # Simple progress bar
    bar_width = 30
    frac = step / total_steps
    filled = int(bar_width * frac)
    bar = "=" * filled + "-" * (bar_width - filled)

    msg = f"\r[{bar}] {step:>6}/{total_steps} ({frac*100:5.1f}%)"
    if loss is not None:
        msg += f"  loss={loss:.4f}"
    if extra:
        msg += f"  {extra}"
    print(msg, end="", flush=True)

def COLORS():
    return {
    "red":   "#D17477",  # red
    "yellow":"#FFDD11",  # yellow
    "blue":  "#5289bb",  # blue
    "green": "#78bb75",  # green
}

def plot_loss(loss_list, save_path='results/result.pdf', figsize=(7, 4), color=COLORS()['red'], marker=False, title="Training Loss over Time", xlabel="Iteration", ylabel="Loss", info_text="", show_plot=True):
    steps = np.arange(1, len(loss_list) + 1)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    if marker == False:
        plt.plot(steps, loss_list, color=color, label='Training Loss', linewidth=1.5)
    else:
        plt.plot(steps, loss_list, color=color, label='Training Loss', linewidth=1.5, marker='o', markersize=6, alpha=0.9)

    ax = plt.gca()
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
    ax.tick_params(axis="both", which="both",
                top=True, right=True, bottom=True, left=True,
                labeltop=False, labelright=False,
                direction="in")
    # main ticks setting
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))  # epoch 通常用整数
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # minor ticks setting
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ticks = list(ax.get_xticks())
    if 1 not in ticks:
        ticks.append(1)
    if steps[-1] not in ticks:
        ticks.append(steps[-1])
    ticks = sorted(set(int(t) for t in ticks if t >= steps[0]))  # 去掉 <1 的，确保整数
    ax.set_xticks(ticks)

    # grid setting
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.18)

    ax.set_xlim(left=steps[0], right=steps[-1])  # left=1, right=max step
    ax.margins(x=0)

    plt.xlabel(xlabel=xlabel, fontsize=14)
    plt.ylabel(ylabel=ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.figtext(0.98, 0.02, info_text, # (0.98, 0.02) on the bottom right corner
            fontsize=10, 
            color='gray', 
            style='italic',
            horizontalalignment='right', 
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3)) # add a semi-transparent background
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, transparent=True)
    if show_plot:
        plt.show()

def plot_2_loss(train_loss_list, val_loss_list, val_steps, save_path='results/result.pdf', 
                figsize=(7, 4), color1='#2ca02c', color2='#d62728', 
                title="Training & Validation Loss", xlabel="Iteration", ylabel="Loss", 
                info_text="", show_plot=True):
    
    # 1. 处理 X 轴坐标
    # 训练集：每一迭代一步
    train_steps = np.arange(1, len(train_loss_list) + 1)
    # 验证集：使用传入的 val_steps (例如 [0, 100, 200...])
    val_steps = np.array(val_steps) + 1 # 转为从 1 开始匹配坐标轴
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    
    # 2. 绘制两条线
    plt.plot(train_steps, train_loss_list, color=color1, label='Training Loss', linewidth=1.5, alpha=0.9)
    plt.plot(val_steps, val_loss_list, color=color2, label='Validation Loss', linewidth=1.5, marker='o', markersize=6, alpha=0.9)

    # 3. 坐标轴与样式设置
    ax = plt.gca()
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
    
    ax.tick_params(axis="both", which="both", top=True, right=True, bottom=True, left=True,
                   labeltop=False, labelright=False, direction="in")
    
    # 刻度定位
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ticks = list(ax.get_xticks())
    if 1 not in ticks:
        ticks.append(1)
    if train_steps[-1] not in ticks:
        ticks.append(train_steps[-1])
    ticks = sorted(set(int(t) for t in ticks if t >= train_steps[0]))  # 去掉 <1 的，确保整数
    ax.set_xticks(ticks)

    # 确保 X 轴范围紧凑
    ax.set_xlim(left=1, right=len(train_loss_list))
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.18)

    # 4. 标签与备注
    plt.xlabel(xlabel=xlabel, fontsize=14)
    plt.ylabel(ylabel=ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.figtext(0.98, 0.02, info_text, 
            fontsize=10, 
            color='gray', 
            style='italic',
            horizontalalignment='right', 
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3)) # 加个半透明背景，防止挡住坐标轴数字
    
    plt.legend(fontsize=11, frameon=True, loc='upper right')
    plt.tight_layout()
    
    # 5. 保存与显示
    plt.savefig(save_path, transparent=False, dpi=300) # PDF 建议不透明以保证文字清晰
    if show_plot:
        plt.show()
    plt.close()