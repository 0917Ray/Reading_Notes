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

def plot_loss(loss_list, save_path='results/result.pdf', figsize=(7, 4), color = COLORS()['red'], show_plot=True):
    steps = np.arange(1, len(loss_list) + 1)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.plot(steps, loss_list, color=color, label='Training Loss', linewidth=1.5)

    ax = plt.gca()
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
    ax.tick_params(axis="both", which="both",
                top=True, right=True, bottom=True, left=True,
                labeltop=False, labelright=False,
                direction="in")
    # 主刻度设置
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))  # epoch 通常用整数
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # 次刻度设置
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ticks = list(ax.get_xticks())
    if 1 not in ticks:
        ticks.append(1)
    if steps[-1] not in ticks:
        ticks.append(steps[-1])
    ticks = sorted(set(int(t) for t in ticks if t >= steps[0]))  # 去掉 <1 的，确保整数
    ax.set_xticks(ticks)

    # 网格（主/次都画一点，便于读数）
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.18)

    ax.set_xlim(left=steps[0], right=steps[-1])  # left=1, right=max step
    ax.margins(x=0)             # 去掉默认留白

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, transparent=True)
    if show_plot:
        plt.show()