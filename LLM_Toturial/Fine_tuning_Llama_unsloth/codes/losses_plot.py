import json
from utilities import plot_lr, plot_training_loss, plot_training_val_loss, COLORS
import os
from datetime import datetime

# All you need is to set the right log_dir and json_path
COLORS = COLORS()
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
log_dir = f"{current_dir}/outputs_20260430_134416/checkpoint-1000"
json_path = f"{log_dir}/trainer_state.json"
plot_id = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(json_path, "r", encoding="utf-8") as f:
    state_data = json.load(f)

history = state_data["log_history"]

train_steps = [entry["step"] for entry in history if "loss" in entry]
train_losses = [entry["loss"] for entry in history if "loss" in entry]

learning_rates = [entry["learning_rate"] for entry in history if "learning_rate" in entry]

eval_steps = [entry["step"] for entry in history if "eval_loss" in entry]
eval_losses = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

# info_text_1 = f"Total Steps: {train_steps[-1]}"
# info_text_2 = f"Final Training Loss: {train_losses[-1]:.4f}, Eval Loss: {eval_losses[-1]:.4f}, Total Steps: {train_steps[-1]}"
plot_lr(learning_rates, color=COLORS["purple"], save_path=f"{log_dir}/figures/learning_rate_schedule_{plot_id}.pdf")
plot_training_loss(train_losses, train_steps, color=COLORS["green"], save_path=f"{log_dir}/figures/training_loss_curves_{plot_id}.pdf")
plot_training_val_loss(train_losses, val_loss_list=eval_losses, train_steps=train_steps, val_steps=eval_steps, train_color=COLORS["green"], val_color=COLORS["purple"], save_path=f"{log_dir}/figures/train_val_loss_curves_{plot_id}.pdf")
print(f"[INFO] Figures have been saved to {log_dir}/figures/")