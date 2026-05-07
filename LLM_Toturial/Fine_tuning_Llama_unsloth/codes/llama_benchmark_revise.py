import json
import os

json_path = "/home/fcr/0_Learn/LLM_Learn/post_train/Fine_Tune_Llama_Unsloth/outputs_20260430_103122/checkpoint-500/benchmark_results_20260506_174100/__home__fcr__0_Learn__LLM_Learn__post_train__Fine_Tune_Llama_Unsloth__outputs_20260430_103122__checkpoint-500/results_2026-05-06T17-48-29.528001.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", {})

print("=" * 80)
print("Benchmark Results")
print("=" * 80)

for task_name, metrics in results.items():
    if task_name == "hellaswag":
        print(f"Task: {task_name} (Context Continuation and Common Sense Reasoning)")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        print("=" * 80)
    elif task_name == "mmlu":
        print(f"Task: {task_name} (Massive Multitask Language Understanding)")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        print("=" * 80)
    elif task_name == "arc_easy":
        print(f"Task: {task_name} (Scientific Reasoning)")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        print("=" * 80)
    elif task_name == "ceval-valid":
        print(f"Task: {task_name} (C-Eval Chinese Evaluation Set)")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        print("=" * 80)