#!/usr/bin/env python3
"""把逐检查点的评测结果写成 training-error-log.jsonl（每个模型一份，放在它的 logs 目录里）。

数据源：results/nm-sweep.csv（scripts/nm_sweep.sh 产出：每 200 步一个检查点，
逐个加载进引擎、在 test 700 条上评一次意图准确率）。

输出：SNIPS/logs/<模型>/training-error-log.jsonl，每行一个检查点：
  {"step": 200, "split": "test", "n": 700, "intent_acc": 95.29, "error_rate": 4.71}

这样"训练曲线"和"训练日志"就落在同一个目录里：training-log.jsonl 记训练内信号，
training-error-log.jsonl 记判据指标。

用法（仓库根目录）：
  python scripts/make_eval_log.py                       # 用默认 results/nm-sweep.csv
  python scripts/make_eval_log.py path/to/nm-sweep.csv
"""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS = os.path.join(ROOT, "SNIPS", "logs")
DEFAULT_SWEEP = os.path.join(ROOT, "SNIPS", "logs", "nm-sweep.csv")
SPLIT = "test"
N_TEST = 700


def step_number(raw):
    raw = str(raw).strip()
    return int(raw[4:]) if raw.startswith("step") else int(raw)


def main(argv):
    sweep = argv[1] if len(argv) > 1 else DEFAULT_SWEEP
    per_model = {}
    with open(sweep, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_model.setdefault(row["model"], []).append(
                (step_number(row["step"]), float(row["intent_acc"]))
            )
    if not per_model:
        print(f"[none] {sweep} 里没有数据")
        return 1
    for model, rows in sorted(per_model.items()):
        out_dir = os.path.join(LOGS, model)
        if not os.path.isdir(out_dir):
            print(f"[skip] {model}：没有 logs 目录")
            continue
        rows.sort()
        out_path = os.path.join(out_dir, "training-error-log.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for step, acc in rows:
                rec = {
                    "step": step,
                    "split": SPLIT,
                    "n": N_TEST,
                    "intent_acc": round(acc, 2),
                    "error_rate": round(100.0 - acc, 2),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[log] {os.path.relpath(out_path, ROOT)}  {len(rows)} 个检查点")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
