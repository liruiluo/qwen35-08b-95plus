#!/usr/bin/env python3
"""从 training-log.jsonl 重建 curves.csv（列：step, mean_reward, roll50）。

训练器（methods/sphere_grpo.py、methods/rl_interp.py）每步往 training-log.jsonl
写一条 JSON，里面有 step 和 mean_reward。这里把这两列取出来，再补一列
roll50＝最近 50 步 mean_reward 的均值，供 scripts/plot_songce_curves.py 画曲线。

用法（仓库根目录）：
  python scripts/make_curves_csv.py                            # 全部已有 log 的模型
  python scripts/make_curves_csv.py SNIPS/logs/GRPO-baseline   # 只做一个目录
"""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS = os.path.join(ROOT, "SNIPS", "logs")
WINDOW = 50


def rows_from_log(log_path):
    """training-log.jsonl -> [(step, mean_reward, roll50), ...]"""
    steps, rewards = [], []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            steps.append(int(rec["step"]))
            rewards.append(float(rec["mean_reward"]))
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    rows, window = [], []
    for i in order:
        window.append(rewards[i])
        seg = window[-WINDOW:]
        rows.append((steps[i], rewards[i], sum(seg) / len(seg)))
    return rows


def write_curves_csv(model_dir):
    log_path = os.path.join(model_dir, "training-log.jsonl")
    if not os.path.isfile(log_path):
        return None
    rows = rows_from_log(log_path)
    out_path = os.path.join(model_dir, "curves.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["step", "mean_reward", "roll50"])
        w.writerows(rows)
    return out_path


def main(argv):
    targets = argv[1:]
    if not targets:
        targets = [
            os.path.join(LOGS, name)
            for name in sorted(os.listdir(LOGS))
            if os.path.isfile(os.path.join(LOGS, name, "training-log.jsonl"))
        ]
    for d in targets:
        out = write_curves_csv(d)
        if out is None:
            print(f"[skip] {d}：没有 training-log.jsonl")
        else:
            print(f"[csv] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main(sys.argv)
