#!/usr/bin/env python3
"""生成 A/B 拼图 SNIPS/logs/training-curves-ab.png：

  A（左）training reward —— 训练过程本身（每步 4 prompt × 16 rollout 的平均奖励，50 步滚动）
  B（右）test error rate —— 判据指标（每 200 步一个检查点评 test 700 条；第 0 步=训练前模型 100%）

两图共用同一套颜色与 x 轴（0–2000 训练步），方便一眼对上"训练信号"和"判据指标"。

数据：SNIPS/logs/<模型>/curves.csv（A）、SNIPS/logs/nm-sweep.csv（B）、
      SNIPS/logs/evaluate-summary.txt 里 RandomInit 的错误率（B 的起点，缺失时按 100%）。

用法（仓库根目录）：
  python scripts/plot_ab_curves.py
"""
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SNIPS", "logs")
MODELS = ["ATLAS", "SPHERE",
          "FIXA", "GRPO-baseline"]
COLORS = {
    "ATLAS": "#1f77b4",
    "SPHERE": "#ff7f0e",
    "FIXA": "#2ca02c",
    "GRPO-baseline": "#d62728",
}
INIT_ERROR_FALLBACK = 100.0


def load_reward(name):
    path = os.path.join(LOGS, name, "curves.csv")
    if not os.path.isfile(path):
        return None
    steps, roll = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            roll.append(float(row["roll50"]))
    return steps, roll


def init_error():
    path = os.path.join(LOGS, "evaluate-summary.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("RandomInit"):
                    m = re.search(r"ERROR_RATE=([\d.]+)%", line)
                    if m:
                        return float(m.group(1))
    except OSError:
        pass
    return INIT_ERROR_FALLBACK


def load_error():
    data = {m: ([0], [init_error()]) for m in MODELS}
    with open(os.path.join(LOGS, "nm-sweep.csv")) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m not in data:
                continue
            data[m][0].append(int(row["step"].removeprefix("step")))
            data[m][1].append(100.0 - float(row["intent_acc"]))
    return data


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)

    for name in MODELS:
        d = load_reward(name)
        if d is None:
            continue
        ax_a.plot(d[0], d[1], linewidth=1.5, color=COLORS[name], label=name)
    ax_a.set_title("A   training reward (rolling mean, 50)", fontsize=12)
    ax_a.set_xlabel("training step")
    ax_a.set_ylabel("mean reward")
    ax_a.set_xlim(0, 2100)
    ax_a.set_ylim(0, 1.0)
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="lower right", fontsize=9)

    errs = load_error()
    for name in MODELS:
        steps, err = errs[name]
        ax_b.plot(steps, err, marker="o", markersize=3.5, linewidth=1.4,
                  color=COLORS[name], label=name)
    ax_b.axhline(4.8, linestyle="--", linewidth=0.9, color="gray", alpha=0.8)
    ax_b.text(2090, 4.8, "4.8% criterion", fontsize=8, color="gray",
              va="bottom", ha="right")
    ax_b.text(40, init_error() - 5, "untrained model\n100% error", fontsize=9,
              color="#444444", va="top", ha="left")
    ax_b.set_title("B   test error rate (checkpoint every 200)", fontsize=12)
    ax_b.set_xlabel("training step")
    ax_b.set_ylabel("test error rate (%)")
    ax_b.set_xlim(0, 2100)
    ax_b.set_ylim(0, 105)
    ax_b.set_xticks(range(0, 2001, 400))
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="center right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(LOGS, "training-curves-ab.png")
    fig.savefig(out)
    print("[plot] A/B ->", out)


if __name__ == "__main__":
    main()
