#!/usr/bin/env python3
"""生成 SNIPS/logs/ 下各模型的训练曲线图（curves.csv -> curves.png），
并输出全部模型的对比总图 curves-all.png。

用法（仓库根目录）：
  python scripts/plot_songce_curves.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SNIPS", "logs")
MODELS = ["RandomInit", "GRPO-baseline", "SPHERE-Last-Adaptive",
          "SPHERE-All-Adaptive", "SPHERE-Last-Fixed"]


def load(name):
    path = os.path.join(LOGS, name, "curves.csv")
    if not os.path.isfile(path):
        return None
    steps, rew, roll = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            rew.append(float(row["mean_reward"]))
            roll.append(float(row["roll50"]))
    return steps, rew, roll


def main():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    plotted = 0
    for name in MODELS:
        d = load(name)
        if d is None:
            continue
        steps, rew, roll = d
        # 单模型图
        f1, a1 = plt.subplots(figsize=(8, 4.5), dpi=150)
        a1.plot(steps, rew, color="#9db8d6", linewidth=0.8, alpha=0.7,
                label="mean reward")
        a1.plot(steps, roll, color="#2b5d8a", linewidth=1.8, label="rolling mean (50)")
        a1.set_xlabel("step")
        a1.set_ylabel("training reward")
        a1.set_title(name)
        a1.grid(alpha=0.3)
        a1.legend(loc="lower right")
        f1.tight_layout()
        out = os.path.join(LOGS, name, "curves.png")
        f1.savefig(out)
        plt.close(f1)
        # 对比总图
        ax.plot(steps, roll, linewidth=1.6, label=name)
        plotted += 1
        print(f"[plot] {name}: {len(steps)} steps -> {out}")
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("training reward (rolling mean, 50)")
        ax.set_title("Training curves")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        out = os.path.join(LOGS, "curves-all.png")
        fig.savefig(out)
        print(f"[plot] overview ({plotted} models) -> {out}")


if __name__ == "__main__":
    main()
