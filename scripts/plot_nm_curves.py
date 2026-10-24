#!/usr/bin/env python3
"""生成 SNIPS/logs/error-rate-curves.png：test 错误率-训练步数曲线。

数据 = SNIPS/logs/nm-sweep.csv（scripts/nm_sweep.sh 逐检查点扫描）：
每 200 步保存的检查点加载权重后跑一次 test 集（700 条）得到的
意图准确率，换算成错误率画曲线，并画出 5% / 4.8% / 4.5% 三条判据线。

用法（仓库根目录）：
  python scripts/plot_nm_curves.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SNIPS", "logs")
MODELS = ["SPHERE-Last-Adaptive", "SPHERE-All-Adaptive",
          "SPHERE-Last-Fixed", "GRPO-baseline"]
THRESHOLDS = [(5.0, "5% (outline)"), (4.8, "4.8%"), (4.5, "4.5%")]


def load():
    data = {m: ([], []) for m in MODELS}
    with open(os.path.join(LOGS, "nm-sweep.csv")) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m not in data:
                continue
            step = int(row["step"].removeprefix("step"))
            err = 100.0 - float(row["intent_acc"])
            data[m][0].append(step)
            data[m][1].append(err)
    return data


def main():
    data = load()
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        steps, errs = data[m]
        ax.plot(steps, errs, marker="o", markersize=3.5, linewidth=1.4, label=m)
    for y, label in THRESHOLDS:
        ax.axhline(y, linestyle="--", linewidth=0.9, color="gray", alpha=0.8)
        ax.text(2040, y, label, fontsize=8, color="gray", va="center")
    ax.set_xlabel("training steps (checkpoint every 200)")
    ax.set_ylabel("test error rate (%)")
    ax.set_xlim(0, 2300)
    ax.set_ylim(2, 6)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(LOGS, "error-rate-curves.png")
    fig.savefig(out, dpi=150)
    print("[plot] error-rate vs step ->", out)


if __name__ == "__main__":
    main()
