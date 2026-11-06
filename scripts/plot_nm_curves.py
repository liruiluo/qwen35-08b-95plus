#!/usr/bin/env python3
"""生成 test 错误率-训练步数曲线（两张）：

  SNIPS/logs/error-rate-curves.png        横轴从 0 步（训练前模型，错误率 100%）画起
  SNIPS/logs/error-rate-curves-zoom.png   放大 2.5–5.5%，看 4.8% 判据线的穿越位置

数据 = SNIPS/logs/nm-sweep.csv（scripts/nm_sweep.sh 逐检查点扫描）：
每 200 步保存的检查点加载权重后跑一次 test 集（700 条）得到的意图准确率，
换算成错误率画曲线；第 0 步用训练前模型（RandomInit）的实测错误率，四条曲线同一起点。

用法（仓库根目录）：
  python scripts/plot_nm_curves.py
"""
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SNIPS", "logs")
MODELS = ["SPHERE-Last-Adaptive", "SPHERE-All-Adaptive",
          "SPHERE-Last-Fixed", "GRPO-baseline"]
THRESHOLDS = [(4.8, "4.8% criterion")]
INIT_ERROR_FALLBACK = 100.0


def init_error():
    """训练前模型（RandomInit）的实测错误率；从 evaluate-summary.txt 读，读不到按 100%。"""
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


def load():
    start = init_error()
    data = {m: ([0], [start]) for m in MODELS}  # 第 0 步 = 训练前模型
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


def draw(data, ylim, filename, y_ticks, note=None, skip_step0=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        steps, errs = data[m]
        if skip_step0:
            steps, errs = steps[1:], errs[1:]
        ax.plot(steps, errs, marker="o", markersize=3.5, linewidth=1.4, label=m)
    for y, label in THRESHOLDS:
        ax.axhline(y, linestyle="--", linewidth=0.9, color="gray", alpha=0.8)
        ax.text(0.995, y, label, fontsize=8, color="gray", va="bottom",
                ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("training steps (checkpoint every 200)")
    ax.set_ylabel("test error rate (%)")
    ax.set_xlim(0, 2150)
    ax.set_xticks(range(0, 2001, 200))
    ax.set_ylim(*ylim)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right" if not skip_step0 else "upper right", fontsize=9)
    if note:
        ax.annotate(note, xy=(1.5, init_error()), xytext=(0.07, 0.88),
                    textcoords="axes fraction", fontsize=9, color="#444444",
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0,
                                    shrinkB=6))
    fig.tight_layout()
    out = os.path.join(LOGS, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] error-rate vs step ->", out)


def main():
    data = load()
    # 全局：从第 0 步（训练前模型，错误率约 100%）一路降到个位数
    draw(data, (0, 105), "error-rate-curves.png", [0, 20, 40, 60, 80, 100],
         note="untrained model: error rate 100%")
    # 放大：只看 200 步以后、2.5–5.5%，判据线穿越位置一目了然
    draw(data, (2.5, 5.5), "error-rate-curves-zoom.png",
         [2.5, 3.0, 3.5, 4.0, 4.5, 4.8, 5.0, 5.5], skip_step0=True)


if __name__ == "__main__":
    main()
