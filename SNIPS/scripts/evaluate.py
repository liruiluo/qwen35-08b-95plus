#!/usr/bin/env python3
"""SNIPS/scripts/evaluate.py —— 测试大纲 4.2.2.4（3）指定的评测入口。

遍历 SNIPS/dataset/test.jsonl 的 700 条测试样本，依次将 5 个模型
（训练前模型、GRPO 基线模型、三种 SPHERE 算法模型）输入测试样本，
计算各模型在测试数据上的错误率（错误率 = 1 − 7 类意图分类准确率）。

用法（在仓库根目录）：
  python SNIPS/scripts/evaluate.py                     # 全部 5 个模型
  python SNIPS/scripts/evaluate.py --only RandomInit   # 只测训练前模型
  python SNIPS/scripts/evaluate.py --limit 20          # 冒烟（前 20 条）

前置条件：
  - SNIPS/models/<名称>/ 下已放置模型权重（见 SNIPS/models/MANIFEST.md
    的 sha256 与获取方式；未放置的模型会被跳过并标注 PENDING）；
  - GPU 环境装有 torch / transformers / sglang（见仓库根 README 的
    Environment 一节）。引擎未启动时本脚本会自动拉起。

判定（测试大纲 4.2.2.5）：
  A = 训练前模型的错误率（预期 100%）；B = 各算法训练后模型的错误率。
  合格判据 B < A×5% = 5%；高效性判据：错误率达到 5% 所需训练步数
  N（SPHERE 各臂）< M（GRPO 基线），见 SNIPS/logs/ 的训练曲线。
"""
import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT, "SNIPS", "models")
DATASET = os.path.join(ROOT, "SNIPS", "dataset", "test.jsonl")
LOGS_DIR = os.path.join(ROOT, "SNIPS", "logs")

# 大纲 4.2.2.2：测试对象 = 训练前模型 + 基线算法模型 + 三种 SPHERE 算法模型
MODELS = ["RandomInit", "GRPO-baseline",
          "SPHERE-Last-Adaptive", "SPHERE-All-Adaptive", "SPHERE-Last-Fixed"]
SPHERE_MODELS = {"SPHERE-Last-Adaptive", "SPHERE-All-Adaptive", "SPHERE-Last-Fixed"}


def engine_up(url, timeout=2):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def launch_engine(model_dir, port):
    """冷启动一个 sglang 引擎（引擎会以该模型权重起服务并等健康检查通过）。"""
    script = os.path.join(ROOT, "engine", "launch_engine.sh")
    subprocess.run(["bash", script, model_dir, str(port)], check=True)


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + "\n" + p.stderr + "\n")
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine-url", default="http://127.0.0.1:30001")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--only", choices=MODELS, help="只评测这一个模型")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = ap.parse_args()

    todo = [args.only] if args.only else MODELS

    # 1) 检查各模型权重是否就位（weights 不入 git，见 MANIFEST.md）
    ready, pending = [], []
    for name in todo:
        d = os.path.join(MODELS_DIR, name)
        if os.path.isfile(os.path.join(d, "model.safetensors")):
            ready.append(name)
        else:
            pending.append(name)

    # 2) 引擎：未启动则用第一个就位模型冷启动
    if ready and not engine_up(args.engine_url):
        print(f"[engine] not up, launching with {ready[0]} on port {args.port} ...")
        launch_engine(os.path.join(MODELS_DIR, ready[0]), args.port)
        assert engine_up(args.engine_url), "engine failed to become healthy"

    # 3) 逐模型：加载权重 → 遍历测试样本 → 解析错误率
    rows = []
    for name in ready:
        mdir = os.path.join(MODELS_DIR, name)
        out = os.path.join(LOGS_DIR, name, "eval-test.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"\n=== {name} ===")
        run([sys.executable, os.path.join(ROOT, "methods", "push_ckpt.py"),
             args.engine_url, mdir])
        cmd = [sys.executable, os.path.join(ROOT, "SNIPS", "eval_snips.py"),
               "--engine-url", args.engine_url, "--model", mdir,
               "--data", DATASET, "--out", out]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        stdout = run(cmd)
        print(stdout.strip())
        m = re.search(r"INTENT_ACC=([\d.]+)%", stdout)
        acc = float(m.group(1)) if m else float("nan")
        rows.append({"model": name, "intent_acc": acc,
                     "error_rate": round(100 - acc, 2) if m else None})

    # 4) 汇总（对齐大纲 4.2.2.5 的合格判据口径）
    print("\n===== SNIPS test-set error rates (700 utterances) =====")
    print(f"{'model':<24}{'intent acc':>12}{'error rate':>12}  note")
    for r in rows:
        note = ""
        if r["model"] == "RandomInit":
            note = ("A（训练前，判据预期错误率 100%）"
                    if r["error_rate"] and abs(r["error_rate"] - 100) < 1e-9
                    else "训练前模型错误率 != 100%，请核查")
        elif r["model"] in SPHERE_MODELS and r["error_rate"] is not None:
            note = "B ≤ 5% 合格" if r["error_rate"] <= 5.0 else "B > 5% 不合格"
        err = "—" if r["error_rate"] is None else f"{r['error_rate']:.2f}%"
        print(f"{r['model']:<24}{r['intent_acc']:>11.2f}%{err:>12}  {note}")
    for name in pending:
        print(f"{name:<24}{'—':>12}{'—':>12}  PENDING（权重未放置，见 models/MANIFEST.md）")

    summary = os.path.join(LOGS_DIR, "evaluate-summary.txt")
    with open(summary, "w") as f:
        f.write(f"generated={time.strftime('%Y-%m-%d %H:%M:%S')} limit={args.limit}\n")
        for r in rows:
            f.write(f"{r['model']}\tINTENT_ACC={r['intent_acc']:.2f}%\t"
                    f"ERROR_RATE={r['error_rate']:.2f}%\n")
        for name in pending:
            f.write(f"{name}\tPENDING\n")
    print(f"\n[summary] written to {summary}")


if __name__ == "__main__":
    main()
