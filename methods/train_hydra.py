#!/usr/bin/env python3
"""Hydra 渲染层：把 configs/songce/<车>.yaml 渲染成 methods/rl_interp.py 的
精确 argv 并执行。训练器本体零改动——argv 恒等即行为恒等；config 文件
成为逐车可 diff 的配方证据。

用法（工作目录 = 仓库根 / pod 的 /root/train95）：
  python methods/train_hydra.py -cn all_adaptive             # 训练（先打印 TRAIN_ARGV）
  python methods/train_hydra.py -cn grpo_baseline run=false  # 只打印 argv 不执行
"""
import os
import shlex
import subprocess
import sys

import hydra

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@hydra.main(config_path="../configs/songce", config_name=None, version_base="1.2")
def main(cfg):
    argv = [
        os.path.join(ROOT, "methods", "rl_interp.py"),
        "--data", str(cfg.data),
        "--base", str(cfg.base),
        "--interp-base", str(cfg.interp_base),
        "--interp-steps", str(cfg.interp_steps),
        "--rollout-url", str(cfg.rollout_url),
        "--arm", str(cfg.arm),
        "--reward", str(cfg.reward),
        "--rho", str(cfg.rho),
        "--out", os.path.join("ckpt", str(cfg.tag)),
        "--steps", str(cfg.steps),
        "--group", str(cfg.group),
        "--prompts-per-step", str(cfg.prompts_per_step),
        "--lr", str(cfg.lr),
        "--max-new-tokens", str(cfg.max_new_tokens),
        "--seed", str(cfg.seed),
        "--save-every", str(cfg.save_every),
    ]
    print("TAG=" + str(cfg.tag))
    print("TRAIN_ARGV=" + " ".join(shlex.quote(a) for a in argv))
    sys.stdout.flush()
    if not cfg.run:
        return 0
    return subprocess.call([sys.executable] + argv)


if __name__ == "__main__":
    main()
