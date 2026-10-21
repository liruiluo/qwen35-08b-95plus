#!/usr/bin/env python3
"""训练入口：读取 configs/songce/<配置名>.yaml 并启动训练。

用法（仓库根目录）：
  python methods/train_hydra.py -cn all_adaptive             # 训练
  python methods/train_hydra.py -cn grpo_baseline run=false  # 只打印训练命令，不执行
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
