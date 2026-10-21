#!/bin/bash
# 顺序训练三个模型：每个算法一个独立脚本（可单独运行/重跑），一个失败不挡下一个。
# 顺序 = All-Adaptive → Last-Fixed → GRPO 基线。
set -u
cd /root/train95
bash scripts/run_songce_all_adaptive.sh
bash scripts/run_songce_last_fixed.sh
bash scripts/run_songce_grpobase.sh
echo "[songce] CAMPAIGN_DONE $(date -u +%FT%TZ)"
