#!/bin/bash
# 三车顺序连发：每车独立脚本（可单独发射/重跑），一车失败不挡下一车。
# 车序 = All-Adaptive → Last-Fixed → GRPO 基线。
set -u
cd /root/train95
bash scripts/run_songce_all_adaptive.sh
bash scripts/run_songce_last_fixed.sh
bash scripts/run_songce_grpobase.sh
echo "[songce] CAMPAIGN_DONE $(date -u +%FT%TZ)"
