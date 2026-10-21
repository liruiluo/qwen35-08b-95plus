#!/bin/bash
# SPHERE-All-Adaptive 臂 · 交付配方（config = configs/songce/all_adaptive.yaml，配方唯一来源；
# 本脚本 = 引擎检查 → 训练 → 热推 → dev/test 评测；pod 布局）
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
CFG=all_adaptive
TAG=$($PY methods/train_hydra.py -cn $CFG run=false | grep ^TAG= | cut -d= -f2)
RES=results/$TAG
mkdir -p "$RES" logs "ckpt/$TAG"

curl -s -m 5 $URL/health >/dev/null 2>&1 \
  && echo "[$TAG] engine 30001 up" || { echo "[$TAG] FATAL engine down"; exit 1; }

echo "[$TAG] TRAIN start $(date -u +%FT%TZ)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/train_hydra.py -cn $CFG > logs/rl-$TAG.log 2>&1
rc=$?
echo "[$TAG] TRAIN_DONE rc=$rc $(date -u +%FT%TZ)"
[ $rc -ne 0 ] && exit $rc

$PY methods/push_ckpt.py $URL ckpt/$TAG > /dev/null 2>&1
echo "[$TAG] eval dev-then-test $(date -u +%FT%TZ)"
(cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
  --data data-jbdef4/dev.jsonl --out ../$RES/final-dev.jsonl \
  2>&1 | tee ../$RES/final-dev-score.txt | tail -2)
(cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
  --data data-jbdef4/test.jsonl --out ../$RES/final-test.jsonl \
  2>&1 | tee ../$RES/final-test-score.txt | tail -2)
echo "[$TAG] ALL_DONE $(date -u +%FT%TZ)"
