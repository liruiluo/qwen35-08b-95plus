#!/bin/bash
# Phase 2c (launched manually after reading the ext2-arm1 calibration):
#   1) SNIPS ext2 arm2 + arm3 (anti-saturation recipe, from their base ckpts)
#   2) BFCL v3 fresh runs: all three arms from base on datav3, temp 1.0,
#      2000 steps, mb=8. The v2 arm-1 continuation stays as data-development
#      evidence; the final per-arm numbers come from these clean runs.
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
mkdir -p results logs ckpt

ensure_engine () {
  if curl -s -m 5 $URL/health >/dev/null 2>&1; then return 0; fi
  echo "[p2c] engine down, relaunching (mem-fraction 0.25)"
  pkill -f launch_server; sleep 12
  CUDA_HOME=/root/h3venv/lib/python3.12/site-packages/nvidia/cu13 \
    nohup $PY -m sglang.launch_server --model-path $BASE \
      --context-length 98304 --mem-fraction-static 0.25 \
      --reasoning-parser qwen3 --cuda-graph-max-bs-decode 32 \
      --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
      --max-running-requests 32 --host 127.0.0.1 --port 30001 \
      > logs/eng-rollout.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s -m 2 -o /dev/null $URL/health && break; sleep 5
  done
  curl -s -m 5 $URL/health >/dev/null 2>&1 \
    && echo "[p2c] engine back up" \
    || { echo "[p2c] FATAL engine relaunch"; exit 1; }
}

echo "[p2c] start $(date -u +%T)"

for arm in all_hidden_adaptive last_fixed; do
  ensure_engine
  bash scripts/extend_arm2.sh snips "$arm" 4000 "ckpt/snips-$arm" \
    "snips-$arm-ext2"
  [ -f "results/final-snips-$arm-ext2-score.txt" ] \
    && echo "[p2c] snips-$arm-ext2:" && cat "results/final-snips-$arm-ext2-score.txt" \
    || echo "[p2c] WARN no score for snips-$arm-ext2"
done

for arm in last_adaptive all_hidden_adaptive last_fixed; do
  ensure_engine
  $PY methods/push_ckpt.py $URL $BASE > /dev/null 2>&1 \
    || { echo "[p2c] FATAL base restore"; exit 1; }
  echo "[p2c] TRAIN bfclv3-$arm steps=2000 $(date -u +%T)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY methods/sphere_grpo.py --data bfcl/datav3/train.jsonl --base $BASE \
      --rollout-url $URL --arm "$arm" --reward bfcl \
      --out "ckpt/bfclv3-$arm" --steps 2000 --group 16 \
      --prompts-per-step 4 --lr 3e-6 --max-new-tokens 256 --seed 53 \
      --microbatch 8 --save-every 500 > "logs/rl-bfclv3-$arm.log" 2>&1
  rc=$?
  echo "[p2c] TRAIN bfclv3-$arm rc=$rc tail: $(grep -E '^step' logs/rl-bfclv3-$arm.log | tail -1)"
  if [ $rc -ne 0 ]; then
    echo "[p2c] FATAL bfclv3-$arm training"
    exit 1
  fi
  $PY methods/push_ckpt.py $URL "ckpt/bfclv3-$arm" > /dev/null 2>&1
  (cd bfcl && $PY eval_bfcl.py --engine-url $URL --model $BASE \
    --questions /root/bfcl-eval/questions.jsonl \
    --answers /root/bfcl-eval/answers.jsonl \
    --out "../results/final-bfclv3-$arm.jsonl" \
    2>&1 | tee "../results/final-bfclv3-$arm-score.txt" | tail -3)
done
echo "[p2c] PHASE2C_DONE $(date -u +%FT%TZ)"
