#!/bin/bash
# Phase 2b (after the base campaign died on a bfcl-last_fixed OOM):
#   0) Retry bfcl-last_fixed on v1 data, microbatch 8 (memory-safe; the
#      microbatch size only chunks the same global-batch computation).
#   1) SNIPS arm-1 extension: 3000 steps from its final ckpt, same recipe.
#   2) BFCL v2 arm-1: fresh from base on the enriched datav2, 1600 steps.
# Reads: two calibration slopes -> remaining arms queued afterwards.
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
mkdir -p results logs ckpt

ensure_engine () {
  if curl -s -m 5 $URL/health >/dev/null 2>&1; then return 0; fi
  echo "[phase2] engine down, relaunching (mem-fraction 0.25)"
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
    && echo "[phase2] engine back up" \
    || { echo "[phase2] FATAL engine relaunch"; exit 1; }
}

echo "[phase2] start $(date -u +%T)"

# ---- 0) retry bfcl-last_fixed on v1 data (ablation row), microbatch 8
ensure_engine
$PY methods/push_ckpt.py $URL $BASE > /dev/null 2>&1 \
  && echo "[phase2] base restored" \
  || { echo "[phase2] FATAL base restore failed"; exit 1; }
echo "[phase2] RETRY bfcl-last_fixed v1 steps=800 mb=8 $(date -u +%T)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/sphere_grpo.py --data bfcl/data/train.jsonl --base $BASE \
    --rollout-url $URL --arm last_fixed --reward bfcl \
    --out ckpt/bfcl-last_fixed --steps 800 --group 16 \
    --prompts-per-step 4 --lr 3e-6 --max-new-tokens 256 --seed 17 \
    --microbatch 8 --save-every 200 > logs/rl-bfcl-last_fixed-retry.log 2>&1
rc=$?
echo "[phase2] RETRY rc=$rc tail: $(grep -E '^step' logs/rl-bfcl-last_fixed-retry.log | tail -1)"
if [ $rc -eq 0 ]; then
  $PY methods/push_ckpt.py $URL ckpt/bfcl-last_fixed > /dev/null 2>&1
  (cd bfcl && $PY eval_bfcl.py --engine-url $URL --model $BASE \
    --questions /root/bfcl-eval/questions.jsonl \
    --answers /root/bfcl-eval/answers.jsonl \
    --out "../results/final-bfcl-last_fixed.jsonl" \
    2>&1 | tee "../results/final-bfcl-last_fixed-score.txt" | tail -3)
else
  echo "[phase2] WARN bfcl-last_fixed retry failed rc=$rc (ablation row stays 2-arm)"
fi

# ---- 1) SNIPS arm-1 extension (3000 steps, same recipe)
ensure_engine
bash scripts/extend_arm.sh snips last_adaptive 3000 ckpt/snips-last_adaptive \
  snips-last_adaptive-ext1
[ -f results/final-snips-last_adaptive-ext1-score.txt ] \
  && echo "[phase2] SNIPS ext1 score:" \
        && cat results/final-snips-last_adaptive-ext1-score.txt \
  || echo "[phase2] WARN snips ext1 produced no score"

# ---- 2) BFCL v2 arm 1 (fresh from base, datav2, 1600 steps, mb=8)
ensure_engine
$PY methods/push_ckpt.py $URL $BASE > /dev/null 2>&1 \
  && echo "[phase2] base restored" \
  || { echo "[phase2] FATAL base restore failed"; exit 1; }
echo "[phase2] TRAIN bfclv2-last_adaptive steps=1600 $(date -u +%T)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/sphere_grpo.py --data bfcl/datav2/train.jsonl --base $BASE \
    --rollout-url $URL --arm last_adaptive --reward bfcl \
    --out ckpt/bfclv2-last_adaptive --steps 1600 --group 16 \
    --prompts-per-step 4 --lr 3e-6 --max-new-tokens 256 --seed 31 \
    --microbatch 8 --save-every 200 > logs/rl-bfclv2-last_adaptive.log 2>&1
rc=$?
echo "[phase2] TRAIN bfclv2-last_adaptive rc=$rc tail: $(grep -E '^step' logs/rl-bfclv2-last_adaptive.log | tail -1)"
if [ $rc -eq 0 ]; then
  $PY methods/push_ckpt.py $URL ckpt/bfclv2-last_adaptive > /dev/null 2>&1
  (cd bfcl && $PY eval_bfcl.py --engine-url $URL --model $BASE \
    --questions /root/bfcl-eval/questions.jsonl \
    --answers /root/bfcl-eval/answers.jsonl \
    --out "../results/final-bfclv2-last_adaptive.jsonl" \
    2>&1 | tee "../results/final-bfclv2-last_adaptive-score.txt" | tail -3)
else
  echo "[phase2] FATAL bfclv2 training rc=$rc"
fi
echo "[phase2] PHASE2B_DONE $(date -u +%FT%TZ)"
