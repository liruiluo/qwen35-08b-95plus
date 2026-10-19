#!/bin/bash
# Phase 2 (calibration pair, run after the base campaign):
#   1) SNIPS arm-1 extension: continue from its final ckpt, 3000 steps,
#      same recipe -> read the exact-F1-per-step slope.
#   2) BFCL v2 arm-1: fresh from base on the enriched datav2, 1600 steps
#      -> read the convention-fix slope.
# The remaining arms are queued after reading these two numbers.
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001

# ---- wait for the base campaign to finish (up to 5h)
for i in $(seq 1 600); do
  grep -q ALL_DONE logs/campaign.log && break
  sleep 30
done
if ! grep -q ALL_DONE logs/campaign.log; then
  echo "[phase2] FATAL campaign not done after 5h"; exit 1
fi
echo "[phase2] campaign finished, starting phase2 $(date -u +%T)"

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

# ---- 1) SNIPS arm-1 extension pass (3000 steps, same recipe)
ensure_engine
bash scripts/extend_arm.sh snips last_adaptive 3000 ckpt/snips-last_adaptive \
  snips-last_adaptive-ext1
[ -f results/final-snips-last_adaptive-ext1-score.txt ] \
  && echo "[phase2] SNIPS ext1 score:" \
        && cat results/final-snips-last_adaptive-ext1-score.txt \
  || echo "[phase2] WARN snips ext1 produced no score"

# ---- 2) BFCL v2 arm 1 (fresh from base, datav2, 1600 steps)
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
    --save-every 200 > logs/rl-bfclv2-last_adaptive.log 2>&1
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
echo "[phase2] PHASE2_DONE $(date -u +%FT%TZ)"
