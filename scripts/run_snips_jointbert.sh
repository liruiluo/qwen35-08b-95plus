#!/bin/bash
# Test-caliber campaign: three SPHERE arms on the JointBERT-standard SNIPS
# split (13,084/700/700), each a SINGLE run from base with the same recipe
# as the original campaign. Per arm: ensure engine -> restore base ->
# train -> eval on test (primary) and dev (secondary).
# Usage: bash run_snips_jointbert.sh [steps]
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
STEPS=${1:-2000}
DATA=snips/data-jb/train.jsonl
mkdir -p results/jb logs ckpt

ensure_engine () {
  if curl -s -m 5 $URL/health >/dev/null 2>&1; then return 0; fi
  echo "[jb] engine down, relaunching (mem-fraction 0.25)"
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
    && echo "[jb] engine back up" \
    || { echo "[jb] FATAL engine relaunch"; exit 1; }
}

restore_base () {
  $PY methods/push_ckpt.py $URL $BASE > /dev/null 2>&1 \
    && echo "[jb] base restored" \
    || { echo "[jb] FATAL base restore failed"; exit 1; }
}

eval_split () {  # tag split
  local tag=$1 split=$2
  (cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
    --data "data-jb/$split.jsonl" \
    --out "../results/jb/$tag-$split.jsonl" \
    2>&1 | tee "../results/jb/$tag-$split-score.txt" | tail -4)
}

# step 0: base zero-shot on the test-caliber corpus (starting point)
if [ ! -f results/jb/base-test-score.txt ]; then
  ensure_engine
  restore_base
  eval_split base test
fi

for arm in last_adaptive all_hidden_adaptive last_fixed; do
  tag="jb-snips-$arm"
  ensure_engine
  restore_base
  echo "[jb] TRAIN $tag steps=$STEPS $(date -u +%T)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY methods/sphere_grpo.py --data $DATA --base $BASE --rollout-url $URL \
      --arm "$arm" --reward snips --out "ckpt/$tag" \
      --steps "$STEPS" --group 16 --prompts-per-step 4 --lr 3e-6 \
      --max-new-tokens 160 --seed 17 --save-every 200 \
      > "logs/rl-$tag.log" 2>&1
  rc=$?
  echo "[jb] TRAIN $tag rc=$rc tail: $(grep -E '^step' logs/rl-$tag.log | tail -1)"
  [ $rc -ne 0 ] && { echo "[jb] FATAL training $tag"; exit 1; }
  $PY methods/push_ckpt.py $URL "ckpt/$tag" > /dev/null 2>&1 \
    || { echo "[jb] FATAL push ckpt $tag"; exit 1; }
  eval_split "$tag" test
  eval_split "$tag" dev
done
echo "[jb] ALL_DONE $(date -u +%FT%TZ)"
