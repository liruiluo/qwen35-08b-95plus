#!/bin/bash
# Serial campaign: three SPHERE arms x two datasets, each from base.
# Per arm: ensure engine -> restore base weights -> train -> full eval.
# Memory-safe settings: engine KV at mem-fraction 0.25; no evals during
# training (a full eval during training overloaded the GPU once).
# Usage: bash run_campaign.sh [snips_steps] [bfcl_steps]
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
SNIPS_STEPS=${1:-2000}
BFCL_STEPS=${2:-1200}
mkdir -p results logs ckpt

ensure_engine () {
  if curl -s -m 5 $URL/health >/dev/null 2>&1; then return 0; fi
  echo "[campaign] engine down, relaunching (mem-fraction 0.25)"
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
    && echo "[campaign] engine back up" \
    || { echo "[campaign] FATAL engine relaunch"; exit 1; }
}

restore_base () {
  $PY methods/push_ckpt.py $URL $BASE > /dev/null 2>&1 \
    && echo "[campaign] base restored" \
    || { echo "[campaign] FATAL base restore failed"; exit 1; }
}

train_arm () {  # dataset arm steps lr pps maxnew
  local ds=$1 arm=$2 steps=$3 lr=$4 pps=$5 mx=$6
  local data="bfcl/data/train.jsonl" reward="bfcl"
  [ "$ds" = snips ] && { data="snips/data/train.jsonl"; reward="snips"; }
  local tag="${ds}-${arm}"
  echo "[campaign] TRAIN $tag steps=$steps lr=$lr pps=$pps $(date -u +%T)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY methods/sphere_grpo.py --data "$data" --base $BASE --rollout-url $URL \
      --arm "$arm" --reward "$reward" --out "ckpt/$tag" \
      --steps "$steps" --group 16 --prompts-per-step "$pps" --lr "$lr" \
      --max-new-tokens "$mx" --seed 17 --save-every 200 \
      > "logs/rl-$tag.log" 2>&1
  local rc=$?
  echo "[campaign] TRAIN $tag rc=$rc tail: $(grep -E '^step' logs/rl-$tag.log | tail -1)"
  [ $rc -ne 0 ] && { echo "[campaign] FATAL training $tag"; exit 1; }
}

eval_arm () {  # dataset tag
  local ds=$1 tag=$2
  if [ "$ds" = snips ]; then
    (cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
      --data data/validate.jsonl --out "../results/final-$tag.jsonl" \
      2>&1 | tee "../results/final-$tag-score.txt" | tail -4)
  else
    $PY methods/push_ckpt.py $URL "ckpt/$tag" > /dev/null 2>&1
    (cd bfcl && $PY eval_bfcl.py --engine-url $URL --model $BASE \
      --questions /root/bfcl-eval/questions.jsonl \
      --answers /root/bfcl-eval/answers.jsonl \
      --out "../results/final-$tag.jsonl" \
      2>&1 | tee "../results/final-$tag-score.txt" | tail -3)
  fi
}

for arm in last_adaptive all_hidden_adaptive last_fixed; do
  ensure_engine
  restore_base
  train_arm snips "$arm" "$SNIPS_STEPS" 3e-6 4 160
  eval_arm snips "snips-$arm"
done
for arm in last_adaptive all_hidden_adaptive last_fixed; do
  ensure_engine
  restore_base
  train_arm bfcl "$arm" "$BFCL_STEPS" 3e-6 4 256
  eval_arm bfcl "bfcl-$arm"
done
echo "[campaign] ALL_DONE $(date -u +%FT%TZ)"
