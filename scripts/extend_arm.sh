#!/bin/bash
# Extension pass: continue one arm from a finished checkpoint with the same
# recipe (group 16, pps 4, lr 3e-6). Used when a base arm finishes below 95.
# Engine must hold the arm's final weights for the first on-policy rollouts.
# Usage: bash extend_arm.sh <snips|bfcl> <arm> <steps> <from_dir> <out_tag>
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
ds=$1; arm=$2; steps=$3; from=$4; out=$5
mkdir -p results logs ckpt

if ! curl -s -m 5 $URL/health >/dev/null 2>&1; then
  echo "[ext] FATAL engine down; relaunch from base first, then rerun"
  exit 1
fi
$PY methods/push_ckpt.py $URL "$from" > /dev/null 2>&1 \
  || { echo "[ext] FATAL cannot push $from into engine"; exit 1; }
echo "[ext] engine now holds $from weights"

data="bfcl/data/train.jsonl"; reward="bfcl"; mx=256
[ "$ds" = snips ] && { data="snips/data/train.jsonl"; reward="snips"; mx=160; }
echo "[ext] TRAIN $out steps=$steps from=$from $(date -u +%T)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/sphere_grpo.py --data "$data" --base "$from" --rollout-url $URL \
    --arm "$arm" --reward "$reward" --out "ckpt/$out" \
    --steps "$steps" --group 16 --prompts-per-step 4 --lr 3e-6 \
    --max-new-tokens "$mx" --seed 23 --save-every 200 \
    > "logs/rl-$out.log" 2>&1
rc=$?
echo "[ext] TRAIN $out rc=$rc tail: $(grep -E '^step' logs/rl-$out.log | tail -1)"
[ $rc -ne 0 ] && { echo "[ext] FATAL training $out"; exit 1; }

if [ "$ds" = snips ]; then
  $PY methods/push_ckpt.py $URL "ckpt/$out" > /dev/null 2>&1
  (cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
    --data data/validate.jsonl --out "../results/final-$out.jsonl" \
    2>&1 | tee "../results/final-$out-score.txt" | tail -4)
else
  $PY methods/push_ckpt.py $URL "ckpt/$out" > /dev/null 2>&1
  (cd bfcl && $PY eval_bfcl.py --engine-url $URL --model $BASE \
    --questions /root/bfcl-eval/questions.jsonl \
    --answers /root/bfcl-eval/answers.jsonl \
    --out "../results/final-$out.jsonl" \
    2>&1 | tee "../results/final-$out-score.txt" | tail -3)
fi
echo "[ext] DONE $out $(date -u +%FT%TZ)"
