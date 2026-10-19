#!/bin/bash
# Extension pass 2 recipe (anti-saturation): same GRPO/SPHERE recipe as the
# base runs, plus --name-credit 0.5 (right-name-wrong-value emission becomes
# net-neutral instead of penalized, keeping emission profitable once groups
# saturate) and --gen-temp 1.15 (restore within-group diversity so GRPO has
# nonzero advantage variance). Eval is unchanged (official exact span F1).
# Usage: bash extend_arm2.sh <snips|bfcl> <arm> <steps> <from_dir> <out_tag>
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
URL=http://127.0.0.1:30001
ds=$1; arm=$2; steps=$3; from=$4; out=$5
mkdir -p results logs ckpt

if ! curl -s -m 5 $URL/health >/dev/null 2>&1; then
  echo "[ext2] FATAL engine down; relaunch from base first, then rerun"
  exit 1
fi
$PY methods/push_ckpt.py $URL "$from" > /dev/null 2>&1 \
  || { echo "[ext2] FATAL cannot push $from into engine"; exit 1; }
echo "[ext2] engine now holds $from weights"

data="bfcl/data/train.jsonl"; reward="bfcl"; mx=256; nc=""
[ "$ds" = snips ] && { data="snips/data/train.jsonl"; reward="snips"; mx=160; nc="--name-credit 0.5"; }
echo "[ext2] TRAIN $out steps=$steps from=$from $(date -u +%T)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/sphere_grpo.py --data "$data" --base "$from" --rollout-url $URL \
    --arm "$arm" --reward "$reward" --out "ckpt/$out" \
    --steps "$steps" --group 16 --prompts-per-step 4 --lr 3e-6 \
    --gen-temp 1.15 $nc \
    --max-new-tokens "$mx" --seed 41 --save-every 500 \
    > "logs/rl-$out.log" 2>&1
rc=$?
echo "[ext2] TRAIN $out rc=$rc tail: $(grep -E '^step' logs/rl-$out.log | tail -1)"
[ $rc -ne 0 ] && { echo "[ext2] FATAL training $out"; exit 1; }

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
echo "[ext2] DONE $out $(date -u +%FT%TZ)"
