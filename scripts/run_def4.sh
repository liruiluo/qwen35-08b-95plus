#!/bin/bash
# v3 definitions: adds see/watch bare-title vs typed-work convention,
# weather-will-it-be, rate-stars, book-lookup rules. 1000 steps, else same.
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
INTERP=/root/models/Qwen3.5-0.8B-RandomInit
URL=http://127.0.0.1:30001
mkdir -p results/jb-interpdef4 logs ckpt/jb-interpdef4-last_adaptive

curl -s -m 5 $URL/health >/dev/null 2>&1 \
  && echo "[def4] engine 30001 up" || { echo "[def4] FATAL engine down"; exit 1; }

echo "[def4] TRAIN jb-interpdef4-last_adaptive start $(date -u +%FT%TZ)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/rl_interp.py --data snips/data-jbdef4/train.jsonl --base $BASE \
  --interp-base $INTERP --interp-steps 100 \
  --rollout-url $URL --arm last_adaptive --reward snips \
  --out ckpt/jb-interpdef4-last_adaptive --steps 1000 --group 16 \
  --prompts-per-step 4 --lr 3e-6 --max-new-tokens 160 --seed 17 \
  --save-every 200 > logs/rl-jb-interpdef4-last_adaptive.log 2>&1
echo "[def4] TRAIN_DONE rc=$? $(date -u +%FT%TZ)"

$PY methods/push_ckpt.py $URL ckpt/jb-interpdef4-last_adaptive > /dev/null 2>&1
echo "[def4] post eval dev-then-test $(date -u +%FT%TZ)"
(cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
  --data data-jbdef4/dev.jsonl --out ../results/jb-interpdef4/final-dev.jsonl \
  2>&1 | tee ../results/jb-interpdef4/final-dev-score.txt | tail -2)
(cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
  --data data-jbdef4/test.jsonl --out ../results/jb-interpdef4/final-test.jsonl \
  2>&1 | tee ../results/jb-interpdef4/final-test-score.txt | tail -2)
echo "[def4] ALL_DONE $(date -u +%FT%TZ)"
