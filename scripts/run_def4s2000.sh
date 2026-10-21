#!/bin/bash
# v4 definitions deliverable, 2000 steps (test 97.14% / dev 99.14%).
# Verbatim record of the pod campaign script that produced the deliverable
# (pod path: /root/train95/scripts/run_def4s2000.sh; the pod tree uses
# lowercase snips/ for the data dir, which is SNIPS/ in this repo).
set -u
cd /root/train95
PY=/root/h3venv/bin/python
BASE=/root/models/Qwen3.5-0.8B
INTERP=/root/models/Qwen3.5-0.8B-RandomInit
URL=http://127.0.0.1:30001
mkdir -p results/jb-interpdef4s2000 logs ckpt/jb-interpdef4s2000-last_adaptive

curl -s -m 5 $URL/health >/dev/null 2>&1 \
  && echo "[def4s2000] engine 30001 up" || { echo "[def4s2000] FATAL engine down"; exit 1; }

echo "[def4s2000] TRAIN jb-interpdef4s2000-last_adaptive start $(date -u +%FT%TZ)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY methods/rl_interp.py --data snips/data-jbdef4/train.jsonl --base $BASE \
  --interp-base $INTERP --interp-steps 100 \
  --rollout-url $URL --arm last_adaptive --reward snips \
  --out ckpt/jb-interpdef4s2000-last_adaptive --steps 2000 --group 16 \
  --prompts-per-step 4 --lr 3e-6 --max-new-tokens 160 --seed 17 \
  --save-every 200 > logs/rl-jb-interpdef4s2000-last_adaptive.log 2>&1
echo "[def4s2000] TRAIN_DONE rc=$? $(date -u +%FT%TZ)"

$PY methods/push_ckpt.py $URL ckpt/jb-interpdef4s2000-last_adaptive > /dev/null 2>&1
echo "[def4s2000] post eval dev-then-test $(date -u +%FT%TZ)"
(cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
  --data data-jbdef4/dev.jsonl --out ../results/jb-interpdef4s2000/final-dev.jsonl \
  2>&1 | tee ../results/jb-interpdef4s2000/final-dev-score.txt | tail -2)
(cd snips && $PY eval_snips.py --engine-url $URL --model $BASE \
  --data data-jbdef4/test.jsonl --out ../results/jb-interpdef4s2000/final-test.jsonl \
  2>&1 | tee ../results/jb-interpdef4s2000/final-test-score.txt | tail -2)
echo "[def4s2000] ALL_DONE $(date -u +%FT%TZ)"
