#!/bin/bash
# Error-rate vs step sweep for the retrained GRPO baseline (lr 3e-7).
set -u
cd /root/train95
PY=/root/h3venv/bin/python
URL=http://127.0.0.1:30001
OUT=results/nm-sweep2.csv

curl -s -m 5 $URL/health >/dev/null 2>&1 || { echo "FATAL engine down"; exit 1; }

echo "model,step,intent_acc" > "$OUT"
for ck in ckpt/jb-grpobase2-s2000/step200 ckpt/jb-grpobase2-s2000/step400 ckpt/jb-grpobase2-s2000/step600 ckpt/jb-grpobase2-s2000/step800 ckpt/jb-grpobase2-s2000/step1000 ckpt/jb-grpobase2-s2000/step1200 ckpt/jb-grpobase2-s2000/step1400 ckpt/jb-grpobase2-s2000/step1600 ckpt/jb-grpobase2-s2000/step1800 ckpt/jb-grpobase2-s2000; do
  step=$(basename "$ck")
  [ "$step" = "jb-grpobase2-s2000" ] && step=2000
  "$PY" methods/push_ckpt.py "$URL" "$ck" >/dev/null 2>&1
  acc=$(cd snips && "$PY" eval_snips.py --engine-url "$URL" --model /root/models/Qwen3.5-0.8B \
    --data data-jbdef4/test.jsonl --out /tmp/nm2.jsonl 2>&1 \
    | grep INTENT_ACC | grep -oE "[0-9]+\.[0-9]+")
  echo "jb-grpobase2-s2000,$step,$acc" >> "$OUT"
  echo "SWEEP_ROW $step $acc"
done
echo SWEEP2_DONE
