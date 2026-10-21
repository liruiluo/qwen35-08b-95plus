#!/bin/bash
# Error-rate vs. training-step sweep over saved checkpoints (N<M efficiency
# evidence). For each model and each 200-step checkpoint: load the weights
# into the running engine, evaluate on the test set, append (model, step,
# INTENT_ACC) to results/nm-sweep.csv.
set -u
cd /root/train95
PY=/root/h3venv/bin/python
URL=http://127.0.0.1:30001
OUT=results/nm-sweep.csv

curl -s -m 5 $URL/health >/dev/null 2>&1 || { echo "FATAL engine down"; exit 1; }

echo "model,step,intent_acc" > "$OUT"
for m in jb-interpdef4s2000-last_adaptive jb-interpdef4s2000-all_hidden_adaptive jb-interpdef4s2000-last_fixed jb-grpobase-s2000; do
  for ck in ckpt/$m/step200 ckpt/$m/step400 ckpt/$m/step600 ckpt/$m/step800 ckpt/$m/step1000 ckpt/$m/step1200 ckpt/$m/step1400 ckpt/$m/step1600 ckpt/$m/step1800 ckpt/$m; do
    step=$(basename "$ck")
    [ "$step" = "$m" ] && step=2000
    "$PY" methods/push_ckpt.py "$URL" "$ck" >/dev/null 2>&1
    acc=$(cd snips && "$PY" eval_snips.py --engine-url "$URL" --model /root/models/Qwen3.5-0.8B \
      --data data-jbdef4/test.jsonl --out /tmp/nm-sweep-eval.jsonl 2>&1 \
      | grep INTENT_ACC | grep -oE "[0-9]+\.[0-9]+")
    echo "$m,$step,$acc" >> "$OUT"
    echo "SWEEP_ROW $m $step $acc"
  done
done
echo SWEEP_DONE
