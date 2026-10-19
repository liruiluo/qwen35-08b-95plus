#!/usr/bin/env bash
# Fetch SNIPS (nlu-benchmark) and BFCL v4 simple_python eval data.
set -euo pipefail
cd "$(dirname "$0")/.."

# SNIPS NLU benchmark (train + validate splits)
if [ ! -d nlu-benchmark ]; then
  git clone --depth 1 https://github.com/snipsco/nlu-benchmark.git
fi

# JointBERT-standard SNIPS split (13,084/700/700): the test-caliber corpus.
# The official nlu-benchmark repo ships no test split, and its train file
# substantially overlaps this split's test set (measured: 464/700 exact after
# case-folding), so test-caliber numbers are trained on THIS split.
JBASE=https://raw.githubusercontent.com/monologg/JointBERT/master/data/snips
mkdir -p jointbert-snips
for split in train dev test; do
  for f in seq.in seq.out label; do
    curl -fsSL "$JBASE/$split/$f" -o "jointbert-snips/$split-$f"
  done
done

# BFCL v4 simple_python: 400 eval questions + official possible answers
BASE=https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data
curl -fsSL "$BASE/BFCL_v4_simple_python.json" -o BFCL_v4_simple_python.json
curl -fsSL "$BASE/possible_answer/BFCL_v4_simple_python.json" \
  -o BFCL_v4_simple_python_possible_answer.json
echo "done: $(wc -l < BFCL_v4_simple_python.json) questions,"
echo "      $(wc -l < BFCL_v4_simple_python_possible_answer.json) answers"
