#!/usr/bin/env bash
# Fetch the SNIPS corpora (nlu-benchmark + JointBERT-standard split).
set -euo pipefail
cd "$(dirname "$0")/.."

# SNIPS NLU benchmark (train + validate splits)
if [ ! -d nlu-benchmark ]; then
  git clone --depth 1 https://github.com/sonos/nlu-benchmark.git
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
echo "done: jointbert-snips/ ready"
