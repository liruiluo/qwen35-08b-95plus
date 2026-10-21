#!/usr/bin/env bash
# Fetch the JointBERT-standard SNIPS split (13,084/700/700), pinned to the
# upstream commit verified byte-for-byte against SNIPS/dataset/.
set -euo pipefail
cd "$(dirname "$0")/.."

# JointBERT-standard SNIPS split: the test-caliber corpus. The official
# snipsco/nlu-benchmark repo ships no test split, and its train file
# substantially overlaps this split's test set (measured: 464/700 exact after
# case-folding), so test-caliber numbers are trained on THIS split.
JBASE=https://raw.githubusercontent.com/monologg/JointBERT/00324f60dda4a491b2913bc3065ae93966caea3f/data/snips
mkdir -p jointbert-snips
for split in train dev test; do
  for f in seq.in seq.out label; do
    curl -fsSL "$JBASE/$split/$f" -o "jointbert-snips/$split-$f"
  done
done
echo "done: jointbert-snips/ ready"
