#!/usr/bin/env bash
# Fetch SNIPS (nlu-benchmark) and BFCL v4 simple_python eval data.
set -euo pipefail
cd "$(dirname "$0")/.."

# SNIPS NLU benchmark (train + validate splits)
if [ ! -d nlu-benchmark ]; then
  git clone --depth 1 https://github.com/snipsco/nlu-benchmark.git
fi

# BFCL v4 simple_python: 400 eval questions + official possible answers
BASE=https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data
curl -fsSL "$BASE/BFCL_v4_simple_python.json" -o BFCL_v4_simple_python.json
curl -fsSL "$BASE/possible_answer/BFCL_v4_simple_python.json" \
  -o BFCL_v4_simple_python_possible_answer.json
echo "done: $(wc -l < BFCL_v4_simple_python.json) questions,"
echo "      $(wc -l < BFCL_v4_simple_python_possible_answer.json) answers"
