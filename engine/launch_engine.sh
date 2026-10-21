#!/usr/bin/env bash
# Launch an sglang engine for evaluation.
#   usage: launch_engine.sh <model_path> [port]
set -euo pipefail
MODEL=${1:?usage: launch_engine.sh <model_path> [port]}
PORT=${2:-30000}

# stop any previous engine
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 10

PY="${PYTHON:-python3}"
nohup "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --context-length 98304 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --cuda-graph-max-bs-decode 32 \
  --host 127.0.0.1 --port "$PORT" \
  > "engine-$PORT.log" 2>&1 &
echo "engine pid $!"

for i in $(seq 1 120); do
  code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" || true)
  [ "$code" = "200" ] && break
  sleep 5
done
# confirm WHICH model is loaded (health only proves an engine is alive)
pgrep -af "sglang.launch_server" | grep -F "$MODEL" \
  && echo "ENGINE_UP model=$MODEL port=$PORT" \
  || { echo "MODEL MISMATCH — engine did not load $MODEL"; exit 1; }
