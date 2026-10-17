#!/usr/bin/env bash
# Start the stream backbone (Kafka + Prometheus) and all five pipeline services.
#   bash scripts/run_all.sh [config]          default config: configs/baselines/b3.yaml
# Services log to runs/live/logs/; PIDs go to runs/.pids (scripts/stop_services.sh stops them).
# Backbone: docker compose (Redpanda + Prometheus) if Docker exists, otherwise Apache Kafka
# (KRaft) + Prometheus inside WSL2 / Linux via scripts/infra_wsl.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/baselines/b3.yaml}"
RUN_ID="${RUN_ID:-live}"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

if command -v docker >/dev/null 2>&1; then
  docker compose up -d --wait
elif command -v wsl.exe >/dev/null 2>&1; then
  wsl.exe -d "${WSL_DISTRO:-Ubuntu}" -- bash scripts/infra_wsl.sh up
else
  bash scripts/infra_wsl.sh up
fi
$PY -c "from xstream.config import load_config; from xstream.bus import ensure_topics; ensure_topics(load_config())"

mkdir -p "runs/$RUN_ID/logs"
: > runs/.pids
for svc in signal_svc windowing_svc vlm_kg_svc rag_svc explain_svc; do
  $PY "services/$svc.py" --config "$CONFIG" --run-id "$RUN_ID" > "runs/$RUN_ID/logs/$svc.log" 2>&1 &
  echo $! >> runs/.pids
  echo "started $svc (pid $!)"
done
echo "all services up; logs in runs/$RUN_ID/logs. Replay with:"
echo "  $PY scripts/replay_producer.py data/videos/accident/cctv_compilation_acc_ved.mp4"
