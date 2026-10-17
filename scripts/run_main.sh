#!/usr/bin/env bash
# Main comparison (buildprompt §2.3 "Must"): B3 and B1 on the test set, McNemar, figures.
#   bash scripts/run_main.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
for c in b3 b1; do   # must-have comparison (B2: run_experiment.py --config configs/baselines/b2.yaml)
  $PY scripts/run_experiment.py --config "configs/baselines/$c.yaml" --run-id "$c-main" || echo "run $c incomplete"
done
$PY scripts/evaluate.py --mcnemar runs/b1-main runs/b3-main
$PY scripts/make_figures.py
for c in b1 b3; do $PY scripts/plot_signal.py "runs/$c-main" cctv_compilation_acc_ved; done
