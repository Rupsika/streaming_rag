# Demo video (≤ 10 min) — recording guide

Close other heavy apps first (the LLM needs ~7 GB RAM).

## Before recording
```powershell
wsl -d Ubuntu -- bash scripts/infra_wsl.sh up          # Kafka + Prometheus
bash scripts/run_all.sh configs/demo.yaml               # 5 services, live namespace (Git Bash)
.venv\Scripts\python -m streamlit run dashboard\app.py  # opens http://localhost:8501
```
Wait ~3 min until `runs/live/logs/rag_svc.log` shows `consuming [...windows.closed]` (models loaded).

## Suggested script
| Time | Show | Say |
|---|---|---|
| 0:00–1:00 | README architecture figure | problem, 3 base papers, what B3 adds |
| 1:00–2:00 | terminal: `services/*.py`, `configs/default.yaml` | 5 separate processes, Kafka topics only, config switches |
| 2:00–2:30 | run `python scripts/replay_producer.py data/videos/accident/cctv_compilation_acc_ved.mp4 --speed 1` | frames go into `frames.raw` at native 30 fps |
| 2:30–5:30 | dashboard: live frame, CALM→ALERT badge, a_t timeline, event card | adaptive windows + back-fill, LLM verdict with cited [IDs] |
| 5:30–7:00 | click evidence IDs; attribution bars, coverage, counterfactual table, KG view | explainability E1–E5 (from a finished run: pick `b3-main` in the sidebar) |
| 7:00–8:00 | Prometheus http://localhost:9090 (`xs_fps`, `xs_stage_latency_seconds`) | monitoring |
| 8:00–9:30 | `results/results_table.md`, `report/figures/main_results.png` | B1 vs B3 numbers, honest limitations |

Note: on this CPU the VLM/LLM are slower than real time, so the live dashboard lags the
video. Say so on camera; the backlog is visible in Kafka (it is buffered, not dropped).

## After recording
```bash
bash scripts/stop_services.sh
wsl -d Ubuntu -- bash scripts/infra_wsl.sh down
```
