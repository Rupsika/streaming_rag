
Fill in the 3 `<<FILL>>` values first.

---

You are the lead engineer for our IOE413 Streaming Data Analytics mini project: **X-StreamRAG — Explainable Time-Aware Streaming RAG for real-time multimodal (video) event detection.** Build the full working pipeline in this repo, milestone by milestone.

## 0. Read these first (all are in this repo)
1. `IOE413-Mini_Project_Instructions-Sep_2026.pdf`: the course guidelines and grading rubric. **These are hard requirements and override anything else.**
2. `Streamingdata_1.docx`: our approved proposal (RQ1–RQ4, base papers, tools, metrics).
3. `plan.md`: the detailed implementation plan. Follow its architecture (§3–4), baselines (§5), metric definitions (§6) and repo layout (§9), **except where this prompt changes them**.
4. `papers/`: the 3 base papers (StreamingRAG, TD-RAG, CityCopilot-X). Skim them where plan.md defers to a paper's detail.
5. `data/`:
   - `data/README.md`
   - `data/annotations_devset.csv`
   - `data/videos/{accident,normal}/` (16 dev clips)
   - `data/ucf_crime/Temporal_Anomaly_Annotation.txt` + `Anomaly_Train.txt`
   - `data/airquality/AirQualityUCI.csv`
   - plus whatever UCF-Crime / TAD videos are present under `data/ucf_crime/videos/` and `data/tad/`

   List what is actually present before relying on it.

## 1. Environment (fill in)
- **Hardware:** `<<FILL: e.g. "laptop, RTX 3050 4 GB, 16 GB RAM, Windows 11 + WSL2"  or  "Kaggle T4 16 GB">>`
- **LLM backend:** `<<FILL: "gemini API (key in .env as GEMINI_API_KEY)"  or  "ollama qwen2.5:7b"  or  "ollama qwen2.5:1.5b">>`
- **VLM backend:** `<<FILL: "blip2-opt-2.7b 8-bit"  or  "SmolVLM-256M (CPU ok)"  or  "gemini vision API">>`

Make every model a config switch (`configs/default.yaml`) behind a small interface (`VLMBackend`, `LLMBackend`, `Embedder`), so we can swap backends without code changes. Use Python 3.10+. Pin all versions in `requirements.txt`.

## 2. Changes to plan.md (required by the guidelines)
1. **Kafka is the stream backbone.** The guidelines reject a static file read once and grade "real data flow between stages, not isolated scripts" (15/50 marks). So:
   - Run Kafka in `docker-compose.yml`. Apache Kafka in KRaft mode, or Redpanda (Kafka-API compatible, lighter). Same compose file: Prometheus. Grafana is optional.
   - A **producer** (`scripts/replay_producer.py`) replays video files (or a webcam / RTSP URL) **at native FPS** into topic `frames.raw`. Each message holds a JPEG-encoded frame at 640 px width, plus `source_id`, `frame_id`, `video_ts` and `ingest_ts`. Use `--speed` for 2×/4× stress tests. Key the messages by `source_id`.
   - Each pipeline stage is a **separate Kafka consumer/producer process** communicating only through topics:

     ```
     frames.raw → [signal svc] → frames.signal → [windowing svc] → windows.keyframes
     windows.keyframes → [vlm/kg svc] → windows.closed
     windows.closed → [rag svc: encode+retrieve+rerank+generate] → events.detected
     events.detected → [explain svc] → events.explained
     ```

     Plus a **consumer** for the Streamlit dashboard. The ring buffer for pre-event back-fill lives in the windowing service.
   - Use `confluent-kafka` (or `kafka-python`). Use JSON messages, except frames, which use raw bytes plus headers. Use one consumer group per service.
   - Provide `scripts/run_all.sh` (or `make up`) that starts compose plus all services, and `make down`.
2. **Tools to show clearly in README and architecture diagram (3+ required):**
   - Kafka
   - River (ADWIN)
   - OpenCV
   - VLM (HF transformers)
   - FAISS
   - NetworkX
   - LangChain (use it for the RAG prompt/chain and the retriever wrapper so the usage is genuine)
   - Streamlit
   - Prometheus (+ OpenTelemetry spans)

   Spark/Flink are **not** required. Do not add them unless everything else is done.
3. **Deadline mode.** Final submission is in the week of Oct 15–21, 2026, so scope is cut to what is gradeable:
   - **Must:** B1 baseline (StreamingRAG re-impl) and B3 enhancement (adaptive windows + window retrieval + explainability), with a quantitative comparison.
   - **Should:** B2 fixed-window, counterfactual explainability, dashboard.
   - **Could:** B0, ablations, Exp-0 AirQualityUCI, TAD, human rating, Grafana.
   - **Test set:** UCF-Crime test RoadAccidents (23) + Explosion (21) + 20 Normal (seed 42, first 3 min), restricted to the videos present on disk. Fall back to the 16 dev clips if UCF videos are missing, and say so loudly in results.

## 3. Milestones (do them in order; after each one, STOP and report)
For each milestone, the report must include:
- what was built (files)
- the exact commands to run it
- a real output excerpt or metric
- what's next

Do not move on while a milestone's acceptance check fails.

**M0. Skeleton (≤ 1 h)**
- Repo layout per plan.md §9 (packages under `xstream/`, services under `services/`).
- `configs/default.yaml`, `xstream/types.py` with `Frame`, `Window`, `EvidenceUnit` and `EventPrediction` exactly as in plan.md §9.
- `docker-compose.yml`, `requirements.txt`, `.env.example`, `Makefile`.
- *Accept:* `make up` starts Kafka + Prometheus. A smoke test produces and consumes one message.

**M1. Stream ingest + change signal**
- Replay producer.
- `signal` service: CLIP ViT-B/32 cosine change + frame diff → `a_t`, at ~10 fps sampled from `frames.raw`.
- Prometheus counters/gauges: `xs_frames_ingested_total`, `xs_fps`, `xs_stage_latency_seconds{stage}`.
- *Accept:* replaying `data/videos/accident/cctv_compilation_acc_ved.mp4` shows `a_t` spikes near the visible crashes (save a plot to `runs/<id>/signal.png`), and the producer keeps native FPS.

**M2. Baseline B1 (StreamingRAG re-implementation) end to end**
- Fixed 2 fps keyframe sampling.
- VLM caption + question bank (≤ 5 VQA questions) + rule-based Visual Query Generator, with KB in `kb/traffic_kb.yaml`.
- NetworkX temporal KG with triple IDs.
- Frame-level kNN retrieval (FAISS).
- LangChain prompt → LLM → JSON event (schema in plan.md §4 M8).
- Label-probability confidence (local model) or self-consistency (API).
- Event open/close debounce.
- **Cache every VLM call** in sqlite, keyed by (video, frame_id, prompt).
- *Accept:* on the 16 dev clips, `events.detected` contains events, and `runs/<id>/events.jsonl` is written.

**M3. Evaluation harness (build it BEFORE the enhancement)**
- `scripts/evaluate.py` computes everything in plan.md §6 that the data supports:
  - video-level Acc/P/R/F1
  - event-level match (tIoU ≥ 0.1 or start within GT ± 2 s)
  - short-event recall (GT ≤ 3 s)
  - detection delay
  - alert latency P50/P95
  - sustained FPS
  - peak GPU memory
  - frame-processing ratio
  - VLM calls/min
  - Retrieval P@5/R@5
- Reads UCF-Crime annotations (frames @30 fps, −1 = none) and `annotations_devset.csv`. If the dev-set start/end columns are empty, **tell me to fill them**; do not invent times.
- Writes `results/<run_id>/metrics.json` and a CSV row to `results/summary.csv`.
- Bootstrap 95 % CIs; McNemar for B1 vs B3.
- *Accept:* B1 metrics on the available test set.

**M4. Enhancement B3 (the graded novelty, 8 marks)**
- Adaptive windowing service (plan.md §4 M3): `river.drift.ADWIN` on `a_t`, CALM/ALERT/EVENT FSM with per-state keyframe rates, 3 s ring-buffer **pre-event back-fill**, and min/max window length.
- Window encoder: text summary + bge-small + mean-CLIP fusion.
- **Case library**, built offline by `scripts/build_case_library.py` from UCF train videos, never from test.
- Session memory with a recency constraint.
- Sequence context (current + previous 2 windows).
- Hybrid dense + BM25 + KG 1-hop retrieval → cross-encoder rerank → top 8 evidence units with per-stage scores logged.
- Also implement B2 (fixed windows W ∈ {2,4,8} s, step W/6) as a config of the same service.
- **Fair-budget rule:** B2/B3 comparisons use matched VLM calls/min (expose `vlm_budget_per_min`).
- *Accept:* B3 vs B1 (and B2) table on the test set, with honest numbers even if B3 loses somewhere.

**M5. Explainability service (plan.md §4 M9)**
- E1 leave-one-out attribution over evidence units.
- E2 temporal heat strip + pointing-game metric.
- E3 NLI claim coverage + token coverage + unsupported %.
- E4 counterfactual removal: top-k vs random-k (k = 1, 2, 3; 5 seeds) → Δconfidence, Δcoverage, flip rate, semantic drift, faithfulness gap.
- E5 KG node/edge/triple perturbation if time allows.
- E6 embedding-drift monitor.
- Runs **off the hot path** (own consumer) and writes `runs/<id>/audit.jsonl`.
- *Accept:* explainability metrics table + one fully worked example event in the audit log.

**M6. Streamlit dashboard + monitoring**
- Layout per plan.md §4 M10: live frame + state badge, window timeline with GT overlay, event cards with clickable evidence IDs, keyframes, attribution bars, coverage, counterfactual table, KG subgraph (pyvis), live metrics from Prometheus.
- OpenTelemetry spans per window, exported to the console or OTLP.
- *Accept:* `make demo` replays one accident clip and the dashboard updates live end to end.

**M7. Experiments + packaging**
- Run B1, B2, B3 (+ ablations if time) via `scripts/run_experiment.py --config ...`. Each run saves its config copy, metrics and plots.
- Generate report-ready figures to `report/figures/`:
  - architecture diagram (PNG/SVG)
  - main results bar chart
  - short-event recall vs VLM budget
  - latency CDF
  - explainability faithfulness plot
- Write `README.md` with:
  - setup and run instructions (Windows/WSL + Linux)
  - **a table of every tool with its exact version**
  - the dataset download steps
  - how to reproduce each results table
  - known limitations

## 4. Engineering rules
- **Never fabricate or "estimate" a result.** Every number in any table must come from a run in `results/`. If something can't run (no GPU, missing videos), say so and give me the command to run it myself.
- **Report negative results honestly.** The rubric rewards it.
- Fix seeds and log config + git commit hash + library versions per run.
- Prefer simple, readable code with docstrings that cite the paper/section each component reproduces, e.g. `# StreamingRAG §3: Visual Query Generator`. We must defend every design choice in a viva.
- Keep GPU memory within the hardware above: 8-bit/4-bit loading, batch CLIP, cache VLM outputs, and offline eval in "fast replay" mode except for latency/throughput runs.
- Add small `pytest` tests for windowing (ADWIN boundaries, back-fill), event matching and metric computation.
- Don't commit large data or model weights. Add a `.gitignore`.
- Ask me before anything destructive or before changing the RQs, baselines or metrics.

Start with M0 now. Before writing code, give me a ≤ 15-line summary of what you understood from the guidelines + proposal + plan, and any blocking questions (e.g. missing videos, hardware limits).