"""Report figures from results/ and runs/ (plan.md §12). Every number is read from a run.

  python scripts/make_figures.py

Writes to report/figures/:
  architecture.png/.svg       pipeline + Kafka topics + tools
  main_results.png            B1/B2/B3 (latest run per config): video F1, event recall, short-event recall, P@5
  budget_curve.png            RQ3: recall vs VLM keyframes/min for B2 vs B3 (configs/sweeps/*)
  latency_cdf.png             per-window pipeline latency CDF per config
  faithfulness.png            RQ4: Δconfidence top-k vs random-k removal per k
Also results/results_table.md (markdown table of the latest run per config).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.config import ROOT

FIG = ROOT / "report" / "figures"
MAIN = ["b1", "b2_w2", "b2", "b2_w8", "b3"]
LABEL = {"b1": "B1 StreamingRAG", "b2_w2": "B2 W=2s", "b2": "B2 W=4s", "b2_w8": "B2 W=8s", "b3": "B3 X-StreamRAG"}


def latest_runs() -> dict[str, dict]:
    p = ROOT / "results" / "summary.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = {}
    for r in rows:                         # run ids end in a timestamp -> lexical max = latest
        if r["config"] not in out or r["run_id"] > out[r["config"]]["run_id"]:
            out[r["config"]] = r
    return out


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    stages = [
        (0.2, "Replay producer\nOpenCV, native FPS\n(file/webcam/RTSP)"),
        (2.45, "signal svc\nCLIP ViT-B/32 + frame diff\na_t @10 fps"),
        (4.7, "windowing svc\nRiver ADWIN + FSM\nring buffer back-fill"),
        (6.95, "vlm/kg svc\nSmolVLM (HF) + VQG\nNetworkX temporal KG"),
        (9.2, "rag svc\nbge+CLIP, FAISS, BM25\ncross-encoder, LangChain→LLM"),
        (11.45, "explain svc\nLOO, NLI coverage,\ncounterfactual, KG perturb"),
    ]
    topics = ["frames.raw", "frames.signal", "windows.keyframes", "windows.closed", "events.detected"]
    for i, (x, text) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, 2.6), 2.0, 1.35, boxstyle="round,pad=0.05", fc="#e8f0fa", ec="#2b5d9b"))
        ax.text(x + 1.0, 3.27, text, ha="center", va="center", fontsize=8.5)
        if i < len(topics):
            ax.annotate("", xy=(x + 2.25, 3.27), xytext=(x + 2.0, 3.27), arrowprops=dict(arrowstyle="->", lw=1.5))
            ax.text(x + 2.12, 4.15, topics[i], ha="center", fontsize=7.5, color="#7a3", rotation=0)
    ax.add_patch(FancyBboxPatch((0.2, 4.5), 13.25, 0.45, boxstyle="round,pad=0.03", fc="#f4f9ec", ec="#7a3"))
    ax.text(6.8, 4.72, "Apache Kafka (KRaft) / Redpanda — every stage is a separate consumer/producer process",
            ha="center", fontsize=9, color="#4a7a1a")
    ax.text(12.45, 2.35, "→ events.explained", ha="center", fontsize=7.5, color="#7a3")
    for x, text in [(3.2, "Case library (FAISS HNSW)\nfrom train videos, offline"),
                    (6.95, "Streamlit dashboard\n(own Kafka consumer)"),
                    (10.5, "Prometheus /metrics + OpenTelemetry spans\nruns/<id>/ events.jsonl, audit.jsonl")]:
        ax.add_patch(FancyBboxPatch((x - 1.5, 0.5), 3.0, 1.1, boxstyle="round,pad=0.05", fc="#fbf3e4", ec="#c08a2b"))
        ax.text(x, 1.05, text, ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(10.2, 2.55), xytext=(4.0, 1.65), arrowprops=dict(arrowstyle="->", color="#c08a2b"))
    ax.set_title("X-StreamRAG architecture", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"architecture.{ext}", dpi=150)
    plt.close(fig)


def main_results(runs: dict) -> None:
    cfgs = [c for c in MAIN if c in runs]
    if not cfgs:
        return
    metrics = [("video_f1", "video F1"), ("event_recall", "event recall"), ("short_event_recall", "short-event recall"),
               ("retrieval_p5", "retrieval P@5")]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        vals = [f(runs[c][k]) for k, _ in metrics]
        xs = np.arange(len(metrics)) + i * w
        ax.bar(xs, np.nan_to_num(vals), w, label=LABEL.get(c, c))
        if c in runs and not np.isnan(f(runs[c]["video_f1_ci_lo"])):
            lo, hi = f(runs[c]["video_f1_ci_lo"]), f(runs[c]["video_f1_ci_hi"])
            ax.errorbar(xs[0], vals[0], yerr=[[vals[0] - lo], [hi - vals[0]]], color="k", capsize=3)
        for x, v in zip(xs, vals):
            ax.text(x, (0 if np.isnan(v) else v) + 0.02, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(metrics)) + 0.4 - w / 2, [m for _, m in metrics])
    ax.set_ylim(0, 1.1)
    ds = runs[cfgs[0]]["dataset"]
    ax.set_title(f"Main comparison on {ds} (n={runs[cfgs[0]]['n_videos']} videos; n/a = needs GT event times)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "main_results.png", dpi=150)
    plt.close(fig)


def budget_curve(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for m, color in (("b2", "#c08a2b"), ("b3", "#2b5d9b")):
        pts = []
        for b in (15, 30, 60, 120):
            r = runs.get(f"{m}_budget{b}")
            if r:
                y = f(r["short_event_recall"])
                key = "short-event recall"
                if np.isnan(y):
                    y, key = f(r["video_r"]), "video recall (no GT times)"
                pts.append((f(r["vlm_kf_per_min"]), y))
        if pts:
            pts.sort()
            ax.plot(*zip(*pts), "o-", color=color, label=m.upper())
            ax.set_ylabel(key)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("measured VLM keyframes / min")
    ax.set_title("RQ3: recall vs VLM budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "budget_curve.png", dpi=150)
    plt.close(fig)


def latency_cdf(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    any_ = False
    for c, r in runs.items():
        p = ROOT / "runs" / r["run_id"] / "verdicts.jsonl"
        if c not in MAIN or not p.exists():
            continue
        lat = sorted(json.loads(x)["verdict_wallclock"] - json.loads(x)["last_ingest_ts"] for x in open(p))
        if lat:
            ax.plot(lat, np.linspace(0, 1, len(lat)), label=f"{LABEL.get(c, c)} (speed {r['speed']})")
            any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("window close → verdict latency incl. queueing (s)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "latency_cdf.png", dpi=150)
    plt.close(fig)


def faithfulness(runs: dict) -> None:
    rows = []
    for c in ("b3", "b2"):
        if c in runs:
            m = json.loads((ROOT / "results" / runs[c]["run_id"] / "metrics.json").read_text())
            for k, v in m["explain"].get("counterfactual", {}).items():
                rows.append((c, int(k), v["d_conf_top"], v["d_conf_random"], v["n"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, color in (("b3", "#2b5d9b"), ("b2", "#c08a2b")):
        rr = sorted(r for r in rows if r[0] == c)
        if rr:
            ks = [r[1] for r in rr]
            ax.plot(ks, [r[2] for r in rr], "o-", color=color, label=f"{c.upper()} top-k removed")
            ax.plot(ks, [r[3] for r in rr], "o--", color=color, alpha=0.6, label=f"{c.upper()} random-k removed")
    ax.set_xlabel("k evidence units removed")
    ax.set_ylabel("Δ confidence of predicted label")
    ax.set_title(f"RQ4 faithfulness (n events = {rows[0][4]})")
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "faithfulness.png", dpi=150)
    plt.close(fig)


def table(runs: dict) -> None:
    cols = [("config", "config"), ("n_videos", "n"), ("video_acc", "Acc"), ("video_p", "P"), ("video_r", "R"),
            ("video_f1", "F1"), ("event_recall", "event R"), ("short_event_recall", "short R"),
            ("detection_delay_mean", "delay s"), ("window_latency_p50", "lat P50 s"),
            ("window_latency_p95", "lat P95 s"), ("frame_processing_ratio", "frame ratio"),
            ("vlm_kf_per_min", "VLM kf/min"), ("sustained_fps", "FPS"), ("retrieval_p5", "P@5"),
            ("claim_coverage", "claim cov"), ("faithfulness_gap_k1", "gap k=1")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for c in sorted(runs, key=lambda c: (c not in MAIN, MAIN.index(c) if c in MAIN else 0, c)):
        r = runs[c]
        cells = []
        for k, _ in cols:
            v = r[k]
            cells.append(f"{float(v):.3f}" if v not in ("", None) and k not in ("config", "n_videos") else (v or "n/a"))
        lines.append("| " + " | ".join(cells) + " |")
    (ROOT / "results" / "results_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    runs = latest_runs()
    architecture()
    main_results(runs)
    budget_curve(runs)
    latency_cdf(runs)
    faithfulness(runs)
    if runs:
        table(runs)
    print("figures in", FIG, "| configs with results:", sorted(runs))
