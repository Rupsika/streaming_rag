"""Plot the change signal a_t of a run (M1 acceptance): runs/<run_id>/signal.png.

  python scripts/plot_signal.py runs/<run_id> [source_id]

Draws a_t (blue), window states from windows.jsonl (bands), ADWIN change points (ticks),
keyframes sent to the VLM (dots) and GT segments if annotated (grey).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.datasets import devset, ucf_test

COLORS = {"CALM": "#4c9a6a", "ALERT": "#e0a030", "EVENT": "#d0453b"}


def plot(run: Path, source: str, out: Path) -> Path:
    df = pd.read_csv(run / "signal" / f"{source}.csv")
    wins, trace = [], {}
    if (run / "windows.jsonl").exists():
        for line in open(run / "windows.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("source_id") != source:
                continue
            if d.get("kind") == "adaptive_trace":
                trace = d
            elif "id" in d:
                wins.append(d)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for w in wins:
        ax.axvspan(w["t0"], w["t1"], color=COLORS[w["state"]], alpha=0.12, lw=0)
    gts = next((v.segments for v in devset() + ucf_test() if v.source_id == source), [])
    for s, e in gts:
        ax.axvspan(s, e, color="black", alpha=0.18, lw=0, label="ground truth")
    ax.plot(df.video_ts, df.a_t, color="#2b5d9b", lw=1, label="a_t")
    kts = [k["video_ts"] for w in wins for k in w.get("keyframes", [])]
    if kts:
        ax.plot(kts, [df.a_t.max() * 1.05] * len(kts), "|", color="#555", ms=8, label="VLM keyframes")
    for cp in trace.get("change_points", []):
        ax.axvline(cp, color="#d0453b", lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("a_t = 0.6(1-cos CLIP) + 0.4 pixel diff")
    ax.set_title(f"{source} — change signal, windows (green CALM / amber ALERT / red EVENT), ADWIN change points")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    run = Path(sys.argv[1])
    sources = [sys.argv[2]] if len(sys.argv) > 2 else [p.stem for p in (run / "signal").glob("*.csv")]
    for s in sources:
        name = "signal.png" if len(sources) == 1 else f"signal_{s}.png"
        print(plot(run, s, run / name))
