"""Evaluate one pipeline run against ground truth (plan.md §6).

  python scripts/evaluate.py runs/<run_id>
  python scripts/evaluate.py --mcnemar runs/<b1_run> runs/<b3_run>

Reads runs/<run_id>/{manifest.json, events.jsonl, verdicts.jsonl, vlm_calls.jsonl, signal/*.csv,
audit.jsonl}; writes results/<run_id>/metrics.json and appends a row to results/summary.csv.

Metrics that need GT event times (event-level match, short-event recall, detection delay,
alert latency, frame AUC, pointing game) are computed only on videos whose start/end are
annotated; for the dev set the team must fill data/annotations_devset.csv first.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.config import ROOT
from xstream.datasets import devset, same_family, ucf_test
from xstream.metrics import bootstrap_ci, match_events, mcnemar, percentiles, video_level

SHORT_S = 3.0


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []


def gt_for(manifest: dict) -> dict:
    pool = ucf_test() if manifest["dataset"] == "ucf_test" else devset()
    by_id = {v.source_id: v for v in pool}
    return {s: by_id[s] for s in manifest["videos"] if s in by_id}


def final_events(run: Path) -> dict[str, list[dict]]:
    last = {}
    for e in jsonl(run / "events.jsonl"):
        if e.get("kind") == "event":
            last[e["event_id"]] = e        # close record supersedes the open record
    out = defaultdict(list)
    for e in last.values():
        out[e["source_id"]].append(e)
    return out


def video_info(path: Path, max_s: float | None) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if max_s:
        n = min(n, int(max_s * fps))
    return n, n / fps


def window_label(gt, t0: float, t1: float) -> str:
    if not gt.has_times:
        return gt.label                       # approximation: video-level label
    return gt.label if any(min(t1, e) - max(t0, s) >= 0 for s, e in gt.segments) else "none"


def evaluate(run: Path) -> dict:
    man = json.loads((run / "manifest.json").read_text())
    gts = gt_for(man)
    events = final_events(run)
    verdicts = jsonl(run / "verdicts.jsonl")
    calls = jsonl(run / "vlm_calls.jsonl")
    verdicts = [v for v in verdicts if v["source_id"] in man["videos"]]      # runs stopped early: kept videos only
    calls = [c for c in calls if c["source_id"] in man["videos"]]
    audit = jsonl(run / "audit.jsonl")
    vids = [s for s in man["videos"] if s in gts]
    m: dict = {"run_id": man["run_id"], "config": man["config_name"], "dataset": man["dataset"],
               "n_videos": len(vids), "speed": man["speed"], "warnings": []}
    if man["dataset"] == "devset":
        m["warnings"].append("FALLBACK TEST SET: 16 dev clips (UCF-Crime test videos not on disk)")

    # ---- video level -----------------------------------------------------------------
    y_true = [gts[s].label != "none" for s in vids]
    y_pred = [len(events.get(s, [])) > 0 for s in vids]
    m["video"] = video_level(y_true, y_pred)
    m["video"]["f1_ci95"] = bootstrap_ci(
        lambda idx: video_level([y_true[i] for i in idx], [y_pred[i] for i in idx])["f1"], len(vids))
    m["video"]["acc_ci95"] = bootstrap_ci(
        lambda idx: video_level([y_true[i] for i in idx], [y_pred[i] for i in idx])["accuracy"], len(vids))
    m["per_video"] = {s: {"gt": gts[s].label, "pred_events": [e["event_type"] for e in events.get(s, [])],
                          "correct": yt == yp} for s, yt, yp in zip(vids, y_true, y_pred)}
    type_ok = [any(same_family(e["event_type"], gts[s].label) for e in events.get(s, []))
               for s in vids if gts[s].label != "none" and events.get(s)]
    m["video"]["type_accuracy_on_detected"] = float(np.mean(type_ok)) if type_ok else None

    # ---- event level (needs GT times) ---------------------------------------------
    timed = [s for s in vids if gts[s].has_times]
    untimed_pos = [s for s in vids if not gts[s].has_times]
    if untimed_pos:
        m["warnings"].append(f"{len(untimed_pos)} positive videos lack GT start/end -> excluded from event-level "
                             f"metrics; fill data/annotations_devset.csv event_start_s/event_end_s: {untimed_pos}")
    n_gt = n_match = n_short = n_short_match = n_pred = n_pred_match = 0
    delays, lat, type_hits, per_video_recall = [], [], [], []
    sig_cache = {}
    for s in timed:
        g, preds = gts[s], sorted(events.get(s, []), key=lambda e: e["alert_video_ts"])
        mt, used = match_events(preds, g.segments)
        n_gt += len(g.segments)
        n_pred += len(preds)
        n_pred_match += len(used)
        for seg, pi in zip(g.segments, mt):
            short = seg[1] - seg[0] <= SHORT_S
            n_short += short
            if pi is None:
                continue
            n_match += 1
            n_short_match += short
            p = preds[pi]
            delays.append(p["alert_video_ts"] - seg[0])
            type_hits.append(same_family(p["event_type"], g.label))
            sig = sig_cache.setdefault(s, _signal(run, s))
            if sig is not None:
                j = int(np.argmin(np.abs(sig[:, 0] - seg[0])))
                lat.append(p["alert_wallclock"] - sig[j, 1])
        if g.segments:
            per_video_recall.append(sum(x is not None for x in mt) / len(g.segments))
    m["event"] = {
        "n_videos_with_times": len(timed), "n_gt_segments": n_gt, "n_pred_events": n_pred,
        "recall": n_match / n_gt if n_gt else None, "precision": n_pred_match / n_pred if n_pred else None,
        "short_event_recall": n_short_match / n_short if n_short else None, "n_short": n_short,
        "type_accuracy": float(np.mean(type_hits)) if type_hits else None,
        "detection_delay_s": {"mean": float(np.mean(delays)) if delays else None, **percentiles(delays)},
        "alert_latency_s": percentiles(lat),
        "alert_latency_note": "meaningful only for real-time runs (speed=1)" if man["speed"] != 1 else "",
        "false_alarms_per_hour_normal": None,
    }
    if m["event"]["recall"] is not None and m["event"]["precision"] is not None:
        r, p = m["event"]["recall"], m["event"]["precision"]
        m["event"]["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    normal = [s for s in vids if gts[s].label == "none"]
    if normal:
        hours = sum(video_info(gts[s].path, man.get("max_seconds"))[1] for s in normal) / 3600
        m["event"]["false_alarms_per_hour_normal"] = sum(len(events.get(s, [])) for s in normal) / hours

    # ---- frame-level ROC-AUC on sampled frames ---------------------------------------
    try:
        from sklearn.metrics import roc_auc_score
        ys, ps = [], []
        by_src = defaultdict(list)
        for v in verdicts:
            by_src[v["source_id"]].append(v)
        for s in timed:
            sig = sig_cache.setdefault(s, _signal(run, s))
            if sig is None:
                continue
            for ts in sig[:, 0]:
                cover = [v["p_event"] for v in by_src[s] if v["t0"] - 0.5 <= ts <= v["t1"] + 0.5]
                ps.append(max(cover) if cover else 0.0)
                ys.append(any(a <= ts <= b for a, b in gts[s].segments))
        m["frame_auc"] = float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else None
    except Exception as e:  # pragma: no cover
        m["frame_auc"] = None
        m["warnings"].append(f"frame AUC failed: {e}")

    # ---- efficiency / streaming --------------------------------------------------------
    n_frames = video_s = 0.0
    for s in vids:
        n, d = video_info(gts[s].path, man.get("max_seconds"))
        n_frames += n
        video_s += d
    uniq_kf = {c["keyframe"] for c in calls}
    m["efficiency"] = {
        "video_seconds": video_s, "frames_ingested": int(n_frames), "keyframes_to_vlm": len(uniq_kf),
        "frame_processing_ratio": len(uniq_kf) / n_frames if n_frames else None,
        "vlm_keyframes_per_min": len(uniq_kf) / (video_s / 60) if video_s else None,
        "vlm_seconds_per_keyframe": percentiles([c["seconds"] for c in calls if not c.get("cached")]),
        "windows": len(verdicts),
        "llm_seconds_per_window": percentiles([v.get("gen_seconds") for v in verdicts]),
    }
    wall = man.get("wall_seconds")
    m["streaming"] = {
        "wall_seconds": wall,
        "sustained_fps": n_frames / wall if wall else None,
        "realtime_factor": video_s / wall if wall else None,
        "window_latency_s": percentiles([v["verdict_wallclock"] - v["last_ingest_ts"] for v in verdicts]),
        "peak_gpu_mem_mb": None, "peak_rss_mb": man.get("resources", {}).get("peak_rss_mb"),
        "cpu_percent_mean": man.get("resources", {}).get("cpu_percent_mean"),
        "note": "CPU only (no CUDA): GPU memory not applicable" if not man.get("cuda") else "",
    }

    # ---- retrieval P@5 / R@5 -------------------------------------------------------------
    lib_counts = _library_counts(man)
    p5, r5 = [], []
    for v in verdicts:
        if v["source_id"] not in gts:
            continue
        top = v.get("retrieval", {}).get("case_top", [])[:5]
        if not top:
            continue
        q = window_label(gts[v["source_id"]], v["t0"], v["t1"])
        rel = [same_family(h["label"], q) if q != "none" else h["label"] == "none" for h in top]
        p5.append(sum(rel) / len(top))
        total_rel = lib_counts(q, v["source_id"])
        if total_rel:
            r5.append(sum(rel) / total_rel)
    m["retrieval"] = {"p_at_5": float(np.mean(p5)) if p5 else None, "r_at_5": float(np.mean(r5)) if r5 else None,
                      "n_queries": len(p5),
                      "label_source": "video-level labels for clips without GT times" if untimed_pos else "GT segments"}

    # ---- explainability (audit.jsonl) ----------------------------------------------------
    m["explain"] = _explain_metrics(audit, gts)
    return m


def _signal(run: Path, source: str):
    p = run / "signal" / f"{source}.csv"
    if not p.exists():
        return None
    arr = np.genfromtxt(p, delimiter=",", skip_header=1, usecols=(1, 2))
    return arr.reshape(-1, 2) if arr.size else None


def _library_counts(man: dict):
    level = "frames" if man["retrieval_level"] == "frame" else "windows"
    p = ROOT / man["case_library_dir"] / level / "meta.jsonl"
    meta = jsonl(p)

    def count(label: str, video: str) -> int:
        return sum(1 for x in meta if x["video"] != video and
                   (same_family(x["label"], label) if label != "none" else x["label"] == "none"))
    return count


def _explain_metrics(audit: list[dict], gts: dict) -> dict:
    ex = [a for a in audit if a.get("explain")]
    if not ex:
        return {"n_events": 0}
    cov = [a["explain"]["coverage"] for a in ex]
    out = {"n_events": len(ex),
           "claim_coverage": _mean([c["claim_coverage"] for c in cov]),
           "token_coverage": _mean([c["token_coverage"] for c in cov]),
           "unsupported": _mean([c["unsupported"] for c in cov]),
           "explain_seconds": percentiles([a["explain"]["explain_seconds"] for a in ex])}
    hits = []
    for a in ex:
        g = gts.get(a["source_id"])
        ts = a["explain"]["temporal"]["top_ts"]
        if g is not None and g.has_times and g.segments and ts is not None:
            hits.append(any(s - 1.0 <= ts <= e + 1.0 for s, e in g.segments))
    out["pointing_game"] = float(np.mean(hits)) if hits else None
    out["pointing_game_n"] = len(hits)
    cf = {}
    for k in ("1", "2", "3"):
        rows = [a["explain"]["counterfactual"][k] for a in ex if k in a["explain"]["counterfactual"]]
        if not rows:
            continue
        cf[k] = {"n": len(rows),
                 "d_conf_top": _mean([r["top"]["d_conf"] for r in rows]),
                 "d_conf_random": _mean([r["random_mean"]["d_conf"] for r in rows]),
                 "faithfulness_gap": _mean([r["faithfulness_gap"] for r in rows]),
                 "flip_top": _mean([float(r["top"]["flip"]) for r in rows]),
                 "flip_random": _mean([r["random_mean"]["flip"] for r in rows]),
                 "d_coverage_top": _mean([r["top"].get("d_coverage") for r in rows]),
                 "d_coverage_random": _mean([r["random_mean"]["d_coverage"] for r in rows]),
                 "drift_top": _mean([r["top"].get("drift") for r in rows]),
                 "drift_random": _mean([r["random_mean"]["drift"] for r in rows])}
    out["counterfactual"] = cf
    infl = [a["explain"]["kg_perturbation"]["most_influential"] for a in ex
            if a["explain"].get("kg_perturbation") and a["explain"]["kg_perturbation"].get("most_influential")]
    out["kg_most_influential"] = infl
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


SUMMARY_COLS = ["run_id", "config", "dataset", "n_videos", "speed", "video_acc", "video_p", "video_r", "video_f1",
                "video_f1_ci_lo", "video_f1_ci_hi", "event_recall", "event_precision", "short_event_recall",
                "detection_delay_mean", "alert_latency_p50", "alert_latency_p95", "window_latency_p50",
                "window_latency_p95", "frame_auc", "frame_processing_ratio", "vlm_kf_per_min", "sustained_fps",
                "realtime_factor", "retrieval_p5", "retrieval_r5", "claim_coverage", "token_coverage",
                "pointing_game", "faithfulness_gap_k1", "faithfulness_gap_k3", "n_explained"]


def summary_row(m: dict) -> dict:
    v, e, s = m["video"], m["event"], m["streaming"]
    ci = v.get("f1_ci95") or [None, None]
    cf = m["explain"].get("counterfactual", {})
    return {"run_id": m["run_id"], "config": m["config"], "dataset": m["dataset"], "n_videos": m["n_videos"],
            "speed": m["speed"], "video_acc": v["accuracy"], "video_p": v["precision"], "video_r": v["recall"],
            "video_f1": v["f1"], "video_f1_ci_lo": ci[0], "video_f1_ci_hi": ci[1],
            "event_recall": e["recall"], "event_precision": e["precision"],
            "short_event_recall": e["short_event_recall"], "detection_delay_mean": e["detection_delay_s"]["mean"],
            "alert_latency_p50": e["alert_latency_s"]["p50"], "alert_latency_p95": e["alert_latency_s"]["p95"],
            "window_latency_p50": s["window_latency_s"]["p50"], "window_latency_p95": s["window_latency_s"]["p95"],
            "frame_auc": m["frame_auc"], "frame_processing_ratio": m["efficiency"]["frame_processing_ratio"],
            "vlm_kf_per_min": m["efficiency"]["vlm_keyframes_per_min"], "sustained_fps": s["sustained_fps"],
            "realtime_factor": s["realtime_factor"], "retrieval_p5": m["retrieval"]["p_at_5"],
            "retrieval_r5": m["retrieval"]["r_at_5"], "claim_coverage": m["explain"].get("claim_coverage"),
            "token_coverage": m["explain"].get("token_coverage"), "pointing_game": m["explain"].get("pointing_game"),
            "faithfulness_gap_k1": cf.get("1", {}).get("faithfulness_gap"),
            "faithfulness_gap_k3": cf.get("3", {}).get("faithfulness_gap"), "n_explained": m["explain"]["n_events"]}


def write(m: dict) -> Path:
    out = ROOT / "results" / m["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    summ = ROOT / "results" / "summary.csv"
    rows = []
    if summ.exists():
        rows = [r for r in csv.DictReader(open(summ, encoding="utf-8")) if r["run_id"] != m["run_id"]]
    rows.append(summary_row(m))
    with open(summ, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    return out / "metrics.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--mcnemar", action="store_true", help="paired test between exactly two runs")
    args = ap.parse_args()
    if args.mcnemar:
        a, b = (evaluate(Path(r)) for r in args.runs)
        common = sorted(set(a["per_video"]) & set(b["per_video"]))
        res = mcnemar([a["per_video"][s]["correct"] for s in common], [b["per_video"][s]["correct"] for s in common])
        res.update(a=a["run_id"], b=b["run_id"], n=len(common))
        print(json.dumps(res, indent=2))
        out = ROOT / "results" / f"mcnemar_{a['config']}_vs_{b['config']}.json"
        out.write_text(json.dumps(res, indent=2))
        return 0
    for r in args.runs:
        m = evaluate(Path(r))
        p = write(m)
        v, eff = m["video"], m["efficiency"]
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
        print(f"{m['run_id']} [{m['config']}] {m['dataset']} n={m['n_videos']}: video Acc={fmt(v['accuracy'])} "
              f"P={fmt(v['precision'])} R={fmt(v['recall'])} F1={fmt(v['f1'])} CI95={v['f1_ci95']} | "
              f"kf/min={fmt(eff['vlm_keyframes_per_min'])} ratio={fmt(eff['frame_processing_ratio'])} "
              f"| P@5={fmt(m['retrieval']['p_at_5'])} -> {p}")
        for w in m["warnings"]:
            print("  WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
