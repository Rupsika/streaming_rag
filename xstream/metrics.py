"""Metric primitives (plan.md §6), kept free of I/O so they are unit-tested."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else float(a == b)


def event_match(pred: tuple[float, float], gt: tuple[float, float], min_tiou: float = 0.1, slack: float = 2.0) -> bool:
    """plan.md §6: tIoU >= 0.1 OR the prediction starts inside the GT segment ± 2 s."""
    return tiou(pred, gt) >= min_tiou or (gt[0] - slack <= pred[0] <= gt[1] + slack)


def match_events(preds: list[dict], gts: list[tuple[float, float]], **kw) -> tuple[list[int | None], set[int]]:
    """Greedy one-to-one matching in GT order. Returns (pred index per GT or None, matched pred indices)."""
    used: set[int] = set()
    out: list[int | None] = []
    for g in gts:
        cand = [i for i, p in enumerate(preds) if i not in used and event_match((p["start_s"], p["end_s"]), g, **kw)]
        best = min(cand, key=lambda i: preds[i]["alert_video_ts"], default=None)
        if best is not None:
            used.add(best)
        out.append(best)
    return out, used


def prf(tp: int, fp: int, fn: int, tn: int = 0) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    n = tp + fp + fn + tn
    return {"precision": p, "recall": r, "f1": f1, "accuracy": (tp + tn) / n if n else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def video_level(y_true: list[bool], y_pred: list[bool]) -> dict:
    yt, yp = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return prf(int((yt & yp).sum()), int((~yt & yp).sum()), int((yt & ~yp).sum()), int((~yt & ~yp).sum()))


def bootstrap_ci(values_fn, n_items: int, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap over items (videos). `values_fn(idx_array) -> float`."""
    if n_items == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = [values_fn(rng.integers(0, n_items, n_items)) for _ in range(n_boot)]
    stats = [s for s in stats if s is not None and not np.isnan(s)]
    if not stats:
        return None
    return [float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))]


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Exact McNemar test on paired per-video correctness."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n01, "b_only": n10, "p_value": float(p)}


def percentiles(xs, qs=(50, 95)) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(xs, q)) for q in qs}
