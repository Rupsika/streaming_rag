"""E1 evidence attribution + E2 temporal attribution (plan.md §4 M9, CityCopilot-X → multimodal).

E1: leave-one-out over the <= 8 evidence units: w_i = P(label | all) - P(label | all \\ {i})
(9 scoring calls per event). E2: unit weights projected onto the timeline; F/T/S units carry
timestamps, H units are "context" with no time.
"""
from __future__ import annotations

import numpy as np

from xstream.types import EvidenceUnit


def leave_one_out(probs_fn, units: list[EvidenceUnit], label: str) -> tuple[float, dict[str, float]]:
    base = probs_fn(units)[label]
    weights = {}
    for i, u in enumerate(units):
        weights[u.id] = round(base - probs_fn(units[:i] + units[i + 1:])[label], 5)
    return base, weights


def temporal_heat(units: list[EvidenceUnit], weights: dict[str, float], t0: float, t1: float,
                  step: float = 0.25) -> dict:
    """Positive weight mass per time bin over [t0, t1] (heat strip) + top-1 timestamped unit."""
    t0 = min([t0] + [u.t0 for u in units if u.t0 is not None])
    t1 = max([t1] + [u.t1 for u in units if u.t1 is not None])
    edges = np.arange(t0, t1 + step, step)
    heat = np.zeros(max(1, len(edges) - 1))
    timed = [u for u in units if u.t0 is not None]
    for u in timed:
        w = max(weights.get(u.id, 0.0), 0.0)
        lo, hi = u.t0, max(u.t1, u.t0)
        idx = [j for j in range(len(heat)) if edges[j] <= hi and edges[j] + step >= lo]
        for j in idx:
            heat[j] += w / len(idx)
    top = max(timed, key=lambda u: weights.get(u.id, 0.0), default=None)
    return {"t0": round(float(t0), 3), "step": step, "heat": [round(float(h), 5) for h in heat],
            "top_unit": top.id if top else None,
            "top_ts": round((top.t0 + top.t1) / 2, 3) if top else None,
            "top_weight": weights.get(top.id) if top else None}
