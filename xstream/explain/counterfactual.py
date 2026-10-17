"""E4 counterfactual evidence removal (RQ4; plan.md §4 M9, CityCopilot-X Δcoverage/Δconfidence).

Remove the top-k attributed units (k = 1, 2, 3) and, as control, k random units (5 seeds);
re-score and (optionally) re-generate. Reported per k:
  Δconfidence = P(label) - P'(label), flip = argmax label changed,
  Δcoverage = coverage - coverage', semantic drift = 1 - cos(expl, expl'),
  faithfulness gap = Δconf(top-k) - mean Δconf(random-k)  (should be > 0).
"""
from __future__ import annotations

import random

import numpy as np

from xstream.types import EvidenceUnit


def _one(probs_fn, regen_fn, cov_fn, embed_fn, units, removed, label, base_p, base_cov, base_vec):
    kept = [u for u in units if u.id not in removed]
    p = probs_fn(kept)
    rec = {"removed": sorted(removed), "p_label": round(p[label], 5), "d_conf": round(base_p - p[label], 5),
           "flip": max(p, key=p.get) != label}
    if regen_fn is not None:
        expl = regen_fn(kept)
        cov = cov_fn(expl, kept)["claim_coverage"] if expl else None
        rec["d_coverage"] = round(base_cov - (cov or 0.0), 4) if base_cov is not None else None
        if base_vec is not None and expl:
            rec["drift"] = round(float(1 - embed_fn([expl])[0] @ base_vec), 4)
        else:
            rec["drift"] = 1.0 if base_vec is not None else None
    return rec


def counterfactual(probs_fn, units: list[EvidenceUnit], weights: dict[str, float], label: str, base_p: float,
                   k_values=(1, 2, 3), seeds: int = 5, regen_fn=None, cov_fn=None, embed_fn=None,
                   base_expl: str = "", base_cov: float | None = None) -> dict:
    base_vec = embed_fn([base_expl])[0] if (regen_fn and base_expl and embed_fn) else None
    ranked = sorted(units, key=lambda u: -weights.get(u.id, 0.0))
    out = {}
    for k in k_values:
        if k >= len(units):
            continue
        top = _one(probs_fn, regen_fn, cov_fn, embed_fn, units, {u.id for u in ranked[:k]}, label, base_p,
                   base_cov, base_vec)
        rand = [_one(probs_fn, regen_fn, cov_fn, embed_fn, units,
                     {u.id for u in random.Random(s).sample(units, k)}, label, base_p, base_cov, base_vec)
                for s in range(seeds)]
        mean = lambda key: (float(np.mean([r[key] for r in rand if r.get(key) is not None]))
                            if any(r.get(key) is not None for r in rand) else None)
        out[str(k)] = {"top": top, "random": rand,
                       "random_mean": {"d_conf": mean("d_conf"), "flip": mean("flip"),
                                       "d_coverage": mean("d_coverage"), "drift": mean("drift")},
                       "faithfulness_gap": round(top["d_conf"] - mean("d_conf"), 5)}
    return out
