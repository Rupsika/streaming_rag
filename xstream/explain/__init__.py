"""Explainability engine (plan.md §4 M9): E1-E5 for one event; E6 lives in drift.py.

Runs in its own Kafka consumer (services/explain_svc.py), off the alert hot path.
"""
from __future__ import annotations

import time

from xstream.explain.attribution import leave_one_out, temporal_heat
from xstream.explain.counterfactual import counterfactual
from xstream.explain.coverage import coverage
from xstream.explain.kg_perturb import kg_perturb
from xstream.types import EvidenceUnit


def explain_event(gen, nli, embedder, cfg: dict, header: str, units: list[EvidenceUnit], label: str,
                  explanation: str, t0: float, t1: float) -> dict:
    ex = cfg["explain"]
    t_start = time.time()
    probs_fn = lambda us: gen.probs(header, us)
    base_p, weights = leave_one_out(probs_fn, units, label)                     # E1
    heat = temporal_heat(units, weights, t0, t1)                                # E2
    cov_fn = lambda text, us: coverage(text, us, nli)
    base_cov = cov_fn(explanation, units)                                       # E3
    regen = None
    if ex.get("regenerate", True):
        regen = lambda us: gen.explain(header, us, label).get("explanation", "")
    cf = counterfactual(probs_fn, units, weights, label, base_p, ex["k_values"], ex["random_seeds"],  # E4
                        regen, cov_fn, embedder.embed, explanation, base_cov["claim_coverage"])
    kgp = kg_perturb(probs_fn, units, label, base_p) if ex.get("kg_perturb", True) else None  # E5
    return {"label": label, "p_label": base_p, "attribution": weights, "temporal": heat, "coverage": base_cov,
            "counterfactual": cf, "kg_perturbation": kgp, "explain_seconds": round(time.time() - t_start, 2)}
