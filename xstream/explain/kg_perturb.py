"""E5 KG perturbation (KGRAG-Ex style, plan.md §4 M9).

Remove one triple, or every triple touching one entity / using one relation, from the
evidence and re-score. The largest Δconfidence names the most influential entity/relation.
"""
from __future__ import annotations

from xstream.types import EvidenceUnit


def kg_perturb(probs_fn, units: list[EvidenceUnit], label: str, base_p: float) -> dict:
    triples = [u for u in units if u.kind == "kg_triple"]
    if not triples:
        return {"triples": {}, "entities": {}, "relations": {}, "most_influential": None}

    def drop(pred):
        kept = [u for u in units if not (u.kind == "kg_triple" and pred(u.source["triple"]))]
        return round(base_p - probs_fn(kept)[label], 5)

    res = {"triples": {u.id: drop(lambda t, i=u.id: t["id"] == i) for u in triples}}
    ents = sorted({t.source["triple"][k] for t in triples for k in ("s", "o")})
    rels = sorted({t.source["triple"]["r"] for t in triples})
    res["entities"] = {e: drop(lambda t, e=e: e in (t["s"], t["o"])) for e in ents}
    res["relations"] = {r: drop(lambda t, r=r: t["r"] == r) for r in rels}
    best = max([("triple", k, v) for k, v in res["triples"].items()] + [("entity", k, v) for k, v in res["entities"].items()]
               + [("relation", k, v) for k, v in res["relations"].items()], key=lambda x: x[2])
    res["most_influential"] = {"kind": best[0], "name": best[1], "d_conf": best[2]}
    return res
