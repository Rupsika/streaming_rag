"""E3 grounded coverage (plan.md §4 M9; CityCopilot-X token coverage).

Claims = sentences of the explanation. A claim is supported if NLI p(entailment) >= 0.5
against the concatenated text of the units it cites (cross-encoder/nli-deberta-v3-small);
a claim with no valid citation is unsupported. Token coverage = share of the explanation's
content tokens that appear in the cited evidence.
"""
from __future__ import annotations

import re

import numpy as np

from xstream.types import EvidenceUnit

_STOP = set("a an the is are was were be been of in on at to and or for with by this that it its as from "
            "there their they which who what when where than then also has have had not no".split())
_CITE = re.compile(r"\[([A-Z]\d+)\]")


def split_claims(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 3]


def content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", _CITE.sub(" ", text).lower()) if t not in _STOP and len(t) > 2]


class NLI:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device="cpu")
        labels = [lab.lower() for lab in self.model.config.id2label.values()]
        self.entail_idx = labels.index("entailment")

    def p_entail(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(pairs, show_progress_bar=False, apply_softmax=False))
        logits = logits - logits.max(-1, keepdims=True)
        p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        return p[:, self.entail_idx].tolist()


def coverage(explanation: str, units: list[EvidenceUnit], nli: NLI, threshold: float = 0.5) -> dict:
    by_id = {u.id: u for u in units}
    claims = split_claims(explanation)
    if not claims:
        return {"claims": [], "claim_coverage": None, "token_coverage": None, "unsupported": None}
    rows, pairs = [], []
    for c in claims:
        cited = [i for i in _CITE.findall(c) if i in by_id]
        rows.append({"claim": c, "cited": cited, "p_entail": 0.0, "supported": False})
        if cited:
            pairs.append((len(rows) - 1, " ".join(by_id[i].text for i in cited), _CITE.sub("", c)))
    for (ri, _, _), p in zip(pairs, nli.p_entail([(prem, hyp) for _, prem, hyp in pairs])):
        rows[ri]["p_entail"] = round(p, 4)
        rows[ri]["supported"] = p >= threshold
    cited_text = " ".join(by_id[i].text for r in rows for i in r["cited"])
    ev_tokens = set(content_tokens(cited_text))
    toks = content_tokens(explanation)
    cov = sum(r["supported"] for r in rows) / len(rows)
    return {"claims": rows, "claim_coverage": round(cov, 4),
            "token_coverage": round(sum(t in ev_tokens for t in toks) / len(toks), 4) if toks else None,
            "unsupported": round(1 - cov, 4)}
