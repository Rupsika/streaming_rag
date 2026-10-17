"""Evidence assembly, hybrid retrieval and rerank (plan.md §4 M7; CityCopilot-X retrieval &
reranking attribution).

Evidence units (plan.md §9 EvidenceUnit):
- F*: captions and VQA answers of the current window's keyframes,
- T*: KG triples of the window plus its 1-hop KG context,
- H*: similar past windows from the case library (label carried along),
- S*: similar earlier windows of this stream (session memory, recency-constrained).

Retrieval over the case library + session memory is wrapped as a LangChain
`BaseRetriever` (dense fused text+CLIP, plus BM25 when `retrieval.hybrid`). The
cross-encoder then reranks all candidates and the top `final_k` are kept. Every stage's
score is stored in `EvidenceUnit.scores` so the dashboard can show why a unit made the cut.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from xstream.index import CaseLibrary, SessionMemory
from xstream.types import EvidenceUnit

_TOK = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def current_units(window: dict, kb: dict, kg_context: bool = True) -> list[EvidenceUnit]:
    """F* and T* units from the current window (no retrieval involved)."""
    qtext = {qid: q["text"] for qid, q in kb["questions"].items()}
    units, n = [], 0
    for kf in window["keyframes"]:
        n += 1
        units.append(EvidenceUnit(f"F{n}", "frame_caption", f"t={kf['video_ts']:.1f} caption: {kf['caption']}",
                                  kf["video_ts"], kf["video_ts"], {"keyframe": kf["id"]}, {}))
        qa = [f"{qtext[q]} {a['a']}" for q, a in (kf.get("vqa") or {}).items()
              if a["a"] == "yes" or a.get("p") is None]
        if qa:
            n += 1
            units.append(EvidenceUnit(f"F{n}", "vqa", f"t={kf['video_ts']:.1f} " + "; ".join(qa),
                                      kf["video_ts"], kf["video_ts"],
                                      {"keyframe": kf["id"], "answers": kf.get("vqa")}, {}))
    triples = list(window.get("triples", [])) + (list(window.get("kg_context", [])) if kg_context else [])
    for t in triples:
        units.append(EvidenceUnit(t["id"], "kg_triple", f"({t['s']} {t['r']} {t['o']}) {t['t0']:.1f}-{t['t1']:.1f} s",
                                  t["t0"], t["t1"], {"triple": t, "context": t["window"] != window["id"]},
                                  {"conf": t["conf"]}))
    return units


class EvidenceRetriever(BaseRetriever):
    """LangChain retriever over the case library (H*) and session memory (S*).

    Query = text of the current window (+ previous windows for sequence context);
    `query_clip`, `now`, `video` are bound per call with `model_copy(update=...)`.
    """
    library: Any = None               # CaseLibrary | None
    session: Any = None               # SessionMemory
    embedder: Any = None
    cfg: dict = {}
    query_clip: Any = None
    now: float = 0.0
    video: str = ""
    exclude_ids: set = set()
    bm25_cache: dict = {}             # shared across model_copy() calls (shallow copy)

    def _bm25_index(self):
        """BM25 over case-library + session texts (rebuilt when session memory grows)."""
        n = (len(self.library) if self.library else 0) + len(self.session.items)
        if self.bm25_cache.get("n") != n:
            docs = [("H", i, m["text"]) for i, m in enumerate(self.library.meta)] if self.library else []
            docs += [("S", i, m["text"]) for i, m in enumerate(self.session.items)]
            self.bm25_cache.update(n=n, index=(BM25Okapi([tokenize(t) for _, _, t in docs]) if docs else None, docs))
        return self.bm25_cache["index"]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        r, e = self.cfg["retrieval"], self.cfg["embedder"]
        qv = self.embedder.embed([query])[0]
        qc = np.asarray(self.query_clip, np.float32) if self.query_clip is not None and r.get("use_clip", True) else None
        w_clip = e["w_clip"] if qc is not None else 0.0
        hits: dict[tuple, dict] = {}
        if self.library is not None:
            excl = self.video if r.get("exclude_same_video", True) else None
            for rank, (i, s) in enumerate(self.library.search(qv, qc, r["case_library_k"], e["w_text"], w_clip, excl)):
                hits[("H", i)] = {"dense": round(s, 4), "dense_rank": rank + 1}
        if r.get("session_k", 0):
            for rank, (i, s) in enumerate(self.session.search(qv, qc, r["session_k"], self.now, self.video,
                                                              e["w_text"], w_clip, self.exclude_ids)):
                hits[("S", i)] = {"dense": round(s, 4), "dense_rank": rank + 1}
        if r.get("hybrid", False) and r.get("bm25_k", 0):
            bm, docs = self._bm25_index()
            if bm is not None:
                sc = bm.get_scores(tokenize(query))
                order, rank = np.argsort(-sc), 0
                for j in order:
                    src, i, _ = docs[j]
                    meta = self.library.meta[i] if src == "H" else self.session.items[i]
                    if src == "H" and r.get("exclude_same_video", True) and meta["video"] == self.video:
                        continue
                    if src == "S" and (meta["video"] != self.video or meta["id"] in self.exclude_ids
                                       or not 0 <= self.now - meta["t1"] <= self.session.recency_s):
                        continue
                    rank += 1
                    hits.setdefault((src, i), {}).update({"bm25": round(float(sc[j]), 4), "bm25_rank": rank})
                    if rank >= r["bm25_k"]:
                        break
        docs = []
        for (src, i), scores in hits.items():
            meta = self.library.meta[i] if src == "H" else self.session.items[i]
            docs.append(Document(page_content=meta["text"], metadata={"src": src, "meta": meta, "scores": scores}))
        docs.sort(key=lambda d: (-d.metadata["scores"].get("dense", -1), -d.metadata["scores"].get("bm25", -1)))
        return docs


def docs_to_units(docs: list[Document], max_chars: int = 400) -> list[EvidenceUnit]:
    units, nh, ns = [], 0, 0
    for d in docs:
        m, s = d.metadata["meta"], d.metadata["scores"]
        if d.metadata["src"] == "H":
            nh += 1
            text = f"past window (label={m['label']}, sim={s.get('dense', 0):.2f}): {d.page_content[:max_chars]}"
            units.append(EvidenceUnit(f"H{nh}", "hist_window", text, None, None,
                                      {"case_id": m["id"], "label": m["label"], "video": m["video"]}, dict(s)))
        else:
            ns += 1
            text = f"earlier window of this stream ({m['t0']:.1f}-{m['t1']:.1f} s): {d.page_content[:max_chars]}"
            units.append(EvidenceUnit(f"S{ns}", "session_window", text, m["t0"], m["t1"],
                                      {"window": m["id"]}, dict(s)))
    return units


class Reranker:
    """cross-encoder/ms-marco-MiniLM-L-6-v2 (Nogueira & Cho 2019) over all candidate units."""

    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device="cpu")

    def rerank(self, query: str, units: list[EvidenceUnit], k: int, min_current: int = 2) -> list[EvidenceUnit]:
        if not units:
            return []
        scores = self.model.predict([(query, u.text) for u in units], show_progress_bar=False)
        for u, s in zip(units, scores):
            u.scores["rerank"] = round(float(s), 4)
        ranked = sorted(units, key=lambda u: -u.scores["rerank"])
        for i, u in enumerate(ranked):
            u.scores["final_rank"] = i + 1
        keep = ranked[:k]
        # keep at least `min_current` units observed in this window, so the LLM always sees the scene
        cur = [u for u in ranked[k:] if u.kind in ("frame_caption", "vqa", "kg_triple")]
        while sum(u.kind in ("frame_caption", "vqa", "kg_triple") for u in keep) < min_current and cur:
            drop = next((u for u in reversed(keep) if u.kind in ("hist_window", "session_window")), None)
            if drop is None:
                break
            keep[keep.index(drop)] = cur.pop(0)
        return keep
