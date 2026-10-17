"""RAG service: windows.closed -> events.detected (plan.md §4 M6-M8).

encode (bge-small + mean CLIP) -> retrieve (LangChain retriever: case library + session
memory, + BM25 if hybrid) -> rerank (cross-encoder, B3) -> generate (LangChain prompt ->
LLM -> JSON, label-probability confidence) -> event debounce.

Emits one `{"kind": "window"}` verdict per window and `{"kind": "event"}` records when an
event opens/closes. Writes runs/<run_id>/verdicts.jsonl and events.jsonl.
Baselines differ only by config: `retrieval.level` frame (B1) | window (B2/B3),
`retrieval.hybrid`, `retrieval.rerank`, `retrieval.sequence_context`.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_json
from xstream.config import ROOT
from xstream.encode import make_embedder
from xstream.events import EventTracker
from xstream.explain.drift import DriftMonitor
from xstream.generate import Generator
from xstream.index import CaseLibrary, SessionMemory
from xstream.retrieve import EvidenceRetriever, Reranker, current_units, docs_to_units
from xstream.service import JsonlWriter, Service, is_eos_json, main_for
from xstream.telemetry import E2E_ALERT_LATENCY, STAGE_LATENCY, get_tracer
from xstream.types import EvidenceUnit
from xstream.vqg import load_kb


def load_library(cfg: dict) -> CaseLibrary | None:
    r = cfg["retrieval"]
    d = ROOT / r["case_library_dir"] / ("frames" if r["level"] == "frame" else "windows")
    if not (d / "meta.jsonl").exists():
        return None
    return CaseLibrary.load(d)


class RagService(Service):
    name = "rag"
    in_keys = ["windows_closed"]
    out_key = "events_detected"

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        r = cfg["retrieval"]
        self.r = r
        self.kb = load_kb(ROOT / cfg["kb_path"])
        self.embedder = make_embedder(cfg)
        self.library = load_library(cfg)
        if self.library is None:
            self.log.warning("NO CASE LIBRARY at %s - run scripts/build_case_library.py; H* units disabled",
                             r["case_library_dir"])
        else:
            self.log.info("case library: %d %s units", len(self.library), r["level"])
        self.reranker = Reranker(r["reranker"]) if r.get("rerank") else None
        self.gen = Generator(cfg, self.kb)
        self.state: dict[str, dict] = {}
        self.verdicts = JsonlWriter(self.dir / "verdicts.jsonl")
        self.events = JsonlWriter(self.dir / "events.jsonl")
        self.tracer = get_tracer("rag", cfg)

    def _src(self, source: str) -> dict:
        if source not in self.state:
            self.state[source] = {
                "session": SessionMemory(self.r["session_recency_s"]), "prev": [],
                "tracker": EventTracker(self.cfg, source),
                "drift": DriftMonitor(self.cfg["explain"]["drift_recent_s"], self.cfg["explain"]["drift_ref_s"],
                                      self.cfg["explain"]["drift_threshold"]),
                "retriever": EvidenceRetriever(library=self.library, embedder=self.embedder, cfg=self.cfg,
                                               session=None),
            }
            st = self.state[source]
            st["retriever"].session = st["session"]
        return self.state[source]

    def handle(self, m) -> None:
        w = decode_json(m.value())
        if is_eos_json(w):
            st = self.state.pop(w["source_id"], None)
            if st:
                for rec in st["tracker"].flush():
                    self._emit_event(rec, st)
            self.emit_eos(w["source_id"])
            return
        if not w["keyframes"]:
            return
        source = w["source_id"]
        st = self._src(source)
        t_start = time.perf_counter()
        with self.tracer.start_as_current_span("window") as span:
            span.set_attribute("window.id", f"{source}/{w['id']}")
            with self.tracer.start_as_current_span("encode"):
                vec = self.embedder.embed([w["summary"]])[0]
                clip = np.asarray(w["clip"], np.float32) if w.get("clip") else None
            n_ctx = self.r.get("sequence_context", 0)
            prev = st["prev"][-n_ctx:] if n_ctx else []
            query = " ".join([p["summary"] for p in prev] + [w["summary"]])
            with self.tracer.start_as_current_span("retrieve"):
                retriever = st["retriever"].model_copy(update={
                    "query_clip": clip, "now": w["t1"], "video": source,
                    "exclude_ids": {p["id"] for p in prev}})
                docs = retriever.invoke(query)
                retrieved = docs_to_units(docs, self.r.get("unit_chars", 400))
                cur = current_units(w, self.kb, self.r.get("kg_context", True))[: self.r.get("max_current_units", 8)]
                seq = [EvidenceUnit(f"P{i + 1}", "session_window",
                                    f"previous window {p['t0']:.1f}-{p['t1']:.1f} s: {p['summary'][:400]}",
                                    p["t0"], p["t1"], {"window": p["id"], "sequence_context": True}, {})
                       for i, p in enumerate(prev)]
            if self.reranker is not None:
                with self.tracer.start_as_current_span("rerank"):
                    rq = " ".join(self.kb["standing_queries"]) + " " + w["summary"][:600]
                    units = self.reranker.rerank(rq, cur + seq + retrieved, self.r["final_k"],
                                                 self.r.get("min_current_units", 2))
            else:
                units = cur + seq + retrieved[: self.r["report_k"]]
            header = f"{w['id']} {w['t0']:.1f}-{w['t1']:.1f} s (state {w['state']})"
            with self.tracer.start_as_current_span("generate"):
                t_gen = time.perf_counter()
                pred = self.gen.predict(header, units, w["t0"], w["t1"])
                gen_s = time.perf_counter() - t_gen
        drift = st["drift"].update(w["t1"], vec)
        lib_hits = [d.metadata for d in docs if d.metadata["src"] == "H"]
        lib_hits.sort(key=lambda h: -h["scores"].get("dense", -1))
        verdict = {
            "kind": "window", "window_id": w["id"], "source_id": source, "t0": w["t0"], "t1": w["t1"],
            "state": w["state"], "mode": w.get("mode"), "header": header, **pred,
            "evidence": [asdict(u) for u in units],
            "retrieval": {"case_top": [{"label": h["meta"]["label"], "video": h["meta"]["video"],
                                        "id": h["meta"]["id"], **h["scores"]} for h in lib_hits[: self.r["report_k"]]],
                          "n_candidates": len(docs) + len(cur) + len(seq)},
            "keyframes": [{"id": k["id"], "video_ts": k["video_ts"], "reason": k["reason"], "priority": k["priority"]}
                          for k in w["keyframes"]],
            "summary": w["summary"], "first_ingest_ts": w["first_ingest_ts"], "last_ingest_ts": w["last_ingest_ts"],
            "closed_wallclock": w.get("closed_wallclock"), "vlm_done_wallclock": w.get("vlm_done_wallclock"),
            "verdict_wallclock": time.time(), "rag_seconds": round(time.perf_counter() - t_start, 3),
            "gen_seconds": round(gen_s, 3), "drift": drift,
        }
        self.emit(verdict, source)
        self.verdicts.write(verdict)
        STAGE_LATENCY.labels("rag").observe(time.perf_counter() - t_start)
        st["session"].add({"id": w["id"], "video": source, "t0": w["t0"], "t1": w["t1"], "text": w["summary"],
                           "state": w["state"]}, vec, clip)
        st["prev"] = (st["prev"] + [{"id": w["id"], "t0": w["t0"], "t1": w["t1"], "summary": w["summary"]}])[-3:]
        for rec in st["tracker"].update(verdict):
            self._emit_event(rec, st)

    def _emit_event(self, rec: dict, st: dict) -> None:
        if rec["status"] == "open":
            E2E_ALERT_LATENCY.observe(rec["alert_wallclock"] - rec["first_ingest_ts"])
        self.emit(rec, rec["source_id"])
        self.events.write(rec)


if __name__ == "__main__":
    main_for(RagService)
