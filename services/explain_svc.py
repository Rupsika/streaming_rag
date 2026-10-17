"""Explain service: events.detected -> events.explained (plan.md §4 M9).

Off the hot path: alerts are already out when this runs. For every closed event it
re-scores the peak window's decision under evidence ablations (E1 LOO, E2 temporal heat,
E3 NLI coverage, E4 counterfactual top-k vs random-k, E5 KG perturbation) and appends
one record per event to runs/<run_id>/audit.jsonl (CityCopilot-X's audit bus idea).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_json
from xstream.config import ROOT
from xstream.service import JsonlWriter, Service, is_eos_json, main_for
from xstream.telemetry import STAGE_LATENCY, get_tracer
from xstream.types import EvidenceUnit
from xstream.vqg import load_kb


class ExplainService(Service):
    name = "explain"
    in_keys = ["events_detected"]
    out_key = "events_explained"

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.enabled = cfg["explain"]["enabled"]
        if self.enabled:
            from xstream.encode import make_embedder
            from xstream.explain.coverage import NLI
            from xstream.generate import Generator

            self.gen = Generator(cfg, load_kb(ROOT / cfg["kb_path"]))
            self.nli = NLI(cfg["explain"]["nli_model"])
            self.embedder = make_embedder(cfg)
        self.audit = JsonlWriter(self.dir / "audit.jsonl")
        self.tracer = get_tracer("explain", cfg)

    def handle(self, m) -> None:
        d = decode_json(m.value())
        if is_eos_json(d):
            self.emit_eos(d["source_id"])
            return
        if d.get("kind") != "event" or d["status"] != "close":
            return
        rec = {k: v for k, v in d.items() if k != "peak_verdict"}
        pv = d.get("peak_verdict")
        if self.enabled and pv and pv.get("evidence"):
            from xstream.explain import explain_event

            units = [EvidenceUnit(**u) for u in pv["evidence"]]
            label = d["event_type"]
            t0 = time.perf_counter()
            with self.tracer.start_as_current_span("explain") as span:
                span.set_attribute("event.id", d["event_id"])
                rec["explain"] = explain_event(self.gen, self.nli, self.embedder, self.cfg, pv["header"], units, label,
                                               pv.get("explanation", ""), pv["t0"], pv["t1"])
            rec["peak_window"] = pv["window_id"]
            rec["evidence"] = pv["evidence"]
            STAGE_LATENCY.labels("explain").observe(time.perf_counter() - t0)
            self.log.info("explained %s in %.1fs", d["event_id"], time.perf_counter() - t0)
        rec["explained_wallclock"] = time.time()
        self.audit.write(rec)
        self.emit(rec, d["source_id"])


if __name__ == "__main__":
    main_for(ExplainService)
