"""VLM + KG service: windows.keyframes -> windows.closed (plan.md §4 M4, M5).

Per keyframe: the Visual Query Generator picks <= 5 questions, the (cached) VLM returns a
caption + answers, and the answers become KG triples. Per window: KG 1-hop context and the
text summary (TD-RAG's window representation) are attached. Keyframe JPEGs are written to
runs/<run_id>/keyframes/ for the dashboard and dropped from the message.
"""
from __future__ import annotations

import base64
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_json
from xstream.config import ROOT
from xstream.kg import TemporalKG, window_summary
from xstream.service import JsonlWriter, Service, is_eos_json, main_for
from xstream.telemetry import STAGE_LATENCY, VLM_CALLS, get_tracer
from xstream.types import Window
from xstream.vlm import CachedVLM
from xstream.vqg import VisualQueryGenerator, load_kb


class VlmKgService(Service):
    name = "vlm"
    in_keys = ["windows_keyframes"]
    out_key = "windows_closed"

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.kb = load_kb(ROOT / cfg["kb_path"])
        self.vlm = CachedVLM(cfg)
        self.vqg = VisualQueryGenerator(self.kb, cfg["vlm"]["max_questions"], cfg["vlm"].get("vqg_memory_s", 4.0))
        self.kgs: dict[str, TemporalKG] = {}
        self.done: OrderedDict = OrderedDict()   # keyframe id -> (caption, vqa): overlapping B2 windows
        self.calls_log = JsonlWriter(self.dir / "vlm_calls.jsonl")
        self.tracer = get_tracer("vlm", cfg)

    def handle(self, m) -> None:
        d = decode_json(m.value())
        if is_eos_json(d):
            src = d["source_id"]
            if src in self.kgs:
                self._snapshot(src)
                self.kgs.pop(src)
            self.vqg.reset(src)
            self.emit_eos(src)
            return
        mode = d.pop("mode", None)
        closed_wall = d.pop("closed_wallclock", None)
        w = Window.from_dict(d)
        kg = self.kgs.setdefault(w.source_id, TemporalKG(self.kb, self.cfg.get("kg_ttl_s", 120.0)))
        t_start = time.perf_counter()
        with self.tracer.start_as_current_span("window") as span:
            span.set_attribute("window.id", f"{w.source_id}/{w.id}")
            with self.tracer.start_as_current_span("extract"):
                for kf in w.keyframes:
                    if kf.id in self.done:
                        kf.caption, kf.vqa = self.done[kf.id]
                    else:
                        jpeg = base64.b64decode(kf.jpeg_b64)
                        qs = self.vqg.questions(w.source_id, kf.video_ts)
                        before = dict(self.vlm.calls)
                        t0 = time.perf_counter()
                        kf.caption, kf.vqa = self.vlm.describe(w.source_id, kf.frame_id, jpeg, qs)
                        for kind in ("caption", "vqa"):
                            VLM_CALLS.labels(kind).inc(self.vlm.calls[kind] - before[kind])
                        self.vqg.observe(w.source_id, kf.video_ts, kf.vqa)
                        self.done[kf.id] = (kf.caption, kf.vqa)
                        while len(self.done) > 512:
                            self.done.popitem(last=False)
                        out = self.dir / "keyframes" / w.source_id
                        out.mkdir(parents=True, exist_ok=True)
                        (out / f"{kf.frame_id}.jpg").write_bytes(jpeg)
                        self.calls_log.write({"source_id": w.source_id, "keyframe": kf.id, "video_ts": kf.video_ts,
                                              "reason": kf.reason, "questions": [q["id"] for q in qs],
                                              "seconds": round(time.perf_counter() - t0, 3),
                                              "cached": self.vlm.calls == before})
                    kf.jpeg_b64 = None
            with self.tracer.start_as_current_span("kg_update"):
                w.triples = kg.add_window(w)
                w.kg_context = kg.one_hop(w.triples, w.t1)
                w.summary = window_summary(w, self.kb, w.triples)
        out = w.to_dict()
        out.update(mode=mode, closed_wallclock=closed_wall, vlm_done_wallclock=time.time())
        self.emit(out, w.source_id)
        STAGE_LATENCY.labels("vlm_kg").observe(time.perf_counter() - t_start)
        if int(w.id[1:]) % 10 == 0:
            self._snapshot(w.source_id)

    def _snapshot(self, source: str) -> None:
        import json

        p = self.dir / "kg" / f"{source}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.kgs[source].snapshot(), default=str))


if __name__ == "__main__":
    main_for(VlmKgService)
