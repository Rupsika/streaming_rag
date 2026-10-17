"""Windowing service: frames.signal -> windows.keyframes (plan.md §4 M3).

Runs one windower per source (adaptive B3 / fixed B2 / per-frame B1, `windowing.mode`)
and owns the 3 s ring buffer used for pre-event back-fill. Closed windows are emitted as
JSON with their keyframes' JPEGs (base64) for the VLM stage.

With `windowing.feedback: true` (live demo) it also reads LLM verdicts from
events.detected to enter the EVENT state. In fast-replay evaluation this feedback is off:
the windowing stage runs far ahead of the LLM there, so feedback would arrive late and make
runs non-deterministic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_json
from xstream.service import JsonlWriter, Service, is_eos_json, main_for, topic
from xstream.telemetry import FRAMES_INGESTED, FRAMES_TO_VLM, STAGE_LATENCY, WINDOW_STATE
from xstream.windowing import STATE_CODE, SignalFrame, make_windower


class WindowingService(Service):
    name = "windowing"
    in_keys = ["frames_signal"]
    out_key = "windows_keyframes"

    def __init__(self, cfg, args):
        if cfg["windowing"].get("feedback", False):
            self.in_keys = ["frames_signal", "events_detected"]
        super().__init__(cfg, args)
        self.windowers: dict = {}
        self.log_w = JsonlWriter(self.dir / "windows.jsonl")
        self.fb_topic = topic(cfg, "events_detected")

    def handle(self, m) -> None:
        if m.topic() == self.fb_topic:
            d = decode_json(m.value())
            if d.get("kind") == "window" and d["source_id"] in self.windowers:
                hazard = any(a["a"] == "yes" for u in d.get("evidence", []) if u["kind"] == "vqa"
                             for a in (u["source"].get("answers") or {}).values())
                self.windowers[d["source_id"]].on_feedback(d["event_type"], d["p_event"], hazard,
                                                           self.cfg["events"]["open_threshold"])
            return
        hdr = dict(m.headers() or [])
        source = hdr["source_id"].decode()
        w = self.windowers.get(source)
        if hdr.get("eos") == b"1":
            if w is not None:
                for win in w.flush():
                    self._emit(win)
                if hasattr(w, "change_points"):
                    self.log_w.write({"kind": "adaptive_trace", "source_id": source, "change_points": w.change_points,
                                      "transitions": w.transitions})
                self.windowers.pop(source)
            self.emit_eos(source)
            return
        if w is None:
            w = self.windowers[source] = make_windower(self.cfg, source)
        t0 = time.perf_counter()
        sf = SignalFrame(frame_id=int(hdr["frame_id"]), video_ts=float(hdr["video_ts"]),
                         ingest_ts=float(hdr["ingest_ts"]), a_t=float(hdr["a_t"]), jpeg=m.value(),
                         clip=np.frombuffer(hdr["clip"], np.float16).astype(np.float32) if "clip" in hdr else None)
        FRAMES_INGESTED.labels("windowing").inc()
        for win in w.step(sf):
            self._emit(win)
        WINDOW_STATE.labels(source).set(STATE_CODE[w.state])
        STAGE_LATENCY.labels("windowing").observe(time.perf_counter() - t0)

    def _emit(self, win) -> None:
        d = win.to_dict()
        d["mode"] = self.cfg["windowing"]["mode"]
        d["closed_wallclock"] = time.time()
        self.emit(d, win.source_id)
        FRAMES_TO_VLM.inc(len(win.keyframes))
        meta = {k: v for k, v in d.items() if k not in ("keyframes", "clip")}
        meta["keyframes"] = [{"id": k.id, "video_ts": k.video_ts, "reason": k.reason, "priority": k.priority}
                             for k in win.keyframes]
        self.log_w.write(meta)


if __name__ == "__main__":
    main_for(WindowingService)
