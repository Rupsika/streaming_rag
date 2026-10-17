"""Signal service: frames.raw -> frames.signal (plan.md §4 M2).

Samples ~`signal.sample_fps` (10) frames/s per source, computes the change signal
a_t = 0.6·(1 − cos CLIP) + 0.4·pixel diff in CLIP batches, and forwards the JPEG with
a_t and the CLIP embedding (float16) in the headers. Also logs a_t per source to
runs/<run_id>/signal/<source>.csv for plots and evaluation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_frame
from xstream.change import ChangeSignal, ClipImageEncoder
from xstream.service import Service, main_for
from xstream.telemetry import CHANGE_SIGNAL, FPS, FRAMES_INGESTED, STAGE_LATENCY, RateMeter


class SignalService(Service):
    name = "signal"
    in_keys = ["frames_raw"]
    out_key = "frames_signal"

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        s = cfg["signal"]
        self.sample_fps, self.batch_size = s["sample_fps"], s["batch_size"]
        self.change = ChangeSignal(ClipImageEncoder(s["clip_model"]), s["w_semantic"], s["w_pixel"])
        self.pending: dict[str, list] = {}        # source -> [(frame, jpeg, headers)]
        self.meter = RateMeter()
        self.csv: dict[str, object] = {}

    def handle(self, m) -> None:
        hdr = dict(m.headers() or [])
        source = hdr["source_id"].decode()
        if hdr.get("eos") == b"1":
            self._process(source)
            self.change.reset(source)
            if source in self.csv:
                self.csv.pop(source).close()
            self.emit_raw(b"", source, [("source_id", source.encode()), ("eos", b"1")])
            self.producer.flush(30)
            self.log.info("EOS %s", source)
            return
        FRAMES_INGESTED.labels("signal").inc()
        fps = float(hdr.get("fps", b"30"))
        stride = max(1, round(fps / self.sample_fps))
        if int(hdr["frame_id"]) % stride:
            return
        frame = decode_frame(m.value(), m.headers())
        self.pending.setdefault(source, []).append((frame, m.value(), m.headers()))
        if len(self.pending[source]) >= self.batch_size:
            self._process(source)

    def on_idle(self) -> None:
        for source in list(self.pending):
            self._process(source)

    def _process(self, source: str) -> None:
        batch = self.pending.pop(source, [])
        if not batch:
            return
        t0 = time.perf_counter()
        sig = self.change.step_batch(source, [f.bgr for f, _, _ in batch])
        dt = (time.perf_counter() - t0) / len(batch)
        if source not in self.csv:
            path = self.dir / "signal" / f"{source}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            self.csv[source] = open(path, "a", encoding="utf-8")
            if new:
                self.csv[source].write("frame_id,video_ts,ingest_ts,a_t,semantic,pixel\n")
        for (frame, jpeg, headers), s in zip(batch, sig):
            extra = [("a_t", repr(s["a_t"]).encode()), ("semantic", repr(s["semantic"]).encode()),
                     ("pixel", repr(s["pixel"]).encode()), ("clip", s["clip"].astype(np.float16).tobytes())]
            self.emit_raw(jpeg, source, list(headers) + extra)
            self.csv[source].write(f"{frame.id},{frame.video_ts:.4f},{frame.ingest_ts:.4f},{s['a_t']:.6f},"
                                   f"{s['semantic']:.6f},{s['pixel']:.6f}\n")
            STAGE_LATENCY.labels("signal").observe(dt)
            FPS.labels("signal").set(self.meter.tick())
            CHANGE_SIGNAL.labels(source).set(s["a_t"])
        self.csv[source].flush()


if __name__ == "__main__":
    cv2.setNumThreads(2)
    main_for(SignalService)
