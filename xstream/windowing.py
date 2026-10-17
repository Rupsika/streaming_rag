"""Temporal windowing: adaptive (B3), fixed (B2) and per-frame (B1) (plan.md §4 M3, §5).

All three share one interface so the windowing service switches between them
with `windowing.mode` only:

    w = make_windower(cfg, source_id)
    for sf in signal_frames:          # SignalFrame, ~10 fps, in video order
        for win in w.step(sf): ...    # closed Window objects
    for win in w.flush(): ...         # end of stream

Keyframe budget (fair-budget rule, plan.md §5): B2 samples uniformly at
`vlm_budget_per_min / 60` keyframes/s; B3 spends the same budget through a token
bucket, so adaptive sampling may only *redistribute* VLM calls, never add them.
B1 keeps StreamingRAG's fixed 2 fps and is not budget-matched (it is the baseline).
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass

import numpy as np
from river.drift import ADWIN

from xstream.types import Keyframe, Window

STATE_CODE = {"CALM": 0, "ALERT": 1, "EVENT": 2}


@dataclass
class SignalFrame:
    """One sampled frame with its change signal (output of the signal service)."""
    frame_id: int
    video_ts: float
    ingest_ts: float
    a_t: float
    jpeg: bytes
    clip: np.ndarray | None = None


class RollingMeanStd:
    """Mean/std of a_t over the last `seconds` of video time (plan.md §4 M3: 30 s)."""

    def __init__(self, seconds: float = 30.0):
        self.seconds = seconds
        self._buf: deque[tuple[float, float]] = deque()

    def add(self, ts: float, x: float) -> None:
        self._buf.append((ts, x))
        while self._buf and ts - self._buf[0][0] > self.seconds:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def mean_std(self) -> tuple[float, float]:
        if not self._buf:
            return 0.0, 0.0
        xs = np.fromiter((x for _, x in self._buf), float)
        return float(xs.mean()), float(xs.std())

    def z(self, x: float, std_floor: float = 1e-3) -> float:
        mu, sd = self.mean_std()
        return (x - mu) / max(sd, std_floor)


class TokenBucket:
    """Average-rate limiter for VLM keyframes (fair-budget rule)."""

    def __init__(self, per_min: float, burst_s: float):
        self.rate = per_min / 60.0
        self.capacity = max(1.0, self.rate * burst_s)
        self.tokens = self.capacity
        self._last: float | None = None

    def refill(self, ts: float) -> None:
        if self._last is not None:
            self.tokens = min(self.capacity, self.tokens + (ts - self._last) * self.rate)
        self._last = ts

    def take(self) -> bool:
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def _keyframe(source_id: str, sf: SignalFrame, reason: str) -> Keyframe:
    return Keyframe(id=f"{source_id}:{sf.frame_id}", frame_id=sf.frame_id, video_ts=sf.video_ts,
                    ingest_ts=sf.ingest_ts, priority=round(sf.a_t, 5), reason=reason,
                    jpeg_b64=base64.b64encode(sf.jpeg).decode())


class _WindowBuilder:
    def __init__(self, wid: str, source_id: str, t0: float, state: str):
        self.id, self.source_id, self.t0, self.state = wid, source_id, t0, state
        self.t1 = t0
        self.keyframes: list[Keyframe] = []
        self.signal: list[list[float]] = []
        self.clips: list[np.ndarray] = []
        self.first_ingest = self.last_ingest = None

    @property
    def length(self) -> float:
        return self.t1 - self.t0

    def add(self, sf: SignalFrame) -> None:
        self.t1 = sf.video_ts
        self.signal.append([round(sf.video_ts, 3), round(sf.a_t, 5)])
        self.first_ingest = sf.ingest_ts if self.first_ingest is None else self.first_ingest
        self.last_ingest = sf.ingest_ts

    def add_keyframe(self, kf: Keyframe, clip: np.ndarray | None) -> None:
        if any(k.frame_id == kf.frame_id for k in self.keyframes):
            return
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.video_ts)
        if clip is not None:
            self.clips.append(clip)

    def close(self) -> Window:
        clip = None
        if self.clips:
            m = np.mean(self.clips, axis=0)
            clip = (m / (np.linalg.norm(m) + 1e-8)).round(5).tolist()
        return Window(id=self.id, source_id=self.source_id, t0=round(self.t0, 3), t1=round(self.t1, 3),
                      state=self.state, keyframes=self.keyframes,
                      a_max=max((a for _, a in self.signal), default=0.0),
                      first_ingest_ts=self.first_ingest or 0.0, last_ingest_ts=self.last_ingest or 0.0,
                      clip=clip, signal=self.signal)


class AdaptiveWindower:
    """B3: ADWIN change points + CALM/ALERT/EVENT FSM + pre-event back-fill (plan.md §4 M3).

    - ADWIN (Bifet & Gavaldà 2007) on a_t marks change points that close windows.
    - CALM -> ALERT when a_t > μ + kσ (rolling 30 s), an ADWIN change, or a hazard
      verdict fed back from downstream; on entering ALERT the top-`backfill_top`
      a_t frames of the 3 s ring buffer are also sent to the VLM, so a 1 s crash
      between two CALM samples is still seen (RQ3).
    - ALERT -> EVENT when the LLM reports an event (feedback); back to CALM after
      `calm_exit_seconds` with a_t < μ + 1σ and no hazard.
    - Within each state's sampling interval the frame with the highest a_t is chosen
      (StreamingRAG's priority frame scheduling).
    """

    def __init__(self, cfg: dict, source_id: str):
        w = cfg["windowing"]
        self.source_id = source_id
        self.rates, self.max_len, self.min_len = w["keyframe_rate"], w["max_len"], w["min_len"]
        self.k_sigma, self.exit_sigma, self.exit_s = w["k_sigma"], w["calm_exit_sigma"], w["calm_exit_seconds"]
        self.warmup_s = w.get("warmup_s", 2.0)
        self.min_rise = w.get("min_rise", 0.02)   # absolute floor: z-trigger also needs a_t - μ > min_rise
        self.backfill_top = w["backfill_top"] if w.get("backfill", True) else 0
        self.adwin = ADWIN(delta=w["adwin_delta"], clock=w.get("adwin_clock", 8))
        self.stats = RollingMeanStd(w["stats_seconds"])
        self.ring: deque[SignalFrame] = deque()
        self.ring_s = w["ring_buffer_seconds"]
        self.bucket = TokenBucket(w["vlm_budget_per_min"], w.get("budget_burst_s", 20.0))
        self.state = "CALM"
        self.n = 0
        self.win: _WindowBuilder | None = None
        self.candidate: SignalFrame | None = None   # best frame of the current sampling interval
        self.next_sample_ts = 0.0
        self.last_hot_ts = 0.0                     # last time the scene looked non-calm
        self.hazard = False
        self.change_points: list[float] = []
        self.transitions: list[tuple[float, str]] = []

    # -- feedback from downstream (vlm hazard answers / LLM verdicts) -------------
    def on_feedback(self, event_type: str, p_event: float, hazard: bool, threshold: float = 0.5) -> None:
        if event_type != "none" and p_event >= threshold:
            self._set_state("EVENT", self.ring[-1].video_ts if self.ring else 0.0)
        self.hazard = hazard or (event_type != "none" and p_event >= threshold)

    def _new_window(self, t0: float) -> None:
        self.n += 1
        self.win = _WindowBuilder(f"W{self.n}", self.source_id, t0, self.state)

    def _close(self) -> Window | None:
        if self.win is None:
            return None
        w = self.win
        self.win = None
        return w.close() if w.keyframes else None

    def _set_state(self, state: str, ts: float) -> None:
        if state != self.state:
            self.state = state
            self.transitions.append((ts, state))
            if self.win is not None:
                self.win.state = max(self.win.state, state, key=STATE_CODE.get)

    def _take_keyframe(self, sf: SignalFrame, reason: str) -> None:
        if self.bucket.take():
            self.win.add_keyframe(_keyframe(self.source_id, sf, reason), sf.clip)

    def step(self, sf: SignalFrame) -> list[Window]:
        ts = sf.video_ts
        out: list[Window] = []
        self.bucket.refill(ts)
        self.ring.append(sf)
        while self.ring and ts - self.ring[0].video_ts > self.ring_s:
            self.ring.popleft()

        self.adwin.update(sf.a_t)
        change = self.adwin.drift_detected
        if change:
            self.change_points.append(ts)
        warm = len(self.stats) and ts - self.stats._buf[0][0] >= self.warmup_s
        z = self.stats.z(sf.a_t) if warm else 0.0
        if warm and sf.a_t - self.stats.mean_std()[0] < self.min_rise:
            z = min(z, self.exit_sigma)            # tiny wiggles in a static scene are not alerts
        self.stats.add(ts, sf.a_t)
        if z > self.exit_sigma or self.hazard:
            self.last_hot_ts = ts

        if self.win is None:
            self._new_window(ts)
        prev_state = self.state
        if self.state == "CALM" and (z > self.k_sigma or change or self.hazard):
            self._set_state("ALERT", ts)
            self.last_hot_ts = ts
        elif self.state != "CALM" and ts - self.last_hot_ts >= self.exit_s:
            self._set_state("CALM", ts)
            self.hazard = False

        # window boundary: change point, state change or max length for the state
        length = ts - self.win.t0
        if ((change or self.state != prev_state) and length >= self.min_len) or length >= self.max_len[self.state]:
            self._flush_candidate()
            if (w := self._close()) is not None:
                out.append(w)
            self._new_window(ts)
            self.next_sample_ts = ts

        if prev_state == "CALM" and self.state == "ALERT" and self.backfill_top:
            # pre-event back-fill from the 3 s ring buffer (excluding frames already chosen)
            chosen = {k.frame_id for k in self.win.keyframes}
            for b in sorted(list(self.ring)[:-1], key=lambda f: f.a_t, reverse=True)[: self.backfill_top]:
                if b.frame_id not in chosen:
                    self._take_keyframe(b, "backfill")

        self.win.add(sf)
        # priority sampling: best a_t frame per 1/rate interval
        if self.candidate is None or sf.a_t > self.candidate.a_t:
            self.candidate = sf
        if ts >= self.next_sample_ts:
            self._flush_candidate()
            self.next_sample_ts = ts + 1.0 / self.rates[self.state]
        return out

    def _flush_candidate(self) -> None:
        if self.candidate is not None and self.win is not None:
            self._take_keyframe(self.candidate, "priority" if self.state != "CALM" else "sample")
        self.candidate = None

    def flush(self) -> list[Window]:
        self._flush_candidate()
        w = self._close()
        return [w] if w else []


class FixedWindower:
    """B2: TD-RAG-style fixed windows of W s with step W/6 (plan.md §4 M3), uniform keyframes
    at the matched average VLM budget. Overlapping windows share keyframes (the VLM cache
    makes repeats free, and calls/min counts unique keyframes)."""

    def __init__(self, cfg: dict, source_id: str):
        w = cfg["windowing"]
        self.source_id = source_id
        self.W = w["fixed_window_s"]
        self.step_s = self.W / w.get("fixed_step_div", 6)
        self.dt = 60.0 / w["vlm_budget_per_min"]
        self.next_sample_ts = 0.0
        self.next_open_ts = 0.0
        self.open: list[_WindowBuilder] = []
        self.n = 0
        self.state = "CALM"

    def on_feedback(self, *a, **k) -> None:
        pass

    def step(self, sf: SignalFrame) -> list[Window]:
        ts = sf.video_ts
        while ts >= self.next_open_ts:
            self.n += 1
            self.open.append(_WindowBuilder(f"W{self.n}", self.source_id, self.next_open_ts, "CALM"))
            self.next_open_ts += self.step_s
        kf = None
        if ts >= self.next_sample_ts:
            kf = _keyframe(self.source_id, sf, "sample")
            self.next_sample_ts += self.dt * max(1, int((ts - self.next_sample_ts) // self.dt) + 1)
        out = []
        for b in list(self.open):
            if ts - b.t0 >= self.W:
                self.open.remove(b)
                if b.keyframes:
                    out.append(b.close())
                continue
            b.add(sf)
            if kf is not None:
                b.add_keyframe(kf, sf.clip)
        return out

    def flush(self) -> list[Window]:
        out = [b.close() for b in self.open if b.keyframes]
        self.open = []
        return out


class FrameWindower:
    """B1: StreamingRAG-style fixed 2 fps keyframes; every keyframe is its own unit
    (frame-level retrieval). Emitted as one-keyframe Windows to keep one wire format."""

    def __init__(self, cfg: dict, source_id: str):
        self.source_id = source_id
        self.dt = 1.0 / cfg["windowing"]["frame_rate"]
        self.next_sample_ts = 0.0
        self.n = 0
        self.state = "CALM"
        self.pending: _WindowBuilder | None = None

    def on_feedback(self, *a, **k) -> None:
        pass

    def step(self, sf: SignalFrame) -> list[Window]:
        out = []
        if self.pending is not None:
            self.pending.add(sf)
        if sf.video_ts >= self.next_sample_ts:
            if self.pending is not None:
                out.append(self.pending.close())
            self.n += 1
            self.pending = _WindowBuilder(f"W{self.n}", self.source_id, sf.video_ts, "CALM")
            self.pending.add(sf)
            self.pending.add_keyframe(_keyframe(self.source_id, sf, "sample"), sf.clip)
            self.next_sample_ts += self.dt * max(1, int((sf.video_ts - self.next_sample_ts) // self.dt) + 1)
        return out

    def flush(self) -> list[Window]:
        out = [self.pending.close()] if self.pending is not None else []
        self.pending = None
        return out


def make_windower(cfg: dict, source_id: str):
    mode = cfg["windowing"]["mode"]
    return {"adaptive": AdaptiveWindower, "fixed": FixedWindower, "frame": FrameWindower}[mode](cfg, source_id)
