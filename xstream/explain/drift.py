"""E6 embedding-drift monitor (plan.md §4 M9; CityCopilot-X drift monitoring).

Rolling centroid of window embeddings over the last `recent_s` vs. the `ref_s` before it;
cosine distance above `threshold` raises a "scene drift" badge (camera moved, night, rain).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class DriftMonitor:
    def __init__(self, recent_s: float = 60.0, ref_s: float = 600.0, threshold: float = 0.15):
        self.recent_s, self.ref_s, self.threshold = recent_s, ref_s, threshold
        self.buf: deque[tuple[float, np.ndarray]] = deque()

    def update(self, ts: float, vec: np.ndarray) -> dict:
        self.buf.append((ts, vec))
        while self.buf and ts - self.buf[0][0] > self.recent_s + self.ref_s:
            self.buf.popleft()
        recent = [v for t, v in self.buf if ts - t <= self.recent_s]
        ref = [v for t, v in self.buf if ts - t > self.recent_s]
        if not ref or not recent:
            return {"drift": None, "alarm": False}
        a, b = np.mean(recent, 0), np.mean(ref, 0)
        d = float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return {"drift": round(d, 5), "alarm": d > self.threshold}
