"""Event open/close debounce (plan.md §4 M8 "Event logic").

- open when P(event != none) >= `open_threshold` in `open_votes` of the last `open_window` windows,
- close after `close_after_none` consecutive windows below threshold.
The alert is the moment an event is opened; alert latency = alert_wallclock - ingest_ts of the
first GT event frame (computed in scripts/evaluate.py).
"""
from __future__ import annotations

import time
from collections import Counter, deque


class EventTracker:
    def __init__(self, cfg: dict, source_id: str):
        e = cfg["events"]
        self.source_id = source_id
        self.th, self.votes, self.close_after = e["open_threshold"], e["open_votes"], e["close_after_none"]
        self.recent: deque[dict] = deque(maxlen=e["open_window"])
        self.open: dict | None = None
        self.none_run = 0
        self.n = 0

    @staticmethod
    def _positive(v: dict, th: float) -> bool:
        return v["p_event"] >= th

    def update(self, v: dict) -> list[dict]:
        """Feed one window verdict; return [] or a list of event records (opened / closed)."""
        out = []
        pos = self._positive(v, self.th)
        self.recent.append(v)
        if self.open is None:
            hits = [r for r in self.recent if self._positive(r, self.th)]
            if pos and len(hits) >= self.votes:
                self.n += 1
                self.open = {
                    "event_id": f"{self.source_id}/E{self.n}", "source_id": self.source_id,
                    "start_s": min(r["t0"] for r in hits), "end_s": v["t1"],
                    "alert_video_ts": v["t1"], "alert_wallclock": time.time(),
                    "alert_window": v["window_id"], "windows": [r["window_id"] for r in hits],
                    "types": Counter(_type(r) for r in hits), "peak": max(hits, key=lambda r: r["p_event"]),
                    "first_ingest_ts": min(r["first_ingest_ts"] for r in hits),
                }
                self.none_run = 0
                out.append(self._record("open"))
        else:
            if pos:
                self.none_run = 0
                self.open["end_s"] = v["t1"]
                self.open["windows"].append(v["window_id"])
                self.open["types"][_type(v)] += 1
                if v["p_event"] > self.open["peak"]["p_event"]:
                    self.open["peak"] = v
            else:
                self.none_run += 1
                if self.none_run >= self.close_after:
                    out.append(self._record("close"))
                    self.open = None
        return out

    def flush(self) -> list[dict]:
        if self.open is None:
            return []
        rec = self._record("close")
        self.open = None
        return [rec]

    def _record(self, status: str) -> dict:
        o = self.open
        peak = o["peak"]
        return {
            "kind": "event", "status": status, "event_id": o["event_id"], "source_id": o["source_id"],
            "event_type": o["types"].most_common(1)[0][0], "start_s": o["start_s"], "end_s": o["end_s"],
            "alert_video_ts": o["alert_video_ts"], "alert_wallclock": o["alert_wallclock"],
            "alert_window": o["alert_window"], "first_ingest_ts": o["first_ingest_ts"], "windows": list(o["windows"]),
            "confidence": peak["p_event"], "peak_window": peak["window_id"],
            "explanation": peak.get("explanation", ""), "cited_ids": peak.get("cited_ids", []),
            "sequence": peak.get("sequence", []),
            # what the explain service needs to re-score this decision
            "peak_verdict": {k: peak[k] for k in ("window_id", "header", "evidence", "t0", "t1", "p", "event_type",
                                                  "explanation", "cited_ids", "explained_label") if k in peak},
        }


def _type(v: dict) -> str:
    if v["event_type"] != "none":
        return v["event_type"]
    return v.get("explained_label") or max((k for k in v["p"] if k != "none"), key=v["p"].get)
