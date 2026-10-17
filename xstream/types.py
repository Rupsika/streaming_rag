"""Core data contracts shared by every service (plan.md §9 "Core types").

These are frozen: services exchange them as JSON over Kafka, so changing a
field changes the wire format. `bgr` pixels never go through JSON; frames
travel on `frames.raw` as JPEG bytes plus headers (see xstream/bus.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

WindowState = Literal["CALM", "ALERT", "EVENT"]
EvidenceKind = Literal["frame_caption", "vqa", "kg_triple", "hist_window", "session_window"]

EVENT_TYPES = [
    "none",
    "vehicle_collision",
    "vehicle_pedestrian_collision",
    "hit_and_run",
    "rollover",
    "fire_explosion",
    "people_commotion",
]


@dataclass
class Frame:
    """plan.md §4 M1: one decoded frame, downscaled to 640 px wide."""
    id: int                      # frame index within the source (frame_id on the wire)
    video_ts: float              # seconds of video time, used for GT matching
    ingest_ts: float             # wall-clock time.time() when the producer emitted it
    source_id: str
    bgr: np.ndarray | None = None
    fps: float = 0.0             # native FPS of the source


@dataclass
class Keyframe:
    """A frame the scheduler chose to send to the VLM (plan.md §4 M3/M4)."""
    id: str                      # "<source_id>:<frame_id>"
    frame_id: int
    video_ts: float
    ingest_ts: float
    priority: float              # a_t at selection time
    reason: str = "sample"       # sample | backfill | priority
    jpeg_b64: str | None = None  # image payload for the VLM stage
    caption: str | None = None
    vqa: dict[str, str] = field(default_factory=dict)


@dataclass
class Window:
    """A closed temporal window (plan.md §4 M3, M6)."""
    id: str                      # "W42"
    source_id: str
    t0: float
    t1: float
    state: WindowState
    keyframes: list[Keyframe] = field(default_factory=list)
    a_max: float = 0.0
    first_ingest_ts: float = 0.0
    summary: str = ""            # text summary filled by the vlm/kg stage
    triples: list[dict] = field(default_factory=list)
    # additive fields (defaults keep the frozen wire format backwards compatible)
    last_ingest_ts: float = 0.0
    clip: list[float] | None = None                       # mean CLIP embedding of keyframes
    signal: list[list[float]] = field(default_factory=list)  # [[video_ts, a_t], ...]
    kg_context: list[dict] = field(default_factory=list)  # KG 1-hop triples (vlm/kg stage)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        d = dict(d)
        d["keyframes"] = [Keyframe(**k) for k in d.get("keyframes", [])]
        return cls(**d)


@dataclass
class EvidenceUnit:
    id: str                 # "F3", "T17", "H2", "S5"
    kind: EvidenceKind
    text: str
    t0: float | None
    t1: float | None        # video seconds (None for case-library units)
    source: dict            # keyframe ids / window id / triple id / case label
    scores: dict            # {"dense":..,"bm25":..,"rerank":..}


@dataclass
class EventPrediction:
    window_id: str
    event_type: str
    p: dict[str, float]
    start_s: float
    end_s: float
    sequence: list[str]
    explanation: str
    cited_ids: list[str]
    evidence: list[EvidenceUnit]
    alert_wallclock: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EventPrediction":
        d = dict(d)
        d["evidence"] = [EvidenceUnit(**e) for e in d.get("evidence", [])]
        return cls(**d)
