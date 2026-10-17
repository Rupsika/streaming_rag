"""Dataset manifests + ground truth (plan.md §2).

- UCF-Crime test: Temporal_Anomaly_Annotation.txt, `video class s1 e1 s2 e2` in frames at
  30 fps (-1 = none). Test set (buildprompt §2.3): RoadAccidents (23) + Explosion (21) +
  20 Normal (seed 42, first 3 min), restricted to videos present on disk.
- Dev set: data/annotations_devset.csv (16 clips). Start/end columns must be annotated by
  the team; until then only video-level metrics are available for it.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from xstream.config import ROOT

DATA = ROOT / "data"
UCF_FPS = 30.0

UCF_TO_EVENT = {"RoadAccidents": "vehicle_collision", "Explosion": "fire_explosion", "Fighting": "people_commotion",
                "Assault": "people_commotion", "Normal": "none", "Testing_Normal_Videos_Anomaly": "none",
                "Training_Normal_Videos_Anomaly": "none"}
DEV_TO_EVENT = {"multiple_collisions": "vehicle_collision", "vehicle_vehicle_collision": "vehicle_collision",
                "vehicle_motorcycle_collision": "vehicle_collision", "rollover": "rollover",
                "fire_explosion": "fire_explosion", "none": "none"}
# event types that count as the same family for type accuracy
COLLISION_FAMILY = {"vehicle_collision", "vehicle_pedestrian_collision", "hit_and_run", "rollover"}


def same_family(a: str, b: str) -> bool:
    return a == b or (a in COLLISION_FAMILY and b in COLLISION_FAMILY)


@dataclass
class VideoGT:
    source_id: str                    # file stem, used as Kafka key
    path: Path | None
    label: str                        # mapped event type ("none" for normal)
    raw_class: str
    segments: list[tuple[float, float]] = field(default_factory=list)   # seconds
    has_times: bool = False           # False -> only video-level metrics
    dataset: str = ""


def _find(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name)) if root.exists() else []
    return hits[0] if hits else None


def ucf_test(seed: int = 42, n_normal: int = 20, classes=("RoadAccidents", "Explosion")) -> list[VideoGT]:
    vids_dir = DATA / "ucf_crime" / "videos"
    rows = [line.split() for line in (DATA / "ucf_crime" / "Temporal_Anomaly_Annotation.txt").read_text().splitlines()
            if line.strip()]
    out, normals = [], []
    for name, cls, s1, e1, s2, e2 in rows:
        segs = [(int(s) / UCF_FPS, int(e) / UCF_FPS) for s, e in ((s1, e1), (s2, e2)) if int(s) >= 0]
        v = VideoGT(Path(name).stem, _find(vids_dir, name), UCF_TO_EVENT.get(cls, "other"), cls, segs, True, "ucf")
        if cls in classes:
            out.append(v)
        elif cls == "Normal":
            normals.append(v)
    out += random.Random(seed).sample(normals, min(n_normal, len(normals)))
    return out


def devset() -> list[VideoGT]:
    out = []
    with open(DATA / "annotations_devset.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = DATA / r["video"]
            label = DEV_TO_EVENT.get(r["event_type"], r["event_type"])
            segs, has = [], label == "none"   # normal clips: "no event" is a complete annotation
            if r["event_start_s"].strip() and r["event_end_s"].strip():
                starts = [float(x) for x in r["event_start_s"].split(";")]
                ends = [float(x) for x in r["event_end_s"].split(";")]
                segs, has = list(zip(starts, ends)), True
            out.append(VideoGT(p.stem, p if p.exists() else None, label, r["event_type"], segs, has, "dev"))
    return out


def test_set(name: str = "auto") -> tuple[list[VideoGT], str]:
    """Return (videos present on disk, dataset name). `auto`: UCF test if any video is on disk, else dev."""
    if name in ("auto", "ucf"):
        ucf = [v for v in ucf_test() if v.path is not None]
        if ucf or name == "ucf":
            return ucf, "ucf_test"
    return [v for v in devset() if v.path is not None], "devset"


def ucf_train(seed: int = 42, n_normal: int = 150) -> list[VideoGT]:
    """Case-library source (plan.md §2.2): train videos only, never test."""
    vids_dir = DATA / "ucf_crime" / "videos"
    rows = [line.strip() for line in (DATA / "ucf_crime" / "Anomaly_Train.txt").read_text().splitlines() if line.strip()]
    keep, normals = [], []
    for rel in rows:
        cls = rel.split("/")[0]
        v = VideoGT(Path(rel).stem, _find(vids_dir, Path(rel).name), UCF_TO_EVENT.get(cls, "other"), cls, [], False,
                    "ucf_train")
        if cls in ("RoadAccidents", "Explosion", "Fighting", "Assault"):
            keep.append(v)
        elif cls.startswith("Training_Normal"):
            normals.append(v)
    keep += random.Random(seed).sample(normals, min(n_normal, len(normals)))
    return [v for v in keep if v.path is not None]
