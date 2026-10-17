"""Video sources paced like a live camera (plan.md §2.3, §4 M1)."""
from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from xstream.types import Frame


def resize_width(bgr, width: int):
    h, w = bgr.shape[:2]
    if w == width:
        return bgr
    return cv2.resize(bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def open_capture(source: str) -> cv2.VideoCapture:
    """File path, webcam index ("0") or rtsp:// / http:// URL."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video source {source!r}")
    return cap


def replay(source: str, source_id: str, width: int = 640, speed: float = 1.0,
           max_seconds: float | None = None) -> Iterator[Frame]:
    """Yield frames at native FPS × speed (speed=0: as fast as possible).

    Files are paced against the wall clock (frame i is released at
    start + i / (fps·speed)), so a slow consumer never slows the camera down;
    live sources are already real-time and are not paced.
    """
    cap = open_capture(source)
    live = source.isdigit() or "://" in source
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    start = time.time()
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            video_ts = i / fps
            if max_seconds and video_ts >= max_seconds:
                break
            if not live and speed > 0:
                delay = start + video_ts / speed - time.time()
                if delay > 0:
                    time.sleep(delay)
            yield Frame(id=i, video_ts=video_ts, ingest_ts=time.time(), source_id=source_id,
                        bgr=resize_width(bgr, width), fps=fps)
            i += 1
    finally:
        cap.release()
