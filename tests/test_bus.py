"""Wire-format tests that need no broker."""
import numpy as np

from xstream.bus import decode_frame, decode_json, encode_frame, encode_json
from xstream.config import load_config
from xstream.types import EventPrediction, EvidenceUnit, Frame, Keyframe, Window


def test_frame_roundtrip():
    img = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    f = Frame(id=12, video_ts=0.4, ingest_ts=1700000000.123456, source_id="cam1", bgr=img, fps=30.0)
    value, headers = encode_frame(f, jpeg_quality=95)
    g = decode_frame(value, headers)
    assert (g.id, g.video_ts, g.ingest_ts, g.source_id, g.fps) == (12, 0.4, 1700000000.123456, "cam1", 30.0)
    assert g.bgr.shape == img.shape


def test_window_and_event_json_roundtrip():
    w = Window(id="W1", source_id="cam1", t0=0.0, t1=2.0, state="ALERT",
               keyframes=[Keyframe(id="cam1:3", frame_id=3, video_ts=0.1, ingest_ts=1.0, priority=0.2)])
    assert Window.from_dict(decode_json(encode_json(w.to_dict()))) == w
    e = EventPrediction(window_id="W1", event_type="none", p={"none": 1.0}, start_s=0, end_s=2,
                        sequence=[], explanation="", cited_ids=[], alert_wallclock=0.0,
                        evidence=[EvidenceUnit("F1", "vqa", "x", 0.1, 0.1, {}, {})])
    assert EventPrediction.from_dict(decode_json(encode_json(e.to_dict()))) == e


def test_config_loads():
    cfg = load_config()
    assert cfg["kafka"]["topics"]["frames_raw"] == "frames.raw"
    assert cfg["windowing"]["mode"] in {"adaptive", "fixed", "frame"}
