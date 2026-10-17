"""M0 acceptance: produce one JSON message and one JPEG frame, consume both back.

Usage: python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import (decode_frame, decode_json, encode_frame, encode_json, ensure_topics,
                         make_consumer, make_producer)
from xstream.config import load_config
from xstream.types import Frame

TOPIC = "xs.smoke"


def main() -> int:
    cfg = load_config()
    ensure_topics(cfg, [TOPIC] + list(cfg["kafka"]["topics"].values()))
    run = uuid.uuid4().hex[:8]

    p = make_producer(cfg)
    sent_at = time.time()
    p.produce(TOPIC, key=b"smoke", value=encode_json({"run": run, "sent_at": sent_at}))
    img = np.full((360, 640, 3), 127, np.uint8)
    value, headers = encode_frame(Frame(id=7, video_ts=0.23, ingest_ts=sent_at, source_id="smoke", bgr=img, fps=30))
    p.produce(TOPIC, key=b"smoke", value=value, headers=headers + [("run", run.encode())])
    p.flush(10)

    c = make_consumer(cfg, group=f"smoke-{run}", topics=[TOPIC])
    got_json = got_frame = None
    deadline = time.time() + 20
    while time.time() < deadline and not (got_json and got_frame):
        msg = c.poll(1.0)
        if msg is None or msg.error():
            continue
        hdr = dict(msg.headers() or [])
        if hdr.get("run") == run.encode():
            got_frame = decode_frame(msg.value(), msg.headers())
        elif not hdr:
            body = decode_json(msg.value())
            if body.get("run") == run:
                got_json = body
    c.close()

    if not (got_json and got_frame):
        print(f"FAIL: json={got_json is not None} frame={got_frame is not None}")
        return 1
    print(f"OK json round-trip {1000 * (time.time() - got_json['sent_at']):.1f} ms; "
          f"frame id={got_frame.id} video_ts={got_frame.video_ts} shape={got_frame.bgr.shape} "
          f"topics={sorted(cfg['kafka']['topics'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
