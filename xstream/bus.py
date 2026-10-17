"""Kafka plumbing shared by all services (buildprompt §2.1).

Wire format:
- frames.raw: value = JPEG bytes, key = source_id, headers carry the Frame metadata.
- every other topic: value = UTF-8 JSON, key = source_id.
Each service uses its own consumer group, so stages are decoupled processes that
communicate only through topics.
"""
from __future__ import annotations

import json
import logging

import cv2
import numpy as np
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from xstream.types import Frame

log = logging.getLogger(__name__)


def make_producer(cfg: dict, **extra) -> Producer:
    conf = {
        "bootstrap.servers": cfg["kafka"]["bootstrap"],
        "message.max.bytes": cfg["kafka"]["max_message_bytes"],
        "linger.ms": 5,
        "compression.type": "none",   # JPEG is already compressed
    }
    conf.update(extra)
    return Producer(conf)


def make_consumer(cfg: dict, group: str, topics: list[str], offset: str = "earliest", **extra) -> Consumer:
    conf = {
        "bootstrap.servers": cfg["kafka"]["bootstrap"],
        "group.id": group,
        "auto.offset.reset": offset,
        "enable.auto.commit": True,
        "fetch.message.max.bytes": cfg["kafka"]["max_message_bytes"],
        # LLM stages can spend minutes on one message (explain: ~4 min/event on CPU);
        # the default 5 min would eject the consumer from its group mid-run.
        "max.poll.interval.ms": 3_600_000,
    }
    conf.update(extra)
    c = Consumer(conf)
    c.subscribe(topics)
    return c


def ensure_topics(cfg: dict, names: list[str] | None = None) -> None:
    """Create the pipeline topics if they do not exist yet."""
    admin = AdminClient({"bootstrap.servers": cfg["kafka"]["bootstrap"]})
    names = names or list(cfg["kafka"]["topics"].values())
    existing = set(admin.list_topics(timeout=10).topics)
    new = [
        NewTopic(n, num_partitions=cfg["kafka"]["partitions"], replication_factor=1,
                 config={"max.message.bytes": str(cfg["kafka"]["max_message_bytes"])})
        for n in names if n not in existing
    ]
    if not new:
        return
    for name, fut in admin.create_topics(new).items():
        try:
            fut.result()
            log.info("created topic %s", name)
        except Exception as e:  # topic may have been created concurrently
            log.warning("topic %s: %s", name, e)


def encode_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def decode_json(raw: bytes):
    return json.loads(raw.decode())


def encode_frame(frame: Frame, jpeg_quality: int = 85) -> tuple[bytes, list[tuple[str, bytes]]]:
    """Frame -> (JPEG bytes, Kafka headers)."""
    ok, buf = cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    headers = [
        ("source_id", frame.source_id.encode()),
        ("frame_id", str(frame.id).encode()),
        ("video_ts", repr(frame.video_ts).encode()),
        ("ingest_ts", repr(frame.ingest_ts).encode()),
        ("fps", repr(frame.fps).encode()),
    ]
    return buf.tobytes(), headers


def decode_frame(value: bytes, headers: list[tuple[str, bytes]] | None, decode_pixels: bool = True) -> Frame:
    h = {k: v.decode() for k, v in (headers or [])}
    bgr = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR) if decode_pixels else None
    return Frame(
        id=int(h["frame_id"]),
        video_ts=float(h["video_ts"]),
        ingest_ts=float(h["ingest_ts"]),
        source_id=h["source_id"],
        bgr=bgr,
        fps=float(h.get("fps", 0.0)),
    )
