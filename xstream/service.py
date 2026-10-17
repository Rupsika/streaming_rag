"""Shared skeleton for the pipeline services (buildprompt §2.1).

Every stage is its own process with its own consumer group, talking only through Kafka:

    frames.raw -> signal -> frames.signal -> windowing -> windows.keyframes
    windows.keyframes -> vlm/kg -> windows.closed -> rag -> events.detected -> explain -> events.explained

End of stream: the producer sends a message with header eos=1 on frames.raw; every stage
flushes that source's state and forwards an EOS (`{"kind": "eos"}` on JSON topics), so
scripts/run_experiment.py knows when a replay has fully drained.

`kafka.namespace` (set per experiment run) prefixes all topic and group names, so runs
never read each other's messages.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

from xstream.bus import decode_json, encode_json, ensure_topics, make_consumer, make_producer
from xstream.config import ROOT, load_config


def topic(cfg: dict, key: str) -> str:
    ns = cfg["kafka"].get("namespace") or ""
    return f"{ns}.{cfg['kafka']['topics'][key]}" if ns else cfg["kafka"]["topics"][key]


def all_topics(cfg: dict) -> list[str]:
    return [topic(cfg, k) for k in cfg["kafka"]["topics"]]


def service_args(name: str) -> tuple[dict, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=f"X-StreamRAG {name} service")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", default="live")
    ap.add_argument("--namespace", default=None, help="topic/group prefix (isolates experiment runs)")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--group-suffix", default="", help="fresh consumer group (re-read a topic from the start)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {name} %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.namespace is not None:
        cfg["kafka"]["namespace"] = args.namespace
    cfg["run"]["run_id"] = args.run_id
    return cfg, args


def run_dir(cfg: dict) -> Path:
    d = ROOT / cfg["run"]["runs_dir"] / cfg["run"]["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, obj) -> None:
        self.f.write(json.dumps(obj, default=_np_default) + "\n")
        self.f.flush()


def _np_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


class Service:
    """Consume -> handle -> produce loop with EOS propagation and graceful shutdown."""

    name = "service"
    in_keys: list[str] = []
    out_key: str | None = None

    def __init__(self, cfg: dict, args):
        self.cfg, self.args = cfg, args
        self.log = logging.getLogger(self.name)
        self.dir = run_dir(cfg)
        ensure_topics(cfg, all_topics(cfg))
        ns = cfg["kafka"].get("namespace") or "live"
        group = f"xs-{self.name}-{ns}{getattr(args, 'group_suffix', '')}"
        self.consumer = make_consumer(cfg, group, [topic(cfg, k) for k in self.in_keys])
        self.producer = make_producer(cfg)
        self.out_topic = topic(cfg, self.out_key) if self.out_key else None
        self._stop = False
        signal.signal(signal.SIGINT, self._sigterm)
        signal.signal(signal.SIGTERM, self._sigterm)
        if not args.no_metrics:
            from xstream.telemetry import start_metrics
            start_metrics(cfg, self.name)

    def _sigterm(self, *_):
        self._stop = True

    # -- helpers ---------------------------------------------------------------------
    def emit(self, obj: dict, key: str, topic_name: str | None = None) -> None:
        self._produce(topic_name or self.out_topic, key.encode(), encode_json(obj), None)

    def emit_raw(self, value: bytes, key: str, headers) -> None:
        self._produce(self.out_topic, key.encode(), value, headers)

    def _produce(self, t, key, value, headers) -> None:
        while True:
            try:
                self.producer.produce(t, key=key, value=value, headers=headers)
                break
            except BufferError:
                self.producer.poll(0.05)
        self.producer.poll(0)

    def emit_eos(self, source_id: str) -> None:
        self.emit({"kind": "eos", "source_id": source_id}, source_id)
        self.producer.flush(30)
        self.log.info("EOS %s", source_id)

    # -- to override -------------------------------------------------------------------
    def handle(self, msg) -> None:
        raise NotImplementedError

    def on_idle(self) -> None:
        pass

    def run(self) -> int:
        self.log.info("consuming %s -> %s", [topic(self.cfg, k) for k in self.in_keys], self.out_topic)
        try:
            while not self._stop:
                msgs = self.consumer.consume(num_messages=64, timeout=0.2)
                if not msgs:
                    self.on_idle()
                    continue
                for m in msgs:
                    if m.error():
                        self.log.warning("kafka error: %s", m.error())
                        continue
                    self.handle(m)
                self.on_idle()
        finally:
            self.producer.flush(10)
            self.consumer.close()
        return 0


def is_eos_json(d: dict) -> bool:
    return d.get("kind") == "eos"


def main_for(cls) -> None:
    cfg, args = service_args(cls.name)
    sys.exit(cls(cfg, args).run())


def wall() -> float:
    return time.time()


__all__ = ["Service", "decode_json", "is_eos_json", "main_for", "topic", "run_dir", "JsonlWriter"]
