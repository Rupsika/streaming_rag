"""Replay a video file / webcam / RTSP stream into Kafka topic frames.raw (buildprompt §2.1).

Each message: value = JPEG (640 px wide), key = source_id, headers = source_id,
frame_id, video_ts, ingest_ts, fps, run_id. A final empty message with header
eos=1 marks end of stream so downstream stages can flush open windows.

Examples:
  python scripts/replay_producer.py data/videos/accident/cctv_compilation_acc_ved.mp4
  python scripts/replay_producer.py 0 --source-id webcam            # live webcam
  python scripts/replay_producer.py clip.mp4 --speed 4              # stress test
  python scripts/replay_producer.py a.mp4 b.mp4 --speed 0 --namespace exp1   # several clips, back to back
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import encode_frame, ensure_topics, make_producer
from xstream.config import load_config
from xstream.ingest import replay
from xstream.service import all_topics, topic
from xstream.telemetry import FPS, FRAMES_INGESTED, STAGE_LATENCY, RateMeter, start_metrics

log = logging.getLogger("producer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="video path(s), webcam index, or rtsp/http URL")
    ap.add_argument("--source-id", default=None, help="only with a single source")
    ap.add_argument("--namespace", default=None, help="topic prefix (experiment isolation)")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--speed", type=float, default=None, help="1 = native FPS, 0 = as fast as possible")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config(args.config)
    if args.namespace is not None:
        cfg["kafka"]["namespace"] = args.namespace
    speed = cfg["stream"]["speed"] if args.speed is None else args.speed
    max_s = cfg["stream"]["max_seconds"] if args.max_seconds is None else args.max_seconds
    t = topic(cfg, "frames_raw")
    if not args.no_metrics:
        start_metrics(cfg, "producer")
    ensure_topics(cfg, all_topics(cfg))
    producer = make_producer(cfg)
    meter = RateMeter()
    run_hdr = ("run_id", args.run_id.encode())

    for source in args.sources:
        source_id = args.source_id or Path(source).stem
        n, t_start, native_fps, last_ts = 0, time.time(), 0.0, 0.0
        for frame in replay(source, source_id, cfg["stream"]["width"], speed, max_s):
            t0 = time.perf_counter()
            value, headers = encode_frame(frame, cfg["stream"]["jpeg_quality"])
            while True:
                try:
                    producer.produce(t, key=source_id.encode(), value=value, headers=headers + [run_hdr])
                    break
                except BufferError:          # local queue full: let librdkafka drain, then retry
                    producer.poll(0.05)
            producer.poll(0)
            STAGE_LATENCY.labels("produce").observe(time.perf_counter() - t0)
            FRAMES_INGESTED.labels("producer").inc()
            FPS.labels("producer").set(meter.tick())
            n, native_fps, last_ts = n + 1, frame.fps, frame.video_ts
            if n % 300 == 0:
                log.info("%s: %d frames, video_ts=%.1fs, wall=%.1fs", source_id, n, frame.video_ts,
                         time.time() - t_start)

        producer.produce(t, key=source_id.encode(), value=b"",
                         headers=[("source_id", source_id.encode()), ("eos", b"1"), run_hdr])
        producer.flush(30)
        wall = time.time() - t_start
        video_len = last_ts + (1 / native_fps if native_fps else 0)
        log.info("done %s: %d frames | video %.2fs @ %.2f native fps | wall %.2fs | achieved %.2f fps "
                 "(target %s) | run_id=%s", source_id, n, video_len, native_fps, wall, n / wall if wall else 0,
                 f"{native_fps * speed:.2f}" if speed else "max", args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
