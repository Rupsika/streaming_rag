"""Build the case library offline (plan.md §2.2, §4 M6; TD-RAG's "historical windows").

Each source video is cut into non-overlapping `--window-s` windows with 1 keyframe/s;
keyframes get the same VLM caption + question bank + KG triples as the online pipeline
(through the shared sqlite VLM cache), and every window/keyframe carries the VIDEO-level
label. Two libraries are written:
  cache/case_library/windows/  (B2/B3 window-level retrieval)
  cache/case_library/frames/   (B1 frame-level kNN: one unit per keyframe)

Sources:
  --source ucf_train   UCF-Crime train videos (plan.md default; never test videos)
  --source devset      FALLBACK when UCF videos are missing: the 16 dev clips. Retrieval then
                       excludes the query's own video (leave-one-video-out), but test and
                       library come from the same small pool - report this loudly.

  python scripts/build_case_library.py --source devset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.change import ClipImageEncoder
from xstream.config import ROOT, load_config
from xstream.datasets import devset, ucf_train
from xstream.encode import make_embedder
from xstream.index import CaseLibrary
from xstream.ingest import replay
from xstream.kg import TemporalKG, window_summary
from xstream.types import Keyframe, Window
from xstream.vlm import CachedVLM
from xstream.vqg import VisualQueryGenerator, load_kb

log = logging.getLogger("case_library")


def video_windows(path: Path, source_id: str, cfg: dict, vlm: CachedVLM, vqg, kb, clip_enc, window_s: float,
                  kf_per_s: float, max_s: float) -> tuple[list[Window], list[np.ndarray]]:
    kg = TemporalKG(kb, cfg.get("kg_ttl_s", 120.0))
    frames = []
    next_ts = 0.0
    for f in replay(str(path), source_id, cfg["stream"]["width"], speed=0, max_seconds=max_s):
        if f.video_ts + 1e-6 >= next_ts:
            frames.append(f)
            next_ts += 1.0 / kf_per_s
    if not frames:
        return [], []
    clips = clip_enc.encode([f.bgr for f in frames])
    windows, wclips = [], []
    n_win = int(np.ceil((frames[-1].video_ts + 1e-6) / window_s))
    for wi in range(n_win):
        idx = [i for i, f in enumerate(frames) if wi * window_s <= f.video_ts < (wi + 1) * window_s]
        if not idx:
            continue
        kfs = []
        for i in idx:
            f = frames[i]
            ok, buf = cv2.imencode(".jpg", f.bgr, [cv2.IMWRITE_JPEG_QUALITY, cfg["stream"]["jpeg_quality"]])
            qs = vqg.questions(source_id, f.video_ts)
            cap, ans = vlm.describe(source_id, f.id, buf.tobytes(), qs)
            vqg.observe(source_id, f.video_ts, ans)
            kfs.append(Keyframe(id=f"{source_id}:{f.id}", frame_id=f.id, video_ts=round(f.video_ts, 3),
                                ingest_ts=0.0, priority=0.0, caption=cap, vqa=ans))
        w = Window(id=f"W{wi + 1}", source_id=source_id, t0=round(wi * window_s, 3),
                   t1=round(min((wi + 1) * window_s, frames[idx[-1]].video_ts), 3), state="CALM", keyframes=kfs)
        w.triples = kg.add_window(w)
        w.summary = window_summary(w, kb, w.triples)
        windows.append(w)
        m = clips[idx].mean(0)
        wclips.append(m / (np.linalg.norm(m) + 1e-8))
    vqg.reset(source_id)
    return windows, wclips, [(frames[i], clips[i]) for i in range(len(frames))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ucf_train", "devset"], default="ucf_train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--kf-per-s", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=180.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    kb = load_kb(ROOT / cfg["kb_path"])

    videos = ucf_train() if args.source == "ucf_train" else devset()
    videos = [v for v in videos if v.path is not None]
    if not videos:
        log.error("no %s videos on disk; download them (data/README.md) or use --source devset", args.source)
        return 1
    if args.source == "devset":
        log.warning("FALLBACK: case library built from the dev clips (leave-one-video-out at query time)")

    vlm = CachedVLM(cfg)
    vqg = VisualQueryGenerator(kb, cfg["vlm"]["max_questions"], cfg["vlm"].get("vqg_memory_s", 4.0))
    clip_enc = ClipImageEncoder(cfg["signal"]["clip_model"])
    emb = make_embedder(cfg)
    win_meta, win_clip, fr_meta, fr_clip = [], [], [], []
    t_start = time.time()
    for n, v in enumerate(videos, 1):
        t0 = time.time()
        windows, wclips, frames = video_windows(v.path, v.source_id, cfg, vlm, vqg, kb, clip_enc, args.window_s,
                                                args.kf_per_s, args.max_seconds)
        for w, c in zip(windows, wclips):
            win_meta.append({"id": f"{v.source_id}/{w.id}", "video": v.source_id, "label": v.label,
                             "raw_class": v.raw_class, "t0": w.t0, "t1": w.t1, "text": w.summary})
            win_clip.append(c)
            for kf in w.keyframes:
                one = Window(id=w.id, source_id=v.source_id, t0=kf.video_ts, t1=kf.video_ts, state="CALM",
                             keyframes=[kf])
                tr = [t for t in w.triples if kf.id in t["src_keyframes"]]
                fr_meta.append({"id": kf.id, "video": v.source_id, "label": v.label, "raw_class": v.raw_class,
                                "t0": kf.video_ts, "t1": kf.video_ts, "text": window_summary(one, kb, tr)})
                fr_clip.append(next(c for f, c in frames if f.id == kf.frame_id))
        log.info("[%d/%d] %s (%s): %d windows in %.1fs, vlm calls so far %s, cache hits %d",
                 n, len(videos), v.source_id, v.label, len(windows), time.time() - t0, vlm.calls, vlm.cache.hits)

    out = ROOT / cfg["retrieval"]["case_library_dir"]
    for name, meta, clips in (("windows", win_meta, win_clip), ("frames", fr_meta, fr_clip)):
        text = emb.embed([m["text"] for m in meta])
        CaseLibrary(text, np.stack(clips).astype(np.float32), meta, cfg["retrieval"].get("hnsw_m", 32)).save(out / name)
    info = {"source": args.source, "videos": len(videos), "windows": len(win_meta), "frames": len(fr_meta),
            "window_s": args.window_s, "kf_per_s": args.kf_per_s, "vlm": cfg["vlm"]["model"],
            "labels": {lab: sum(m["label"] == lab for m in win_meta) for lab in sorted({m["label"] for m in win_meta})},
            "built_seconds": round(time.time() - t_start, 1), "fallback": args.source == "devset"}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    log.info("case library: %s", info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
