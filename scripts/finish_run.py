"""Finish a run whose detection was stopped early: keep the videos that fully reached
events.detected, run the explain service on them through Kafka, then evaluate.

  python scripts/finish_run.py runs/b3-main --videos-file runs/b3_detected.txt --detect-wall 4159.3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_json, make_consumer
from xstream.config import ROOT, load_config
from xstream.service import topic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--videos-file", required=True)
    ap.add_argument("--detect-wall", type=float, required=True, help="seconds until the last kept video was detected")
    args = ap.parse_args()
    run = Path(args.run)
    man = json.loads((run / "manifest.json").read_text())
    keep = [v.strip() for v in open(args.videos_file) if v.strip()]
    dropped = [v for v in man["videos"] if v not in keep]
    cfg = load_config(man["config_path"])
    cfg["kafka"]["namespace"] = man["namespace"]

    # watch only NEW end-of-stream markers (earlier attempts may have left some in the topic)
    watcher = make_consumer(cfg, f"xs-finish-{man['namespace']}-{int(time.time())}", [topic(cfg, "events_explained")],
                            offset="latest")
    while not watcher.assignment():
        watcher.poll(0.5)
    log = open(run / "logs" / "explain_svc.log", "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "services" / "explain_svc.py"), "--config", man["config_path"],
                             "--run-id", man["run_id"], "--namespace", man["namespace"],
                             f"--group-suffix=-finish{int(time.time())}"], cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT)
    done, t0 = set(), time.time()
    try:
        while not set(keep) <= done:
            m = watcher.poll(1.0)
            if m is not None and not m.error():
                d = decode_json(m.value())
                if d.get("kind") == "eos":
                    done.add(d["source_id"])
                    print(f"[{time.time() - t0:6.1f}s] explained {d['source_id']} ({len(done & set(keep))}/{len(keep)})",
                          flush=True)
            if proc.poll() is not None:
                print("explain service died; see", run / "logs" / "explain_svc.log")
                return 2
    finally:
        watcher.close()
        proc.terminate()
    man.update(videos=keep, dropped_videos=dropped, completed=True, wall_seconds=args.detect_wall,
               total_wall_seconds=round(args.detect_wall + time.time() - t0, 1), explain_mode="after_detection",
               note=f"detection stopped early to save time; {len(dropped)} videos not processed: {dropped}")
    (run / "manifest.json").write_text(json.dumps(man, indent=2))
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(run)], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
