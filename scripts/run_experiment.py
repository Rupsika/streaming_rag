"""Run one configuration end to end through Kafka and evaluate it (plan.md §7).

  python scripts/run_experiment.py --config configs/baselines/b3.yaml
  python scripts/run_experiment.py --config configs/baselines/b1.yaml --videos cctv_compilation_acc_ved --speed 1

Starts the five services (signal, windowing, vlm/kg, rag, explain) as separate processes on
topics namespaced by the run id, replays the test set through scripts/replay_producer.py, waits
until every video's end-of-stream marker reaches events.explained, then stops the services and
runs scripts/evaluate.py. Explainability runs off the hot path: by default (`--explain-after`)
the explain service starts once the rag service has drained every video and been stopped, so only
one copy of the LLM is in RAM (16 GB CPU laptop); it then consumes events.detected from the
beginning through Kafka as usual. Saves the merged config, git commit, library versions, per-service
logs and CPU/RSS samples under runs/<run_id>/.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xstream.bus import decode_json, ensure_topics, make_consumer
from xstream.config import ROOT, load_config
from xstream.datasets import test_set
from xstream.service import all_topics, topic

SERVICES = ["signal_svc", "windowing_svc", "vlm_kg_svc", "rag_svc", "explain_svc"]
LIBS = ["torch", "transformers", "sentence-transformers", "faiss-cpu", "river", "networkx", "langchain-core",
        "confluent-kafka", "opencv-python-headless", "streamlit", "prometheus-client", "opentelemetry-sdk",
        "rank-bm25", "scikit-learn", "numpy", "pyvis", "google-genai"]


def versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="auto", choices=["auto", "ucf", "devset"])
    ap.add_argument("--videos", nargs="*", help="subset of source ids")
    ap.add_argument("--speed", type=float, default=0.0, help="0 = fast replay (default for offline eval)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--concurrent-explain", action="store_true",
                    help="run explain concurrently with detection (needs RAM for two LLM copies)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = Path(args.config).stem
    run_id = args.run_id or f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    ns = f"{run_id}-{time.strftime('%H%M%S')}".replace("_", "-")   # fresh topics + groups on every attempt
    if (ROOT / cfg["run"]["runs_dir"] / run_id).exists():
        print(f"runs/{run_id} exists; use a new --run-id or delete it first")
        return 1
    cfg["kafka"]["namespace"] = ns
    run = ROOT / cfg["run"]["runs_dir"] / run_id
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    videos, dataset = test_set(args.dataset)
    if args.videos:
        videos = [v for v in videos if v.source_id in set(args.videos)]
    if not videos:
        print("no videos to run")
        return 1
    if dataset == "devset":
        print("WARNING: UCF-Crime test videos not on disk -> FALLBACK to the 16 dev clips")
    max_s = args.max_seconds if args.max_seconds is not None else cfg["stream"]["max_seconds"]
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    manifest = {"run_id": run_id, "config_name": name, "config_path": args.config, "dataset": dataset,
                "videos": [v.source_id for v in videos], "speed": args.speed, "max_seconds": max_s,
                "namespace": ns, "git_commit": git_commit(), "versions": versions(), "cuda": cuda,
                "retrieval_level": cfg["retrieval"]["level"], "case_library_dir": cfg["retrieval"]["case_library_dir"],
                "platform": platform.platform(), "seed": cfg["run"]["seed"]}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ensure_topics(cfg, all_topics(cfg))
    def start(svc: str) -> subprocess.Popen:
        log = open(run / "logs" / f"{svc}.log", "w", encoding="utf-8")
        return subprocess.Popen([sys.executable, str(ROOT / "services" / f"{svc}.py"), "--config", args.config,
                                 "--run-id", run_id, "--namespace", ns], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    sequential = not args.concurrent_explain
    procs = {svc: start(svc) for svc in SERVICES if not (sequential and svc == "explain_svc")}
    stage = "detect" if sequential else "explain"
    watcher = make_consumer(cfg, f"xs-runner-{ns}", [topic(cfg, "events_detected"), topic(cfg, "events_explained")])
    time.sleep(3)
    t_start = time.time()
    prod_log = open(run / "logs" / "producer.log", "w", encoding="utf-8")
    producer = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "replay_producer.py"),
                                 *[str(v.path) for v in videos], "--speed", str(args.speed), "--max-seconds",
                                 str(max_s), "--config", args.config, "--namespace", ns, "--run-id", run_id],
                                cwd=ROOT, stdout=prod_log, stderr=subprocess.STDOUT)
    procs["producer"] = producer

    done: set[str] = set()
    detected: set[str] = set()
    want = {v.source_id for v in videos}
    detect_wall = None
    rss_peak, cpu = 0.0, []
    ps = {k: psutil.Process(p.pid) for k, p in procs.items()}
    last_sample = 0.0
    ok = True
    try:
        while done != want:
            msg = watcher.poll(1.0)
            if msg is not None and not msg.error():
                d = decode_json(msg.value())
                if d.get("kind") == "eos" and msg.topic() == topic(cfg, "events_detected"):
                    detected.add(d["source_id"])
                    print(f"[{time.time() - t_start:7.1f}s] detected {d['source_id']} ({len(detected)}/{len(want)})",
                          flush=True)
                    if detected == want:
                        detect_wall = time.time() - t_start
                        if stage == "detect":   # free the rag LLM, then explain off the hot path
                            for k in ("rag_svc", "vlm_kg_svc", "signal_svc", "windowing_svc"):
                                procs.pop(k).terminate()
                            time.sleep(3)
                            procs["explain_svc"] = start("explain_svc")
                            ps["explain_svc"] = psutil.Process(procs["explain_svc"].pid)
                            ps = {k: v for k, v in ps.items() if k in procs}
                            stage = "explain"
                            print(f"[{time.time() - t_start:7.1f}s] detection done; explain stage started", flush=True)
                elif d.get("kind") == "eos":
                    done.add(d["source_id"])
                    print(f"[{time.time() - t_start:7.1f}s] explained {d['source_id']} ({len(done)}/{len(want)})",
                          flush=True)
            if time.time() - last_sample > 2:
                last_sample = time.time()
                rss, c = 0.0, 0.0
                for k, p in ps.items():
                    try:
                        rss += p.memory_info().rss
                        c += p.cpu_percent()
                    except psutil.Error:
                        pass
                rss_peak, _ = max(rss_peak, rss / 2**20), cpu.append(c)
            dead = [k for k, p in procs.items() if k != "producer" and p.poll() is not None]
            if dead:
                print(f"service(s) died: {dead}; see {run / 'logs'}")
                ok = False
                break
            if time.time() - t_start > args.timeout:
                print("timeout")
                ok = False
                break
    except KeyboardInterrupt:
        ok = False
    finally:
        wall = time.time() - t_start
        watcher.close()
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    manifest.update(wall_seconds=round(detect_wall or wall, 2), total_wall_seconds=round(wall, 2),
                    explain_mode="after_detection" if sequential else "concurrent", completed=ok, finished_videos=sorted(done),
                    resources={"peak_rss_mb": round(rss_peak, 1),
                               "cpu_percent_mean": round(sum(cpu) / len(cpu), 1) if cpu else None})
    manifest["videos"] = [v for v in manifest["videos"] if v in done] if not ok else manifest["videos"]
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"run {run_id}: {'OK' if ok else 'INCOMPLETE'} in {wall:.0f}s -> {run}")
    if not args.no_eval and done:
        subprocess.call([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(run)], cwd=ROOT)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
