"""Prometheus metrics + OpenTelemetry tracing (plan.md §4 M11).

Each service process calls `start_metrics(cfg, "<service>")`, which exposes
/metrics on its own port (configs/default.yaml telemetry.ports). Prometheus in
docker-compose scrapes all of them every 1 s.
"""
from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram, start_http_server

log = logging.getLogger(__name__)

FRAMES_INGESTED = Counter("xs_frames_ingested_total", "Frames seen by a stage", ["stage"])
FRAMES_TO_VLM = Counter("xs_frames_to_vlm_total", "Keyframes sent to the VLM")
VLM_CALLS = Counter("xs_vlm_calls_total", "VLM calls", ["kind"])
FPS = Gauge("xs_fps", "Frames per second processed by a stage", ["stage"])
STAGE_LATENCY = Histogram(
    "xs_stage_latency_seconds", "Per-stage processing latency", ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
E2E_ALERT_LATENCY = Histogram(
    "xs_e2e_alert_latency_seconds", "Ingest-to-alert latency",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
GPU_MEM = Gauge("xs_gpu_mem_bytes", "torch.cuda.memory_allocated (0 on CPU)")
WINDOW_STATE = Gauge("xs_window_state", "0=CALM 1=ALERT 2=EVENT", ["source_id"])
CHANGE_SIGNAL = Gauge("xs_change_signal", "Latest a_t", ["source_id"])


class RateMeter:
    """Events per second over a sliding window (feeds the xs_fps gauge)."""

    def __init__(self, window_s: float = 2.0):
        from collections import deque
        self.window_s = window_s
        self._t = deque()

    def tick(self, now: float | None = None) -> float:
        import time
        now = time.time() if now is None else now
        self._t.append(now)
        while self._t and now - self._t[0] > self.window_s:
            self._t.popleft()
        span = now - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


def start_metrics(cfg: dict, service: str) -> int:
    port = cfg["telemetry"]["ports"][service]
    try:
        start_http_server(port)
    except OSError as e:          # another run already owns the port: keep running without /metrics
        log.warning("metrics port %d busy (%s); %s runs without /metrics", port, e, service)
        return 0
    log.info("metrics for %s on :%d/metrics", service, port)
    return port


def get_tracer(service: str, cfg: dict):
    """OpenTelemetry tracer; console exporter by default (no collector needed)."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider(resource=Resource.create({"service.name": f"xstream-{service}"}))
        if cfg["telemetry"]["otel_exporter"] == "console":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    return trace.get_tracer(f"xstream.{service}")
