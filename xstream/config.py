"""Config loading: configs/default.yaml deep-merged with an optional override file."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _overrides(path: Path) -> dict:
    """An override file may name a `base:` (relative to configs/) that it extends."""
    over = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = over.pop("base", None)
    return _merge(_overrides(ROOT / "configs" / base), over) if base else over


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load default.yaml, merge `path` (and its `base:` chain) on top, then apply env overrides."""
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = _merge(cfg, _overrides(Path(path)))
        cfg["config_name"] = Path(path).stem
    if os.getenv("KAFKA_BOOTSTRAP"):
        cfg["kafka"]["bootstrap"] = os.environ["KAFKA_BOOTSTRAP"]
    return cfg
