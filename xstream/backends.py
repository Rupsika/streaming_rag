"""Swappable model interfaces (buildprompt §1).

Concrete backends (Gemini, SmolVLM, Ollama, ...) register themselves with
`register()` and are selected by the `backend` key in configs/default.yaml,
so swapping a model never requires a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_REGISTRY: dict[tuple[str, str], type] = {}


def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY[(kind, name)] = cls
        return cls
    return deco


def build(kind: str, section: dict):
    try:
        cls = _REGISTRY[(kind, section["backend"])]
    except KeyError:
        known = sorted(n for k, n in _REGISTRY if k == kind)
        raise ValueError(f"unknown {kind} backend {section.get('backend')!r}; known: {known}")
    return cls(section)


class VLMBackend(ABC):
    """StreamingRAG §3: lightweight VLM used for captions and the VQA question bank."""

    @abstractmethod
    def caption(self, jpeg: bytes) -> str: ...

    @abstractmethod
    def vqa(self, jpeg: bytes, questions: list[str]) -> dict[str, str]:
        """Answer up to `max_questions` questions about one image in a single call."""


class LLMBackend(ABC):
    """Grounded generator (plan.md §4 M8)."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float | None = None) -> str: ...


class Embedder(ABC):
    """Text embedder for window summaries (plan.md §4 M6)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (n, d)."""
