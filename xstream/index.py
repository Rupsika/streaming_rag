"""Vector indexes (plan.md §4 M6).

- `CaseLibrary`: FAISS IndexHNSWFlat, inner product on L2-normalised vectors (cosine), the
  same family TD-RAG used. Built offline by scripts/build_case_library.py; one library
  for windows (B2/B3) and one for single keyframes (B1 frame-level kNN).
- `SessionMemory`: windows of the current stream with a recency constraint (last 10 min,
  mirroring TD-RAG's 90-day constraint). Small and append-only, so it is exact brute force.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from xstream.encode import fused_scores


class CaseLibrary:
    def __init__(self, text: np.ndarray, clip: np.ndarray | None, meta: list[dict], hnsw_m: int = 32):
        self.text, self.clip, self.meta = text.astype(np.float32), clip, meta
        self.index = faiss.IndexHNSWFlat(text.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = 128
        if len(text):
            self.index.add(self.text)

    def __len__(self) -> int:
        return len(self.meta)

    def search(self, q_text: np.ndarray, q_clip: np.ndarray | None, k: int, w_text: float, w_clip: float,
               exclude_video: str | None = None) -> list[tuple[int, float]]:
        """HNSW text candidates, re-scored with the fused text+CLIP similarity."""
        if not len(self):
            return []
        n = min(len(self), max(4 * k, 50))
        _, idx = self.index.search(q_text[None].astype(np.float32), n)
        idx = [i for i in idx[0] if i >= 0 and (exclude_video is None or self.meta[i]["video"] != exclude_video)]
        if not idx:
            return []
        clips = self.clip[idx] if self.clip is not None else None
        s = fused_scores(q_text, q_clip, self.text[idx], clips, w_text, w_clip)
        order = np.argsort(-s)[:k]
        return [(idx[j], float(s[j])) for j in order]

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "text.npy", self.text)
        if self.clip is not None:
            np.save(d / "clip.npy", self.clip)
        with open(d / "meta.jsonl", "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m) + "\n")

    @classmethod
    def load(cls, d: str | Path) -> "CaseLibrary":
        d = Path(d)
        clip = np.load(d / "clip.npy") if (d / "clip.npy").exists() else None
        meta = [json.loads(line) for line in open(d / "meta.jsonl", encoding="utf-8")]
        return cls(np.load(d / "text.npy"), clip, meta)


class SessionMemory:
    def __init__(self, recency_s: float = 600.0):
        self.recency_s = recency_s
        self.items: list[dict] = []       # {"id","video","t0","t1","text","state"}
        self._text: list[np.ndarray] = []
        self._clip: list[np.ndarray | None] = []

    def add(self, meta: dict, text_vec: np.ndarray, clip_vec: np.ndarray | None) -> None:
        self.items.append(meta)
        self._text.append(text_vec)
        self._clip.append(clip_vec)

    def search(self, q_text: np.ndarray, q_clip: np.ndarray | None, k: int, now: float, video: str,
               w_text: float, w_clip: float, exclude_ids: set[str] = frozenset()) -> list[tuple[int, float]]:
        idx = [i for i, m in enumerate(self.items)
               if m["video"] == video and m["id"] not in exclude_ids and 0 <= now - m["t1"] <= self.recency_s]
        if not idx:
            return []
        texts = np.stack([self._text[i] for i in idx])
        has_clip = q_clip is not None and all(self._clip[i] is not None for i in idx)
        clips = np.stack([self._clip[i] for i in idx]) if has_clip else None
        s = fused_scores(q_text, q_clip if has_clip else None, texts, clips, w_text, w_clip)
        order = np.argsort(-s)[:k]
        return [(idx[j], float(s[j])) for j in order]
