"""Window encoder (plan.md §4 M6, TD-RAG's semantic window representation).

Text: BAAI/bge-small-en-v1.5 (384-d) over the window summary (TD-RAG used OpenAI
text-embedding-3-large; we use a free local model). Visual: mean CLIP embedding of the
window's keyframes (512-d, computed upstream by the signal service). Fused similarity
s = w_text·cos_text + w_clip·cos_clip.
"""
from __future__ import annotations

import numpy as np

from xstream.backends import Embedder, build, register


@register("embedder", "sentence-transformers")
class STEmbedder(Embedder):
    def __init__(self, section: dict):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(section["text_model"], device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), np.float32)
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False).astype(np.float32)


def make_embedder(cfg: dict) -> Embedder:
    section = dict(cfg["embedder"])
    section.setdefault("backend", "sentence-transformers")
    return build("embedder", section)


def fused_scores(q_text: np.ndarray, q_clip: np.ndarray | None, texts: np.ndarray, clips: np.ndarray | None,
                 w_text: float = 0.7, w_clip: float = 0.3) -> np.ndarray:
    """s = w_text·cos_text + w_clip·cos_clip (vectors are L2-normalised). Falls back to text only."""
    s = texts @ q_text
    if q_clip is not None and clips is not None and len(clips) and w_clip > 0:
        s = w_text * s + w_clip * (clips @ q_clip)
    return s
