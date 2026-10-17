"""Cheap change signal a_t (plan.md §4 M2).

    a_t = w_sem · (1 − cos(clip_t, clip_{t−1})) + w_pix · mean(|gray_t − gray_{t−1}|) / 255

The semantic term follows StreamingRAG's idea of only waking the expensive VLM
when the scene changes; the pixel term catches fast motion that CLIP smooths over.
The CLIP embedding is returned too, so later stages reuse it for visual retrieval
(window encoder, plan.md §4 M6) instead of recomputing it.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

GRAY_WIDTH = 160   # pixel term on a small image: robust to JPEG noise and cheap


class ClipImageEncoder:
    """CLIP ViT-B/32 image embeddings (Radford et al., 2021), L2-normalised."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def encode(self, bgr_frames: list[np.ndarray]) -> np.ndarray:
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float32)


def small_gray(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (GRAY_WIDTH, max(1, round(h * GRAY_WIDTH / w))), interpolation=cv2.INTER_AREA)


class ChangeSignal:
    """Stateful per-source a_t computer. Feed frames in order; batch for CLIP speed."""

    def __init__(self, encoder: ClipImageEncoder, w_semantic: float = 0.6, w_pixel: float = 0.4):
        self.encoder = encoder
        self.w_sem, self.w_pix = w_semantic, w_pixel
        self._prev: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # source_id -> (clip, gray)

    def reset(self, source_id: str) -> None:
        self._prev.pop(source_id, None)

    def step_batch(self, source_id: str, frames: list[np.ndarray]) -> list[dict]:
        """Return one dict per frame: a_t, semantic, pixel and the CLIP embedding."""
        if not frames:
            return []
        embs = self.encoder.encode(frames)
        out = []
        for bgr, emb in zip(frames, embs):
            gray = small_gray(bgr)
            prev = self._prev.get(source_id)
            if prev is None or prev[1].shape != gray.shape:
                sem = pix = 0.0           # first frame of a stream: no change defined
            else:
                sem = float(1.0 - np.dot(emb, prev[0]))
                pix = float(np.mean(cv2.absdiff(gray, prev[1]))) / 255.0
            self._prev[source_id] = (emb, gray)
            out.append({
                "a_t": self.w_sem * sem + self.w_pix * pix,
                "semantic": sem,
                "pixel": pix,
                "clip": emb,
            })
        return out
