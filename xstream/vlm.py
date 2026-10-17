"""VLM extraction: caption + VQA question bank per keyframe (plan.md §4 M4, StreamingRAG §3).

Backends (configs/default.yaml `vlm.backend`):
- `smolvlm`: HuggingFaceTB/SmolVLM-256M/500M-Instruct, runs on CPU. Yes/no questions are
  answered by comparing the next-token logits of "Yes" vs "No" (deterministic, and gives
  P(yes) for free); count questions are a 4-token generation.
- `gemini`: Gemini vision API, one JSON call per keyframe (caption + all answers).
- `blip2`: Salesforce/blip2-opt-2.7b in 8-bit (needs CUDA; not tested on the dev laptop).

Every call goes through `CachedVLM`, keyed by (model, source, frame_id, prompt).
"""
from __future__ import annotations

import io
import json
import logging
import os
import time

import numpy as np
from PIL import Image

from xstream.backends import VLMBackend, build, register
from xstream.cache import KVCache

log = logging.getLogger(__name__)

CAPTION_PROMPT = "Describe this traffic camera image in one sentence. Mention vehicles, people, damage, smoke or fire."


def _pil(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


@register("vlm", "smolvlm")
class SmolVLM(VLMBackend):
    """SmolVLM on CPU. The image is encoded once per keyframe (vision encoder + connector,
    ~1.2 s) and the features are reused for the caption and every question (~0.3 s each)."""

    def __init__(self, section: dict):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.name = section.get("model", "HuggingFaceTB/SmolVLM-256M-Instruct")
        torch.set_num_threads(section.get("threads", 8))
        self.processor = AutoProcessor.from_pretrained(self.name)
        size = section.get("image_size", 512)
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.size = {"longest_edge": size}
        self.model = AutoModelForVision2Seq.from_pretrained(self.name, dtype=torch.float32).eval()
        tok = self.processor.tokenizer
        self.yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        self.no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
        self._feats: tuple[int, object] | None = None      # (hash(jpeg), image features)

    def _inputs(self, img, text: str):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        return self.processor(text=prompt, images=[img], return_tensors="pt")

    def _prepared(self, jpeg: bytes, text: str) -> dict:
        """Token inputs + cached image features for (jpeg, text)."""
        inputs = self._inputs(_pil(jpeg), text)
        key = hash(jpeg)
        if self._feats is None or self._feats[0] != key:
            with self.torch.inference_mode():
                feats = self.model.model.get_image_features(inputs["pixel_values"], inputs.get("pixel_attention_mask"))
            self._feats = (key, feats)
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                "image_hidden_states": self._feats[1]}

    def _generate(self, jpeg: bytes, text: str, max_new_tokens: int) -> str:
        inputs = self._prepared(jpeg, text)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def p_yes(self, jpeg: bytes, question: str) -> float:
        inputs = self._prepared(jpeg, question + " Answer yes or no.")
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        y = self.torch.logsumexp(logits[self.yes_ids], 0)
        n = self.torch.logsumexp(logits[self.no_ids], 0)
        return float(self.torch.sigmoid(y - n))

    def caption(self, jpeg: bytes) -> str:
        return self._generate(jpeg, CAPTION_PROMPT, 40)

    def vqa(self, jpeg: bytes, questions: list[dict]) -> dict[str, dict]:
        out = {}
        for q in questions:
            if q["kind"] == "yesno":
                p = self.p_yes(jpeg, q["text"])
                out[q["id"]] = {"a": "yes" if p >= 0.5 else "no", "p": round(p, 4)}
            else:
                out[q["id"]] = {"a": self._generate(jpeg, q["text"] + " Answer with a number.", 4), "p": None}
        return out


@register("vlm", "gemini")
class GeminiVLM(VLMBackend):
    """One call per keyframe returning caption + answers as JSON (saves API quota)."""

    def __init__(self, section: dict):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.name = section.get("model", "gemini-2.5-flash")
        self.min_interval = 60.0 / section.get("rpm", 10)
        self._last = 0.0

    def _call(self, jpeg: bytes, prompt: str) -> str:
        from google.genai import types

        for attempt in range(6):
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.client.models.generate_content(
                    model=self.name,
                    contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
                return r.text
            except Exception as e:  # rate limits / transient errors: back off
                log.warning("gemini vlm error (%s), retry %d", e, attempt)
                time.sleep(2 ** attempt * 5)
        raise RuntimeError("gemini vlm failed")

    def caption(self, jpeg: bytes) -> str:
        return self.describe(jpeg, [])[0]

    def vqa(self, jpeg: bytes, questions: list[dict]) -> dict[str, dict]:
        return self.describe(jpeg, questions)[1]

    def describe(self, jpeg: bytes, questions: list[dict]):
        qs = {q["id"]: q["text"] + (" (yes/no)" if q["kind"] == "yesno" else " (number)") for q in questions}
        prompt = (f"{CAPTION_PROMPT}\nThen answer each question. Return JSON "
                  f'{{"caption": str, "answers": {{id: answer}}}} for questions {json.dumps(qs)}')
        d = json.loads(self._call(jpeg, prompt))
        ans = {}
        for q in questions:
            a = str(d.get("answers", {}).get(q["id"], "no")).strip().lower()
            if q["kind"] == "yesno":
                a = "yes" if a.startswith("y") else "no"
                ans[q["id"]] = {"a": a, "p": 1.0 if a == "yes" else 0.0}
            else:
                ans[q["id"]] = {"a": a, "p": None}
        return d.get("caption", ""), ans


@register("vlm", "blip2")
class Blip2VLM(SmolVLM):
    """BLIP-2 OPT-2.7B, 8-bit (plan.md default on a T4). Untested here: the dev laptop has no CUDA."""

    def __init__(self, section: dict):
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

        self.torch = torch
        self.name = section.get("model", "Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained(self.name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto").eval()
        tok = self.processor.tokenizer
        self.yes_ids = [tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" Yes", add_special_tokens=False)[0]]
        self.no_ids = [tok.encode(" no", add_special_tokens=False)[0], tok.encode(" No", add_special_tokens=False)[0]]

    def _inputs(self, img, text: str):
        return self.processor(images=img, text=f"Question: {text} Answer:", return_tensors="pt").to(self.model.device)

    def _prepared(self, jpeg: bytes, text: str) -> dict:
        return dict(self._inputs(_pil(jpeg), text))


class CachedVLM:
    """Caching wrapper used by the vlm/kg service and build_case_library.py."""

    def __init__(self, cfg: dict):
        self.section = cfg["vlm"]
        self.cache = KVCache(self.section["cache_path"])
        self._backend = None
        self.calls = {"caption": 0, "vqa": 0}

    @property
    def backend(self) -> VLMBackend:
        if self._backend is None:       # lazy: fully cached reruns never load the model
            self._backend = build("vlm", self.section)
        return self._backend

    @property
    def model_name(self) -> str:
        return self.section.get("model", self.section["backend"])

    def describe(self, source_id: str, frame_id: int, jpeg: bytes, questions: list[dict]) -> tuple[str, dict]:
        """Caption + answers for one keyframe. Each (frame, prompt) pair is cached separately."""
        base = (self.section["backend"], self.model_name, source_id, frame_id)
        ck = KVCache.key(*base, CAPTION_PROMPT)
        cap = self.cache.get(ck)
        answers, missing = {}, []
        for q in questions:
            v = self.cache.get(KVCache.key(*base, q["text"]))
            if v is None:
                missing.append(q)
            else:
                answers[q["id"]] = v
        if cap is None or missing:
            be = self.backend
            if isinstance(be, GeminiVLM):
                c, a = be.describe(jpeg, missing)
                self.calls["caption"] += 1
            else:
                c = be.caption(jpeg) if cap is None else cap
                a = be.vqa(jpeg, missing) if missing else {}
                self.calls["caption"] += cap is None
            self.calls["vqa"] += len(missing)
            if cap is None:
                cap = c
                self.cache.put(ck, cap)
            for q in missing:
                answers[q["id"]] = a[q["id"]]
                self.cache.put(KVCache.key(*base, q["text"]), a[q["id"]])
        return cap, answers


def jpeg_to_array(jpeg: bytes) -> np.ndarray:
    return np.asarray(_pil(jpeg))
