"""Grounded generation: LangChain prompt -> LLM -> JSON event (plan.md §4 M8).

Confidence is never self-reported (plan.md §4 M8):
- local `hf` backend: log-probability of each label string after the forced prefix
  `{"event_type": "`, softmaxed over the 7 labels (deterministic);
- `gemini` backend: self-consistency, the share of n samples at T=0.7 that agree.

To keep CPU cost bounded, the full JSON (times, sequence, cited explanation) is only
generated when the top label is not `none` or P(event) >= `llm.explain_threshold`;
otherwise the prediction is `none` with an empty explanation. `llm.generate: false` skips the
explanation text entirely (B1: detection uses only the label probabilities, so the decision is
unchanged, and B1 has no explainability stage).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from xstream.backends import LLMBackend, build, register
from xstream.cache import KVCache
from xstream.types import EVENT_TYPES, EvidenceUnit

log = logging.getLogger(__name__)

TEMPLATE = """You are a traffic-incident analyst watching a live camera. Standing queries:
{standing}
Decide whether an incident is happening in the CURRENT window {header}.
Use ONLY the evidence units below. Captions and Q/A come from a small, noisy vision model.
Most windows show normal traffic: answer "none" unless the evidence shows a collision, vehicle damage,
an overturned vehicle, a person lying on the road, smoke/fire, or people running/gathering.
Past windows (H) are similar cases from other videos; earlier windows (S) are from this stream.
Evidence:
{evidence}

Return only JSON: {{"event_type": one of {labels}, "start_s": float, "end_s": float,
"sequence": [stages, e.g. {stages}], "explanation": "2-4 sentences, cite every claim with [ID]",
"cited_ids": [IDs]}}"""

PREFIX = '{"event_type": "'


def format_evidence(units: list[EvidenceUnit]) -> str:
    return "\n".join(f"[{u.id}] {u.text}" for u in units)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        frag = m.group(0)
        for cut in range(len(frag), 0, -1):          # tolerate trailing junk after the object
            if frag[cut - 1] != "}":
                continue
            try:
                return json.loads(frag[:cut])
            except json.JSONDecodeError:
                continue
    d = {}
    for key in ("event_type", "explanation"):
        mm = re.search(rf'"{key}"\s*:\s*"([^"]*)', text)
        if mm:
            d[key] = mm.group(1)
    return d


@register("llm", "hf")
class HFLLM(LLMBackend):
    """Local instruct model via transformers (default Qwen2.5-0.5B/1.5B-Instruct on CPU)."""

    def __init__(self, section: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(section.get("threads", os.cpu_count() or 4))
        self.tok = AutoTokenizer.from_pretrained(section["model"])
        self.model = AutoModelForCausalLM.from_pretrained(section["model"], torch_dtype=torch.float32).eval()
        self.max_new_tokens = section.get("max_new_tokens", 200)

    def _chat(self, prompt: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, prompt: str, temperature: float | None = None, prefix: str = "") -> str:
        text = self._chat(prompt) + prefix
        ids = self.tok(text, return_tensors="pt")
        with self.torch.inference_mode():
            kw = {"do_sample": bool(temperature), "temperature": temperature} if temperature else {"do_sample": False}
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.eos_token_id,
                                      **kw)
        return prefix + self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def label_probs(self, prompt: str, labels: list[str], prefix: str = PREFIX) -> dict[str, float]:
        """softmax over labels of sum log P(label tokens + closing quote | prompt + prefix)."""
        torch = self.torch
        ids = self.tok(self._chat(prompt) + prefix, return_tensors="pt")["input_ids"]
        conts = [self.tok.encode(lab + '"', add_special_tokens=False) for lab in labels]
        L = max(map(len, conts))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        batch = torch.tensor([c + [pad] * (L - len(c)) for c in conts])
        with torch.inference_mode():
            out = self.model(ids, use_cache=True)
            first = torch.log_softmax(out.logits[0, -1], -1)
            cache = out.past_key_values
            cache.batch_repeat_interleave(len(labels))
            attn = torch.cat([torch.ones(len(labels), ids.shape[1], dtype=torch.long),
                              torch.tensor([[1] * len(c) + [0] * (L - len(c)) for c in conts])], 1)
            logits = torch.log_softmax(self.model(batch, past_key_values=cache, attention_mask=attn).logits, -1)
        scores = []
        for i, c in enumerate(conts):
            s = first[c[0]].item() + sum(logits[i, j - 1, c[j]].item() for j in range(1, len(c)))
            scores.append(s)
        z = torch.softmax(torch.tensor(scores), 0).tolist()
        return {lab: round(p, 5) for lab, p in zip(labels, z)}


@register("llm", "gemini")
class GeminiLLM(LLMBackend):
    def __init__(self, section: dict):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.name = section["model"]
        self.min_interval = 60.0 / section.get("rpm", 10)
        self._last = 0.0

    def generate(self, prompt: str, temperature: float | None = None, prefix: str = "") -> str:
        from google.genai import types

        for attempt in range(6):
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.client.models.generate_content(
                    model=self.name, contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature or 0.0,
                                                       response_mime_type="application/json"))
                return r.text
            except Exception as e:
                log.warning("gemini llm error (%s), retry %d", e, attempt)
                time.sleep(2 ** attempt * 5)
        raise RuntimeError("gemini llm failed")


@register("llm", "ollama")
class OllamaLLM(LLMBackend):
    """Ollama HTTP API (e.g. qwen2.5:7b). Not tested on the dev laptop (Ollama not installed)."""

    def __init__(self, section: dict):
        self.name, self.url = section["model"], section.get("url", "http://localhost:11434")

    def generate(self, prompt: str, temperature: float | None = None, prefix: str = "") -> str:
        import requests

        r = requests.post(f"{self.url}/api/generate", timeout=300, json={
            "model": self.name, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": temperature or 0.0}})
        r.raise_for_status()
        return r.json()["response"]


class Generator:
    """Prompt building + confidence + JSON parsing, shared by the rag and explain services."""

    def __init__(self, cfg: dict, kb: dict):
        self.cfg, self.section = cfg, cfg["llm"]
        self.kb = kb
        self.cache = KVCache(self.section["cache_path"])
        self._backend = None
        self.calls = 0
        self.prompt = PromptTemplate.from_template(TEMPLATE)
        # LangChain chain: prompt -> LLM -> JSON parser
        self.chain = (self.prompt | RunnableLambda(lambda pv: self._gen(pv.to_string(), None, ""))
                      | RunnableLambda(parse_json))

    @property
    def backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = build("llm", self.section)
        return self._backend

    def prompt_text(self, header: str, units: list[EvidenceUnit]) -> dict:
        stages = sorted({s for v in self.kb["event_templates"].values() for s in v})
        return {"standing": "\n".join(f"- {q}" for q in self.kb["standing_queries"]), "header": header,
                "evidence": format_evidence(units) or "(no evidence)", "labels": json.dumps(EVENT_TYPES),
                "stages": json.dumps(stages[:6])}

    def _gen(self, prompt: str, temperature: float | None, prefix: str) -> str:
        k = KVCache.key("gen", self.section["backend"], self.section["model"], prompt, temperature, prefix)
        v = self.cache.get(k)
        if v is None:
            self.calls += 1
            v = self.backend.generate(prompt, temperature=temperature, prefix=prefix)
            self.cache.put(k, v)
        return v

    def probs(self, header: str, units: list[EvidenceUnit]) -> dict[str, float]:
        """P(label) for the evidence set (used for detection, E1 LOO and E4 counterfactuals)."""
        prompt = self.prompt.format(**self.prompt_text(header, units))
        k = KVCache.key("probs", self.section["backend"], self.section["model"], prompt)
        v = self.cache.get(k)
        if v is not None:
            return v
        self.calls += 1
        if hasattr(self.backend, "label_probs"):
            p = self.backend.label_probs(prompt, EVENT_TYPES)
        else:   # API: self-consistency over n samples at T=0.7
            n, t = self.section["self_consistency_n"], self.section["self_consistency_temperature"]
            votes = Counter()
            for i in range(n):
                lab = parse_json(self._gen(prompt + f"\n(sample {i})", t, "")).get("event_type", "none")
                votes[lab if lab in EVENT_TYPES else "none"] += 1
            p = {lab: votes[lab] / n for lab in EVENT_TYPES}
        self.cache.put(k, p)
        return p

    def explain(self, header: str, units: list[EvidenceUnit], label: str) -> dict:
        """Full JSON with times, sequence and a cited explanation, conditioned on `label`."""
        pv = self.prompt_text(header, units)
        if self.section["backend"] == "hf":
            raw = self._gen(self.prompt.format(**pv), None, PREFIX + label + '", ')
            d = parse_json(raw)
        else:
            d = self.chain.invoke(pv)
        d["event_type"] = label
        return d

    def predict(self, header: str, units: list[EvidenceUnit], t0: float, t1: float) -> dict:
        p = self.probs(header, units)
        label = max(p, key=p.get)
        p_event = 1.0 - p.get("none", 0.0)
        out = {"event_type": label, "p": p, "p_event": round(p_event, 5), "start_s": t0, "end_s": t1,
               "sequence": [], "explanation": "", "cited_ids": []}
        if label == "none" and p_event >= self.section.get("explain_threshold", 0.3):
            label = max((k for k in p if k != "none"), key=p.get)     # explain the best event hypothesis
        if label != "none" and self.section.get("generate", True):
            d = self.explain(header, units, label)
            ids = {u.id for u in units}
            cited = [c for c in d.get("cited_ids", []) if isinstance(c, str) and c in ids]
            cited += [c for c in re.findall(r"\[([A-Z]\d+)\]", d.get("explanation", "")) if c in ids and c not in cited]
            out.update(sequence=[s for s in d.get("sequence", []) if isinstance(s, str)][:6],
                       explanation=str(d.get("explanation", "")), cited_ids=cited, explained_label=label)
            for key in ("start_s", "end_s"):
                try:
                    out[f"llm_{key}"] = float(d[key])
                except (KeyError, TypeError, ValueError):
                    pass
        return out
