"""Visual Query Generator: Q_{t+k} = VQG(S_t, KB) (StreamingRAG §3, plan.md §4 M4).

A rule table from kb/traffic_kb.yaml plus the stream's recent answers decides which
<= `max_questions` questions the VLM is asked about the next keyframe. Example: once
"are two vehicles touching?" was answered yes, the next keyframes are asked whether
a person is lying near the vehicles and whether a vehicle is leaving (hit-and-run).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_kb(path: str | Path) -> dict:
    kb = yaml.safe_load(Path(path).read_text())
    for qid, q in kb["questions"].items():
        q["id"] = qid
    return kb


class VisualQueryGenerator:
    def __init__(self, kb: dict, max_questions: int = 5, memory_s: float = 4.0, max_followups: int = 2):
        self.kb, self.max_q, self.memory_s, self.max_follow = kb, max_questions, memory_s, max_followups
        self._yes: dict[str, dict[str, float]] = {}   # source -> question id -> last video_ts answered yes

    def questions(self, source_id: str, ts: float) -> list[dict]:
        recent = {q for q, t in self._yes.get(source_id, {}).items() if ts - t <= self.memory_s}
        follow: list[str] = []
        for rule in self.kb["vqg_rules"]:
            if rule["if_yes"] in recent:
                follow += [q for q in rule["ask"] if q not in follow and q not in recent]
        follow = follow[: self.max_follow]
        base = [q for q in self.kb["base_questions"] if q not in follow]
        chosen = base[: self.max_q - len(follow)] + follow
        return [self.kb["questions"][q] for q in chosen]

    def observe(self, source_id: str, ts: float, answers: dict[str, dict]) -> None:
        mem = self._yes.setdefault(source_id, {})
        for qid, a in answers.items():
            if a["a"] == "yes":
                mem[qid] = ts

    def reset(self, source_id: str) -> None:
        self._yes.pop(source_id, None)
