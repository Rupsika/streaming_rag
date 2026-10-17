"""Temporal knowledge graph (plan.md §4 M5; StreamingRAG keeps a NetworkX KG too).

`nx.MultiDiGraph` with Window, Keyframe and Entity nodes. VQA "yes" answers become
triples via kb/traffic_kb.yaml `triple_rules`; each triple has a stable ID ("T17") so the
explainability engine can cite and remove it. Nodes older than `ttl_s` (120 s) are pruned,
like StreamingRAG's reset policy. Object identity across frames is not tracked: entities
are generic role names (vehicle_1, person_1, ...).
"""
from __future__ import annotations

import networkx as nx

from xstream.types import Window


class TemporalKG:
    def __init__(self, kb: dict, ttl_s: float = 120.0):
        self.kb, self.ttl_s = kb, ttl_s
        self.g = nx.MultiDiGraph()
        self._n_triples = 0
        self._prev_window: str | None = None

    def add_window(self, w: Window) -> list[dict]:
        """Insert a window's keyframes + answers; return the window's triples (with IDs)."""
        wn = f"win:{w.id}"
        self.g.add_node(wn, kind="Window", t0=w.t0, t1=w.t1, state=w.state)
        if self._prev_window:
            self.g.add_edge(self._prev_window, wn, rel="NEXT", t0=w.t0, t1=w.t0)
        self._prev_window = wn

        merged: dict[tuple, dict] = {}
        for kf in w.keyframes:
            kn = f"kf:{kf.id}"
            self.g.add_node(kn, kind="Keyframe", ts=kf.video_ts)
            self.g.add_edge(kn, wn, rel="IN", t0=kf.video_ts, t1=kf.video_ts)
            for qid, a in (kf.vqa or {}).items():
                rule = self.kb["triple_rules"].get(qid)
                if not rule or a["a"] != "yes":
                    continue
                s, r, o = rule
                p = a["p"] if a.get("p") is not None else 1.0
                m = merged.setdefault((s, r, o), {"s": s, "r": r, "o": o, "t0": kf.video_ts, "t1": kf.video_ts,
                                                  "ps": [], "src_keyframes": [], "questions": set()})
                m["t1"] = kf.video_ts
                m["ps"].append(p)
                m["src_keyframes"].append(kf.id)
                m["questions"].add(qid)
        triples = []
        for m in merged.values():
            self._n_triples += 1
            t = {"id": f"T{self._n_triples}", "s": m["s"], "r": m["r"], "o": m["o"],
                 "t0": round(m["t0"], 3), "t1": round(m["t1"], 3), "conf": round(sum(m["ps"]) / len(m["ps"]), 3),
                 "src_keyframes": m["src_keyframes"], "questions": sorted(m["questions"]), "window": w.id}
            for ent in (t["s"], t["o"]):
                self.g.add_node(f"ent:{ent}", kind="Entity", type=ent.split("_")[0], last_ts=t["t1"])
            self.g.add_edge(f"ent:{t['s']}", f"ent:{t['o']}", key=t["id"], rel=t["r"], **{k: t[k] for k in
                            ("id", "t0", "t1", "conf", "src_keyframes", "window")})
            for kid in t["src_keyframes"]:
                self.g.add_edge(f"ent:{t['s']}", f"kf:{kid}", rel="OBSERVED_IN", t0=t["t0"], t1=t["t1"])
            triples.append(t)
        self.prune(w.t1)
        return triples

    def one_hop(self, triples: list[dict], now: float) -> list[dict]:
        """KG 1-hop context: earlier triples (within ttl) touching the current window's entities."""
        ents = {f"ent:{t['s']}" for t in triples} | {f"ent:{t['o']}" for t in triples}
        own = {t["id"] for t in triples}
        out, seen = [], set()
        for u, v, d in self.g.edges(data=True):
            tid = d.get("id")
            if not tid or tid in own or tid in seen or (u not in ents and v not in ents):
                continue
            if now - d["t1"] <= self.ttl_s:
                seen.add(tid)
                out.append({"id": tid, "s": u[4:], "r": d["rel"], "o": v[4:], "t0": d["t0"], "t1": d["t1"],
                            "conf": d["conf"], "src_keyframes": d["src_keyframes"], "window": d["window"]})
        return sorted(out, key=lambda t: t["t1"], reverse=True)[:10]

    def prune(self, now: float) -> None:
        old = [n for n, d in self.g.nodes(data=True)
               if (d.get("t1", d.get("ts", d.get("last_ts", now)))) < now - self.ttl_s]
        self.g.remove_nodes_from(old)

    def snapshot(self) -> dict:
        return nx.node_link_data(self.g, edges="links")


def window_summary(w: Window, kb: dict, triples: list[dict]) -> str:
    """TD-RAG's semantic window representation, rendered for video (plan.md §4 M6)."""
    qtext = {qid: q["text"] for qid, q in kb["questions"].items()}
    parts = [f"[{w.id} | {w.t0:.1f}-{w.t1:.1f} s | {w.state}]"]
    for kf in w.keyframes:
        parts.append(f't={kf.video_ts:.1f} caption: "{kf.caption}"')
        yes = [qid for qid, a in (kf.vqa or {}).items() if a["a"] == "yes"]
        for qid, a in (kf.vqa or {}).items():
            if qid in yes or a.get("p") is None:
                parts.append(f"t={kf.video_ts:.1f} Q: {qtext[qid]} A: {a['a']}")
        if not yes:
            parts.append(f"t={kf.video_ts:.1f} no hazard answers")
    if triples:
        parts.append("triples: " + " ".join(f"({t['s']} {t['r']} {t['o']} @{t['t0']:.1f})" for t in triples))
    return "; ".join(parts)
