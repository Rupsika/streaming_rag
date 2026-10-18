"""X-StreamRAG dashboard (plan.md §4 M10).

  streamlit run dashboard/app.py

A background Kafka consumer (one per namespace, cached across Streamlit reruns) follows
frames.signal, windows.closed, events.detected and events.explained; past runs are read
from runs/<run_id>/ files. The page polls every second while "live" is on.

Left: live frame + CALM/ALERT/EVENT badge, a_t timeline with window states and GT overlay.
Centre: event cards with clickable evidence IDs. Right: evidence panel (keyframes, E1
attribution bars, E2 heat strip, E3 coverage, E4 counterfactual table, E5 KG subgraph).
Bottom: live metrics from Prometheus.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xstream.config import load_config  # noqa: E402

STATE_COLOR = {"CALM": "#4c9a6a", "ALERT": "#e0a030", "EVENT": "#d0453b"}
st.set_page_config(page_title="X-StreamRAG", layout="wide")


# ---------------------------------------------------------------- Kafka live consumer
class LiveFeed:
    """Background consumer for one topic namespace (the dashboard's own consumer group)."""

    def __init__(self, namespace: str):
        from xstream.bus import decode_json, make_consumer
        from xstream.service import topic

        cfg = load_config()
        cfg["kafka"]["namespace"] = namespace
        self.decode_json = decode_json
        self.t = {k: topic(cfg, k) for k in ("frames_signal", "windows_closed", "events_detected", "events_explained")}
        self.consumer = make_consumer(cfg, f"xs-dashboard-{namespace or 'live'}-{int(time.time())}",
                                      list(self.t.values()), offset="earliest")
        self.frame: dict[str, bytes] = {}
        self.signal: dict[str, deque] = defaultdict(lambda: deque(maxlen=6000))
        self.windows: dict[str, list] = defaultdict(list)
        self.verdicts: dict[str, list] = defaultdict(list)
        self.events: dict[str, dict] = {}
        self.audit: dict[str, dict] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            for m in self.consumer.consume(256, 0.5):
                if m.error():
                    continue
                with self.lock:
                    self._handle(m)

    def _handle(self, m):
        if m.topic() == self.t["frames_signal"]:
            h = dict(m.headers() or [])
            if h.get("eos") == b"1":
                return
            s = h["source_id"].decode()
            self.frame[s] = m.value()
            self.signal[s].append((float(h["video_ts"]), float(h["a_t"])))
            return
        d = self.decode_json(m.value())
        if d.get("kind") == "eos":
            return
        if m.topic() == self.t["windows_closed"]:
            self.windows[d["source_id"]].append({k: d[k] for k in ("id", "t0", "t1", "state")})
        elif m.topic() == self.t["events_detected"]:
            if d.get("kind") == "window":
                self.verdicts[d["source_id"]].append(d)
            else:
                self.events[d["event_id"]] = d
        elif m.topic() == self.t["events_explained"]:
            self.audit[d["event_id"]] = d


@st.cache_resource
def live_feed(namespace: str) -> LiveFeed:
    return LiveFeed(namespace)


# ---------------------------------------------------------------- run files
def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p, encoding="utf-8")] if p.exists() else []


def load_run(run: Path) -> dict:
    data = {"signal": {}, "windows": defaultdict(list), "verdicts": defaultdict(list), "events": {}, "audit": {}}
    for p in (run / "signal").glob("*.csv"):
        df = pd.read_csv(p)
        data["signal"][p.stem] = list(zip(df.video_ts, df.a_t))
    for w in jsonl(run / "windows.jsonl"):
        if "id" in w:
            data["windows"][w["source_id"]].append(w)
    for v in jsonl(run / "verdicts.jsonl"):
        data["verdicts"][v["source_id"]].append(v)
    for e in jsonl(run / "events.jsonl"):
        data["events"][e["event_id"]] = e
    for a in jsonl(run / "audit.jsonl"):
        data["audit"][a["event_id"]] = a
    return data


def gt_segments(source: str) -> list[tuple[float, float]]:
    from xstream.datasets import devset, ucf_test

    for v in devset() + ucf_test():
        if v.source_id == source:
            return v.segments
    return []


# ---------------------------------------------------------------- UI
runs = sorted([p for p in (ROOT / "runs").glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
# default to the newest run that has recorded data (an idle runs/live is just empty files)
default = next((i for i, p in enumerate(runs) if any((p / "signal").glob("*.csv"))), 0)
with st.sidebar:
    st.title("X-StreamRAG")
    choice = st.selectbox("Run", [p.name for p in runs] or ["live"], index=default if runs else 0)
    run_dir = ROOT / "runs" / choice
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    namespace = st.text_input("Kafka namespace", manifest.get("namespace", ""))
    use_kafka = st.toggle("Live from Kafka", value=not manifest.get("completed", False))
    auto = st.toggle("Auto-refresh (1 s)", value=use_kafka)

data = load_run(run_dir)
if use_kafka:
    try:
        feed = live_feed(namespace)
        with feed.lock:
            for s, sig in feed.signal.items():
                data["signal"][s] = list(sig)
            for s, ws in feed.windows.items():
                data["windows"][s] = list(ws)
            for s, vs in feed.verdicts.items():
                data["verdicts"][s] = list(vs)
            data["events"].update(feed.events)
            data["audit"].update(feed.audit)
            frames = dict(feed.frame)
    except Exception as e:
        st.sidebar.error(f"Kafka: {e}")
        frames = {}
else:
    frames = {}

sources = sorted(set(data["signal"]) | set(data["verdicts"]))
if not sources:
    st.info("No data yet. Start a replay (make demo) or pick a finished run.")
    st.stop()
source = st.sidebar.selectbox("Source", sources)

left, centre, right = st.columns([1.2, 1.0, 1.3])

# ---- left: live frame + timeline
with left:
    wins = data["windows"].get(source, [])
    state = wins[-1]["state"] if wins else "CALM"
    st.markdown(f"<span style='background:{STATE_COLOR[state]};color:white;padding:4px 10px;border-radius:6px;"
                f"font-weight:600'>{state}</span> &nbsp; <b>{source}</b>", unsafe_allow_html=True)
    if source in frames:
        st.image(frames[source], use_container_width=True)
    else:
        kfdir = run_dir / "keyframes" / source
        imgs = sorted(kfdir.glob("*.jpg"), key=lambda p: int(p.stem)) if kfdir.exists() else []
        if imgs:
            st.image(str(imgs[-1]), caption=f"last keyframe {imgs[-1].stem}", use_container_width=True)
    sig = pd.DataFrame(data["signal"].get(source, []), columns=["t", "a_t"])
    layers = []
    if wins:
        wdf = pd.DataFrame(wins)[["t0", "t1", "state"]]
        layers.append(alt.Chart(wdf).mark_rect(opacity=0.25).encode(
            x="t0:Q", x2="t1:Q", color=alt.Color("state:N", scale=alt.Scale(domain=list(STATE_COLOR),
                                                                             range=list(STATE_COLOR.values())))))
    gts = gt_segments(source)
    if gts:
        layers.append(alt.Chart(pd.DataFrame(gts, columns=["s", "e"])).mark_rect(color="black", opacity=0.15)
                      .encode(x="s:Q", x2="e:Q"))
    if not sig.empty:
        layers.append(alt.Chart(sig).mark_line(color="#2b5d9b").encode(x=alt.X("t:Q", title="video time (s)"),
                                                                       y=alt.Y("a_t:Q", title="change a_t")))
    ver = pd.DataFrame([(v["t1"], v["p_event"]) for v in data["verdicts"].get(source, [])], columns=["t", "p_event"])
    if not ver.empty:
        layers.append(alt.Chart(ver).mark_line(color="#d0453b", strokeDash=[4, 2]).encode(x="t:Q", y="p_event:Q"))
    if layers:
        st.altair_chart(alt.layer(*layers).properties(height=220), use_container_width=True)
        st.caption("blue: a_t · red dashed: P(event) · bands: window state · grey: ground truth")

# ---- centre: event cards
with centre:
    st.subheader("Events")
    evs = [e for e in data["events"].values() if e["source_id"] == source]
    if not evs:
        st.write("No events.")
    if "sel" not in st.session_state:
        st.session_state.sel = None
    for e in sorted(evs, key=lambda e: e["alert_video_ts"]):
        with st.container(border=True):
            st.markdown(f"**{e['event_type']}** · {e['start_s']:.1f}-{e['end_s']:.1f} s · "
                        f"conf {e['confidence']:.2f} · {e['status']}")
            st.write(e.get("explanation") or "_(no explanation)_")
            cols = st.columns(max(1, len(e.get("cited_ids", []))))
            for c, cid in zip(cols, e.get("cited_ids", [])):
                if c.button(cid, key=f"{e['event_id']}-{cid}"):
                    st.session_state.sel = (e["event_id"], cid)
            if st.button("evidence ▸", key=f"open-{e['event_id']}"):
                st.session_state.sel = (e["event_id"], None)

# ---- right: evidence panel
with right:
    st.subheader("Evidence")
    sel = st.session_state.sel
    ev = data["events"].get(sel[0]) if sel else (evs[-1] if evs else None)
    if ev:
        aud = data["audit"].get(ev["event_id"], {})
        units = aud.get("evidence") or ev.get("peak_verdict", {}).get("evidence", [])
        exp = aud.get("explain", {})
        focus = sel[1] if sel else None
        for u in units:
            hl = "**" if u["id"] == focus else ""
            sc = ", ".join(f"{k}={v}" for k, v in u["scores"].items() if k in ("dense", "bm25", "rerank", "final_rank"))
            st.markdown(f"{hl}[{u['id']}]{hl} `{u['kind']}` {u['text'][:160]}  \n<small>{sc}</small>",
                        unsafe_allow_html=True)
            kf = u["source"].get("keyframe")
            if kf and u["id"] == focus:
                p = run_dir / "keyframes" / kf.split(":")[0] / f"{kf.split(':')[1]}.jpg"
                if p.exists():
                    st.image(str(p), caption=f"{kf} @ {u['t0']:.1f}s")
        if exp:
            att = pd.DataFrame(exp["attribution"].items(), columns=["unit", "weight"])
            st.markdown("**E1 attribution (leave-one-out)**")
            st.altair_chart(alt.Chart(att).mark_bar().encode(x="weight:Q", y=alt.Y("unit:N", sort="-x"),
                                                             color=alt.condition("datum.weight > 0",
                                                                                 alt.value("#2b5d9b"),
                                                                                 alt.value("#bbbbbb")))
                            .properties(height=180), use_container_width=True)
            heat = exp["temporal"]
            hdf = pd.DataFrame({"t": [heat["t0"] + i * heat["step"] for i in range(len(heat["heat"]))],
                                "w": heat["heat"]})
            st.markdown(f"**E2 temporal heat** (top unit {heat['top_unit']} @ {heat['top_ts']} s)")
            st.altair_chart(alt.Chart(hdf).mark_rect().encode(x="t:O", color=alt.Color("w:Q", legend=None))
                            .properties(height=40), use_container_width=True)
            cov = exp["coverage"]
            st.markdown(f"**E3 coverage** claims {cov['claim_coverage']} · tokens {cov['token_coverage']}")
            for c in cov["claims"]:
                color = "#2e7d32" if c["supported"] else "#c62828"
                st.markdown(f"<span style='color:{color}'>{c['claim']} (p={c['p_entail']})</span>",
                            unsafe_allow_html=True)
            rows = [{"k": k, "Δconf top": v["top"]["d_conf"], "Δconf rand": v["random_mean"]["d_conf"],
                     "gap": v["faithfulness_gap"], "flip top": v["top"]["flip"], "flip rand": v["random_mean"]["flip"],
                     "Δcov top": v["top"].get("d_coverage"), "drift top": v["top"].get("drift")}
                    for k, v in exp["counterfactual"].items()]
            st.markdown("**E4 counterfactual removal**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            kgp = exp.get("kg_perturbation") or {}
            if kgp.get("most_influential"):
                st.markdown(f"**E5 KG** most influential: {kgp['most_influential']}")
            kgfile = run_dir / "kg" / f"{source}.json"
            if kgfile.exists() and st.toggle("Show KG subgraph (pyvis)"):
                from pyvis.network import Network

                g = json.loads(kgfile.read_text())
                net = Network(height="320px", width="100%", directed=True)
                keep = {n["id"] for n in g["nodes"] if str(n["id"]).startswith("ent:")}
                for n in g["nodes"]:
                    if n["id"] in keep:
                        net.add_node(n["id"], label=n["id"][4:], color="#2b5d9b")
                for e_ in g["links"]:
                    if e_["source"] in keep and e_["target"] in keep:
                        net.add_edge(e_["source"], e_["target"], label=f"{e_.get('rel')} {e_.get('id', '')}")
                st.components.v1.html(net.generate_html(), height=340)
        else:
            st.caption("explanation audit not available yet (explain service runs off the hot path)")

if auto:
    time.sleep(1)
    st.rerun()
