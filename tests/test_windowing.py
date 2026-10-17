"""Windowing: ADWIN boundaries, FSM, pre-event back-fill, fair budget, fixed/frame modes."""
import copy

import numpy as np

from xstream.config import load_config
from xstream.windowing import AdaptiveWindower, FixedWindower, FrameWindower, SignalFrame

CFG = load_config()


def frames(a_fn, seconds=40.0, fps=10.0):
    rng = np.random.default_rng(0)
    for i in range(int(seconds * fps)):
        ts = i / fps
        yield SignalFrame(frame_id=i * 3, video_ts=ts, ingest_ts=1000 + ts, a_t=float(a_fn(ts) + rng.normal(0, 0.002)),
                          jpeg=b"\xff\xd8", clip=None)


def run(w, gen):
    out = []
    for sf in gen:
        out += w.step(sf)
    return out + w.flush()


def test_level_shift_closes_window_and_raises_alert():
    w = AdaptiveWindower(CFG, "cam")
    wins = run(w, frames(lambda t: 0.02 if t < 20 else 0.25))
    assert any(19.9 <= x.t1 <= 22.0 for x in wins), [(x.t0, x.t1) for x in wins]   # boundary at the shift
    assert any(abs(ts - 20.0) < 1.0 and s == "ALERT" for ts, s in w.transitions)
    assert any(19.5 <= cp <= 26.0 for cp in w.change_points)   # ADWIN sees it (with a few s of delay)
    for x in wins:
        assert x.t1 - x.t0 <= CFG["windowing"]["max_len"]["CALM"] + 0.11


def test_calm_stream_has_long_windows_and_low_rate():
    w = AdaptiveWindower(CFG, "cam")
    wins = run(w, frames(lambda t: 0.02, seconds=60))
    assert all(x.state == "CALM" for x in wins)
    n_kf = sum(len(x.keyframes) for x in wins)
    assert n_kf <= 60 * CFG["windowing"]["keyframe_rate"]["CALM"] + len(wins)   # + 1 opening keyframe per window


def test_backfill_recovers_short_event_between_calm_samples():
    cfg = copy.deepcopy(CFG)
    cfg["windowing"]["keyframe_rate"]["CALM"] = 0.05     # one CALM sample per 20 s
    cfg["windowing"]["k_sigma"] = 1e9                     # only the hazard feedback triggers ALERT
    cfg["windowing"]["adwin_delta"] = 1e-12
    w = AdaptiveWindower(cfg, "cam")
    wins = []
    for sf in frames(lambda t: 0.3 if abs(t - 14.0) < 0.05 else 0.02, seconds=20):
        if abs(sf.video_ts - 15.0) < 1e-6:
            w.on_feedback("none", 0.0, hazard=True)
        wins += w.step(sf)
    wins += w.flush()
    kfs = [k for x in wins for k in x.keyframes]
    back = [k for k in kfs if k.reason == "backfill"]
    assert any(abs(k.video_ts - 14.0) < 0.06 for k in back), [(k.video_ts, k.reason) for k in kfs]


def test_backfill_disabled_ablation():
    cfg = copy.deepcopy(CFG)
    cfg["windowing"].update(backfill=False, k_sigma=1e9, adwin_delta=1e-12)
    cfg["windowing"]["keyframe_rate"]["CALM"] = 0.05
    w = AdaptiveWindower(cfg, "cam")
    wins = []
    for sf in frames(lambda t: 0.3 if abs(t - 14.0) < 0.05 else 0.02, seconds=20):
        if abs(sf.video_ts - 15.0) < 1e-6:
            w.on_feedback("none", 0.0, hazard=True)
        wins += w.step(sf)
    assert not any(k.reason == "backfill" for x in wins + w.flush() for k in x.keyframes)


def test_adaptive_respects_vlm_budget():
    cfg = copy.deepcopy(CFG)
    cfg["windowing"]["vlm_budget_per_min"] = 30
    w = AdaptiveWindower(cfg, "cam")
    wins = run(w, frames(lambda t: 0.02 + 0.3 * (int(t) % 5 == 0), seconds=120))   # busy scene
    n_kf = sum(len(x.keyframes) for x in wins)
    cap = 30 * 120 / 60 + 30 / 60 * cfg["windowing"]["budget_burst_s"]
    assert n_kf <= cap + 1


def test_fixed_windows_length_and_step():
    cfg = copy.deepcopy(CFG)
    cfg["windowing"].update(fixed_window_s=4.0, vlm_budget_per_min=60)
    wins = run(FixedWindower(cfg, "cam"), frames(lambda t: 0.02, seconds=20))
    starts = sorted(x.t0 for x in wins)
    assert np.allclose(np.diff(starts)[:5], 4.0 / 6, atol=0.11)
    assert all(x.t1 - x.t0 <= 4.0 for x in wins)
    full = [x for x in wins if x.t1 < 19]
    assert all(3.8 <= x.t1 - x.t0 for x in full)
    uniq = {k.frame_id for x in wins for k in x.keyframes}
    assert 19 <= len(uniq) <= 21                                                 # 1 keyframe / s


def test_frame_mode_two_fps():
    wins = run(FrameWindower(CFG, "cam"), frames(lambda t: 0.02, seconds=10))
    assert 19 <= len(wins) <= 21 and all(len(x.keyframes) == 1 for x in wins)
