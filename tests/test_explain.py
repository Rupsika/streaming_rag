"""Explainability engine with a stub scorer (no models): E1, E2, E4, E5 wiring."""
from xstream.explain.attribution import leave_one_out, temporal_heat
from xstream.explain.counterfactual import counterfactual
from xstream.explain.kg_perturb import kg_perturb
from xstream.types import EVENT_TYPES, EvidenceUnit


def units():
    t = {"id": "T1", "s": "vehicle_1", "r": "collides_with", "o": "vehicle_2", "t0": 3.0, "t1": 3.5, "conf": 0.9,
         "src_keyframes": [], "window": "W1"}
    return [EvidenceUnit("F1", "frame_caption", "car hits truck", 3.0, 3.0, {}, {}),
            EvidenceUnit("F2", "frame_caption", "sunny road", 1.0, 1.0, {}, {}),
            EvidenceUnit("T1", "kg_triple", "(vehicle_1 collides_with vehicle_2)", 3.0, 3.5, {"triple": t}, {}),
            EvidenceUnit("H1", "hist_window", "past crash", None, None, {"label": "vehicle_collision"}, {})]


def probs(us):
    """P(collision) grows with crash evidence: F1 counts 0.4, T1 0.3, H1 0.1."""
    w = {"F1": 0.4, "T1": 0.3, "H1": 0.1}
    pc = 0.1 + sum(w.get(u.id, 0.0) for u in us)
    p = {k: 0.0 for k in EVENT_TYPES}
    p.update(none=1 - pc, vehicle_collision=pc)
    return p


def test_loo_and_temporal_pointing():
    base, wts = leave_one_out(probs, units(), "vehicle_collision")
    assert abs(base - 0.9) < 1e-9 and max(wts, key=wts.get) == "F1" and wts["F2"] == 0
    heat = temporal_heat(units(), wts, 0.0, 4.0)
    assert heat["top_unit"] == "F1" and heat["top_ts"] == 3.0


def test_counterfactual_gap_positive_and_kg():
    base, wts = leave_one_out(probs, units(), "vehicle_collision")
    cf = counterfactual(probs, units(), wts, "vehicle_collision", base, (1, 2), seeds=5)
    assert cf["1"]["top"]["removed"] == ["F1"] and cf["1"]["faithfulness_gap"] > 0
    assert cf["2"]["top"]["flip"] is True                         # 0.9 - 0.7 = 0.2 < 0.5
    kg = kg_perturb(probs, units(), "vehicle_collision", base)
    assert kg["most_influential"]["d_conf"] > 0.29
