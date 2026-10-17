"""Event matching, metric computation and event debounce."""
import pytest

from xstream.config import load_config
from xstream.events import EventTracker
from xstream.metrics import event_match, match_events, mcnemar, prf, tiou, video_level


def test_tiou():
    assert tiou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert tiou((0, 1), (2, 3)) == 0.0


def test_event_match_rules():
    assert event_match((10.0, 12.0), (10.5, 11.0))           # overlap
    assert event_match((12.5, 30.0), (10.0, 11.0))           # starts within GT end + 2 s, tIoU tiny
    assert not event_match((14.0, 30.0), (10.0, 11.0))       # starts too late, no overlap
    assert event_match((8.5, 8.9), (10.0, 10.5))             # starts within GT start - 2 s


def test_match_events_one_to_one():
    preds = [{"start_s": 1.0, "end_s": 2.0, "alert_video_ts": 2.0}, {"start_s": 1.2, "end_s": 2.2, "alert_video_ts": 2.5}]
    m, used = match_events(preds, [(1.0, 2.0), (1.5, 2.0)])
    assert m == [0, 1] and used == {0, 1}
    m, used = match_events(preds[:1], [(1.0, 2.0), (1.5, 2.0)])
    assert m == [0, None]


def test_prf_and_video_level():
    r = prf(tp=3, fp=1, fn=1, tn=5)
    assert r["precision"] == 0.75 and r["recall"] == 0.75 and r["accuracy"] == 0.8
    v = video_level([True, True, False, False], [True, False, True, False])
    assert (v["tp"], v["fn"], v["fp"], v["tn"]) == (1, 1, 1, 1)


def test_mcnemar():
    res = mcnemar([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
    assert res["a_only"] == 10 and res["b_only"] == 2 and res["p_value"] < 0.05


def _v(i, p):
    return {"window_id": f"W{i}", "t0": float(i), "t1": float(i + 1), "p_event": p, "first_ingest_ts": 0.0,
            "event_type": "vehicle_collision" if p >= 0.5 else "none",
            "p": {"none": 1 - p, "vehicle_collision": p}}


def test_debounce_open_two_of_three_and_close_after_two_none():
    t = EventTracker(load_config(), "cam")
    seq = [0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.1, 0.1]
    recs = [r for i, p in enumerate(seq) for r in t.update(_v(i, p))]
    assert [r["status"] for r in recs] == ["open", "close"]
    assert recs[0]["alert_window"] == "W3" and recs[0]["start_s"] == 1.0   # 2 of last 3 at W3
    assert recs[1]["end_s"] == 5.0


def test_single_spike_does_not_open():
    t = EventTracker(load_config(), "cam")
    assert not [r for i, p in enumerate([0.1, 0.9, 0.1, 0.1]) for r in t.update(_v(i, p))]
