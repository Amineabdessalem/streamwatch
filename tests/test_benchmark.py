import numpy as np

from streamwatch import EWMADetector, RollingZScoreDetector
from streamwatch.benchmark import events, make_streams, point_adjusted_f1, run


def test_events_and_f1():
    labels = np.array([0, 1, 1, 0, 0, 1, 0])
    assert events(labels) == [(1, 3), (5, 6)]
    preds = np.array([1, 0, 1, 0, 0, 0, 0])
    m = point_adjusted_f1(labels, preds)
    assert m["recall"] == 0.5 and m["precision"] == 0.5


def test_tolerance_counts_late_detection():
    labels = np.array([0, 1, 0, 0, 0])
    preds = np.array([0, 0, 0, 1, 0])
    assert point_adjusted_f1(labels, preds)["recall"] == 0
    assert point_adjusted_f1(labels, preds, tolerance=2)["recall"] == 1


def test_streams_are_deterministic_and_zscore_finds_spikes():
    a, b = make_streams(seed=4), make_streams(seed=4)
    assert all(np.array_equal(x.values, y.values) for x, y in zip(a, b))
    rows = run({"z": lambda: RollingZScoreDetector(window=50, threshold=4)}, streams=[a[0]])
    assert rows[0]["recall"] == 1.0


def test_run_shape():
    rows = run({"ewma": EWMADetector}, streams=make_streams(n=500))
    assert len(rows) == 3 and all(0 <= r["f1"] <= 1 for r in rows)
