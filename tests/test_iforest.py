import numpy as np

from streamwatch.iforest import IsolationForest, StreamingIsolationForest


def test_batch_forest_ranks_outlier_highest():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    X[0] = [8, 8]
    scores = IsolationForest(seed=0).fit(X).score(X)
    assert scores.argmax() == 0
    assert scores[0] > 0.7 > np.median(scores)


def test_streaming_forest_catches_level_shift_burst():
    rng = np.random.default_rng(2)
    t = np.arange(800)
    data = np.sin(t / 10) + rng.normal(0, 0.1, size=800)
    data[600:604] += 4
    det = StreamingIsolationForest(seed=0)
    flagged = [d.index for d in map(det.update, data) if d.is_anomaly]
    assert any(600 <= i <= 611 for i in flagged)
    assert sum(1 for i in flagged if i < 590) <= 5
