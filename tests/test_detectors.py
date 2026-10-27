import math
import random

import pytest

from streamwatch import EWMADetector, RollingZScoreDetector


def noisy(n, seed=0):
    rng = random.Random(seed)
    return [10 + rng.gauss(0, 1) for _ in range(n)]


@pytest.mark.parametrize("det", [RollingZScoreDetector(window=30, threshold=4.5), EWMADetector(alpha=0.1, threshold=4.5)])
def test_flags_spike_and_not_noise(det):
    data = noisy(300)
    data[200] = 25.0
    flagged = [d.index for d in map(det.update, data) if d.is_anomaly]
    assert 200 in flagged
    assert len(flagged) <= 3


def test_zscore_matches_numpy_window():
    import numpy as np
    det = RollingZScoreDetector(window=20, threshold=100)
    data = noisy(60, seed=3)
    out = [det.update(x) for x in data]
    w = np.array(data[39:59])
    expected = abs(data[59] - w.mean()) / w.std()
    assert math.isclose(out[59].score, expected, rel_tol=1e-9)


def test_constant_series_then_change():
    det = RollingZScoreDetector(window=10)
    out = [det.update(5.0) for _ in range(20)] + [det.update(6.0)]
    assert not any(d.is_anomaly for d in out[:-1])
    assert out[-1].is_anomaly and out[-1].score == math.inf


def test_spike_does_not_poison_window():
    det = RollingZScoreDetector(window=30)
    data = noisy(100)
    data[50] = data[52] = 40.0
    flagged = {d.index for d in map(det.update, data) if d.is_anomaly}
    assert {50, 52} <= flagged


def test_validation():
    with pytest.raises(ValueError):
        RollingZScoreDetector(window=1)
    with pytest.raises(ValueError):
        EWMADetector(alpha=0)
