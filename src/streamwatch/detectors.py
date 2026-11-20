"""Online detectors that score one observation at a time in O(1) memory per step."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class Detection:
    index: int
    value: float
    score: float
    is_anomaly: bool


class RollingZScoreDetector:
    """Flags points more than `threshold` standard deviations from a rolling window mean.

    The window keeps running sums so each update is O(1). Anomalous points are not
    added to the window by default, so one spike does not mask the next.
    """

    def __init__(self, window: int = 50, threshold: float = 3.0, warmup: int | None = None,
                 include_anomalies: bool = False) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.threshold = threshold
        self.warmup = warmup if warmup is not None else min(window, 10)
        self.include_anomalies = include_anomalies
        self._buf: deque[float] = deque()
        self._sum = 0.0
        self._sumsq = 0.0
        self._n = 0

    def _push(self, x: float) -> None:
        self._buf.append(x)
        self._sum += x
        self._sumsq += x * x
        if len(self._buf) > self.window:
            old = self._buf.popleft()
            self._sum -= old
            self._sumsq -= old * old

    def update(self, x: float) -> Detection:
        idx = self._n
        self._n += 1
        k = len(self._buf)
        if k < self.warmup:
            self._push(x)
            return Detection(idx, x, 0.0, False)
        mean = self._sum / k
        var = max(self._sumsq / k - mean * mean, 0.0)
        std = math.sqrt(var)
        score = abs(x - mean) / std if std > 1e-12 else (0.0 if x == mean else math.inf)
        anomaly = score > self.threshold
        if not anomaly or self.include_anomalies:
            self._push(x)
        return Detection(idx, x, score, anomaly)


class EWMADetector:
    """Exponentially weighted mean and variance; reacts faster than a fixed window."""

    def __init__(self, alpha: float = 0.1, threshold: float = 3.0, warmup: int = 10) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.threshold = threshold
        self.warmup = warmup
        self._mean: float | None = None
        self._var = 0.0
        self._n = 0

    def update(self, x: float) -> Detection:
        idx = self._n
        self._n += 1
        if self._mean is None:
            self._mean = x
            return Detection(idx, x, 0.0, False)
        diff = x - self._mean
        std = math.sqrt(self._var)
        if idx < self.warmup:
            score, anomaly = 0.0, False
        else:
            score = abs(diff) / std if std > 1e-12 else (0.0 if diff == 0 else math.inf)
            anomaly = score > self.threshold
        if not anomaly:
            a = self.alpha
            self._mean += a * diff
            self._var = (1 - a) * (self._var + a * diff * diff)
        return Detection(idx, x, score, anomaly)


class RobustZScoreDetector:
    """Flags points far from the window median, scaled by the median absolute deviation (MAD).

    Unlike the mean/std detector, a few outliers in the window barely move the median or the MAD,
    so they cannot hide the next spike. The MAD is scaled by 1.4826 so it matches a standard
    deviation for normal data. Each update sorts the window, so it is O(w log w) per point.
    """

    _MAD_SCALE = 1.4826

    def __init__(self, window: int = 50, threshold: float = 3.5, warmup: int | None = None) -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        self.window = window
        self.threshold = threshold
        self.warmup = warmup if warmup is not None else min(window, 10)
        self._buf: deque[float] = deque(maxlen=window)
        self._n = 0

    def update(self, x: float) -> Detection:
        idx = self._n
        self._n += 1
        if len(self._buf) < self.warmup:
            self._buf.append(x)
            return Detection(idx, x, 0.0, False)
        med = median(self._buf)
        mad = median(abs(v - med) for v in self._buf) * self._MAD_SCALE
        if mad > 1e-12:
            score = abs(x - med) / mad
        else:
            score = 0.0 if x == med else math.inf
        anomaly = score > self.threshold
        if not anomaly:
            self._buf.append(x)
        return Detection(idx, x, score, anomaly)
