"""Streaming isolation forest over sliding feature windows (numpy only).

The forest is fit on a reference buffer of recent points and refit every
`refit_every` observations, so it follows slow drift while still isolating
sudden, unusual shapes. Each point is scored on the window ending at it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from streamwatch.detectors import Detection

EULER = 0.5772156649


def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + EULER) - 2.0 * (n - 1) / n


class _Tree:
    __slots__ = ("feat", "split", "left", "right", "size")

    def __init__(self, X: np.ndarray, depth: int, max_depth: int, rng: np.random.Generator):
        self.size = len(X)
        self.feat = -1
        self.split = 0.0
        self.left = self.right = None
        if depth >= max_depth or len(X) <= 1:
            return
        lo, hi = X.min(axis=0), X.max(axis=0)
        candidates = np.flatnonzero(hi > lo)
        if candidates.size == 0:
            return
        self.feat = int(rng.choice(candidates))
        self.split = float(rng.uniform(lo[self.feat], hi[self.feat]))
        mask = X[:, self.feat] < self.split
        self.left = _Tree(X[mask], depth + 1, max_depth, rng)
        self.right = _Tree(X[~mask], depth + 1, max_depth, rng)

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        node = self
        while node.feat >= 0:
            node = node.left if x[node.feat] < node.split else node.right
            depth += 1
        return depth + _c(node.size)


class IsolationForest:
    def __init__(self, n_trees: int = 100, sample_size: int = 128, seed: int | None = 0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.trees: list[_Tree] = []
        self._psi = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        psi = min(self.sample_size, len(X))
        max_depth = int(np.ceil(np.log2(max(psi, 2))))
        self.trees = [
            _Tree(X[self.rng.choice(len(X), psi, replace=False)], 0, max_depth, self.rng)
            for _ in range(self.n_trees)
        ]
        self._psi = psi
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in (0, 1]; around 0.5 is normal, close to 1 is anomalous."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean_path = np.array([np.mean([t.path_length(x) for t in self.trees]) for x in X])
        return 2.0 ** (-mean_path / _c(self._psi))


class StreamingIsolationForest:
    """Scores each point using the `shingle` most recent values as its feature vector."""

    def __init__(self, shingle: int = 8, reference: int = 256, refit_every: int = 64,
                 threshold: float = 0.65, n_trees: int = 50, seed: int | None = 0):
        self.shingle = shingle
        self.reference = reference
        self.refit_every = refit_every
        self.threshold = threshold
        self.forest = IsolationForest(n_trees=n_trees, seed=seed)
        self._recent: deque[float] = deque(maxlen=shingle)
        self._ref: deque[np.ndarray] = deque(maxlen=reference)
        self._since_fit = 0
        self._fitted = False
        self._n = 0

    def update(self, x: float) -> Detection:
        idx = self._n
        self._n += 1
        self._recent.append(x)
        if len(self._recent) < self.shingle:
            return Detection(idx, x, 0.0, False)
        vec = np.array(self._recent)
        score = float(self.forest.score(vec)[0]) if self._fitted else 0.0
        anomaly = self._fitted and score > self.threshold
        if not anomaly:
            self._ref.append(vec)
        self._since_fit += 1
        if len(self._ref) >= min(self.reference, 64) and (not self._fitted or self._since_fit >= self.refit_every):
            self.forest.fit(np.array(self._ref))
            self._fitted = True
            self._since_fit = 0
        return Detection(idx, x, score, anomaly)
