"""Synthetic benchmark: labelled streams with injected anomalies, scored with point-adjusted F1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stream:
    name: str
    values: np.ndarray
    labels: np.ndarray  # 1 where an anomaly was injected


def make_streams(n: int = 2000, seed: int = 0) -> list[Stream]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    out = []

    v = rng.normal(0, 1, n)
    lab = np.zeros(n, int)
    for i in rng.choice(np.arange(200, n), 8, replace=False):
        v[i] += rng.choice([-1, 1]) * rng.uniform(6, 9)
        lab[i] = 1
    out.append(Stream("spikes", v, lab))

    v = 5 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 0.3, n)
    lab = np.zeros(n, int)
    starts = np.arange(300, n - 20, 150)
    for s in rng.choice(starts, min(4, len(starts)), replace=False):
        v[s:s + 10] += 4
        lab[s:s + 10] = 1
    out.append(Stream("seasonal-bursts", v, lab))

    v = np.cumsum(rng.normal(0, 0.05, n)) + rng.normal(0, 0.5, n)
    lab = np.zeros(n, int)
    for i in rng.choice(np.arange(200, n), 6, replace=False):
        v[i] += rng.choice([-1, 1]) * 5
        lab[i] = 1
    out.append(Stream("drift-spikes", v, lab))
    return out


def events(labels: np.ndarray) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, x in enumerate(labels):
        if x and start is None:
            start = i
        elif not x and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def point_adjusted_f1(labels: np.ndarray, preds: np.ndarray, tolerance: int = 0) -> dict[str, float]:
    """Counts an event as detected if any prediction lands in it (plus `tolerance` after it).

    False positives are predictions outside every (tolerance-extended) event.
    """
    spans = events(labels)
    covered = np.zeros(len(labels), bool)
    hit = 0
    for s, e in spans:
        e2 = min(len(labels), e + tolerance)
        covered[s:e2] = True
        hit += bool(preds[s:e2].any())
    fp = int((preds.astype(bool) & ~covered).sum())
    recall = hit / len(spans) if spans else 1.0
    precision = hit / (hit + fp) if hit + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run(detector_factories: dict, streams: list[Stream] | None = None, tolerance: int = 5):
    streams = streams or make_streams()
    rows = []
    for name, factory in detector_factories.items():
        for s in streams:
            det = factory()
            preds = np.array([det.update(float(x)).is_anomaly for x in s.values])
            rows.append({"detector": name, "stream": s.name, **point_adjusted_f1(s.labels, preds, tolerance)})
    return rows
