# streamwatch

[![CI](https://github.com/Amineabdessalem/streamwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Amineabdessalem/streamwatch/actions/workflows/ci.yml)

Streaming time-series anomaly detection in plain Python and numpy. Every
detector consumes one value at a time with `update(x)` and returns a
`Detection(index, value, score, is_anomaly)`, so it fits behind a metrics
pipeline, a log tail or a message queue consumer.

## Detectors

| Detector | Idea | Good at |
|---|---|---|
| `RollingZScoreDetector` | distance from a rolling window mean, in standard deviations (O(1) per point) | isolated spikes on a stable baseline |
| `EWMADetector` | exponentially weighted mean and variance | spikes on slowly drifting series |
| `StreamingIsolationForest` | isolation forest over shape features of the last few points, refit periodically | pattern breaks in seasonal data |

Points flagged as anomalous are kept out of each detector's baseline, so a
burst does not hide the points that follow it.

## Usage

```python
from streamwatch import RollingZScoreDetector

det = RollingZScoreDetector(window=50, threshold=4)
for value in stream:
    d = det.update(value)
    if d.is_anomaly:
        alert(d)
```

```bash
pip install -e .
streamwatch detect metrics.csv --column latency_ms --detector ewma
tail -f values.txt | streamwatch detect --detector zscore --window 100
streamwatch bench
```

`detect` prints one JSON line per anomaly (`--all` prints every point).

## Benchmark

`streamwatch bench` builds three labelled synthetic streams (spikes on noise,
level bursts on a seasonal signal, spikes on a random-walk drift) and reports
point-adjusted precision, recall and F1 (an event counts as found if any alert
lands in it or within 5 points after it). Default settings, seed 0:

```
detector  stream             precision  recall     f1
zscore    spikes                  0.89    1.00   0.94
zscore    seasonal-bursts         0.05    1.00   0.09
zscore    drift-spikes            0.75    1.00   0.86
ewma      spikes                  0.89    1.00   0.94
ewma      seasonal-bursts         1.00    0.00   0.00
ewma      drift-spikes            0.67    1.00   0.80
iforest   spikes                  0.62    0.62   0.62
iforest   seasonal-bursts         1.00    0.50   0.67
iforest   drift-spikes            0.86    1.00   0.92
```

The statistical detectors win on simple spikes; only the isolation forest
copes with the seasonal stream, where a fixed mean is the wrong baseline.

## Development

```bash
pip install -e '.[dev]'
pytest
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

MIT
