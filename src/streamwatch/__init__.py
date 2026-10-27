"""streamwatch: streaming time-series anomaly detection."""

from streamwatch.detectors import Detection, EWMADetector, RollingZScoreDetector
from streamwatch.iforest import IsolationForest, StreamingIsolationForest

__all__ = [
    "Detection",
    "EWMADetector",
    "IsolationForest",
    "RollingZScoreDetector",
    "StreamingIsolationForest",
]
__version__ = "0.1.0"
