"""streamwatch: streaming time-series anomaly detection."""

from streamwatch.detectors import (
    CusumDetector,
    Detection,
    EnsembleDetector,
    EWMADetector,
    RobustZScoreDetector,
    RollingZScoreDetector,
)
from streamwatch.iforest import IsolationForest, StreamingIsolationForest

__all__ = [
    "CusumDetector",
    "Detection",
    "EWMADetector",
    "EnsembleDetector",
    "IsolationForest",
    "RobustZScoreDetector",
    "RollingZScoreDetector",
    "StreamingIsolationForest",
]
__version__ = "0.1.0"
