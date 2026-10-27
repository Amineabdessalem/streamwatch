"""streamwatch: streaming time-series anomaly detection."""

from streamwatch.detectors import Detection, EWMADetector, RollingZScoreDetector

__all__ = ["Detection", "EWMADetector", "RollingZScoreDetector"]
__version__ = "0.1.0"
