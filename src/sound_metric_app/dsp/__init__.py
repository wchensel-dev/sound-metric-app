"""DSP layer: pure, testable acoustic metric functions and the processor."""

from .graphing import (
    SMOOTHING_FAST,
    SMOOTHING_INSTANT,
    SMOOTHING_SLOW,
    MetricTrace,
    build_metric_trace,
)
from .metrics import (
    find_onset,
    pa_to_db,
    positive_phase_impulse_pa_ms,
    rms_pa,
    running_leq_rms,
    signed_peak_pa,
    window_samples,
)
from .processor import MetricsProcessor
from .weighting import a_weighting_sos, apply_a_weighting

__all__ = [
    "a_weighting_sos",
    "apply_a_weighting",
    "find_onset",
    "window_samples",
    "pa_to_db",
    "signed_peak_pa",
    "rms_pa",
    "positive_phase_impulse_pa_ms",
    "running_leq_rms",
    "MetricsProcessor",
    "MetricTrace",
    "build_metric_trace",
    "SMOOTHING_INSTANT",
    "SMOOTHING_FAST",
    "SMOOTHING_SLOW",
]
