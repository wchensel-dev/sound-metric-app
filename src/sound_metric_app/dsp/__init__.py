"""DSP layer: pure, testable acoustic metric functions and the processor.

:mod:`.quantile` is a visual aid only and feeds no reported number.
"""

from .graphing import SIGNED_METRICS, MetricTrace, build_metric_trace
from .metrics import (
    find_onset,
    pa_to_db,
    positive_phase_impulse_pa_ms,
    pretrigger_floor_pa,
    rms_pa,
    running_leq_rms,
    signed_peak_pa,
    window_samples,
)
from .processor import MetricsProcessor
from .quantile import (
    DEFAULT_QUANTILE,
    QUANTILE_METHOD,
    QuantileCurve,
    fit_quantile_curve,
)
from .weighting import a_weighting_sos, apply_a_weighting

__all__ = [
    "SIGNED_METRICS",
    "a_weighting_sos",
    "apply_a_weighting",
    "DEFAULT_QUANTILE",
    "QUANTILE_METHOD",
    "QuantileCurve",
    "fit_quantile_curve",
    "find_onset",
    "window_samples",
    "pa_to_db",
    "signed_peak_pa",
    "rms_pa",
    "positive_phase_impulse_pa_ms",
    "pretrigger_floor_pa",
    "running_leq_rms",
    "MetricsProcessor",
    "MetricTrace",
    "build_metric_trace",
]
