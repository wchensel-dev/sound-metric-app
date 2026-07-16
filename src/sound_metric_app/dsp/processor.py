"""Orchestrates the four metrics for a Frame into a MetricResult."""

from __future__ import annotations

from ..models import Frame, MetricResult
from .metrics import leq_db, peak_db, peak_impulse_db
from .weighting import apply_a_weighting


class MetricsProcessor:
    """Compute Peak dB, Peak dBA, Peak Impulse, and LIAeq100ms for a frame.

    Stateless by design: each DewesoftX file is one self-contained 100 ms frame,
    so no filter/integrator state is carried between frames.
    """

    def process(self, frame: Frame) -> MetricResult:
        p = frame.samples
        fs = frame.sample_rate
        p_a = apply_a_weighting(p, fs)

        return MetricResult(
            peak_db=peak_db(p),
            peak_dba=peak_db(p_a),
            peak_impulse_db=peak_impulse_db(p_a, fs),
            liaeq_100ms_db=leq_db(p_a),
            source_file=frame.source_file,
            channel=frame.channel,
            sample_rate=fs,
            n_samples=frame.n_samples,
            timestamp=frame.timestamp,
        )
