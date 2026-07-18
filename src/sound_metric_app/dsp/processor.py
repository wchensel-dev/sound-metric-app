"""Orchestrates the four metrics for a Frame into a MetricResult."""

from __future__ import annotations

import math
import warnings

from ..config import WINDOW_MS
from ..models import Frame, MetricResult
from .metrics import (
    impulse_max_from_levels,
    impulse_weighted_level,
    leq_db,
    peak_db,
    peak_impulse_from_levels,
)
from .weighting import apply_a_weighting

#: Fractional tolerance on frame duration before a warning is emitted. The
#: Impulse metric integrates over the frame (MATH.md §6), so off-nominal lengths
#: shift its baseline and make shots non-comparable; small acquisition drift is
#: harmless, so only deviations beyond this are flagged.
_DURATION_REL_TOL = 0.01


class MetricsProcessor:
    """Compute Peak dB, Peak dBA, Peak Impulse, LAImax, and LIAeq100ms for a frame.

    Stateless by design: each DewesoftX file is one self-contained 100 ms frame,
    so no filter/integrator state is carried between frames.
    """

    def process(self, frame: Frame) -> MetricResult:
        p = frame.samples
        fs = frame.sample_rate
        self._warn_if_off_nominal_duration(frame)
        p_a = apply_a_weighting(p, fs)

        # The per-sample Impulse smoother is the frame's most expensive op, so
        # run it once and derive both Impulse metrics from the shared level array.
        impulse_levels = impulse_weighted_level(p_a, fs)

        return MetricResult(
            peak_db=peak_db(p),
            peak_dba=peak_db(p_a),
            peak_impulse_db=peak_impulse_from_levels(impulse_levels, fs),
            laimax_db=impulse_max_from_levels(impulse_levels),
            liaeq_100ms_db=leq_db(p_a),
            source_file=frame.source_file,
            channel=frame.channel,
            sample_rate=fs,
            n_samples=frame.n_samples,
            timestamp=frame.timestamp,
        )

    @staticmethod
    def _warn_if_off_nominal_duration(frame: Frame) -> None:
        """Warn when a frame is not the nominal ``WINDOW_MS`` length.

        ``peak_impulse_db`` integrates over the whole frame, so its value scales
        with capture duration; a frame longer or shorter than the nominal window
        yields an Impulse that is not comparable to nominal-length shots. Nominal
        parameters "drive validation warnings only" (MATH.md §2.3), so this is a
        warning, not a hard rejection.
        """
        duration_ms = frame.duration_s * 1000.0
        if not math.isclose(duration_ms, WINDOW_MS, rel_tol=_DURATION_REL_TOL):
            warnings.warn(
                f"Frame {frame.source_file!r} channel {frame.channel!r} is "
                f"{duration_ms:.1f} ms ({frame.n_samples} samples at "
                f"{frame.sample_rate:.0f} Hz), not the nominal {WINDOW_MS:.0f} ms; "
                f"peak_impulse_db integrates over the frame and will not be "
                f"comparable to nominal-length shots.",
                stacklevel=2,
            )
