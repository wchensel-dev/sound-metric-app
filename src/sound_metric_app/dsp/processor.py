"""Orchestrates the per-shot metrics for a Frame into a MetricResult.

Every metric is anchored to a single detected onset (first raw-pressure sample
above the 1 Pa threshold) and computed over a fixed window from there, matching
TBAC's ``process_string.m``:

* Peak dB / Peak dBA — largest *signed* pressure in ``[onset, onset+100 ms]``;
* Peak Impulse — positive-phase ``∫p·dt`` (unweighted) in the same window;
* Peak 10 ms-Leq — max of the rectangular 10 ms running Leq (A-weighted) within
  ``[onset, onset+25 ms]``;
* LIAeq,100ms — A-weighted equivalent level over ``[onset, onset+100 ms]`` (our
  proprietary free-field divergence).

Each metric carries both its linear magnitude (Pa or Pa·ms) and the dB level, so
group aggregation can average in the linear domain (MATH.md §9).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..config import (
    CAPTURE_MS,
    LEQ_SEARCH_MS,
    LIAEQ_WINDOW_MS,
    ONSET_THRESHOLD_PA,
    PEAK_WINDOW_MS,
)
from ..models import Frame, MetricResult
from .metrics import (
    find_onset,
    pa_to_db,
    positive_phase_impulse_pa_ms,
    rms_pa,
    running_leq_rms,
    signed_peak_pa,
    window_samples,
)
from .weighting import apply_a_weighting


class MetricsProcessor:
    """Compute the onset-anchored per-shot metrics for a frame.

    Stateless by design: each DewesoftX file is one self-contained capture, so no
    filter/integrator state is carried between frames.
    """

    def process(self, frame: Frame) -> MetricResult:
        p = frame.samples
        fs = frame.sample_rate

        onset = find_onset(p)
        self._warn_if_off_nominal(frame, onset)
        if onset is None:
            # No shot detected; analyse from the frame start so the pipeline still
            # yields numbers (the warning above flags them as suspect).
            onset = 0

        p_a = apply_a_weighting(p, fs)

        peak_n = window_samples(fs, PEAK_WINDOW_MS)
        leq_search_n = window_samples(fs, LEQ_SEARCH_MS)
        liaeq_n = window_samples(fs, LIAEQ_WINDOW_MS)

        peak_seg = p[onset : onset + peak_n]
        peak_a_seg = p_a[onset : onset + peak_n]

        peak_pa = signed_peak_pa(peak_seg)
        peak_a_pa = signed_peak_pa(peak_a_seg)
        impulse_pa_ms = positive_phase_impulse_pa_ms(peak_seg, fs)

        rms_trace = running_leq_rms(p_a, fs)
        leq_window = rms_trace[onset : onset + leq_search_n]
        leq10ms_pa = float(np.max(leq_window)) if leq_window.size else 0.0

        liaeq_pa = rms_pa(p_a[onset : onset + liaeq_n])

        return MetricResult(
            peak_pa=peak_pa,
            peak_db=pa_to_db(peak_pa),
            peak_a_pa=peak_a_pa,
            peak_dba=pa_to_db(peak_a_pa),
            impulse_pa_ms=impulse_pa_ms,
            peak_impulse_db=pa_to_db(impulse_pa_ms),
            leq10ms_pa=leq10ms_pa,
            leq10ms_db=pa_to_db(leq10ms_pa),
            liaeq_pa=liaeq_pa,
            liaeq_100ms_db=pa_to_db(liaeq_pa),
            source_file=frame.source_file,
            channel=frame.channel,
            sample_rate=fs,
            n_samples=frame.n_samples,
            timestamp=frame.timestamp,
        )

    @staticmethod
    def _warn_if_off_nominal(frame: Frame, onset: int | None) -> None:
        """Warn when a frame can't support the onset-anchored analysis windows.

        Two things matter now that every metric is onset-anchored (MATH.md §2):
        the shot must be *detectable* (a sample above the 1 Pa onset threshold),
        and there must be at least ``LIAEQ_WINDOW_MS`` of capture after it so the
        100 ms LIAeq window is not truncated. Total frame length is otherwise
        irrelevant, so this is a warning, not a hard rejection.
        """
        ident = f"Frame {frame.source_file!r} channel {frame.channel!r}"
        if onset is None:
            warnings.warn(
                f"{ident}: no sample exceeds the {ONSET_THRESHOLD_PA:g} Pa onset "
                f"threshold; metrics are computed from the frame start and may be "
                f"meaningless (expected a triggered {CAPTURE_MS:.0f} ms capture).",
                stacklevel=2,
            )
            return
        post_onset_ms = (frame.n_samples - onset) / frame.sample_rate * 1000.0
        if post_onset_ms + 1e-9 < LIAEQ_WINDOW_MS:
            warnings.warn(
                f"{ident}: only {post_onset_ms:.1f} ms of capture after onset, less "
                f"than the {LIAEQ_WINDOW_MS:.0f} ms LIAeq window; LIAeq and the peak "
                f"windows are truncated and not comparable to full-length shots.",
                stacklevel=2,
            )
