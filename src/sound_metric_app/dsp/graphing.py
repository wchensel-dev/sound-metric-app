"""Build the per-metric time series a Report graph draws from a raw frame.

Kept Qt-free (numpy only) so the trace math is unit-testable without a GUI, and
so the heavy read + A-weight + smooth work can run on a worker thread. Each of
the report metrics maps to one curve over the capture's time axis, plus a
single annotation that explains where the reported number comes from:

* the peak metrics (raw Pascals, dB, and dBA) mark the largest *signed* sample in
  the onset-anchored peak window with a vertical bar;
* the Impulse metric plots the cumulative ``∫p·dt`` curve (Pa·ms) and marks its
  positive-phase peak; the Peak-10 ms-Leq metric marks the max of its running
  rectangular level;
* the energy-average metric (LIAeq) has no single peak, so it carries a
  horizontal reference line at the reported level instead.

Every curve spans the whole capture, not just the window its metric was computed
over, so the operator can see what the rest of the frame was doing — a second
blast or a reflection is visible even when it lands outside the window and
therefore changes nothing. ``window_start_index`` and ``window_end_index`` bracket
that window, separating the samples the reported number came from from the ones
drawn for context only. Widening the *drawn* span never widens a *computed* one.

Each SPL-over-time metric can be drawn two ways, chosen by the caller's
``smoothing`` argument: the raw per-sample instantaneous level
(:data:`SMOOTHING_INSTANT`), or an exponentially time-weighted RMS envelope at
the standard sound-level-meter Fast/Slow time constants (:data:`SMOOTHING_FAST`,
:data:`SMOOTHING_SLOW`). Instantaneous SPL swings from the 0 dB floor to the
cycle peak on every acoustic cycle, so it reads as a point cloud; the
time-weighted envelope is the continuous line a level meter shows.

The full sample count (thousands of points) is returned unthinned — the widget,
not this layer, decides how to render density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..config import (
    FAST_TIME_S,
    LEQ_SEARCH_MS,
    LIAEQ_WINDOW_MS,
    P_REF,
    PEAK_WINDOW_MS,
    SLOW_TIME_S,
)
from ..models import Frame
from .metrics import (
    _positive_phase_impulse,
    find_onset,
    leq_window_samples,
    pa_to_db,
    positive_phase_peak_index,
    rms_pa,
    running_leq_rms,
    window_samples,
)
from .weighting import apply_a_weighting

#: SPL-over-time rendering modes for :func:`build_metric_trace`.
SMOOTHING_INSTANT = "instant"  # raw per-sample level, no time weighting
SMOOTHING_FAST = "fast"  # 125 ms exponential RMS (SLM "Fast")
SMOOTHING_SLOW = "slow"  # 1 s exponential RMS (SLM "Slow")

#: Time constant (seconds) for each exponentially time-weighted mode.
_TIME_WEIGHT_TAU = {SMOOTHING_FAST: FAST_TIME_S, SMOOTHING_SLOW: SLOW_TIME_S}

#: Every accepted ``smoothing`` value.
_SMOOTHING_MODES = (SMOOTHING_INSTANT, SMOOTHING_FAST, SMOOTHING_SLOW)


@dataclass
class MetricTrace:
    """One metric's curve over a capture, ready to hand straight to a plot.

    ``peak_index`` and ``level`` are mutually exclusive annotations: a peak-style
    metric sets ``peak_index`` (vertical marker) and leaves ``level`` ``None``; the
    energy-average metric sets ``level`` (horizontal marker) and leaves
    ``peak_index`` ``None``. Either may be ``None`` (e.g. a fully silent frame).

    ``window_start_index`` and ``window_end_index`` bracket *this* metric's
    onset-anchored calculation window — the samples that produced the reported
    number, inside the whole capture that every curve now draws. Both bounds are
    *inclusive*: ``window_end_index`` is the last sample the metric used, not the
    exclusive slice bound. They are purely
    annotations: no value in ``values`` depends on either. The start is the
    detected onset for every metric except Peak-10 ms-Leq, whose trailing 10 ms
    RMS makes its peak integrate up to ``L-1`` pre-onset samples and so opens the
    window that much earlier; the end differs too (Peak-10 ms-Leq closes at
    25 ms, the rest at 100 ms). Both are therefore stored per trace rather than
    read from a single constant.

    The A-weighted traces carry a residual caveat the brackets cannot express:
    :func:`~sound_metric_app.dsp.weighting.apply_a_weighting` is an IIR filter
    run over the whole capture, so pre-window pressure influences the weighted
    samples inside it with an exponentially decaying tail rather than a bounded
    lookback.
    """

    t_ms: np.ndarray  # time axis, milliseconds from capture start
    values: np.ndarray  # y values, dB (may contain NaN gaps for silent samples)
    y_label: str
    title: str
    peak_index: int | None = None  # sample index for a vertical peak marker
    level: float | None = None  # y for a horizontal reference line
    connected: bool = False  # draw as a joined line (envelope) vs a point cloud
    window_start_index: int | None = None  # sample index where the window opens
    window_end_index: int | None = None  # last sample index inside this metric's window


def _spl_db(pressure: np.ndarray) -> np.ndarray:
    """Instantaneous SPL of a pressure signal: ``20*log10(|p|/p_ref)`` (dB).

    Magnitude is floored at ``P_REF`` (0 dB) so zero-crossings become a clean
    0 dB floor rather than ``-inf`` spikes that would wreck the plot's autoscale.
    """
    mag = np.maximum(np.abs(pressure), P_REF)
    return 20.0 * np.log10(mag / P_REF)


def _exp_rms_spl_db(pressure: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """Exponentially time-weighted RMS level (dB), sound-level-meter Fast/Slow style.

    Squared pressure is passed through a single-pole exponential average with
    time constant ``tau_s`` (Fast = 125 ms, Slow = 1 s), then converted to dB.
    This is the integration a level meter applies: it turns the per-cycle swing
    of the raw waveform into a continuous level envelope. The mean-square is
    floored at ``P_REF**2`` (0 dB) so quiet stretches never fall to ``-inf``.
    """
    a = float(np.exp(-1.0 / (fs * tau_s)))
    sq = pressure**2
    # y[n] = a*y[n-1] + (1-a)*x[n]: a first-order IIR low-pass on the power.
    mean_sq = signal.lfilter([1.0 - a], [1.0, -a], sq)
    mean_sq = np.maximum(mean_sq, P_REF**2)
    return 10.0 * np.log10(mean_sq / P_REF**2)


def _exp_rms_pa(pressure: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """Exponentially time-weighted RMS pressure in Pascals (linear, not dB).

    The Pascal-domain analogue of :func:`_exp_rms_spl_db`: the same single-pole
    average over squared pressure, but returned as the square-root RMS envelope
    in Pa rather than converted to a dB level. Used for the raw ``peak_pa`` trace
    so its Fast/Slow curve stays in the same units as the instantaneous one.
    """
    a = float(np.exp(-1.0 / (fs * tau_s)))
    mean_sq = signal.lfilter([1.0 - a], [1.0, -a], pressure**2)
    return np.sqrt(np.maximum(mean_sq, 0.0))


def _onset_window(fs: float, onset: int | None, window_ms: float) -> tuple[int, int]:
    """``[start, stop)`` sample bounds of the onset-anchored ``window_ms`` window.

    Falls back to the frame start when no onset is detected, mirroring the
    processor so a graph annotates the same window the stored metric used.
    """
    start = onset if onset is not None else 0
    return start, start + window_samples(fs, window_ms)


def _window_bounds(start: int, stop: int, n_samples: int) -> tuple[int | None, int | None]:
    """Sample indices to draw the calculation window's start/end markers at.

    ``stop`` is the exclusive bound the metrics slice with, so the end marker
    goes at ``stop - 1`` — the last sample that actually fed the reported
    number. Marking ``stop`` itself would put the line one sample past the data
    it brackets, and would drop the marker whenever the window ends exactly at
    the capture end (``stop == n_samples``), which is the common case.

    Either is ``None`` when that edge falls outside the capture, since there is
    then no boundary inside the plot to mark — a window running past the last
    sample means the whole tail is window, with nothing to separate it from.
    An empty window (``stop <= start``) has no last sample and so no end marker.
    """
    last = stop - 1
    return (
        start if 0 <= start < n_samples else None,
        last if start <= last < n_samples else None,
    )


def _signed_peak_index(signal: np.ndarray, start: int, stop: int) -> int | None:
    """Index of the largest *signed* sample in ``signal[start:stop]``, or None.

    Matches the reported peak (``max``, the positive overpressure — not
    ``max|p|``) over the onset-anchored window.
    """
    seg = signal[start:stop]
    if seg.size == 0:
        return None
    return start + int(np.argmax(seg))


def build_metric_trace(
    frame: Frame, metric_key: str, smoothing: str = SMOOTHING_INSTANT
) -> MetricTrace:
    """Turn a raw :class:`Frame` into the :class:`MetricTrace` for ``metric_key``.

    ``metric_key`` is one of the report's stored metric columns: ``peak_pa``,
    ``peak_db``, ``peak_dba``, ``peak_impulse_db``, ``leq10ms_db``,
    ``liaeq_100ms_db``.

    ``smoothing`` selects how an SPL-over-time curve is drawn — the raw per-sample
    level (:data:`SMOOTHING_INSTANT`) or a Fast/Slow time-weighted RMS envelope
    (:data:`SMOOTHING_FAST` / :data:`SMOOTHING_SLOW`). It shapes only that curve;
    the reported scalar (the peak sample, the LIAeq level) is unchanged, so the
    marker and reference line stay put. The Impulse and Peak-10 ms-Leq traces
    carry their own dedicated curves and ignore ``smoothing``.
    """
    if smoothing not in _SMOOTHING_MODES:
        raise ValueError(f"Unknown smoothing mode: {smoothing!r}")

    p = np.asarray(frame.samples, dtype=float)
    fs = float(frame.sample_rate)
    t_ms = np.arange(p.shape[0]) / fs * 1000.0
    onset = find_onset(p)

    def spl(sig: np.ndarray) -> tuple[np.ndarray, bool]:
        """SPL-over-time values for ``sig`` plus whether to join them as a line."""
        if smoothing == SMOOTHING_INSTANT:
            return _spl_db(sig), False
        return _exp_rms_spl_db(sig, fs, _TIME_WEIGHT_TAU[smoothing]), True

    if metric_key == "peak_pa":
        # Raw pressure, unconverted. Instantaneous is the literal waveform (Pa);
        # Fast/Slow is the RMS pressure envelope in the same units.
        start, stop = _onset_window(fs, onset, PEAK_WINDOW_MS)
        w_start, w_end = _window_bounds(start, stop, p.shape[0])
        if smoothing == SMOOTHING_INSTANT:
            values, connected = p, False
        else:
            values = _exp_rms_pa(p, fs, _TIME_WEIGHT_TAU[smoothing])
            connected = True
        return MetricTrace(
            t_ms, values, "Pressure (Pa)", "Peak Pa",
            peak_index=_signed_peak_index(p, start, stop), connected=connected,
            window_start_index=w_start, window_end_index=w_end,
        )

    if metric_key == "peak_db":
        start, stop = _onset_window(fs, onset, PEAK_WINDOW_MS)
        w_start, w_end = _window_bounds(start, stop, p.shape[0])
        values, connected = spl(p)
        return MetricTrace(
            t_ms, values, "SPL (dB)", "Peak dB",
            peak_index=_signed_peak_index(p, start, stop), connected=connected,
            window_start_index=w_start, window_end_index=w_end,
        )

    if metric_key == "peak_dba":
        p_a = apply_a_weighting(p, fs)
        start, stop = _onset_window(fs, onset, PEAK_WINDOW_MS)
        w_start, w_end = _window_bounds(start, stop, p.shape[0])
        values, connected = spl(p_a)
        return MetricTrace(
            t_ms, values, "SPL (dBA)", "Peak dBA",
            peak_index=_signed_peak_index(p_a, start, stop), connected=connected,
            window_start_index=w_start, window_end_index=w_end,
        )

    if metric_key in ("peak_impulse_db", "impulse_pa_ms"):
        # The cumulative positive-phase impulse ∫p·dt (Pa·ms); the marker sits at
        # the peak of the positive phase (the reported value), found before the
        # running integral turns over into its minimum.
        #
        # The curve is drawn from onset to the end of the capture, but the
        # reported peak comes from the window alone: the peak search runs over
        # `q_draw[: stop - start]`, not all of `q_draw`, precisely so extending
        # the drawn curve cannot reach the marker. That slice *is* the window's
        # own integral — a cumulative sum of the same samples from the same
        # start — so the marker matches the reported scalar exactly while the
        # drawn curve continues past it rather than restarting. Pre-onset
        # samples stay NaN: ∫p·dt is defined from the onset, so a flat 0 there
        # would draw a value the integral does not have.
        start, stop = _onset_window(fs, onset, PEAK_WINDOW_MS)
        w_start, w_end = _window_bounds(start, stop, p.shape[0])
        q_draw, _ = _positive_phase_impulse(p[start:], fs)
        local_peak = positive_phase_peak_index(q_draw[: stop - start])
        values = np.full(p.shape[0], np.nan)
        values[start : start + q_draw.size] = q_draw
        peak_index = start + local_peak if local_peak is not None else None
        return MetricTrace(
            t_ms, values, "Impulse ∫p·dt (Pa·ms)", "Peak Impulse",
            peak_index=peak_index, connected=True,
            window_start_index=w_start, window_end_index=w_end,
        )

    if metric_key == "leq10ms_db":
        p_a = apply_a_weighting(p, fs)
        rms = running_leq_rms(p_a, fs)
        with np.errstate(divide="ignore"):
            values = 20.0 * np.log10(np.maximum(rms, P_REF) / P_REF)
        start, stop = _onset_window(fs, onset, LEQ_SEARCH_MS)
        # Each running-Leq value is a *trailing* 10 ms RMS, so a peak found at
        # `start + k` with k < L integrates L-1-k samples from before the onset.
        # Draw the window from `start - (L-1)`: the earliest sample that can feed
        # any value in the search span, which keeps the "only samples between the
        # lines reached the reported number" reading true (a superset — samples
        # after the marked peak contribute nothing either). This is the one metric
        # whose window start is not the onset itself.
        lookback = leq_window_samples(fs) - 1
        w_start, w_end = _window_bounds(max(0, start - lookback), stop, p.shape[0])
        seg = rms[start:stop]
        peak_index = start + int(np.argmax(seg)) if seg.size else None
        return MetricTrace(
            t_ms, values, "Leq 10 ms (dBA)", "Peak Leq 10 ms",
            peak_index=peak_index, connected=True,
            window_start_index=w_start, window_end_index=w_end,
        )

    if metric_key == "liaeq_100ms_db":
        p_a = apply_a_weighting(p, fs)
        start, stop = _onset_window(fs, onset, LIAEQ_WINDOW_MS)
        w_start, w_end = _window_bounds(start, stop, p.shape[0])
        values, connected = spl(p_a)
        return MetricTrace(
            t_ms, values, "SPL (dBA)", "LIAeq,100ms",
            level=pa_to_db(rms_pa(p_a[start:stop])), connected=connected,
            window_start_index=w_start, window_end_index=w_end,
        )

    raise ValueError(f"Unknown metric key: {metric_key!r}")
