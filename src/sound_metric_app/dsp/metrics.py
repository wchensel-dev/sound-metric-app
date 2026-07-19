"""Core acoustic metric primitives, aligned to TBAC's ``process_string.m``.

All functions operate on a 1-D pressure signal in Pascals. Every reported metric
is computed over a fixed window anchored to the shot onset; this module provides
the pure, window-agnostic operators (the caller slices the window it wants) plus
the onset detector. Exact definitions live in ``MATH.md``.

Levels are ``20*log10(magnitude / p_ref)`` where the magnitude is a pressure (Pa)
or a positive-phase impulse (Pa·ms) — TBAC reports both that way.
"""

from __future__ import annotations

import numpy as np

from ..config import LEQ_TAU_S, ONSET_THRESHOLD_PA, P_REF


def find_onset(pressure: np.ndarray, threshold_pa: float = ONSET_THRESHOLD_PA) -> int | None:
    """Index of the first sample whose *signed* pressure exceeds ``threshold_pa``.

    TBAC's shot-onset detector (``find(Y>1.)``): the first raw-pressure sample
    above 1 Pa. Returns ``None`` when no sample crosses the threshold (a silent /
    non-shot frame) or the frame is empty, leaving the caller to decide how to
    handle it.
    """
    p = np.asarray(pressure)
    if p.size == 0:
        return None
    above = p > threshold_pa
    idx = int(np.argmax(above))
    return idx if bool(above[idx]) else None


def window_samples(fs: float, window_ms: float) -> int:
    """Number of samples spanning ``window_ms`` at rate ``fs`` (rounded)."""
    return int(round(window_ms * fs / 1000.0))


def pa_to_db(pa: float) -> float:
    """Level of a linear magnitude: ``20*log10(pa / p_ref)`` (dB).

    Works for a pressure (Pa) or an impulse (Pa·ms). Returns ``-inf`` for a
    non-positive magnitude (silent segment).
    """
    value = float(pa)
    if value <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(value / P_REF)


def signed_peak_pa(pressure: np.ndarray) -> float:
    """Largest *signed* sample of the (already-windowed) segment, Pa.

    TBAC reports ``max(Y)``, not ``max|Y|``: the blast overpressure peak, not the
    largest magnitude (which could be a rarefaction trough). Returns ``-inf`` for
    an empty segment.
    """
    p = np.asarray(pressure)
    if p.size == 0:
        return float("-inf")
    return float(np.max(p))


def rms_pa(pressure: np.ndarray) -> float:
    """Root-mean-square pressure of an array, Pa: ``sqrt(mean(p**2))``.

    The linear magnitude behind an equivalent-continuous (Leq) level; ``pa_to_db``
    of it gives the dB. Returns 0.0 for an empty segment.
    """
    p = np.asarray(pressure)
    if p.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(p**2)))


def _positive_phase_impulse(segment: np.ndarray, fs: float) -> tuple[np.ndarray, int | None]:
    """Running impulse ``∫p·dt`` (Pa·ms) of a segment and its positive-phase peak.

    Single source of truth shared by :func:`positive_phase_impulse_pa_ms` (which
    reports the scalar peak) and the Report graph's Impulse trace (which draws the
    ``q`` curve and marks the peak) so the two can never drift. Returns
    ``(q, peak_index)`` where ``q`` is the cumulative-trapezoid integral (same
    length as ``segment``, ``q[0] = 0``) and ``peak_index`` indexes ``q`` at the
    positive-phase peak — the max of ``q`` up to its minimum (the end of the
    negative phase), per TBAC's min-bounding rule. ``peak_index`` is ``None`` only
    for an empty segment.
    """
    seg = np.asarray(segment, dtype=float)
    n = seg.size
    if n == 0:
        return np.zeros(0), None
    if n < 2:
        return np.zeros(n), 0
    dt_ms = 1000.0 / fs
    # Cumulative trapezoidal integral, same length as seg, q[0] = 0 (no scipy dep).
    q = np.concatenate(([0.0], np.cumsum((seg[:-1] + seg[1:]) * 0.5 * dt_ms)))
    i_min = int(np.argmin(q))
    upper = q if i_min == 0 else q[: i_min + 1]
    return q, int(np.argmax(upper))


def positive_phase_impulse_pa_ms(pressure: np.ndarray, fs: float) -> float:
    """Peak positive-phase acoustic impulse ``∫p·dt`` over the segment, in Pa·ms.

    The running (cumulative-trapezoid) integral of pressure vs time rises through
    the blast's positive-overpressure phase and falls once pressure turns
    negative. Its peak is the positive impulse. Following TBAC, the peak is taken
    *before* the running integral's minimum (the deepest point of the negative
    phase), so a later secondary rise cannot inflate it::

        Q       = cumtrapz(p, dt_ms)          # Pa·ms, Q[0] = 0
        i_min   = argmin(Q)                   # end of the negative phase
        impulse = max(Q[: i_min + 1])         # peak of the positive phase

    The min-bounding rejects a later (e.g. reflected) rise **only when the negative
    phase drives Q below its start** (``i_min > 0``) — the usual free-field case,
    where the rarefaction pulls the running integral negative after the positive
    peak. When Q never dips below its start (``i_min == 0``), the impulse is the
    global max over the whole window; a within-window reflection could then inflate
    it. Free-field capture (no early reflections) is what makes this safe here;
    MATH.md §6 spells out the caveat.

    Time is integrated in **milliseconds**, so the result is Pa·ms — matching TBAC
    (whose ``dB*ms`` is ``pa_to_db`` of this value). A NaN in the input propagates
    so contaminated data surfaces instead of a plausible-looking value.
    """
    q, peak_index = _positive_phase_impulse(pressure, fs)
    if peak_index is None or q.size < 2:
        return 0.0
    if np.isnan(q).any():
        return float("nan")
    return max(float(q[peak_index]), 0.0)


def running_leq_rms(pressure: np.ndarray, fs: float, tau_s: float = LEQ_TAU_S) -> np.ndarray:
    """Rectangular running RMS (Pa), same length as the input.

    A causal trailing moving-RMS of the pressure over ``L = floor(fs*tau)``
    samples — the rectangular-kernel form of Tougaard & Beedholm's ``Leq_fast``
    (the mean-square is a boxcar sum divided by ``L``, then square-rooted). Unlike
    ``Leq_fast``'s FFT (circular) convolution this is strictly causal, so the
    leading ``L`` samples ramp up from zero state rather than wrapping the array
    tail; an onset-anchored search window sits past that ramp, so reported maxima
    match. The caller takes the max over its search window.
    """
    p = np.asarray(pressure, dtype=float)
    n = p.size
    L = int(np.floor(fs * tau_s))
    if L < 1 or n == 0:
        return np.abs(p)
    csum = np.cumsum(p**2)
    ms = np.empty(n, dtype=float)
    upto = min(L, n)
    ms[:upto] = csum[:upto] / L  # causal ramp-up: partial window / L
    if n > L:
        ms[L:] = (csum[L:] - csum[:-L]) / L  # trailing L-sample mean of p²
    return np.sqrt(np.maximum(ms, 0.0))
