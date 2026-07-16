"""Core acoustic metric primitives.

All functions operate on a 1-D pressure signal in Pascals and return decibel
levels referenced to 20 microPascals.

NOTE: ``peak_db`` and A-weighted peak are unambiguous. The Impulse time-weighting
and the exact LIAeq definition are PROVISIONAL pending validation against the
values DewesoftX reports for the same file.
"""

from __future__ import annotations

import numpy as np

from ..config import IMPULSE_FALL_S, IMPULSE_RISE_S, P_REF


def peak_db(pressure: np.ndarray) -> float:
    """Peak level: 20*log10(|p|_max / p_ref)."""
    peak = float(np.max(np.abs(pressure)))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak / P_REF)


def leq_db(pressure: np.ndarray) -> float:
    """Equivalent continuous level over the whole array: 10*log10(<p^2>/p_ref^2)."""
    ms = float(np.mean(pressure**2))
    if ms <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(ms / P_REF**2)


def impulse_weighted_level(
    pressure: np.ndarray,
    fs: float,
    rise_s: float = IMPULSE_RISE_S,
    fall_s: float = IMPULSE_FALL_S,
) -> np.ndarray:
    """Instantaneous Impulse ('I') time-weighted level, sample by sample (dB).

    Squared pressure is exponentially smoothed with a fast rise (35 ms) and a
    slow fall (1500 ms) time constant, then converted to dB.
    """
    a_rise = np.exp(-1.0 / (fs * rise_s))
    a_fall = np.exp(-1.0 / (fs * fall_s))
    sq = pressure**2

    smoothed = np.empty_like(sq)
    acc = 0.0
    for i in range(sq.shape[0]):
        xi = sq[i]
        a = a_rise if xi > acc else a_fall
        acc = a * acc + (1.0 - a) * xi
        smoothed[i] = acc

    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(smoothed / P_REF**2)


def peak_impulse_db(pressure: np.ndarray, fs: float) -> float:
    """Maximum of the Impulse time-weighted level (dB)."""
    return float(np.max(impulse_weighted_level(pressure, fs)))
