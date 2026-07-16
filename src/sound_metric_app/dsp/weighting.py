"""Frequency weighting filters (IEC 61672)."""

from __future__ import annotations

import functools

import numpy as np
from scipy import signal

# A-weighting pole frequencies (Hz), IEC 61672 / ANSI S1.4.
_F1, _F2, _F3, _F4 = 20.598997, 107.65265, 737.86223, 12194.217


@functools.lru_cache(maxsize=8)
def a_weighting_sos(fs: float) -> np.ndarray:
    """A-weighting filter as second-order sections, normalized to 0 dB @ 1 kHz.

    Built from the analog IEC-61672 prototype and bilinear-transformed to the
    given sample rate. Cached per sample rate.
    """
    zeros = [0.0, 0.0, 0.0, 0.0]
    poles = [
        -2 * np.pi * _F1,
        -2 * np.pi * _F1,
        -2 * np.pi * _F2,
        -2 * np.pi * _F3,
        -2 * np.pi * _F4,
        -2 * np.pi * _F4,
    ]
    gain = (2 * np.pi * _F4) ** 2

    zd, pd, kd = signal.bilinear_zpk(zeros, poles, gain, fs)
    sos = signal.zpk2sos(zd, pd, kd)

    # Normalize so magnitude at 1 kHz is exactly unity (0 dB).
    _, h = signal.sosfreqz(sos, worN=[1000.0], fs=fs)
    sos[0, :3] /= np.abs(h[0])
    return sos


def apply_a_weighting(x: np.ndarray, fs: float) -> np.ndarray:
    """Apply the A-weighting filter to a pressure signal (Pascals in/out)."""
    sos = a_weighting_sos(fs)
    return signal.sosfilt(sos, x)
