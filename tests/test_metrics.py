"""Unit tests for DSP metrics against analytically-known signals."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from sound_metric_app.config import P_REF
from sound_metric_app.dsp.metrics import leq_db, peak_db
from sound_metric_app.dsp.weighting import a_weighting_sos

FS = 200_000.0


def _sine(freq: float, amp_pa: float, dur_s: float = 0.5) -> np.ndarray:
    t = np.arange(int(FS * dur_s)) / FS
    return amp_pa * np.sin(2 * np.pi * freq * t)


def test_peak_db_known_amplitude():
    # 1 Pa peak -> 20*log10(1 / 20e-6) = 93.98 dB
    x = _sine(1000.0, amp_pa=1.0)
    assert peak_db(x) == pytest.approx(20 * np.log10(1.0 / P_REF), abs=0.01)


def test_leq_db_of_sine():
    # RMS of a unit-amplitude sine is 1/sqrt(2); Leq should match 20*log10(rms/pref).
    amp = 2.0
    x = _sine(1000.0, amp_pa=amp)
    expected = 20 * np.log10((amp / np.sqrt(2)) / P_REF)
    assert leq_db(x) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize(
    "freq, expected_db, tol",
    [
        (100.0, -19.1, 0.5),   # IEC 61672 A-weighting relative response
        (1000.0, 0.0, 0.1),    # normalized to 0 dB at 1 kHz
        (10000.0, -2.5, 0.7),
    ],
)
def test_a_weighting_response(freq, expected_db, tol):
    sos = a_weighting_sos(FS)
    _, h = signal.sosfreqz(sos, worN=[freq], fs=FS)
    mag_db = 20 * np.log10(np.abs(h[0]))
    assert mag_db == pytest.approx(expected_db, abs=tol)
