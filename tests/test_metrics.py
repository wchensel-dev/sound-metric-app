"""Unit tests for DSP metrics against analytically-known signals."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import signal

from sound_metric_app.config import P_REF, WINDOW_MS
from sound_metric_app.dsp.metrics import leq_db, peak_db, peak_impulse_db
from sound_metric_app.dsp.processor import MetricsProcessor
from sound_metric_app.dsp.weighting import a_weighting_sos
from sound_metric_app.models import Frame

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


def test_peak_impulse_db_silent_frame_integrates_to_zero():
    # An all-zero (silent) frame: every sample is -inf and is skipped, so the
    # integral is exactly 0 rather than -inf.
    x = np.zeros(int(FS * 0.1), dtype=np.float64)
    assert peak_impulse_db(x, FS) == 0.0


def test_peak_impulse_db_nan_input_surfaces_as_nan():
    # A NaN-contaminated frame must NOT be silently reported as 0.0 (a plausible
    # silent-frame value); the NaN has to propagate so the corruption is visible.
    x = _sine(1000.0, amp_pa=1.0, dur_s=0.1)
    x[len(x) // 2] = np.nan
    assert np.isnan(peak_impulse_db(x, FS))


def _frame(n_samples: int) -> Frame:
    return Frame(
        samples=np.zeros(n_samples, dtype=np.float64),
        sample_rate=FS,
        channel="AI 1",
        source_file="SUP_AR15_001.dxd",
    )


def test_processor_no_warning_for_nominal_duration():
    # 100 ms at 200 kHz -> 20 000 samples: no off-nominal warning.
    frame = _frame(int(FS * WINDOW_MS / 1000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MetricsProcessor().process(frame)


def test_processor_warns_for_off_nominal_duration():
    # 120 ms frame: peak_impulse_db integrates over the frame, so the extra
    # length makes it non-comparable; the processor should warn.
    frame = _frame(int(FS * 0.120))
    with pytest.warns(UserWarning, match="not the nominal"):
        MetricsProcessor().process(frame)
