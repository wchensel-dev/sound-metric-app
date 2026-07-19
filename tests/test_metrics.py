"""Unit tests for DSP metrics against analytically-known signals.

Metric definitions follow TBAC's ``process_string.m`` (see MATH.md); several tests
assert that alignment directly (signed peak, unweighted ∫p·dt impulse, 1 Pa
onset, rectangular 10 ms-Leq).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import signal

from sound_metric_app.config import (
    LEQ_TAU_S,
    LIAEQ_WINDOW_MS,
    ONSET_THRESHOLD_PA,
    P_REF,
)
from sound_metric_app.dsp.metrics import (
    find_onset,
    pa_to_db,
    positive_phase_impulse_pa_ms,
    rms_pa,
    running_leq_rms,
    signed_peak_pa,
    window_samples,
)
from sound_metric_app.dsp.processor import MetricsProcessor
from sound_metric_app.dsp.weighting import a_weighting_sos
from sound_metric_app.models import Frame

FS = 200_000.0


def _sine(freq: float, amp_pa: float, dur_s: float = 0.5) -> np.ndarray:
    t = np.arange(int(FS * dur_s)) / FS
    return amp_pa * np.sin(2 * np.pi * freq * t)


def _blast_frame(peak_pa: float = 2000.0, n: int = 42_000, lead_ms: float = 10.0) -> Frame:
    """A Friedlander-like blast: quiet lead, then a decaying overpressure pulse."""
    t = np.arange(n) / FS
    lead = int(lead_ms / 1000.0 * FS)
    tau = 0.0008
    p = np.zeros(n)
    td = t[lead:] - t[lead]
    p[lead:] = peak_pa * (1 - td / tau) * np.exp(-td / tau)
    p += np.random.default_rng(0).normal(0, 1e-3, n)  # realistic quiet floor
    return Frame(samples=p, sample_rate=FS, channel="AI 1", source_file="SUP_AR15_001.dxd")


# --------------------------------------------------------------------------- #
# Onset
# --------------------------------------------------------------------------- #


def test_find_onset_is_first_sample_above_threshold():
    p = np.zeros(1000)
    p[400] = ONSET_THRESHOLD_PA + 0.5  # first crossing
    p[600] = 50.0
    assert find_onset(p) == 400


def test_find_onset_is_strict_and_signed():
    # Exactly at the threshold does not count (strict >), and a large *negative*
    # excursion is ignored — onset is a positive-overpressure crossing.
    p = np.zeros(100)
    p[10] = ONSET_THRESHOLD_PA  # equal, not above
    p[20] = -500.0  # negative, ignored
    p[30] = ONSET_THRESHOLD_PA + 1e-6
    assert find_onset(p) == 30


def test_find_onset_none_when_never_crossed():
    assert find_onset(np.full(100, 0.5)) is None


def test_find_onset_empty_array_returns_none():
    # np.argmax raises on an empty array; the guard must return None instead.
    assert find_onset(np.array([])) is None


# --------------------------------------------------------------------------- #
# Base operators
# --------------------------------------------------------------------------- #


def test_signed_peak_takes_positive_overpressure_not_magnitude():
    # A rarefaction trough larger in magnitude than the overpressure peak must NOT
    # win: TBAC reports max(Y), the positive peak.
    x = np.array([0.0, 3.0, -9.0, 2.0])
    assert signed_peak_pa(x) == 3.0


def test_pa_to_db_known_values():
    assert pa_to_db(P_REF) == pytest.approx(0.0)
    assert pa_to_db(1.0) == pytest.approx(20 * np.log10(1.0 / P_REF))
    assert pa_to_db(0.0) == float("-inf")


def test_rms_pa_of_sine():
    amp = 2.0
    assert rms_pa(_sine(1000.0, amp)) == pytest.approx(amp / np.sqrt(2), rel=1e-3)


def test_window_samples_rounds():
    assert window_samples(FS, 100.0) == 20_000
    assert window_samples(262_144.0, 10.0) == 2621


# --------------------------------------------------------------------------- #
# Positive-phase impulse  (∫p·dt, TBAC)
# --------------------------------------------------------------------------- #


def test_positive_phase_impulse_matches_analytic_triangle():
    # A positive triangular pulse of height A falling to 0 over d samples has area
    # 0.5 * (d*dt) * A in Pa·ms. cumtrapz + max should recover it.
    d = 100
    A = 1000.0
    seg = np.zeros(500)
    k = np.arange(d + 1)
    seg[: d + 1] = A * (1 - k / d)
    dt_ms = 1000.0 / FS
    expected = 0.5 * (d * dt_ms) * A  # = 250 Pa·ms
    assert positive_phase_impulse_pa_ms(seg, FS) == pytest.approx(expected, rel=1e-3)


def test_positive_phase_impulse_stops_before_negative_phase():
    # Positive rectangle then an equal negative rectangle: the running integral
    # rises to the positive-phase peak, then the negative phase pulls it down. The
    # reported impulse is the peak, not the (near-zero) end value.
    A = 500.0
    seg = np.concatenate([np.full(300, A), np.full(300, -A)])
    dt_ms = 1000.0 / FS
    expected = A * 300 * dt_ms  # positive-phase area (≈, trapezoid ends aside)
    got = positive_phase_impulse_pa_ms(seg, FS)
    assert got == pytest.approx(expected, rel=1e-2)


def test_positive_phase_impulse_nan_propagates():
    seg = _sine(1000.0, 100.0, dur_s=0.01)
    seg[len(seg) // 2] = np.nan
    assert np.isnan(positive_phase_impulse_pa_ms(seg, FS))


# --------------------------------------------------------------------------- #
# Rectangular running Leq
# --------------------------------------------------------------------------- #


def test_running_leq_rms_settles_to_sine_rms():
    amp = 3.0
    r = running_leq_rms(_sine(1000.0, amp), FS)
    # Past the L-sample ramp the trailing RMS equals the sine RMS.
    L = int(np.floor(FS * LEQ_TAU_S))
    assert np.median(r[3 * L :]) == pytest.approx(amp / np.sqrt(2), rel=1e-2)
    assert r.shape[0] == _sine(1000.0, amp).shape[0]


# --------------------------------------------------------------------------- #
# A-weighting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "freq, expected_db, tol",
    [
        (100.0, -19.1, 0.5),
        (1000.0, 0.0, 0.1),
        (10000.0, -2.5, 0.7),
    ],
)
def test_a_weighting_response(freq, expected_db, tol):
    sos = a_weighting_sos(FS)
    _, h = signal.sosfreqz(sos, worN=[freq], fs=FS)
    mag_db = 20 * np.log10(np.abs(h[0]))
    assert mag_db == pytest.approx(expected_db, abs=tol)


# --------------------------------------------------------------------------- #
# Processor (end-to-end, onset-anchored)
# --------------------------------------------------------------------------- #


def test_processor_metrics_are_self_consistent_and_sane():
    frame = _blast_frame(peak_pa=2000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nominal 42k frame: no warning
        r = MetricsProcessor().process(frame)

    # Peak dB of a 2000 Pa overpressure.
    assert r.peak_pa == pytest.approx(2000.0, rel=1e-3)
    assert r.peak_db == pytest.approx(20 * np.log10(2000.0 / P_REF), abs=0.01)
    # Every dB field is exactly its linear magnitude converted (the store relies
    # on this to average linear then convert).
    assert r.peak_db == pytest.approx(pa_to_db(r.peak_pa))
    assert r.peak_dba == pytest.approx(pa_to_db(r.peak_a_pa))
    assert r.peak_impulse_db == pytest.approx(pa_to_db(r.impulse_pa_ms))
    assert r.leq10ms_db == pytest.approx(pa_to_db(r.leq10ms_pa))
    assert r.liaeq_100ms_db == pytest.approx(pa_to_db(r.liaeq_pa))
    assert r.impulse_pa_ms > 0.0


def test_processor_impulse_is_unweighted():
    # TBAC integrates the raw (not A-weighted) pressure. The processor's impulse
    # must equal the positive-phase impulse of the raw signal over its window.
    from sound_metric_app.config import PEAK_WINDOW_MS

    frame = _blast_frame(peak_pa=2000.0)
    r = MetricsProcessor().process(frame)
    onset = find_onset(frame.samples)
    seg = frame.samples[onset : onset + window_samples(FS, PEAK_WINDOW_MS)]
    assert r.impulse_pa_ms == pytest.approx(positive_phase_impulse_pa_ms(seg, FS))


def test_processor_10ms_leq_exceeds_100ms_liaeq_for_a_transient():
    # A blast concentrates energy in a few ms, so the 10 ms-window Leq reads higher
    # than the 100 ms LIAeq — the deliberate window divergence.
    r = MetricsProcessor().process(_blast_frame(peak_pa=2000.0))
    assert r.leq10ms_db > r.liaeq_100ms_db


def test_processor_warns_when_no_onset():
    frame = Frame(
        samples=np.full(42_000, 0.5), sample_rate=FS, channel="AI 1", source_file="q.dxd"
    )
    with pytest.warns(UserWarning, match="onset threshold"):
        MetricsProcessor().process(frame)


def test_processor_warns_when_post_onset_shorter_than_liaeq_window():
    # A 100 ms frame with the shot 10 ms in leaves only 90 ms after onset, less
    # than the 100 ms LIAeq window.
    frame = _blast_frame(peak_pa=2000.0, n=20_000, lead_ms=10.0)
    with pytest.warns(UserWarning, match="LIAeq window"):
        MetricsProcessor().process(frame)


def test_processor_no_warning_for_nominal_capture():
    frame = _blast_frame(peak_pa=2000.0, n=42_000, lead_ms=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MetricsProcessor().process(frame)  # 200 ms after onset >= 100 ms window


def test_liaeq_window_is_100ms_from_onset():
    frame = _blast_frame(peak_pa=2000.0)
    r = MetricsProcessor().process(frame)
    onset = find_onset(frame.samples)
    from sound_metric_app.dsp.weighting import apply_a_weighting

    p_a = apply_a_weighting(frame.samples, FS)
    n = window_samples(FS, LIAEQ_WINDOW_MS)
    assert r.liaeq_pa == pytest.approx(rms_pa(p_a[onset : onset + n]))


def test_processor_handles_empty_frame_gracefully():
    # A malformed / zero-length capture must degrade to a result, not raise
    # (neither find_onset nor the A-weighting filter may crash on empty input).
    frame = Frame(samples=np.array([]), sample_rate=FS, channel="AI 1", source_file="e.dxd")
    with pytest.warns(UserWarning):  # "no onset" warning
        r = MetricsProcessor().process(frame)
    assert r.n_samples == 0
    assert r.impulse_pa_ms == 0.0
    assert r.peak_db == float("-inf")
