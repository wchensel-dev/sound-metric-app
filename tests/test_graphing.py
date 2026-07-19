"""Unit tests for the Report graph's trace builder (Qt-free DSP layer)."""

from __future__ import annotations

import numpy as np
import pytest

from sound_metric_app.config import LIAEQ_WINDOW_MS, PEAK_WINDOW_MS
from sound_metric_app.dsp import (
    SMOOTHING_FAST,
    SMOOTHING_INSTANT,
    SMOOTHING_SLOW,
    build_metric_trace,
)
from sound_metric_app.dsp.metrics import (
    find_onset,
    pa_to_db,
    positive_phase_impulse_pa_ms,
    rms_pa,
    window_samples,
)
from sound_metric_app.dsp.weighting import apply_a_weighting
from sound_metric_app.models import Frame

FS = 200_000.0
N = 42_000


def _shot_frame() -> Frame:
    """A blast that crosses the 1 Pa onset ~10 ms in: quiet lead, decaying pulse."""
    t = np.arange(N) / FS
    lead = int(0.010 * FS)
    tau = 0.0008
    samples = np.zeros(N)
    td = t[lead:] - t[lead]
    samples[lead:] = 2000.0 * (1 - td / tau) * np.exp(-td / tau)
    return Frame(samples=samples, sample_rate=FS, channel="AI 1", source_file="x.dxd")


def _onset_window(p):
    onset = find_onset(p) or 0
    return onset, onset + window_samples(FS, PEAK_WINDOW_MS)


def test_peak_db_marks_signed_peak_in_onset_window():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "peak_db")
    start, stop = _onset_window(frame.samples)

    assert trace.t_ms.shape == (N,)
    assert trace.values.shape == (N,)
    assert trace.level is None
    # The bar sits on the largest *signed* sample within the onset window.
    assert trace.peak_index == start + int(np.argmax(frame.samples[start:stop]))
    assert np.all(np.isfinite(trace.values))
    assert trace.values.min() >= 0.0


def test_peak_dba_uses_a_weighted_signal():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "peak_dba")
    p_a = apply_a_weighting(frame.samples, FS)
    start, stop = _onset_window(frame.samples)
    assert trace.peak_index == start + int(np.argmax(p_a[start:stop]))
    assert trace.y_label == "SPL (dBA)"


def test_peak_pa_trace_is_the_raw_pressure_waveform():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "peak_pa")
    start, stop = _onset_window(frame.samples)
    assert trace.y_label == "Pressure (Pa)"
    assert trace.connected is False
    assert np.array_equal(trace.values, np.asarray(frame.samples, dtype=float))
    assert trace.peak_index == start + int(np.argmax(frame.samples[start:stop]))
    assert trace.level is None


def test_peak_pa_time_weighted_is_a_connected_pascal_envelope():
    trace = build_metric_trace(_shot_frame(), "peak_pa", SMOOTHING_FAST)
    assert trace.y_label == "Pressure (Pa)"
    assert trace.connected is True
    assert np.all(trace.values >= 0.0)


def test_impulse_trace_is_the_cumulative_integral_curve():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "peak_impulse_db")
    # ∫p·dt is drawn only over the onset window; the rest is NaN gaps.
    assert "Pa·ms" in trace.y_label
    assert trace.connected is True
    assert np.any(np.isnan(trace.values))
    assert not np.any(np.isneginf(trace.values))
    # The marker sits on the positive-phase peak, with a positive impulse value.
    assert trace.peak_index is not None
    assert trace.values[trace.peak_index] > 0.0


def test_impulse_marker_equals_reported_scalar():
    # Regression guard: the graph's Impulse marker must land on exactly the value
    # positive_phase_impulse_pa_ms reports. Both now share one implementation
    # (_positive_phase_impulse), so a change to the min-bounding rule can't move
    # the marker off the reported number.
    frame = _shot_frame()
    trace = build_metric_trace(frame, "peak_impulse_db")
    start, stop = _onset_window(frame.samples)
    reported = positive_phase_impulse_pa_ms(frame.samples[start:stop], FS)
    assert trace.peak_index is not None
    assert trace.values[trace.peak_index] == pytest.approx(reported)


def test_impulse_pa_ms_key_graphs_the_same_curve():
    frame = _shot_frame()
    a = build_metric_trace(frame, "peak_impulse_db")
    b = build_metric_trace(frame, "impulse_pa_ms")
    assert b.peak_index == a.peak_index
    np.testing.assert_array_equal(np.nan_to_num(a.values), np.nan_to_num(b.values))


def test_leq10ms_trace_marks_running_level_peak():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "leq10ms_db")
    assert trace.connected is True
    assert trace.y_label == "Leq 10 ms (dBA)"
    assert trace.peak_index is not None
    assert trace.level is None


def test_liaeq_trace_has_level_line_not_peak():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "liaeq_100ms_db")
    onset = find_onset(frame.samples) or 0
    p_a = apply_a_weighting(frame.samples, FS)
    n = window_samples(FS, LIAEQ_WINDOW_MS)
    assert trace.peak_index is None
    assert trace.level == pytest.approx(pa_to_db(rms_pa(p_a[onset : onset + n])))


def test_default_smoothing_is_instantaneous_point_cloud():
    assert build_metric_trace(_shot_frame(), "peak_db").connected is False


@pytest.mark.parametrize("smoothing", [SMOOTHING_FAST, SMOOTHING_SLOW])
def test_time_weighted_trace_is_a_connected_smoother_curve(smoothing):
    frame = _shot_frame()
    instant = build_metric_trace(frame, "peak_db", SMOOTHING_INSTANT)
    weighted = build_metric_trace(frame, "peak_db", smoothing)
    assert weighted.connected is True
    assert weighted.values.shape == instant.values.shape
    assert np.nanmax(np.abs(np.diff(weighted.values))) < np.nanmax(
        np.abs(np.diff(instant.values))
    )
    assert weighted.peak_index == instant.peak_index


def test_liaeq_smoothing_keeps_its_level_line():
    frame = _shot_frame()
    trace = build_metric_trace(frame, "liaeq_100ms_db", SMOOTHING_FAST)
    onset = find_onset(frame.samples) or 0
    p_a = apply_a_weighting(frame.samples, FS)
    n = window_samples(FS, LIAEQ_WINDOW_MS)
    assert trace.connected is True
    assert trace.peak_index is None
    assert trace.level == pytest.approx(pa_to_db(rms_pa(p_a[onset : onset + n])))


def test_unknown_smoothing_raises():
    with pytest.raises(ValueError):
        build_metric_trace(_shot_frame(), "peak_db", "medium")


def test_unknown_metric_key_raises():
    with pytest.raises(ValueError):
        build_metric_trace(_shot_frame(), "not_a_metric")


def test_time_axis_spans_the_capture_in_ms():
    trace = build_metric_trace(_shot_frame(), "peak_db")
    assert trace.t_ms[0] == 0.0
    assert trace.t_ms[-1] == pytest.approx((N - 1) / FS * 1000.0)


def test_build_metric_trace_handles_empty_frame():
    # A zero-length capture must not crash any trace (find_onset + A-weighting
    # both guard empty input); every trace comes back empty.
    frame = Frame(samples=np.array([]), sample_rate=FS, channel="AI 1", source_file="e.dxd")
    for key in ("peak_pa", "peak_db", "peak_dba", "peak_impulse_db", "leq10ms_db", "liaeq_100ms_db"):
        trace = build_metric_trace(frame, key)
        assert trace.values.shape == (0,)
