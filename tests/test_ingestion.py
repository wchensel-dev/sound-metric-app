"""Ingestion tests against a real Dewesoft file (skipped if not present)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sound_metric_app.dsp import MetricsProcessor
from sound_metric_app.ingestion import list_channels, read_frame

# Point SMA_SAMPLE_DXD at a real file to enable these tests, or drop one in data/.
_CANDIDATES = [
    os.environ.get("SMA_SAMPLE_DXD"),
    str(Path(__file__).resolve().parents[1] / "data" / "Test_0009.dxd"),
    r"C:\Users\silen\Downloads\Test_0009.dxd",
]
SAMPLE = next((p for p in _CANDIDATES if p and Path(p).exists()), None)

pytestmark = pytest.mark.skipif(SAMPLE is None, reason="no sample .dxd available")


def test_channels_have_pressure():
    channels = list_channels(SAMPLE)
    assert any(c.unit.strip().lower() == "pa" for c in channels)


def test_read_frame_shape():
    frame = read_frame(SAMPLE)
    assert frame.sample_rate > 0
    assert frame.n_samples > 0
    assert frame.samples.ndim == 1


def test_processor_produces_finite_metrics():
    frame = read_frame(SAMPLE)
    result = MetricsProcessor().process(frame)
    for value in (
        result.peak_db,
        result.peak_dba,
        result.peak_impulse_db,
        result.liaeq_100ms_db,
    ):
        assert value == value  # not NaN
        assert value not in (float("inf"), float("-inf"))
