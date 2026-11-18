"""Ingestion tests: channel tagging (pure) + real-file reads (skipped if absent)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sound_metric_app.dsp import MetricsProcessor
from sound_metric_app.ingestion import list_channels, read_capture, read_frame, tag_channels
from sound_metric_app.models import Frame, MicPosition

# Point SMA_SAMPLE_DXD at a real file to enable these tests, or drop one in data/.
_CANDIDATES = [
    os.environ.get("SMA_SAMPLE_DXD"),
    str(Path(__file__).resolve().parents[1] / "data" / "Test_0009.dxd"),
    r"C:\Users\silen\Downloads\Test_0009.dxd",
]
SAMPLE = next((p for p in _CANDIDATES if p and Path(p).exists()), None)

requires_sample = pytest.mark.skipif(SAMPLE is None, reason="no sample .dxd available")


# --- channel tagging (no real file needed) --------------------------------- #


def _frame(name: str) -> Frame:
    return Frame(samples=np.zeros(4), sample_rate=200_000.0, channel=name, source_file="f.dxd")


def test_tag_two_channels():
    frames = [_frame("AI 1"), _frame("AI 2")]
    tagged = tag_channels(frames, {"AI 1": MicPosition.SE, "AI 2": MicPosition.ML})
    got = {t.position: t.frame.channel for t in tagged}
    assert got == {MicPosition.SE: "AI 1", MicPosition.ML: "AI 2"}


def test_tag_single_channel_ok():
    # A single-mic test condition tags just one channel.
    tagged = tag_channels([_frame("AI 1")], {"AI 1": MicPosition.SE})
    assert len(tagged) == 1 and tagged[0].position is MicPosition.SE


def test_tag_unknown_channel_rejected():
    with pytest.raises(ValueError):
        tag_channels([_frame("AI 1")], {"AI 9": MicPosition.SE})


def test_tag_duplicate_position_rejected():
    frames = [_frame("AI 1"), _frame("AI 2")]
    with pytest.raises(ValueError):
        tag_channels(frames, {"AI 1": MicPosition.SE, "AI 2": MicPosition.SE})


def test_tag_empty_mapping_rejected():
    with pytest.raises(ValueError):
        tag_channels([_frame("AI 1")], {})


# --- real-file reads ------------------------------------------------------- #


@requires_sample
def test_read_capture_yields_distinct_channels():
    frames = read_capture(SAMPLE)
    assert len(frames) >= 1
    names = [f.channel for f in frames]
    assert len(names) == len(set(names))  # distinct channel names
    for f in frames:
        assert f.sample_rate > 0 and f.n_samples > 0 and f.samples.ndim == 1


@requires_sample
def test_channels_have_pressure():
    channels = list_channels(SAMPLE)
    assert any(c.unit.strip().lower() == "pa" for c in channels)


@requires_sample
def test_read_frame_shape():
    frame = read_frame(SAMPLE)
    assert frame.sample_rate > 0
    assert frame.n_samples > 0
    assert frame.samples.ndim == 1


@requires_sample
def test_processor_produces_finite_metrics():
    frame = read_frame(SAMPLE)
    result = MetricsProcessor().process(frame)
    for value in (
        result.peak_pa,
        result.peak_db,
        result.peak_dba,
        result.peak_impulse_db,
        result.liaeq_100ms_db,
    ):
        assert value == value  # not NaN
        assert value not in (float("inf"), float("-inf"))


# --- multi-shot files ------------------------------------------------------- #


def test_split_segments_breaks_on_gaps_in_the_time_axis():
    from sound_metric_app.ingestion import split_segments

    fs = 200_000.0
    block = np.arange(42_000) / fs
    times = np.concatenate([block + 7.686, block + 14.5786])
    assert split_segments(times, fs) == [slice(0, 42_000), slice(42_000, 84_000)]
    assert split_segments(block, fs) == [slice(0, 42_000)]
    assert split_segments(np.array([]), fs) == []


_MULTI = r"C:\Users\Public\Documents\Dewesoft\Data\CAF556\105\caf556_105_03_0000.dxd"


@pytest.mark.skipif(not Path(_MULTI).exists(), reason="no two-trigger sample .dxd available")
def test_two_trigger_file_reads_as_two_shots():
    from sound_metric_app.ingestion import count_shots

    assert count_shots(_MULTI) == 2
    first, second = read_capture(_MULTI, 0), read_capture(_MULTI, 1)
    assert first[0].n_samples == second[0].n_samples == 42_000
    assert second[0].timestamp > first[0].timestamp
    # None reads the file whole, as files ingested before multi-shot support are.
    assert read_capture(_MULTI)[0].n_samples == 84_000
