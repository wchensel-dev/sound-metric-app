"""Tests for the GUI's headless view-model (:class:`WorkflowController`).

The controller carries all the GUI's logic with no Qt dependency, so the full
ingest -> mark -> close -> report flow is exercised here without a live window —
mirroring how ``tests/test_cli.py`` drives the CLI. Fake channel/capture readers
stand in for a real ``.dxd``; ``mark`` still runs the real DSP over sine frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from sound_metric_app.ingestion import ChannelInfo
from sound_metric_app.models import Frame, MicPosition
from sound_metric_app.ui.controller import WorkflowController

FS = 200_000.0


def _sine_frame(path: str, channel: str, amp: float = 1.0) -> Frame:
    t = np.arange(20_000) / FS
    return Frame(
        samples=amp * np.sin(2 * np.pi * 1000.0 * t),
        sample_rate=FS,
        channel=channel,
        source_file=path,
        timestamp=None,
    )


def _fake_channels(path: str) -> list[ChannelInfo]:
    return [
        ChannelInfo(name="AI 1", unit="Pa", sample_rate=FS, n_samples=20_000),
        ChannelInfo(name="AI 2", unit="Pa", sample_rate=FS, n_samples=20_000),
    ]


def _fake_capture(path: str) -> list[Frame]:
    return [_sine_frame(path, "AI 1"), _sine_frame(path, "AI 2")]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    """Controller on an isolated tmp DB + settings file, with fake readers."""
    monkeypatch.setenv("SMA_CONFIG", str(tmp_path / "sma_config.json"))
    return WorkflowController(
        tmp_path / "wf.db",
        channel_reader=_fake_channels,
        capture_reader=_fake_capture,
    )


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    return folder


def _touch(folder, name):
    (folder / name).write_bytes(b"")


def test_full_cycle_ingest_mark_close_report(controller, inbox):
    _touch(inbox, "SUP-1_AR15_001.dxd")
    _touch(inbox, "SUP-1_AR15_002.dxd")

    # Ingest from an explicit folder.
    report = controller.ingest(inbox, validate=False)
    assert report.n_ingested == 2
    assert len(controller.unmarked_shots()) == 2

    # Channel choices for the mark form come from the (fake) reader.
    shots = controller.unmarked_shots()
    assert [c.name for c in controller.channels_for(shots[0].source_file)] == ["AI 1", "AI 2"]

    # Mark both shots with SE + MR.
    for shot in shots:
        marked = controller.mark(
            shot.id,
            ammo="M855",
            channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
        )
        assert set(marked.metrics) == {MicPosition.SE, MicPosition.MR}

    assert controller.unmarked_shots() == []

    # One batch, one group, two shots, SE and MR averaged separately.
    batches = controller.batches()
    assert len(batches) == 1
    rep = controller.batch_report(batches[0].id)
    assert len(rep.groups) == 1
    group_avg = rep.groups[0]
    assert group_avg.n_shots == 2
    assert group_avg.averages[MicPosition.SE]["n"] == 2
    assert group_avg.averages[MicPosition.MR]["n"] == 2

    # Close the batch; it reports as closed and further marks would open a new one.
    controller.close_batch(batches[0].id)
    assert controller.get_batch(batches[0].id).closed is True


def test_batch_tree_nests_batches_groups_shots(controller, inbox):
    _touch(inbox, "SUP-1_AR15_001.dxd")
    _touch(inbox, "SUP-1_AR15_002.dxd")
    controller.ingest(inbox, validate=False)
    for shot in controller.unmarked_shots():
        controller.mark(
            shot.id,
            ammo="M855",
            channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
        )

    tree = controller.batch_tree()
    assert len(tree) == 1
    batch_node = tree[0]
    assert batch_node.batch.id == controller.batches()[0].id
    assert len(batch_node.groups) == 1
    group_node = batch_node.groups[0]
    # Shot count for the tree comes from the materialized shots, no COUNT query.
    assert len(group_node.shots) == 2


def test_ingest_without_folder_raises(controller):
    with pytest.raises(ValueError, match="No input folder"):
        controller.ingest()


def test_ingest_uses_configured_folder(controller, inbox):
    _touch(inbox, "SUP-1_AR15_001.dxd")
    controller.set_input_folder(inbox)
    report = controller.ingest(validate=False)  # no explicit folder
    assert report.n_ingested == 1


def test_rescan_adds_zero(controller, inbox):
    _touch(inbox, "SUP-1_AR15_001.dxd")
    assert controller.ingest(inbox, validate=False).n_ingested == 1
    second = controller.ingest(inbox, validate=False)
    assert second.n_ingested == 0
    assert len(second.already_present) == 1


def test_mark_unknown_shot_raises(controller):
    with pytest.raises(LookupError):
        controller.mark(999, ammo="M855", channel_map={"AI 1": MicPosition.SE})


def test_report_unknown_batch_raises(controller):
    with pytest.raises(LookupError):
        controller.batch_report(42)
