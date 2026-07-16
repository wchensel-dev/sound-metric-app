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


def _ingest_and_mark_one(controller, inbox, name="SUP-1_AR15_001.dxd", ammo="M855"):
    _touch(inbox, name)
    controller.ingest(inbox, validate=False)
    shot = controller.unmarked_shots()[0]
    controller.mark(
        shot.id,
        ammo=ammo,
        channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
    )
    return shot.id


def test_rename_batch_relabels_without_moving_shots(controller, inbox):
    shot_id = _ingest_and_mark_one(controller, inbox)
    batch = controller.batches()[0]

    controller.rename_batch(batch.id, "SUP-2")

    assert controller.get_batch(batch.id).sku == "SUP-2"
    # The shot stays in the same batch; only the label changed.
    tree = controller.batch_tree()
    assert len(tree) == 1
    assert tree[0].batch.sku == "SUP-2"
    assert tree[0].groups[0].shots[0].id == shot_id


def test_rename_open_batch_onto_another_open_sku_raises(controller, inbox):
    _ingest_and_mark_one(controller, inbox, "SUP-1_AR15_001.dxd")
    _ingest_and_mark_one(controller, inbox, "SUP-2_AR15_001.dxd")
    first, second = controller.batches()

    with pytest.raises(ValueError, match="already has an open batch"):
        controller.rename_batch(first.id, second.sku)


def test_rename_batch_to_same_sku_is_allowed(controller, inbox):
    _ingest_and_mark_one(controller, inbox)
    batch = controller.batches()[0]
    controller.rename_batch(batch.id, batch.sku)  # no-op, must not raise
    assert controller.get_batch(batch.id).sku == batch.sku


def test_rename_empty_sku_raises(controller, inbox):
    _ingest_and_mark_one(controller, inbox)
    batch = controller.batches()[0]
    with pytest.raises(ValueError, match="cannot be empty"):
        controller.rename_batch(batch.id, "   ")


def test_edit_marked_shot_moves_it_to_a_new_group(controller, inbox):
    # Re-marking a marked shot with a corrected ammo re-clusters it, so the report
    # reflects the fix — the mechanism the GUI's Batches-tab edit relies on.
    shot_id = _ingest_and_mark_one(controller, inbox, ammo="WRONG")
    controller.mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
    )
    batch = controller.batches()[0]
    groups = controller.groups_for_batch(batch.id)
    # The corrected group holds the shot; the emptied wrong-ammo group is dropped.
    by_ammo = {g.ammo: controller.shots_by_group(g.id) for g in groups}
    assert "WRONG" not in by_ammo
    assert [s.id for s in by_ammo["M855"]] == [shot_id]


def test_batch_tree_is_a_pure_read(controller, inbox):
    # batch_tree() must not mutate: a stray empty group survives a load and is
    # only removed by the explicit sweep_empty() maintenance pass.
    _ingest_and_mark_one(controller, inbox, ammo="M855")
    batch = controller.batches()[0]
    with controller._repo() as repo:
        stray = repo.upsert_group(batch.id, "AR15", "STRAY")

    tree = controller.batch_tree()  # read only

    assert stray in {g.group.id for node in tree for g in node.groups}


def test_sweep_empty_removes_pre_existing_empty_groups(controller, inbox):
    # Simulate a shot-less group left behind by an edit before per-re-mark
    # cleanup existed; the refresh path's sweep_empty() drops it.
    shot_id = _ingest_and_mark_one(controller, inbox, ammo="M855")
    batch = controller.batches()[0]
    with controller._repo() as repo:
        stray = repo.upsert_group(batch.id, "AR15", "STRAY")
    assert stray in {g.id for g in controller.groups_for_batch(batch.id)}

    controller.sweep_empty()

    group_ids = {g.id for g in controller.groups_for_batch(batch.id)}
    assert stray not in group_ids
    assert controller.get_shot(shot_id).group_id in group_ids  # marked shot's group kept


def test_remark_out_of_closed_batch_prunes_the_empty_batch(controller, inbox):
    # A closed batch with its sole shot re-marked into a new open batch must not
    # leave the emptied closed batch behind as a shell.
    shot_id = _ingest_and_mark_one(controller, inbox, ammo="M855")
    closed_batch = controller.batches()[0]
    controller.close_batch(closed_batch.id)

    # Re-mark the shot: a closed batch is never the SKU's open batch, so this
    # re-clusters into a new open batch and empties the closed one.
    controller.mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
    )

    assert controller.get_batch(closed_batch.id) is None
    batches = controller.batches()
    assert len(batches) == 1 and not batches[0].closed


def test_get_shot_returns_marked_shot(controller, inbox):
    shot_id = _ingest_and_mark_one(controller, inbox)
    shot = controller.get_shot(shot_id)
    assert shot is not None and shot.marked is True
    assert controller.get_shot(999) is None
