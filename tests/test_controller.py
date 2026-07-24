"""Tests for the GUI's headless view-model (:class:`WorkflowController`).

The controller carries all the GUI's logic with no Qt dependency, so the full
ingest -> mark -> include -> report flow is exercised here without a live window —
mirroring how ``tests/test_cli.py`` drives the CLI. Fake channel/capture readers
stand in for a real ``.dxd``; ``mark`` still runs the real DSP over sine frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from sound_metric_app.ingestion import ChannelInfo
from sound_metric_app.models import Frame, MicPosition, ShotRole
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


def _ingest_and_mark_one(controller, inbox, name="SUP-1_AR15_01_001.dxd", ammo="M855"):
    _touch(inbox, name)
    controller.ingest(inbox, validate=False)
    shot = next(s for s in controller.unmarked_shots() if s.source_file.endswith(name))
    controller.mark(shot.id, ammo=ammo)
    return shot.id


def _ingest_and_mark_cluster(controller, inbox, cluster: int, n: int, ammo="M855"):
    """Ingest + mark one string of fire of ``n`` shots; return their ids in order."""
    for order in range(n):  # 0-based: each cluster's 0000 is its FRP
        _touch(inbox, f"SUP-1_AR15_{cluster:02d}_{order:04d}.dxd")
    controller.ingest(inbox, validate=False)
    ids = []
    for shot in controller.unmarked_shots():
        controller.mark(shot.id, ammo=ammo)
        ids.append(shot.id)
    return ids


# --- full cycle ------------------------------------------------------------- #


def test_full_cycle_ingest_mark_include_report(controller, inbox):
    _touch(inbox, "SUP-1_AR15_01_0000.dxd")  # Dewesoft's 0-based counter: the FRP
    _touch(inbox, "SUP-1_AR15_01_0001.dxd")

    # Ingest from an explicit folder.
    report = controller.ingest(inbox, validate=False)
    assert report.n_ingested == 2
    assert len(controller.unmarked_shots()) == 2

    # Channel choices for the mark form come from the (fake) reader, and the
    # DAQ convention pre-fills the tagging.
    shots = controller.unmarked_shots()
    assert [c.name for c in controller.channels_for(shots[0].source_file)] == ["AI 1", "AI 2"]
    assert controller.suggested_channel_map(shots[0].source_file) == {
        "AI 1": MicPosition.ML,
        "AI 2": MicPosition.SE,
    }

    for shot in shots:
        marked = controller.mark(shot.id, ammo="M855")
        assert set(marked.metrics) == {MicPosition.SE, MicPosition.ML}

    assert controller.unmarked_shots() == []

    # One combination, one batch. Nothing is averaged until brought forward.
    batches = controller.batches()
    assert len(batches) == 1
    rep = controller.batch_averages(batches[0].id)
    assert rep.n_shots == 2 and rep.n_included == 0
    assert rep.averages == {}

    # Bring the cluster forward: the four slots fill in.
    cluster = controller.data_bank()[0].batches[0].clusters[0]
    assert controller.include_cluster(cluster.cluster.id) == 2
    rep = controller.batch_averages(batches[0].id)
    assert rep.n_included == 2
    assert rep.averages[(MicPosition.ML, ShotRole.FRP)]["n"] == 1
    assert rep.averages[(MicPosition.SE, ShotRole.REGULAR)]["n"] == 1

    # Close the batch; further marks would open a new session.
    controller.close_batch(batches[0].id)
    assert controller.get_batch(batches[0].id).closed is True


def test_data_bank_nests_combination_batch_cluster_shot(controller, inbox):
    _ingest_and_mark_cluster(controller, inbox, 1, 2)

    tree = controller.data_bank()
    assert len(tree) == 1
    combination_node = tree[0]
    assert combination_node.combination.label == "SUP-1 / AR15 / M855"
    assert len(combination_node.batches) == 1
    batch_node = combination_node.batches[0]
    assert batch_node.batch.id == controller.batches()[0].id
    assert batch_node.n_shots == 2
    assert len(batch_node.clusters) == 1
    cluster_node = batch_node.clusters[0]
    # Shot count comes from the materialized shots, no COUNT query.
    assert len(cluster_node.shots) == 2
    assert cluster_node.n_included == 0
    # Progress against the soft targets rides along with the node.
    assert batch_node.status.progress[ShotRole.FRP].target == 3


def test_data_bank_shows_idle_shots_alongside_included_ones(controller, inbox):
    # Nothing is hidden for being left out — the bank is the complete archive.
    ids = _ingest_and_mark_cluster(controller, inbox, 1, 3)
    controller.include_shot(ids[0])

    cluster_node = controller.data_bank()[0].batches[0].clusters[0]
    assert len(cluster_node.shots) == 3
    assert [s.included for s in cluster_node.shots] == [True, False, False]


def test_skus_lists_distinct_sorted_values(controller, inbox):
    # Two SKUs, and SUP-1 appears under two platforms — the list dedupes to the
    # distinct SKU values, sorted, regardless of how many combinations each spans.
    _ingest_and_mark_one(controller, inbox, name="SUP-2_AR15_01_001.dxd")
    _ingest_and_mark_one(controller, inbox, name="SUP-1_AR15_01_001.dxd")
    _ingest_and_mark_one(controller, inbox, name="SUP-1_MK18_01_001.dxd")

    assert controller.skus() == ["SUP-1", "SUP-2"]


def test_skus_is_empty_with_no_combinations(controller):
    assert controller.skus() == []


def test_data_bank_filters_to_one_sku(controller, inbox):
    _ingest_and_mark_one(controller, inbox, name="SUP-1_AR15_01_001.dxd")
    _ingest_and_mark_one(controller, inbox, name="SUP-2_AR15_01_001.dxd")

    # Unfiltered: the whole archive, both SKUs.
    assert {n.combination.sku for n in controller.data_bank()} == {"SUP-1", "SUP-2"}

    # Filtered: only the requested SKU's combinations are built.
    filtered = controller.data_bank(sku="SUP-1")
    assert [n.combination.sku for n in filtered] == ["SUP-1"]

    # A SKU with no combinations yields an empty tree, not an error.
    assert controller.data_bank(sku="SUP-9") == []


# --- ingest ----------------------------------------------------------------- #


def test_ingest_without_folder_raises(controller):
    with pytest.raises(ValueError, match="No input folder"):
        controller.ingest()


def test_ingest_uses_configured_folder(controller, inbox):
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    controller.set_input_folder(inbox)
    report = controller.ingest(validate=False)  # no explicit folder
    assert report.n_ingested == 1


def test_rescan_adds_zero(controller, inbox):
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    assert controller.ingest(inbox, validate=False).n_ingested == 1
    second = controller.ingest(inbox, validate=False)
    assert second.n_ingested == 0
    assert len(second.already_present) == 1


def test_mark_unknown_shot_raises(controller):
    with pytest.raises(LookupError):
        controller.mark(999, ammo="M855")


# --- inclusion -------------------------------------------------------------- #


def test_include_shot_and_status(controller, inbox):
    ids = _ingest_and_mark_cluster(controller, inbox, 1, 3)
    batch_id = controller.batches()[0].id

    controller.include_shot(ids[0])  # the FRP
    controller.include_shot(ids[1])
    status = controller.inclusion_status(batch_id)
    assert status.progress[ShotRole.FRP].included == 1
    assert status.progress[ShotRole.REGULAR].included == 1
    assert status.summary() == "FRP: 1/3   Regular: 1/5"


def test_exclude_records_a_reason_then_including_clears_it(controller, inbox):
    ids = _ingest_and_mark_cluster(controller, inbox, 1, 1)
    controller.include_shot(ids[0], False, reason="ambient noise")
    shot = controller.get_shot(ids[0])
    assert shot.included is False and shot.exclusion_reason == "ambient noise"

    controller.include_shot(ids[0], True)
    shot = controller.get_shot(ids[0])
    assert shot.included is True and shot.exclusion_reason is None


def test_include_cluster_then_drop_shots_to_hit_an_exact_count(controller, inbox):
    # The bring-forward shortcut plus per-shot control: exactly what lands a
    # batch on 5 regulars when clusters come in uneven sizes.
    first = _ingest_and_mark_cluster(controller, inbox, 1, 3)
    second = _ingest_and_mark_cluster(controller, inbox, 2, 4)
    batch_id = controller.batches()[0].id

    clusters = controller.data_bank()[0].batches[0].clusters
    controller.include_cluster(clusters[0].cluster.id)  # 1 FRP + 2 regulars
    controller.include_cluster(clusters[1].cluster.id)  # + 1 FRP + 3 regulars
    assert controller.inclusion_status(batch_id).summary() == "FRP: 2/3   Regular: 5/5"

    # Drop one regular back out; the FRP count is untouched (independent axes).
    controller.include_shot(second[-1], False, reason="high winds")
    status = controller.inclusion_status(batch_id)
    assert status.progress[ShotRole.REGULAR].included == 4
    assert status.progress[ShotRole.FRP].included == 2
    assert len(first) == 3 and len(second) == 4


def test_include_unknown_shot_raises(controller):
    with pytest.raises(LookupError):
        controller.include_shot(999)


# --- batch session metadata ------------------------------------------------- #


def test_update_batch_writes_session_context(controller, inbox):
    _ingest_and_mark_one(controller, inbox)
    batch_id = controller.batches()[0].id

    controller.update_batch(
        batch_id,
        label="  Morning string  ",
        session_date="2026-07-22",
        wind_speed=4.0,
        temp=88.0,
        relative_humidity=35.0,
        notes="  clear  ",
    )
    batch = controller.get_batch(batch_id)
    # Text fields are trimmed on the way in.
    assert batch.label == "Morning string" and batch.notes == "clear"
    assert batch.session_date == "2026-07-22"
    assert (batch.wind_speed, batch.temp, batch.relative_humidity) == (4.0, 88.0, 35.0)

    # Full-form write: a blank field clears the stored value.
    controller.update_batch(batch_id, label="   ")
    batch = controller.get_batch(batch_id)
    assert batch.label is None and batch.session_date is None and batch.notes is None


def test_update_unknown_batch_raises(controller):
    with pytest.raises(LookupError):
        controller.update_batch(42, label="nope")


# --- ammo presets ----------------------------------------------------------- #


def test_ammo_definitions_default_to_builtins(controller):
    # A fresh settings file yields the built-in presets so the mark form is never
    # empty out of the box.
    assert controller.ammo_definitions() == [
        "LC M193 (5.56)",
        "LC M855 (5.56)",
        "Black Hills 77gr OTM (5.56)",
    ]


def test_set_ammo_definitions_persists_and_normalizes(controller):
    stored = controller.set_ammo_definitions(
        ["  LC M855 (5.56) ", "Custom 62gr", "LC M855 (5.56)", "  "]
    )
    # Trimmed, de-duplicated, blanks dropped, order preserved.
    assert stored == ["LC M855 (5.56)", "Custom 62gr"]
    assert controller.ammo_definitions() == ["LC M855 (5.56)", "Custom 62gr"]


def test_ammo_definitions_are_cached_between_reads(controller, monkeypatch):
    # The GUI re-reads presets on every view refresh; only the first read should
    # touch the settings file, and callers can't mutate the cached list.
    from sound_metric_app import config

    controller.ammo_definitions()  # warm the cache

    def _fail() -> list[str]:
        raise AssertionError("get_ammo_definitions re-read after the cache was warm")

    monkeypatch.setattr(config, "get_ammo_definitions", _fail)
    first = controller.ammo_definitions()
    first.append("mutated")
    assert "mutated" not in controller.ammo_definitions()


def test_set_ammo_definitions_refreshes_the_cache(controller):
    controller.ammo_definitions()  # warm the cache with the built-in defaults
    controller.set_ammo_definitions(["Custom 62gr"])
    # A write invalidates the stale cache rather than serving the old list.
    assert controller.ammo_definitions() == ["Custom 62gr"]


def test_malformed_ammo_definitions_are_not_cached(controller, monkeypatch):
    # A ValueError must keep surfacing on every call until the settings are fixed,
    # so a failed read is never cached as a successful one.
    from sound_metric_app import config

    monkeypatch.setattr(
        config,
        "get_ammo_definitions",
        lambda: (_ for _ in ()).throw(ValueError("bad settings")),
    )
    with pytest.raises(ValueError):
        controller.ammo_definitions()
    with pytest.raises(ValueError):
        controller.ammo_definitions()


# --- report ----------------------------------------------------------------- #


def test_report_unknown_batch_raises(controller):
    with pytest.raises(LookupError):
        controller.batch_averages(42)


def test_metric_trace_reads_the_shots_capture(controller, inbox):
    shot_id = _ingest_and_mark_one(controller, inbox)

    trace = controller.metric_trace(shot_id, MicPosition.SE, "peak_db")

    # A full-length curve over the capture, with a peak marker for a peak metric.
    assert trace.t_ms.shape == (20_000,)
    assert trace.values.shape == (20_000,)
    assert trace.peak_index is not None


def test_metric_trace_missing_mic_raises(controller, inbox):
    # Mark only ML, then ask for the SE graph the shot doesn't have.
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    controller.ingest(inbox, validate=False)
    shot = controller.unmarked_shots()[0]
    controller.mark(shot.id, ammo="M855", channel_map={"AI 1": MicPosition.ML})

    with pytest.raises(ValueError, match="Shooter's Ear"):
        controller.metric_trace(shot.id, MicPosition.SE, "peak_db")


def test_metric_trace_unknown_shot_raises(controller):
    with pytest.raises(LookupError):
        controller.metric_trace(999, MicPosition.SE, "peak_db")


# --- re-marking & cleanup --------------------------------------------------- #


def test_edit_marked_shot_moves_it_to_a_new_combination(controller, inbox):
    # Re-marking a marked shot with a corrected ammo re-places it, so the report
    # reflects the fix — the mechanism the GUI's data-bank edit relies on.
    shot_id = _ingest_and_mark_one(controller, inbox, ammo="WRONG")
    controller.mark(shot_id, ammo="M855")

    combinations = controller.combinations()
    # The corrected combination holds the shot; the emptied wrong one is dropped.
    assert [c.ammo for c in combinations] == ["M855"]
    tree = controller.data_bank()
    assert tree[0].batches[0].clusters[0].shots[0].id == shot_id


def test_remark_into_a_new_cluster_moves_the_shot(controller, inbox):
    shot_id = _ingest_and_mark_one(controller, inbox)
    controller.mark(shot_id, ammo="M855", cluster_index=3)

    clusters = controller.data_bank()[0].batches[0].clusters
    assert [c.cluster.cluster_index for c in clusters] == [3]
    assert clusters[0].shots[0].id == shot_id


def test_data_bank_is_a_pure_read(controller, inbox):
    # data_bank() must not mutate: a stray empty cluster survives a load and is
    # only removed by the explicit sweep_empty() maintenance pass.
    _ingest_and_mark_one(controller, inbox)
    batch = controller.batches()[0]
    with controller._repo() as repo:
        stray = repo.upsert_cluster(batch.id, 9)

    tree = controller.data_bank()  # read only

    assert stray in {c.cluster.id for n in tree for b in n.batches for c in b.clusters}


def test_sweep_empty_removes_pre_existing_empty_clusters(controller, inbox):
    # Simulate a shot-less cluster left behind by an edit; the refresh path's
    # sweep_empty() drops it while leaving the populated one alone.
    shot_id = _ingest_and_mark_one(controller, inbox)
    batch = controller.batches()[0]
    with controller._repo() as repo:
        stray = repo.upsert_cluster(batch.id, 9)

    controller.sweep_empty()

    tree = controller.data_bank()
    cluster_ids = {c.cluster.id for n in tree for b in n.batches for c in b.clusters}
    assert stray not in cluster_ids
    assert controller.get_shot(shot_id).cluster_id in cluster_ids


def test_sweep_empty_walks_up_to_combinations(controller, inbox):
    # Pruning a combination's last empty batch leaves the SKU/platform/ammo path
    # an empty shell, which the sweep removes too.
    with controller._repo() as repo:
        combination_id = repo.upsert_combination("SUP-9", "MK18", "M193")
        repo.create_batch(combination_id)

    controller.sweep_empty()

    assert controller.get_combination(combination_id) is None


def test_remark_out_of_closed_batch_prunes_the_empty_batch(controller, inbox):
    # A closed batch with its sole shot re-marked into a new open session must not
    # leave the emptied closed batch behind as a shell.
    shot_id = _ingest_and_mark_one(controller, inbox)
    closed_batch = controller.batches()[0]
    controller.close_batch(closed_batch.id)

    # Re-mark the shot: a closed batch is never the combination's open batch, so
    # this re-places it in a new open session and empties the closed one.
    controller.mark(shot_id, ammo="M855")

    assert controller.get_batch(closed_batch.id) is None
    batches = controller.batches()
    assert len(batches) == 1 and not batches[0].closed


def test_get_shot_returns_marked_shot(controller, inbox):
    shot_id = _ingest_and_mark_one(controller, inbox)
    shot = controller.get_shot(shot_id)
    assert shot is not None and shot.marked is True
    assert controller.get_shot(999) is None
