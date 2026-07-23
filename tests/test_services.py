"""Unit tests for the Phase B headless workflow services.

Ingestion, marking, clustering, inclusion, and aggregation are exercised with
injected capture/channel readers and a deterministic DSP double, so none of these
tests need a real ``.dxd`` file.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import numpy as np
import pytest

from sound_metric_app.dsp.metrics import pa_to_db
from sound_metric_app.models import Frame, MetricResult, MicPosition, ShotRole
from sound_metric_app.services import (
    AggregationService,
    ClosedBatchError,
    ClusteringService,
    InclusionService,
    IngestionService,
    MarkingService,
)
from sound_metric_app.storage import WorkflowRepository


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeCaptureReader:
    """Returns frames per source path; each frame's single sample encodes a value.

    Paired with :class:`PeakProcessor`, the value flows straight through to every
    metric, so a shot's ML/SE metrics are whatever value the test configured.
    Channels are named for the DAQ inputs so auto-tagging applies by default.
    """

    def __init__(self, default=(("AI 1", 1.0), ("AI 2", 1.0))):
        self.default = list(default)
        self.by_path: dict[str, list[tuple[str, float]]] = {}

    def set(self, path: str, channels: list[tuple[str, float]]) -> None:
        self.by_path[path] = channels

    def __call__(self, path: str) -> list[Frame]:
        specs = self.by_path.get(path, self.default)
        return [
            Frame(
                samples=np.array([value], dtype=np.float64),
                sample_rate=200_000.0,
                channel=name,
                source_file=path,
            )
            for name, value in specs
        ]


class PeakProcessor:
    """Deterministic DSP double: every linear magnitude equals the frame's peak.

    dB fields are the matching ``pa_to_db``, mirroring a real MetricResult so the
    store's linear-then-dB averaging is exercised faithfully.
    """

    def process(self, frame: Frame) -> MetricResult:
        v = float(np.max(np.abs(frame.samples)))
        db = pa_to_db(v)
        return MetricResult(
            peak_pa=v, peak_db=db,
            peak_a_pa=v, peak_dba=db,
            impulse_pa_ms=v, peak_impulse_db=db,
            leq10ms_pa=v, leq10ms_db=db,
            liaeq_pa=v, liaeq_100ms_db=db,
            source_file=frame.source_file,
            channel=frame.channel,
            sample_rate=frame.sample_rate,
            n_samples=frame.n_samples,
        )


@pytest.fixture
def repo(tmp_path):
    with WorkflowRepository(tmp_path / "wf.db") as r:
        yield r


# --------------------------------------------------------------------------- #
# Task 4 — IngestionService
# --------------------------------------------------------------------------- #


def _touch(folder, name):
    p = folder / name
    p.write_bytes(b"")
    return p


def test_scan_ingests_new_files_and_rescans_add_zero(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("SUP-1_AR15_01_001.dxd", "SUP-1_AR15_01_002.dxd", "SUP-2_MK18_01_001.d7d"):
        _touch(inbox, name)

    svc = IngestionService(repo, reader=lambda p: [])  # validate = no-op

    first = svc.scan(inbox)
    assert first.n_ingested == 3
    assert all(not s.marked for s in first.ingested)
    # Nothing arrives included: a fresh shot sits idle in the data bank.
    assert all(not s.included for s in first.ingested)
    assert len(repo.unmarked_shots()) == 3

    second = svc.scan(inbox)
    assert second.n_ingested == 0
    assert len(second.already_present) == 3
    assert len(repo.unmarked_shots()) == 3


def test_scan_seeds_provisional_keys_from_filename(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1234_AR15_02_003.dxd")
    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    shot = report.ingested[0]
    assert (shot.suppressor_sku, shot.test_platform) == ("SUP-1234", "AR15")
    # The cluster comes from the filename, so a shot knows its string of fire
    # and its position in it — and therefore its role — from the moment it lands.
    assert (shot.cluster_index, shot.shot_order) == (2, 3)
    assert shot.role is ShotRole.REGULAR


def test_scan_seeds_the_frp_from_shot_order_zero(repo, tmp_path):
    # Dewesoft's export counter starts at zero, so the trailing 0000 is the FRP.
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_01_0000.dxd")
    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    assert report.ingested[0].shot_order == 0
    assert report.ingested[0].role is ShotRole.FRP


def test_scan_reports_malformed_names_without_ingesting(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_01_001.dxd")  # good
    _touch(inbox, "not_enough.dxd")  # 2 fields
    _touch(inbox, "SUP_AR15_001.dxd")  # 3 fields — the old convention
    _touch(inbox, "SUP_AR15_01_notanumber.dxd")  # non-numeric order

    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    assert report.n_ingested == 1
    assert len(report.malformed) == 3
    assert len(repo.unmarked_shots()) == 1


def test_scan_ignores_non_capture_files(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_01_001.dxd")
    _touch(inbox, "README.txt")
    _touch(inbox, "notes.csv")

    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    assert report.n_ingested == 1
    assert report.malformed == []  # non-capture files are simply skipped, not flagged


def test_scan_reports_unreadable_files(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_01_001.dxd")

    def boom(path):
        raise OSError("cannot open")

    report = IngestionService(repo, reader=boom).scan(inbox)
    assert report.n_ingested == 0
    assert len(report.unreadable) == 1
    assert repo.unmarked_shots() == []


def test_scan_validate_false_skips_reader(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_01_001.dxd")

    def boom(path):
        raise AssertionError("reader must not be called when validate=False")

    report = IngestionService(repo, reader=boom).scan(inbox, validate=False)
    assert report.n_ingested == 1


def test_scan_missing_folder_raises(repo, tmp_path):
    with pytest.raises(FileNotFoundError):
        IngestionService(repo).scan(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# Task 6 — ClusteringService & batch lifecycle
# --------------------------------------------------------------------------- #


def test_resolve_placement_creates_then_reuses_the_open_batch(repo):
    svc = ClusteringService(repo)
    a = svc.resolve_placement("SUP-1", "AR15", "M855", 1)
    b = svc.resolve_placement("SUP-1", "AR15", "M855", 1)
    c = svc.resolve_placement("SUP-1", "AR15", "M855", 2)  # same batch, new cluster
    assert a.combination.id == b.combination.id == c.combination.id
    assert a.batch.id == b.batch.id == c.batch.id
    assert a.cluster_id == b.cluster_id
    assert c.cluster_id != a.cluster_id


def test_resolve_placement_separates_combinations(repo):
    svc = ClusteringService(repo)
    base = svc.resolve_placement("SUP-1", "AR15", "M855", 1)
    other_ammo = svc.resolve_placement("SUP-1", "AR15", "M193", 1)
    other_platform = svc.resolve_placement("SUP-1", "MK18", "M855", 1)
    other_sku = svc.resolve_placement("SUP-2", "AR15", "M855", 1)

    ids = {base.combination.id, other_ammo.combination.id,
           other_platform.combination.id, other_sku.combination.id}
    assert len(ids) == 4
    # Each combination gets its own session, so batches never span conditions.
    assert len({base.batch.id, other_ammo.batch.id,
                other_platform.batch.id, other_sku.batch.id}) == 4


def test_resolve_placement_rejects_a_non_positive_cluster(repo):
    with pytest.raises(ValueError, match="1 or greater"):
        ClusteringService(repo).resolve_placement("SUP-1", "AR15", "M855", 0)


def test_closed_batch_starts_a_new_session_on_next_resolve(repo):
    svc = ClusteringService(repo)
    first = svc.resolve_placement("SUP-1", "AR15", "M855", 1)
    svc.close_batch(first.batch.id)

    second = svc.resolve_placement("SUP-1", "AR15", "M855", 1)
    assert second.batch.id != first.batch.id
    assert second.batch.closed is False
    # Same combination, new session — and cluster 1 of the new session is a
    # different string of fire from cluster 1 of the closed one.
    assert second.combination.id == first.combination.id
    assert second.cluster_id != first.cluster_id


def test_ensure_open_guards_closed_and_unknown(repo):
    svc = ClusteringService(repo)
    resolved = svc.resolve_placement("SUP-1", "AR15", "M855", 1)
    assert svc.ensure_open(resolved.batch.id).id == resolved.batch.id

    svc.close_batch(resolved.batch.id)
    with pytest.raises(ClosedBatchError):
        svc.ensure_open(resolved.batch.id)
    with pytest.raises(LookupError):
        svc.ensure_open(9999)


def test_open_batch_raises_if_created_batch_vanishes(repo):
    """A just-created batch disappearing (e.g. concurrent delete) fails loudly."""
    svc = ClusteringService(repo)
    # Simulate the race: no open batch, create succeeds, but the follow-up
    # fetch finds nothing.
    repo.get_batch = lambda batch_id: None
    with pytest.raises(LookupError):
        svc.resolve_placement("SUP-1", "AR15", "M855", 1)


# --------------------------------------------------------------------------- #
# Task 5 — MarkingService
# --------------------------------------------------------------------------- #


def _marking_service(repo, reader=None):
    reader = reader or FakeCaptureReader()
    clustering = ClusteringService(repo)
    return MarkingService(repo, clustering, reader=reader, processor=PeakProcessor())


def _unmarked(repo, name="SUP-1_AR15_01_003.dxd", *, cluster=1, order=3):
    return repo.add_unmarked_shot(name, "SUP-1", "AR15", cluster, order)


def test_mark_places_the_shot_and_produces_both_mic_rows(repo):
    shot_id = _unmarked(repo)
    svc = _marking_service(repo)

    result = svc.mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.ML, "AI 2": MicPosition.SE},
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
    )

    assert set(result.metrics) == {MicPosition.SE, MicPosition.ML}
    assert result.combination.label == "SUP-1 / AR15 / M855"
    assert result.batch.closed is False
    assert result.cluster.cluster_index == 1

    stored = {m["mic_position"] for m in repo.metrics_for_shot(shot_id)}
    assert stored == {"SE", "ML"}

    shot = repo.get_shot(shot_id)
    assert shot.marked is True
    assert shot.ammo == "M855" and shot.cluster_id == result.cluster.id
    assert (shot.ml_channel, shot.se_channel) == ("AI 1", "AI 2")
    assert (shot.wind_speed, shot.temp, shot.relative_humidity) == (5.0, 72.0, 40.0)
    assert shot.shot_order == 3 and shot.cluster_index == 1  # preserved from ingest
    assert repo.unmarked_shots() == []


def test_mark_auto_tags_the_daq_channels_when_no_map_is_given(repo):
    # AI 1 is the muzzle-left transducer and AI 2 the shooter's-ear one, so a
    # conforming capture needs no manual tagging at all.
    shot_id = _unmarked(repo)
    result = _marking_service(repo).mark(shot_id, ammo="M855")

    assert set(result.metrics) == {MicPosition.ML, MicPosition.SE}
    shot = repo.get_shot(shot_id)
    assert (shot.ml_channel, shot.se_channel) == ("AI 1", "AI 2")


def test_explicit_channel_map_overrides_the_auto_tagging(repo):
    # A capture that breaks the convention (or a swapped rig) must still be
    # taggable by hand, and the manual map must win.
    shot_id = _unmarked(repo)
    result = _marking_service(repo).mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.ML},
    )
    assert set(result.metrics) == {MicPosition.ML, MicPosition.SE}
    shot = repo.get_shot(shot_id)
    assert (shot.se_channel, shot.ml_channel) == ("AI 1", "AI 2")


def test_mark_raises_when_auto_tagging_finds_no_daq_channels(repo):
    # No AI 1 / AI 2 to key off and no explicit map: fail loudly rather than
    # guessing which stream is which mic.
    reader = FakeCaptureReader(default=(("Mic A", 1.0), ("Mic B", 1.0)))
    shot_id = _unmarked(repo)
    with pytest.raises(ValueError, match="auto-tag"):
        _marking_service(repo, reader=reader).mark(shot_id, ammo="M855")
    assert repo.get_shot(shot_id).marked is False


def test_mark_leaves_the_shot_idle_in_the_data_bank(repo):
    # Marking is about test context; bringing a shot forward is a separate,
    # explicit action, so a freshly marked shot feeds no average yet.
    shot_id = _unmarked(repo)
    marked = _marking_service(repo).mark(shot_id, ammo="M855")

    assert repo.get_shot(shot_id).included is False
    assert repo.batch_averages(marked.batch.id) == {}


def test_remark_preserves_inclusion(repo):
    shot_id = _unmarked(repo)
    svc = _marking_service(repo)
    svc.mark(shot_id, ammo="M855")
    repo.set_shot_included(shot_id, True)

    svc.mark(shot_id, ammo="M193")
    assert repo.get_shot(shot_id).included is True


def test_mark_records_capture_timestamp_from_frames(repo):
    # The capture reader supplies the file's start-store time on each frame;
    # marking should persist it as the shot's captured_at (ISO-8601).
    fired_at = datetime(2026, 7, 15, 9, 30, 15)

    def reader(path):
        return [
            Frame(
                samples=np.array([1.0]),
                sample_rate=200_000.0,
                channel=name,
                source_file=path,
                timestamp=fired_at,
            )
            for name in ("AI 1", "AI 2")
        ]

    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo, reader=reader)
    svc.mark(shot_id, ammo="M855")

    assert repo.get_shot(shot_id).captured_at == fired_at.isoformat()


def test_mark_leaves_captured_at_none_when_file_has_no_timestamp(repo):
    # The default fake reader builds frames without a timestamp; captured_at
    # stays None rather than raising.
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    _marking_service(repo).mark(shot_id, ammo="M855")
    assert repo.get_shot(shot_id).captured_at is None


def test_mark_single_mic_yields_one_row(repo):
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)
    result = svc.mark(shot_id, ammo="M855", channel_map={"AI 1": MicPosition.ML})
    assert set(result.metrics) == {MicPosition.ML}
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"ML"}


def test_remark_two_mic_shot_as_single_mic_drops_the_other(repo):
    # First mark tags both mics, then re-mark drops the SE mic (a noisy channel).
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)
    first = svc.mark(shot_id, ammo="M855")
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"SE", "ML"}
    repo.set_shot_included(shot_id, True)

    result = svc.mark(shot_id, ammo="M855", channel_map={"AI 1": MicPosition.ML})

    # No stale SE row survives to corrupt aggregation, and the tag is cleared.
    assert set(result.metrics) == {MicPosition.ML}
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"ML"}
    shot = repo.get_shot(shot_id)
    assert (shot.ml_channel, shot.se_channel) == ("AI 1", None)
    slots = repo.batch_averages(first.batch.id)
    assert not any(position is MicPosition.SE for position, _role in slots)


def test_remark_into_a_new_cluster_drops_the_emptied_one(repo):
    # A shot re-marked into a different string of fire leaves its former cluster
    # empty; that cluster must be pruned so the tree does not accrue it.
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", cluster=1, order=1)
    svc = _marking_service(repo)
    first = svc.mark(shot_id, ammo="M855")

    second = svc.mark(shot_id, ammo="M855", cluster_index=2)

    assert second.cluster.id != first.cluster.id
    assert repo.get_cluster(first.cluster.id) is None  # emptied cluster gone
    assert [c.id for c in repo.clusters_for_batch(second.batch.id)] == [second.cluster.id]


def test_remark_keeps_a_cluster_that_still_has_other_shots(repo):
    # Two shots share a cluster; re-marking one out must not delete the cluster
    # the other still lives in.
    keep_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", cluster=1, order=1)
    move_id = _unmarked(repo, "SUP-1_AR15_01_002.dxd", cluster=1, order=2)
    svc = _marking_service(repo)
    shared = svc.mark(keep_id, ammo="M855")
    svc.mark(move_id, ammo="M855")

    svc.mark(move_id, ammo="M855", cluster_index=2)

    assert repo.get_cluster(shared.cluster.id) is not None
    assert repo.count_shots_in_cluster(shared.cluster.id) == 1


def test_remark_within_the_same_cluster_keeps_it(repo):
    # Re-marking that resolves back to the same cluster must never delete it.
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)
    first = svc.mark(shot_id, ammo="M855")

    again = svc.mark(shot_id, ammo="M855", channel_map={"AI 1": MicPosition.ML})

    assert again.cluster.id == first.cluster.id
    assert repo.get_cluster(first.cluster.id) is not None


def test_remark_onto_new_ammo_prunes_the_whole_emptied_branch(repo):
    # Changing ammo changes the combination, so the shot's old cluster, its
    # batch, and the combination itself are all left empty and must be swept.
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)
    first = svc.mark(shot_id, ammo="M855")

    second = svc.mark(shot_id, ammo="M193")

    assert second.combination.id != first.combination.id
    assert repo.get_cluster(first.cluster.id) is None
    assert repo.get_batch(first.batch.id) is None
    assert repo.get_combination(first.combination.id) is None


def test_mark_rolls_back_when_processing_fails_midway(repo):
    # DSP fails on the second mic: the shot must stay unmarked with no metrics,
    # so it re-surfaces for marking rather than becoming a half-marked shot.
    class ExplodingProcessor:
        def __init__(self):
            self.calls = 0

        def process(self, frame):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("DSP blew up")
            return PeakProcessor().process(frame)

    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = MarkingService(
        repo, ClusteringService(repo), reader=FakeCaptureReader(), processor=ExplodingProcessor()
    )

    with pytest.raises(RuntimeError):
        svc.mark(shot_id, ammo="M855")

    shot = repo.get_shot(shot_id)
    assert shot.marked is False
    assert shot.cluster_id is None
    assert (shot.se_channel, shot.ml_channel) == (None, None)
    assert repo.metrics_for_shot(shot_id) == []
    assert repo.unmarked_shots() != []


def test_mark_rolls_back_when_a_metric_write_fails(repo):
    # A DB failure after the mark + first metric are written must roll back the
    # whole marking, not leave the shot marked with only one mic's metric.
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)

    real_save = repo.save_channel_metric
    calls = {"n": 0}

    def flaky_save(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("disk full")
        return real_save(*args, **kwargs)

    repo.save_channel_metric = flaky_save
    try:
        with pytest.raises(sqlite3.OperationalError):
            svc.mark(shot_id, ammo="M855")
    finally:
        repo.save_channel_metric = real_save

    shot = repo.get_shot(shot_id)
    assert shot.marked is False
    assert shot.cluster_id is None
    assert repo.metrics_for_shot(shot_id) == []


def test_mark_override_keys(repo):
    # Filename said SUP-1/AR15/cluster 1, but the user corrects all three.
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", cluster=1, order=1)
    result = _marking_service(repo).mark(
        shot_id,
        ammo="M855",
        suppressor_sku="SUP-9",
        test_platform="MK18",
        cluster_index=4,
    )
    assert result.combination.sku == "SUP-9"
    assert result.combination.platform == "MK18"
    assert result.cluster.cluster_index == 4
    assert repo.get_shot(shot_id).cluster_index == 4


def test_mark_unknown_shot_raises(repo):
    with pytest.raises(LookupError):
        _marking_service(repo).mark(9999, ammo="M855")


def test_mark_without_resolvable_keys_raises(repo):
    # Shot ingested with no provisional keys and none supplied at marking.
    shot_id = repo.add_unmarked_shot("mystery.dat")
    with pytest.raises(ValueError):
        _marking_service(repo).mark(shot_id, ammo="M855")


def test_mark_without_a_cluster_raises(repo):
    # SKU and platform are known but no cluster: there is no string of fire to
    # place the shot in, so refuse rather than inventing one.
    shot_id = repo.add_unmarked_shot("x.dxd", "SUP-1", "AR15", None, 1)
    with pytest.raises(ValueError, match="cluster"):
        _marking_service(repo).mark(shot_id, ammo="M855")


def test_mark_bad_channel_name_raises_before_marking(repo):
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)
    with pytest.raises(ValueError):
        svc.mark(shot_id, ammo="M855", channel_map={"AI 9": MicPosition.SE})
    # Shot stays unmarked because tagging failed before any DB write.
    assert repo.get_shot(shot_id).marked is False


def test_mark_empty_channel_map_raises_before_marking(repo):
    shot_id = _unmarked(repo, "SUP-1_AR15_01_001.dxd", order=1)
    svc = _marking_service(repo)
    with pytest.raises(ValueError):
        svc.mark(shot_id, ammo="M855", channel_map={})
    # An empty map must not silently produce a marked shot with zero metrics.
    assert repo.get_shot(shot_id).marked is False


# --------------------------------------------------------------------------- #
# InclusionService — the data-bank -> batch-average gate
# --------------------------------------------------------------------------- #


def _mark_cluster(repo, svc, cluster: int, n: int, ammo: str = "M855"):
    """Ingest + mark ``n`` shots as one string of fire; return the MarkedShots."""
    out = []
    for order in range(n):  # 0-based: the 0000 of each string is its FRP
        shot_id = repo.add_unmarked_shot(
            f"SUP-1_AR15_{cluster:02d}_{order:04d}.dxd", "SUP-1", "AR15", cluster, order
        )
        out.append(svc.mark(shot_id, ammo=ammo))
    return out


def test_include_shot_toggles_and_records_a_reason(repo):
    svc = _marking_service(repo)
    marked = _mark_cluster(repo, svc, 1, 1)[0]
    inclusion = InclusionService(repo)

    inclusion.include_shot(marked.shot.id)
    assert repo.get_shot(marked.shot.id).included is True

    inclusion.include_shot(marked.shot.id, False, reason="high winds")
    shot = repo.get_shot(marked.shot.id)
    assert shot.included is False and shot.exclusion_reason == "high winds"


def test_include_shot_refuses_an_unordered_shot(repo):
    # No shot order means no derivable role, so the shot could not land in an
    # FRP or regular slot — including it would silently do nothing.
    svc = _marking_service(repo)
    marked = _mark_cluster(repo, svc, 1, 1)[0]
    repo.mark_shot(
        marked.shot.id, cluster_id=marked.cluster.id, ammo="M855", replace_optional=True
    )
    with pytest.raises(ValueError, match="shot order"):
        InclusionService(repo).include_shot(marked.shot.id)


def test_include_shot_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        InclusionService(repo).include_shot(9999)


def test_include_cluster_brings_the_whole_string_forward(repo):
    svc = _marking_service(repo)
    marked = _mark_cluster(repo, svc, 1, 3)
    inclusion = InclusionService(repo)

    assert inclusion.include_cluster(marked[0].cluster.id) == 3
    assert all(repo.get_shot(m.shot.id).included for m in marked)


def test_include_cluster_skips_an_unordered_shot(repo):
    # The bring-forward shortcut must not do what per-shot inclusion refuses:
    # an unordered shot has no role, so every roll-up filters it out. Flagging
    # it included would show it feeding an average it never reaches.
    svc = _marking_service(repo)
    marked = _mark_cluster(repo, svc, 1, 3)
    stray = marked[-1]
    repo.mark_shot(stray.shot.id, cluster_id=stray.cluster.id, ammo="M855", replace_optional=True)
    inclusion = InclusionService(repo)

    assert inclusion.include_cluster(stray.cluster.id) == 2
    assert repo.get_shot(stray.shot.id).included is False
    assert all(repo.get_shot(m.shot.id).included for m in marked[:-1])
    # What the tree shows now matches what the average actually counts.
    counts = repo.inclusion_counts(stray.batch.id)
    assert counts[ShotRole.FRP] + counts[ShotRole.REGULAR] == 2

    # Idling still covers the whole cluster, unordered shots and all.
    assert inclusion.include_cluster(stray.cluster.id, False, reason="high winds") == 3
    assert not any(repo.get_shot(m.shot.id).included for m in marked)


def test_include_cluster_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        InclusionService(repo).include_cluster(9999)


def test_status_tracks_progress_against_the_soft_targets(repo):
    svc = _marking_service(repo)
    marked = _mark_cluster(repo, svc, 1, 3)
    batch_id = marked[0].batch.id
    inclusion = InclusionService(repo)

    status = inclusion.status(batch_id)
    assert status.progress[ShotRole.FRP].included == 0
    assert status.progress[ShotRole.FRP].target == 3
    assert status.progress[ShotRole.REGULAR].target == 5
    assert status.complete is False

    inclusion.include_cluster(marked[0].cluster.id)
    status = inclusion.status(batch_id)
    # A 3-shot cluster contributes one FRP and two regulars.
    assert status.progress[ShotRole.FRP].included == 1
    assert status.progress[ShotRole.REGULAR].included == 2
    assert status.progress[ShotRole.FRP].remaining == 2
    assert status.summary() == "FRP: 1/3   Regular: 2/5"


def test_targets_are_soft_not_hard_caps(repo):
    # Nothing refuses an inclusion that overshoots; the status just reports it.
    svc = _marking_service(repo)
    marked = _mark_cluster(repo, svc, 1, 7)
    inclusion = InclusionService(repo)
    inclusion.include_cluster(marked[0].cluster.id)

    progress = inclusion.status(marked[0].batch.id).progress[ShotRole.REGULAR]
    assert progress.included == 6 and progress.target == 5
    assert progress.over is True and progress.met is True and progress.remaining == 0


def test_shot_level_inclusion_lands_on_exact_counts_across_clusters(repo):
    # The directive's case: a 3-shot cluster contributes two regulars and a
    # 4-shot cluster three, so whole clusters cannot sum to exactly 5 —
    # shot-level inclusion is what gets there.
    svc = _marking_service(repo)
    first = _mark_cluster(repo, svc, 1, 3)
    second = _mark_cluster(repo, svc, 2, 4)
    batch_id = first[0].batch.id
    inclusion = InclusionService(repo)

    inclusion.include_cluster(first[0].cluster.id)  # 1 FRP + 2 regulars
    for marked in second:
        if marked.shot.role is ShotRole.REGULAR:
            inclusion.include_shot(marked.shot.id)  # + 3 regulars

    status = inclusion.status(batch_id)
    assert status.progress[ShotRole.REGULAR].included == 5
    assert status.progress[ShotRole.FRP].included == 1


def test_status_unknown_batch_raises(repo):
    with pytest.raises(LookupError):
        InclusionService(repo).status(9999)


# --------------------------------------------------------------------------- #
# Task 7 — AggregationService
# --------------------------------------------------------------------------- #


def test_batch_averages_fill_the_four_slots(repo):
    reader = FakeCaptureReader()
    marking = MarkingService(
        repo, ClusteringService(repo), reader=reader, processor=PeakProcessor()
    )

    # One FRP (150 ML / 140 SE) and two regulars: ML 160/170 -> 165,
    # SE 150/160 -> 155. Positions and roles never mixed.
    batch_id = None
    for order, (ml, se) in enumerate([(150.0, 140.0), (160.0, 150.0), (170.0, 160.0)]):
        source = f"SUP-1_AR15_01_{order:04d}.dxd"
        reader.set(source, [("AI 1", ml), ("AI 2", se)])
        shot_id = repo.add_unmarked_shot(source, "SUP-1", "AR15", 1, order)
        marked = marking.mark(shot_id, ammo="M855")
        repo.set_shot_included(shot_id, True)
        batch_id = marked.batch.id

    report = AggregationService(repo).batch_averages(batch_id)
    assert report.combination.label == "SUP-1 / AR15 / M855"
    assert report.n_shots == 3 and report.n_included == 3
    assert set(report.averages) == {
        (MicPosition.ML, ShotRole.FRP),
        (MicPosition.ML, ShotRole.REGULAR),
        (MicPosition.SE, ShotRole.FRP),
        (MicPosition.SE, ShotRole.REGULAR),
    }
    frp_ml = report.averages[(MicPosition.ML, ShotRole.FRP)]
    assert frp_ml["n"] == 1 and frp_ml["peak_pa"] == pytest.approx(150.0)
    # Linear-Pa mean, then converted once to dB.
    reg_ml = report.averages[(MicPosition.ML, ShotRole.REGULAR)]
    assert reg_ml["peak_pa"] == pytest.approx(165.0)
    assert reg_ml["peak_db"] == pytest.approx(pa_to_db(165.0))
    reg_se = report.averages[(MicPosition.SE, ShotRole.REGULAR)]
    assert reg_se["peak_pa"] == pytest.approx(155.0)
    # The drill-down explains each slot and matches its keys exactly.
    assert set(report.shots) == set(report.averages)
    assert len(report.shots[(MicPosition.ML, ShotRole.REGULAR)]) == 2


def test_batch_averages_ignore_idle_shots(repo):
    reader = FakeCaptureReader()
    marking = MarkingService(
        repo, ClusteringService(repo), reader=reader, processor=PeakProcessor()
    )
    batch_id = None
    for order, ml in enumerate([150.0, 160.0, 999.0]):
        source = f"SUP-1_AR15_01_{order:04d}.dxd"
        reader.set(source, [("AI 1", ml), ("AI 2", ml)])
        shot_id = repo.add_unmarked_shot(source, "SUP-1", "AR15", 1, order)
        marked = marking.mark(shot_id, ammo="M855")
        repo.set_shot_included(shot_id, order != 2)  # leave the wild one idle
        batch_id = marked.batch.id

    report = AggregationService(repo).batch_averages(batch_id)
    # The data-bank total still counts all three; only the average filters.
    assert report.n_shots == 3 and report.n_included == 2
    reg = report.averages[(MicPosition.ML, ShotRole.REGULAR)]
    assert reg["n"] == 1 and reg["peak_pa"] == pytest.approx(160.0)


def test_combination_report_covers_multiple_sessions(repo):
    svc = _marking_service(repo)
    clustering = ClusteringService(repo)

    first = _mark_cluster(repo, svc, 1, 2)[0]
    clustering.close_batch(first.batch.id)
    # A new session under the same combination, after the close.
    second_id = repo.add_unmarked_shot("SUP-1_AR15_02_001.dxd", "SUP-1", "AR15", 2, 1)
    second = svc.mark(second_id, ammo="M855")

    assert second.batch.id != first.batch.id
    report = AggregationService(repo).combination_report(first.combination.id)
    assert report.combination.id == first.combination.id
    assert [b.batch.id for b in report.batches] == [first.batch.id, second.batch.id]


def test_aggregation_unknown_ids_raise(repo):
    agg = AggregationService(repo)
    with pytest.raises(LookupError):
        agg.batch_averages(9999)
    with pytest.raises(LookupError):
        agg.combination_report(9999)
