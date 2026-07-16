"""Unit tests for the Phase B headless workflow services.

Ingestion, marking, clustering, and aggregation are exercised with injected
capture/channel readers and a deterministic DSP double, so none of these tests
need a real ``.dxd`` file.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from sound_metric_app.models import Frame, MetricResult, MicPosition
from sound_metric_app.services import (
    AggregationService,
    ClosedBatchError,
    ClusteringService,
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
    metric, so a shot's SE/MR metrics are whatever value the test configured.
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
    """Deterministic DSP double: every metric equals the frame's peak sample."""

    def process(self, frame: Frame) -> MetricResult:
        v = float(np.max(np.abs(frame.samples)))
        return MetricResult(
            peak_db=v,
            peak_dba=v,
            peak_impulse_db=v,
            liaeq_100ms_db=v,
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
    for name in ("SUP-1_AR15_001.dxd", "SUP-1_AR15_002.dxd", "SUP-2_MK18_001.d7d"):
        _touch(inbox, name)

    svc = IngestionService(repo, reader=lambda p: [])  # validate = no-op

    first = svc.scan(inbox)
    assert first.n_ingested == 3
    assert all(not s.marked for s in first.ingested)
    assert len(repo.unmarked_shots()) == 3

    second = svc.scan(inbox)
    assert second.n_ingested == 0
    assert len(second.already_present) == 3
    assert len(repo.unmarked_shots()) == 3


def test_scan_seeds_provisional_keys_from_filename(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1234_AR15_003.dxd")
    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    shot = report.ingested[0]
    assert (shot.suppressor_sku, shot.test_platform, shot.shot_order) == ("SUP-1234", "AR15", 3)


def test_scan_reports_malformed_names_without_ingesting(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_001.dxd")  # good
    _touch(inbox, "not_enough.dxd")  # 2 fields
    _touch(inbox, "SUP_AR15_notanumber.dxd")  # non-numeric order

    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    assert report.n_ingested == 1
    assert len(report.malformed) == 2
    assert len(repo.unmarked_shots()) == 1


def test_scan_ignores_non_capture_files(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_001.dxd")
    _touch(inbox, "README.txt")
    _touch(inbox, "notes.csv")

    report = IngestionService(repo, reader=lambda p: []).scan(inbox)
    assert report.n_ingested == 1
    assert report.malformed == []  # non-capture files are simply skipped, not flagged


def test_scan_reports_unreadable_files(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_001.dxd")

    def boom(path):
        raise OSError("cannot open")

    report = IngestionService(repo, reader=boom).scan(inbox)
    assert report.n_ingested == 0
    assert len(report.unreadable) == 1
    assert repo.unmarked_shots() == []


def test_scan_validate_false_skips_reader(repo, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox, "SUP-1_AR15_001.dxd")

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


def test_resolve_group_creates_then_reuses_open_batch(repo):
    svc = ClusteringService(repo)
    a = svc.resolve_group("SUP-1", "AR15", "M855")
    b = svc.resolve_group("SUP-1", "AR15", "M855")
    c = svc.resolve_group("SUP-1", "AR15", "M193")  # same batch, new group
    assert a.batch.id == b.batch.id == c.batch.id
    assert a.group_id == b.group_id
    assert c.group_id != a.group_id


def test_closed_batch_starts_a_new_batch_on_next_resolve(repo):
    svc = ClusteringService(repo)
    first = svc.resolve_group("SUP-1", "AR15", "M855")
    svc.close_batch(first.batch.id)

    second = svc.resolve_group("SUP-1", "AR15", "M855")
    assert second.batch.id != first.batch.id
    assert second.batch.closed is False


def test_ensure_open_guards_closed_and_unknown(repo):
    svc = ClusteringService(repo)
    resolved = svc.resolve_group("SUP-1", "AR15", "M855")
    assert svc.ensure_open(resolved.batch.id).id == resolved.batch.id

    svc.close_batch(resolved.batch.id)
    with pytest.raises(ClosedBatchError):
        svc.ensure_open(resolved.batch.id)
    with pytest.raises(LookupError):
        svc.ensure_open(9999)


# --------------------------------------------------------------------------- #
# Task 5 — MarkingService
# --------------------------------------------------------------------------- #


def _marking_service(repo, reader=None):
    reader = reader or FakeCaptureReader()
    clustering = ClusteringService(repo)
    return MarkingService(repo, clustering, reader=reader, processor=PeakProcessor())


def test_mark_produces_se_and_mr_rows_and_leaves_unmarked_list(repo):
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_003.dxd", "SUP-1", "AR15", 3)
    svc = _marking_service(repo)

    result = svc.mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
    )

    assert set(result.metrics) == {MicPosition.SE, MicPosition.MR}
    assert result.batch.sku == "SUP-1" and result.batch.closed is False

    stored = {m["mic_position"] for m in repo.metrics_for_shot(shot_id)}
    assert stored == {"SE", "MR"}

    shot = repo.get_shot(shot_id)
    assert shot.marked is True
    assert shot.ammo == "M855" and shot.group_id == result.group_id
    assert (shot.se_channel, shot.mr_channel) == ("AI 1", "AI 2")
    assert (shot.wind_speed, shot.temp, shot.relative_humidity) == (5.0, 72.0, 40.0)
    assert shot.shot_order == 3  # preserved from ingest
    assert repo.unmarked_shots() == []


def test_mark_single_mic_yields_one_row(repo):
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    svc = _marking_service(repo)
    result = svc.mark(shot_id, ammo="M855", channel_map={"AI 1": MicPosition.SE})
    assert set(result.metrics) == {MicPosition.SE}
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"SE"}


def test_remark_two_mic_shot_as_single_mic_drops_mr(repo):
    # First mark tags both mics, then re-mark drops the MR mic (a noisy channel).
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    svc = _marking_service(repo)
    first = svc.mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
    )
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"SE", "MR"}

    result = svc.mark(shot_id, ammo="M855", channel_map={"AI 1": MicPosition.SE})

    # No stale MR row survives to corrupt aggregation, and the tag is cleared.
    assert set(result.metrics) == {MicPosition.SE}
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"SE"}
    shot = repo.get_shot(shot_id)
    assert (shot.se_channel, shot.mr_channel) == ("AI 1", None)
    assert MicPosition.MR not in repo.group_averages(first.group_id)


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

    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    svc = MarkingService(
        repo, ClusteringService(repo), reader=FakeCaptureReader(), processor=ExplodingProcessor()
    )

    with pytest.raises(RuntimeError):
        svc.mark(
            shot_id,
            ammo="M855",
            channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
        )

    shot = repo.get_shot(shot_id)
    assert shot.marked is False
    assert shot.group_id is None
    assert (shot.se_channel, shot.mr_channel) == (None, None)
    assert repo.metrics_for_shot(shot_id) == []
    assert repo.unmarked_shots() != []


def test_mark_rolls_back_when_a_metric_write_fails(repo):
    # A DB failure after the mark + first metric are written must roll back the
    # whole marking, not leave the shot marked with only one mic's metric.
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
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
            svc.mark(
                shot_id,
                ammo="M855",
                channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
            )
    finally:
        repo.save_channel_metric = real_save

    shot = repo.get_shot(shot_id)
    assert shot.marked is False
    assert shot.group_id is None
    assert repo.metrics_for_shot(shot_id) == []


def test_mark_override_keys(repo):
    # Filename said SUP-1/AR15, but the user corrects both at marking.
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    svc = _marking_service(repo)
    result = svc.mark(
        shot_id,
        ammo="M855",
        channel_map={"AI 1": MicPosition.SE},
        suppressor_sku="SUP-9",
        test_platform="MK18",
    )
    assert result.batch.sku == "SUP-9"
    assert repo.get_group(result.group_id).test_platform == "MK18"


def test_mark_unknown_shot_raises(repo):
    with pytest.raises(LookupError):
        _marking_service(repo).mark(9999, ammo="M855", channel_map={"AI 1": MicPosition.SE})


def test_mark_without_resolvable_keys_raises(repo):
    # Shot ingested with no provisional keys and none supplied at marking.
    shot_id = repo.add_unmarked_shot("mystery.dat")
    with pytest.raises(ValueError):
        _marking_service(repo).mark(shot_id, ammo="M855", channel_map={"AI 1": MicPosition.SE})


def test_mark_bad_channel_name_raises_before_marking(repo):
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    svc = _marking_service(repo)
    with pytest.raises(ValueError):
        svc.mark(shot_id, ammo="M855", channel_map={"AI 9": MicPosition.SE})
    # Shot stays unmarked because tagging failed before any DB write.
    assert repo.get_shot(shot_id).marked is False


def test_mark_empty_channel_map_raises_before_marking(repo):
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    svc = _marking_service(repo)
    with pytest.raises(ValueError):
        svc.mark(shot_id, ammo="M855", channel_map={})
    # An empty map must not silently produce a marked shot with zero metrics.
    assert repo.get_shot(shot_id).marked is False


# --------------------------------------------------------------------------- #
# Task 7 — AggregationService
# --------------------------------------------------------------------------- #


def test_group_and_batch_report_keep_se_mr_separate(repo):
    reader = FakeCaptureReader()
    marking = MarkingService(
        repo, ClusteringService(repo), reader=reader, processor=PeakProcessor()
    )

    # Two shots in one group: SE = 160/170 (avg 165), MR = 150/160 (avg 155).
    group_id = None
    for order, (se, mr) in enumerate([(160.0, 150.0), (170.0, 160.0)], start=1):
        source = f"SUP-1_AR15_00{order}.dxd"
        reader.set(source, [("AI 1", se), ("AI 2", mr)])
        shot_id = repo.add_unmarked_shot(source, "SUP-1", "AR15", order)
        marked = marking.mark(
            shot_id,
            ammo="M855",
            channel_map={"AI 1": MicPosition.SE, "AI 2": MicPosition.MR},
        )
        group_id = marked.group_id
        batch_id = marked.batch.id

    agg = AggregationService(repo)

    group = agg.group_averages(group_id)
    assert group.n_shots == 2
    assert set(group.averages) == {MicPosition.SE, MicPosition.MR}
    assert group.averages[MicPosition.SE]["peak_db"] == pytest.approx(165.0)
    assert group.averages[MicPosition.MR]["peak_db"] == pytest.approx(155.0)
    assert group.averages[MicPosition.SE]["n"] == 2

    report = agg.batch_report(batch_id)
    assert report.batch.id == batch_id
    assert len(report.groups) == 1
    assert report.groups[0].averages[MicPosition.SE]["peak_db"] == pytest.approx(165.0)


def test_batch_report_covers_multiple_groups(repo):
    marking = _marking_service(repo)
    clustering = ClusteringService(repo)
    batch = clustering.open_batch_for("SUP-1")

    # Two groups under one SKU: different ammo.
    for order, ammo in enumerate(["M855", "M193"], start=1):
        source = f"SUP-1_AR15_00{order}.dxd"
        shot_id = repo.add_unmarked_shot(source, "SUP-1", "AR15", order)
        marking.mark(shot_id, ammo=ammo, channel_map={"AI 1": MicPosition.SE})

    report = AggregationService(repo).batch_report(batch.id)
    assert {g.group.ammo for g in report.groups} == {"M855", "M193"}


def test_aggregation_unknown_ids_raise(repo):
    agg = AggregationService(repo)
    with pytest.raises(LookupError):
        agg.group_averages(9999)
    with pytest.raises(LookupError):
        agg.batch_report(9999)
