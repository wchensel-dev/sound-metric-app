"""Round-trip and aggregation tests for the WorkflowRepository hierarchy."""

from __future__ import annotations

import pytest

from sound_metric_app.models import MetricResult, MicPosition
from sound_metric_app.storage import WorkflowRepository


def _metric(peak: float, channel: str = "AI 1") -> MetricResult:
    """A MetricResult whose four metrics all equal ``peak`` (easy to average)."""
    return MetricResult(
        peak_db=peak,
        peak_dba=peak,
        peak_impulse_db=peak,
        liaeq_100ms_db=peak,
        source_file="f.dxd",
        channel=channel,
        sample_rate=200_000.0,
        n_samples=20_000,
    )


@pytest.fixture
def repo(tmp_path):
    with WorkflowRepository(tmp_path / "wf.db") as r:
        yield r


def test_schema_created_fresh(tmp_path):
    db_path = tmp_path / "fresh.db"
    assert not db_path.exists()
    with WorkflowRepository(db_path) as r:
        names = {
            row["name"]
            for row in r._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {"batches", "groups", "shots", "channel_metrics"} <= names
    assert db_path.exists()


def test_batch_group_shot_metrics_round_trip(repo):
    batch_id = repo.create_batch("SUP-1234")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")

    shot_id = repo.add_unmarked_shot(
        "SUP-1234_AR15_003.dxd", suppressor_sku="SUP-1234", test_platform="AR15", shot_order=3
    )
    repo.mark_shot(
        shot_id,
        group_id=group_id,
        ammo="M855",
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
        se_channel="AI 1",
        mr_channel="AI 2",
    )
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0, "AI 1"))
    repo.save_channel_metric(shot_id, MicPosition.MR, _metric(150.0, "AI 2"))

    # Batch / group read back.
    batch = repo.get_batch(batch_id)
    assert batch.sku == "SUP-1234" and batch.closed is False
    group = repo.get_group(group_id)
    assert (group.test_platform, group.ammo) == ("AR15", "M855")

    # Shot read back with marking metadata.
    shot = repo.get_shot(shot_id)
    assert shot.marked is True
    assert shot.ammo == "M855" and shot.group_id == group_id
    assert shot.wind_speed == 5.0 and shot.temp == 72.0 and shot.relative_humidity == 40.0
    assert shot.se_channel == "AI 1" and shot.mr_channel == "AI 2"
    assert shot.shot_order == 3  # preserved from ingest

    # Two mic metric rows, one per position.
    metrics = repo.metrics_for_shot(shot_id)
    assert {m["mic_position"] for m in metrics} == {"SE", "MR"}

    # Shot no longer unmarked; it shows up under its group.
    assert repo.unmarked_shots() == []
    assert [s.id for s in repo.shots_by_group(group_id)] == [shot_id]


def test_remark_shot_preserves_unsupplied_fields(repo):
    batch_id = repo.create_batch("SUP-1")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_002.dxd", "SUP-1", "AR15", 2)

    repo.mark_shot(
        shot_id,
        group_id=group_id,
        ammo="M855",
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
        se_channel="AI 1",
        mr_channel="AI 2",
    )

    # Re-mark to correct only ammo/group; omit environment + channel tags.
    new_group_id = repo.upsert_group(batch_id, "AR15", "M193")
    repo.mark_shot(shot_id, group_id=new_group_id, ammo="M193")

    shot = repo.get_shot(shot_id)
    assert shot.ammo == "M193" and shot.group_id == new_group_id
    # Previously stored values survive the partial re-mark.
    assert shot.wind_speed == 5.0 and shot.temp == 72.0 and shot.relative_humidity == 40.0
    assert shot.se_channel == "AI 1" and shot.mr_channel == "AI 2"
    assert shot.shot_order == 2


def test_add_unmarked_shot_is_idempotent(repo):
    a = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    b = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    assert a == b
    assert len(repo.unmarked_shots()) == 1


def test_upsert_group_returns_same_id(repo):
    batch_id = repo.create_batch("SUP-1")
    g1 = repo.upsert_group(batch_id, "AR15", "M855")
    g2 = repo.upsert_group(batch_id, "AR15", "M855")
    g3 = repo.upsert_group(batch_id, "AR15", "M193")  # different ammo -> new group
    assert g1 == g2
    assert g3 != g1


def test_close_batch(repo):
    batch_id = repo.create_batch("SUP-1")
    repo.close_batch(batch_id)
    assert repo.get_batch(batch_id).closed is True
    assert repo.open_batch_for_sku("SUP-1") is None  # no open batch remains


def test_group_averages_keep_se_and_mr_separate(repo):
    batch_id = repo.create_batch("SUP-1")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")

    # Two shots: SE = 160/170 (avg 165), MR = 150/160 (avg 155). Never mixed.
    for order, (se, mr) in enumerate([(160.0, 150.0), (170.0, 160.0)], start=1):
        shot_id = repo.add_unmarked_shot(f"SUP-1_AR15_00{order}.dxd", "SUP-1", "AR15", order)
        repo.mark_shot(shot_id, group_id=group_id, ammo="M855")
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(se))
        repo.save_channel_metric(shot_id, MicPosition.MR, _metric(mr))

    averages = repo.group_averages(group_id)
    assert set(averages) == {MicPosition.SE, MicPosition.MR}
    assert averages[MicPosition.SE]["peak_db"] == pytest.approx(165.0)
    assert averages[MicPosition.MR]["peak_db"] == pytest.approx(155.0)
    assert averages[MicPosition.SE]["n"] == 2
    # All four metrics averaged the same way in this fixture.
    for field in ("peak_dba", "peak_impulse_db", "liaeq_100ms_db"):
        assert averages[MicPosition.SE][field] == pytest.approx(165.0)


def test_save_channel_metric_returns_updated_row_id_on_upsert(repo):
    """Re-saving a position must return that position's row id, not the last insert's."""
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)

    se_id = repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))
    mr_id = repo.save_channel_metric(shot_id, MicPosition.MR, _metric(150.0))
    assert se_id != mr_id

    # Conflict path (UPDATE of the SE row) must return the SE row id again.
    se_id_again = repo.save_channel_metric(shot_id, MicPosition.SE, _metric(165.0))
    assert se_id_again == se_id


def test_group_averages_single_mic_group(repo):
    """A single-mic test condition yields only that position's average."""
    batch_id = repo.create_batch("SUP-1")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    repo.mark_shot(shot_id, group_id=group_id, ammo="M855", se_channel="AI 1")
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))

    averages = repo.group_averages(group_id)
    assert set(averages) == {MicPosition.SE}
