"""Round-trip and aggregation tests for the WorkflowRepository hierarchy."""

from __future__ import annotations

import sqlite3

import pytest

from sound_metric_app.dsp.metrics import pa_to_db
from sound_metric_app.models import MetricResult, MicPosition
from sound_metric_app.storage import ResultsDatabase, WorkflowRepository

#: Every stored metric column (linear magnitude + its derived dB level).
_METRIC_KEYS_ALL = (
    "peak_pa", "peak_db", "peak_a_pa", "peak_dba",
    "impulse_pa_ms", "peak_impulse_db",
    "leq10ms_pa", "leq10ms_db", "liaeq_pa", "liaeq_100ms_db",
)


def _metric(pa: float, channel: str = "AI 1") -> MetricResult:
    """A MetricResult whose every linear magnitude equals ``pa`` (easy to average).

    dB fields are the matching ``pa_to_db(pa)``, so a group's linear-then-dB
    average of identical shots is ``pa`` (linear) and ``pa_to_db(pa)`` (dB).
    """
    db = pa_to_db(pa)
    return MetricResult(
        peak_pa=pa, peak_db=db,
        peak_a_pa=pa, peak_dba=db,
        impulse_pa_ms=pa, peak_impulse_db=db,
        leq10ms_pa=pa, leq10ms_db=db,
        liaeq_pa=pa, liaeq_100ms_db=db,
        source_file="f.dxd",
        channel=channel,
        sample_rate=200_000.0,
        n_samples=42_000,
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


def test_migrate_adds_captured_at_to_legacy_shots_table(tmp_path):
    # A database created before captured_at existed: a shots table without the
    # column. Opening the repo must add it (idempotently) so marking can write it.
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        # The shots schema as it stood before captured_at was added.
        conn.execute(
            """
            CREATE TABLE shots (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file       TEXT NOT NULL UNIQUE,
                suppressor_sku    TEXT,
                test_platform     TEXT,
                ammo              TEXT,
                shot_order        INTEGER,
                wind_speed        REAL,
                temp              REAL,
                relative_humidity REAL,
                se_channel        TEXT,
                mr_channel        TEXT,
                marked            INTEGER NOT NULL DEFAULT 0,
                group_id          INTEGER,
                created_at        TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute("INSERT INTO shots (source_file) VALUES ('SUP-1_AR15_001.dxd')")

    with WorkflowRepository(db_path) as repo:
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(shots)")}
        assert "captured_at" in cols
        # The pre-existing row survived and reads back with captured_at = None.
        shot = repo.get_shot_by_source("SUP-1_AR15_001.dxd")
        assert shot is not None and shot.captured_at is None

    # Re-opening runs _migrate again; adding the column a second time is a no-op.
    with WorkflowRepository(db_path) as repo:
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(shots)")}
        assert "captured_at" in cols


def test_migrate_v2_blanks_pre_alignment_metric_columns(tmp_path):
    # The metrics were realigned to TBAC's onset-anchored definitions and now
    # store a linear magnitude per metric. Rows written before the realignment
    # can't be converted, so opening the repo blanks every metric column.
    db_path = tmp_path / "stale.db"
    with WorkflowRepository(db_path) as repo:
        batch_id = repo.create_batch("SUP-1")
        group_id = repo.upsert_group(batch_id, "AR15", "M855")
        shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", shot_order=1)
        repo.mark_shot(shot_id, group_id=group_id, ammo="M855", se_channel="AI 1")
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(200.0))
        # Pretend the row predates the realignment.
        repo._conn.execute("DELETE FROM schema_version")
        repo._conn.commit()

    with WorkflowRepository(db_path) as repo:
        se = repo.group_averages(group_id)[MicPosition.SE]
        for key in _METRIC_KEYS_ALL:
            assert se[key] is None
        assert repo._schema_version() == 2


def test_migrate_leaves_current_metric_values_alone(tmp_path):
    # A row written at the current version must survive later re-opens.
    db_path = tmp_path / "current.db"
    with WorkflowRepository(db_path) as repo:
        batch_id = repo.create_batch("SUP-1")
        group_id = repo.upsert_group(batch_id, "AR15", "M855")
        shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", shot_order=1)
        repo.mark_shot(shot_id, group_id=group_id, ammo="M855", se_channel="AI 1")
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(200.0))

    with WorkflowRepository(db_path) as repo:
        se = repo.group_averages(group_id)[MicPosition.SE]
        assert se["peak_pa"] == pytest.approx(200.0)
        assert se["peak_db"] == pytest.approx(pa_to_db(200.0))


def test_migrate_backfills_missing_linear_column(tmp_path):
    # A database whose channel_metrics predates a linear column must gain it on
    # open. Because the group dB is *derived* from the linear magnitude, dropping
    # peak_pa nulls both its average and the peak_db derived from it — while a
    # metric whose linear column survives still averages.
    db_path = tmp_path / "no_peak_pa.db"
    with WorkflowRepository(db_path) as repo:
        batch_id = repo.create_batch("SUP-1")
        group_id = repo.upsert_group(batch_id, "AR15", "M855")
        shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", shot_order=1)
        repo.mark_shot(shot_id, group_id=group_id, ammo="M855", se_channel="AI 1")
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(150.0))
        repo._conn.execute("ALTER TABLE channel_metrics DROP COLUMN peak_pa")
        repo._conn.commit()
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(channel_metrics)")}
        assert "peak_pa" not in cols

    with WorkflowRepository(db_path) as repo:
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(channel_metrics)")}
        assert "peak_pa" in cols
        se = repo.group_averages(group_id)[MicPosition.SE]
        assert se["peak_pa"] is None
        assert se["peak_db"] is None  # derived from the now-null peak_pa
        assert se["peak_dba"] == pytest.approx(pa_to_db(150.0))  # untouched


def test_both_stores_migrate_when_sharing_one_file(tmp_path):
    # The two stores share a .db, so the version marker must be per-store: the
    # first one to connect must not stamp the second out of its own migration.
    db_path = tmp_path / "shared.db"
    with ResultsDatabase(db_path) as db:
        db.add_result(_metric(150.0))
    with WorkflowRepository(db_path) as repo:
        shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", shot_order=1)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(150.0))
        repo._conn.execute("DELETE FROM schema_version")
        repo._conn.commit()

    # WorkflowRepository connects first and blanks its own stale metric columns...
    with WorkflowRepository(db_path) as repo:
        row = repo._conn.execute(
            "SELECT peak_impulse_db, peak_pa FROM channel_metrics"
        ).fetchone()
        assert row["peak_impulse_db"] is None and row["peak_pa"] is None
    # ...which must not stop ResultsDatabase from blanking its own stale row.
    with ResultsDatabase(db_path) as db:
        r0 = db.all_results()[0]
        assert r0["peak_impulse_db"] is None and r0["peak_pa"] is None


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
        captured_at="2026-07-15T09:30:15",
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
    assert shot.captured_at == "2026-07-15T09:30:15"
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
        captured_at="2026-07-15T09:30:15",
    )

    # Re-mark to correct only ammo/group; omit environment + channel tags.
    new_group_id = repo.upsert_group(batch_id, "AR15", "M193")
    repo.mark_shot(shot_id, group_id=new_group_id, ammo="M193")

    shot = repo.get_shot(shot_id)
    assert shot.ammo == "M193" and shot.group_id == new_group_id
    # Previously stored values survive the partial re-mark.
    assert shot.wind_speed == 5.0 and shot.temp == 72.0 and shot.relative_humidity == 40.0
    assert shot.se_channel == "AI 1" and shot.mr_channel == "AI 2"
    assert shot.captured_at == "2026-07-15T09:30:15"
    assert shot.shot_order == 2


def test_remark_shot_replace_optional_blanks_cleared_fields(repo):
    batch_id = repo.create_batch("SUP-1")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_002.dxd", "SUP-1", "AR15", 2)

    repo.mark_shot(
        shot_id,
        group_id=group_id,
        ammo="M855",
        shot_order=2,
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
        se_channel="AI 1",
        mr_channel="AI 2",
        captured_at="2026-07-15T09:30:15",
    )

    # A full-form edit re-mark that clears the optional fields (passes None):
    # replace_optional writes them exactly, so they are blanked, not preserved.
    repo.mark_shot(shot_id, group_id=group_id, ammo="M855", replace_optional=True)

    shot = repo.get_shot(shot_id)
    assert shot.shot_order is None
    assert shot.wind_speed is None and shot.temp is None and shot.relative_humidity is None
    # Channels and captured_at are not governed by replace_optional; they persist.
    assert shot.se_channel == "AI 1" and shot.mr_channel == "AI 2"
    assert shot.captured_at == "2026-07-15T09:30:15"


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


def test_delete_group_if_empty_removes_only_empty_groups(repo):
    batch_id = repo.create_batch("SUP-1")
    empty = repo.upsert_group(batch_id, "AR15", "M193")
    occupied = repo.upsert_group(batch_id, "AR15", "M855")
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    repo.mark_shot(shot_id, group_id=occupied, ammo="M855")

    assert repo.delete_group_if_empty(empty) is True
    assert repo.get_group(empty) is None
    # The name is now free to be re-created without colliding on the unique key.
    assert repo.upsert_group(batch_id, "AR15", "M193") != empty

    # A group that still holds a shot is left untouched.
    assert repo.delete_group_if_empty(occupied) is False
    assert repo.get_group(occupied) is not None


def test_delete_empty_groups_sweeps_all_shot_less_groups(repo):
    batch_id = repo.create_batch("SUP-1")
    empty_a = repo.upsert_group(batch_id, "AR15", "M193")
    empty_b = repo.upsert_group(batch_id, "MK18", "M855")
    occupied = repo.upsert_group(batch_id, "AR15", "M855")
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    repo.mark_shot(shot_id, group_id=occupied, ammo="M855")

    assert repo.delete_empty_groups() == 2
    assert repo.get_group(empty_a) is None
    assert repo.get_group(empty_b) is None
    assert repo.get_group(occupied) is not None
    # Idempotent: a second sweep with nothing empty removes nothing.
    assert repo.delete_empty_groups() == 0


def test_delete_batch_if_empty_removes_only_group_less_batches(repo):
    empty = repo.create_batch("SUP-1")
    occupied = repo.create_batch("SUP-2")
    repo.upsert_group(occupied, "AR15", "M855")

    assert repo.delete_batch_if_empty(empty) is True
    assert repo.get_batch(empty) is None

    # A batch that still holds a group is left untouched.
    assert repo.delete_batch_if_empty(occupied) is False
    assert repo.get_batch(occupied) is not None


def test_delete_empty_batches_sweeps_all_group_less_batches(repo):
    empty_a = repo.create_batch("SUP-1")
    empty_b = repo.create_batch("SUP-2")
    occupied = repo.create_batch("SUP-3")
    repo.upsert_group(occupied, "AR15", "M855")

    assert repo.delete_empty_batches() == 2
    assert repo.get_batch(empty_a) is None
    assert repo.get_batch(empty_b) is None
    assert repo.get_batch(occupied) is not None
    # Idempotent: a second sweep with nothing empty removes nothing.
    assert repo.delete_empty_batches() == 0


def test_close_batch(repo):
    batch_id = repo.create_batch("SUP-1")
    repo.close_batch(batch_id)
    assert repo.get_batch(batch_id).closed is True
    assert repo.open_batch_for_sku("SUP-1") is None  # no open batch remains


def test_close_batch_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.close_batch(9999)


def test_mark_shot_unknown_id_raises(repo):
    batch_id = repo.create_batch("SUP-1")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")
    with pytest.raises(LookupError):
        repo.mark_shot(9999, group_id=group_id, ammo="M855")


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
    # Linear-Pa mean (165), then converted to dB — not a mean of the dB values.
    assert averages[MicPosition.SE]["peak_pa"] == pytest.approx(165.0)
    assert averages[MicPosition.SE]["peak_db"] == pytest.approx(pa_to_db(165.0))
    assert averages[MicPosition.MR]["peak_pa"] == pytest.approx(155.0)
    assert averages[MicPosition.MR]["peak_db"] == pytest.approx(pa_to_db(155.0))
    assert averages[MicPosition.SE]["n"] == 2
    # Every metric averages its linear magnitude, then converts once.
    for lin, db in (
        ("peak_a_pa", "peak_dba"),
        ("impulse_pa_ms", "peak_impulse_db"),
        ("leq10ms_pa", "leq10ms_db"),
        ("liaeq_pa", "liaeq_100ms_db"),
    ):
        assert averages[MicPosition.SE][lin] == pytest.approx(165.0)
        assert averages[MicPosition.SE][db] == pytest.approx(pa_to_db(165.0))


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


def test_set_shot_channels_clears_unsupplied_tag(repo):
    batch_id = repo.create_batch("SUP-1")
    group_id = repo.upsert_group(batch_id, "AR15", "M855")
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    repo.mark_shot(shot_id, group_id=group_id, ammo="M855", se_channel="AI 1", mr_channel="AI 2")

    # Unlike mark_shot, a None tag is written through, not preserved.
    repo.set_shot_channels(shot_id, se_channel="AI 1", mr_channel=None)

    shot = repo.get_shot(shot_id)
    assert (shot.se_channel, shot.mr_channel) == ("AI 1", None)


def test_set_shot_channels_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.set_shot_channels(9999, se_channel="AI 1", mr_channel=None)


def test_delete_channel_metrics_except(repo):
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_001.dxd", "SUP-1", "AR15", 1)
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))
    repo.save_channel_metric(shot_id, MicPosition.MR, _metric(150.0))

    repo.delete_channel_metrics_except(shot_id, [MicPosition.SE])
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"SE"}

    # An empty keep set clears every row for the shot.
    repo.delete_channel_metrics_except(shot_id, [])
    assert repo.metrics_for_shot(shot_id) == []
