"""Round-trip, inclusion, and roll-up tests for the WorkflowRepository tree."""

from __future__ import annotations

import sqlite3

import pytest

from sound_metric_app.dsp.metrics import pa_to_db
from sound_metric_app.models import MetricResult, MicPosition, ShotRole
from sound_metric_app.storage import ResultsDatabase, WorkflowRepository

#: Every stored metric column (linear magnitude + its derived dB level).
_METRIC_KEYS_ALL = (
    "peak_pa", "peak_db", "peak_a_pa", "peak_dba",
    "impulse_pa_ms", "peak_impulse_db",
    "leq10ms_pa", "leq10ms_db", "liaeq_pa", "liaeq_100ms_db",
)

#: The current WorkflowRepository schema version, asserted by the migration tests.
_CURRENT_VERSION = 5


def _metric(pa: float, channel: str = "AI 1") -> MetricResult:
    """A MetricResult whose every linear magnitude equals ``pa`` (easy to average).

    dB fields are the matching ``pa_to_db(pa)``, so a slot's linear-then-dB
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


def _placed_shot(
    repo,
    cluster_id: int,
    *,
    order: int,
    ammo: str = "M855",
    name: str | None = None,
    se_channel: str | None = None,
    ml_channel: str | None = None,
    **mark_kwargs,
) -> int:
    """Ingest and mark one shot into ``cluster_id``; return its id.

    Channel tags go through ``set_shot_channels`` (as the marking service does),
    since ``mark_shot`` deliberately does not own them.
    """
    name = name or f"SUP-1_AR15_01_{order:04d}.dxd"
    shot_id = repo.add_unmarked_shot(name, "SUP-1", "AR15", 1, order)
    repo.mark_shot(shot_id, cluster_id=cluster_id, ammo=ammo, shot_order=order, **mark_kwargs)
    if se_channel or ml_channel:
        repo.set_shot_channels(shot_id, se_channel=se_channel, ml_channel=ml_channel)
    return shot_id


@pytest.fixture
def batch(repo):
    """A combination + open batch + one cluster, the common setup for these tests."""
    combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
    batch_id = repo.create_batch(combination_id)
    cluster_id = repo.upsert_cluster(batch_id, 1)
    return combination_id, batch_id, cluster_id


# --- schema & migrations ---------------------------------------------------- #


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
    assert {"combinations", "batches", "clusters", "shots", "channel_metrics"} <= names
    assert db_path.exists()


def test_migrate_v5_rebuilds_the_pre_containment_tree_schema(tmp_path):
    # The old shape nested Group (platform + ammo) under Batch (a SKU). "Batch"
    # now means a test session and a Cluster sits between it and the shot, so old
    # rows carry no session boundary to reshape into: v5 drops and recreates.
    db_path = tmp_path / "pre_v5.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE batches (id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT NOT NULL,"
            " closed INTEGER NOT NULL DEFAULT 0, created_at TEXT, closed_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE groups (id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id INTEGER,"
            " test_platform TEXT, ammo TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE shots (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " source_file TEXT NOT NULL UNIQUE, suppressor_sku TEXT, test_platform TEXT,"
            " ammo TEXT, shot_order INTEGER, wind_speed REAL, temp REAL,"
            " relative_humidity REAL, se_channel TEXT, ml_channel TEXT,"
            " marked INTEGER NOT NULL DEFAULT 0, group_id INTEGER, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE channel_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " shot_id INTEGER, mic_position TEXT, channel TEXT, sample_rate REAL,"
            " n_samples INTEGER, created_at TEXT)"
        )
        conn.execute("INSERT INTO batches (sku) VALUES ('SUP-1')")
        conn.execute("INSERT INTO shots (source_file) VALUES ('SUP-1_AR15_001.dxd')")

    with WorkflowRepository(db_path) as repo:
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(shots)")}
        assert "cluster_id" in cols and "included" in cols
        assert "group_id" not in cols
        # The old rows are gone, not half-migrated into an invented session.
        assert repo.all_batches() == []
        assert repo.unmarked_shots() == []
        tables = {
            r["name"]
            for r in repo._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "groups" not in tables
        assert repo._schema_version() == _CURRENT_VERSION


def test_migrate_v5_leaves_the_flat_results_table_alone(tmp_path):
    # Only this store's tables are rebuilt; the sma-analyze path's flat store
    # shares the file and must survive untouched.
    db_path = tmp_path / "shared_pre_v5.db"
    with ResultsDatabase(db_path) as db:
        db.add_result(_metric(150.0))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE shots (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " source_file TEXT NOT NULL UNIQUE, group_id INTEGER,"
            " marked INTEGER NOT NULL DEFAULT 0, created_at TEXT)"
        )

    with WorkflowRepository(db_path):
        pass
    with ResultsDatabase(db_path) as db:
        assert len(db.all_results()) == 1


def test_migrate_v5_is_a_no_op_on_a_fresh_database(tmp_path):
    # A brand-new database is already the new shape; the rebuild must not fire
    # and wipe rows written before the second open.
    db_path = tmp_path / "fresh_reopen.db"
    with WorkflowRepository(db_path) as repo:
        combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
        batch_id = repo.create_batch(combination_id)
        cluster_id = repo.upsert_cluster(batch_id, 1)
        _placed_shot(repo, cluster_id, order=1)

    with WorkflowRepository(db_path) as repo:
        assert len(repo.all_batches()) == 1
        assert len(repo.shots_for_batch(batch_id)) == 1
        assert repo._schema_version() == _CURRENT_VERSION


def test_migrate_v2_blanks_pre_alignment_metric_columns(tmp_path):
    # The metrics were realigned to TBAC's onset-anchored definitions and now
    # store a linear magnitude per metric. Rows written before the realignment
    # can't be converted, so opening the repo blanks every metric column.
    db_path = tmp_path / "stale.db"
    with WorkflowRepository(db_path) as repo:
        combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
        batch_id = repo.create_batch(combination_id)
        cluster_id = repo.upsert_cluster(batch_id, 1)
        shot_id = _placed_shot(repo, cluster_id, order=0, se_channel="AI 1")
        repo.set_shot_included(shot_id, True)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(200.0))
        # Pretend the row predates the realignment, but keep the v5 stamp so the
        # tree rebuild does not fire and delete the row we are checking.
        repo._set_schema_version(0)
        repo._conn.commit()

    with WorkflowRepository(db_path) as repo:
        repo._set_schema_version(_CURRENT_VERSION)
        se = repo.batch_averages(batch_id)[(MicPosition.SE, ShotRole.FRP)]
        for key in _METRIC_KEYS_ALL:
            assert se[key] is None


def test_migrate_v3_blanks_only_peak_window_columns(tmp_path):
    # PEAK_WINDOW_MS widened 75 ms -> 100 ms, so peak / peak dBA / impulse rows
    # written under the old window are not comparable and must be blanked, while
    # the Leq and LIAeq columns (their own windows) survive untouched.
    db_path = tmp_path / "old_window.db"
    with WorkflowRepository(db_path) as repo:
        combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
        batch_id = repo.create_batch(combination_id)
        cluster_id = repo.upsert_cluster(batch_id, 1)
        shot_id = _placed_shot(repo, cluster_id, order=0, se_channel="AI 1")
        repo.set_shot_included(shot_id, True)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(200.0))
        # Pretend the row was computed under the 75 ms window (v2, pre-widening).
        repo._set_schema_version(2)
        repo._conn.commit()

    with WorkflowRepository(db_path) as repo:
        se = repo.batch_averages(batch_id)[(MicPosition.SE, ShotRole.FRP)]
        for key in ("peak_pa", "peak_db", "peak_a_pa", "peak_dba",
                    "impulse_pa_ms", "peak_impulse_db"):
            assert se[key] is None, key
        for key in ("leq10ms_pa", "leq10ms_db", "liaeq_pa", "liaeq_100ms_db"):
            assert se[key] is not None, key
        assert se["leq10ms_pa"] == pytest.approx(200.0)
        assert se["liaeq_100ms_db"] == pytest.approx(pa_to_db(200.0))
        assert repo._schema_version() == _CURRENT_VERSION


def test_migrate_leaves_current_metric_values_alone(tmp_path):
    # A row written at the current version must survive later re-opens.
    db_path = tmp_path / "current.db"
    with WorkflowRepository(db_path) as repo:
        combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
        batch_id = repo.create_batch(combination_id)
        cluster_id = repo.upsert_cluster(batch_id, 1)
        shot_id = _placed_shot(repo, cluster_id, order=0, se_channel="AI 1")
        repo.set_shot_included(shot_id, True)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(200.0))

    with WorkflowRepository(db_path) as repo:
        se = repo.batch_averages(batch_id)[(MicPosition.SE, ShotRole.FRP)]
        assert se["peak_pa"] == pytest.approx(200.0)
        assert se["peak_db"] == pytest.approx(pa_to_db(200.0))


def test_migrate_backfills_missing_linear_column(tmp_path):
    # A database whose channel_metrics predates a linear column must gain it on
    # open. Because the slot dB is *derived* from the linear magnitude, dropping
    # peak_pa nulls both its average and the peak_db derived from it — while a
    # metric whose linear column survives still averages.
    db_path = tmp_path / "no_peak_pa.db"
    with WorkflowRepository(db_path) as repo:
        combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
        batch_id = repo.create_batch(combination_id)
        cluster_id = repo.upsert_cluster(batch_id, 1)
        shot_id = _placed_shot(repo, cluster_id, order=0, se_channel="AI 1")
        repo.set_shot_included(shot_id, True)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(150.0))
        repo._conn.execute("ALTER TABLE channel_metrics DROP COLUMN peak_pa")
        repo._conn.commit()
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(channel_metrics)")}
        assert "peak_pa" not in cols

    with WorkflowRepository(db_path) as repo:
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(channel_metrics)")}
        assert "peak_pa" in cols
        se = repo.batch_averages(batch_id)[(MicPosition.SE, ShotRole.FRP)]
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
        shot_id = repo.add_unmarked_shot("SUP-1_AR15_01_001.dxd", "SUP-1", "AR15", 1, 1)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(150.0))
        # Rewind *both* stores' stamps so each has a stale row to blank; the
        # point of the test is that neither store's migration starves the other.
        repo._conn.execute("DELETE FROM schema_version")
        repo._conn.commit()

    # WorkflowRepository connects first and blanks its own stale metric columns...
    with WorkflowRepository(db_path) as repo:
        repo._set_schema_version(_CURRENT_VERSION)
        row = repo._conn.execute(
            "SELECT peak_impulse_db, peak_pa FROM channel_metrics"
        ).fetchone()
        assert row["peak_impulse_db"] is None and row["peak_pa"] is None
    # ...which must not stop ResultsDatabase from blanking its own stale row.
    with ResultsDatabase(db_path) as db:
        r0 = db.all_results()[0]
        assert r0["peak_impulse_db"] is None and r0["peak_pa"] is None


def test_results_db_migrate_v3_blanks_only_peak_window_columns(tmp_path):
    # The flat CLI store carries the same peak-window widening as the workflow one.
    db_path = tmp_path / "flat_old_window.db"
    with ResultsDatabase(db_path) as db:
        db.add_result(_metric(150.0))
        db._set_schema_version(2)  # pretend the row predates the 100 ms window
        db._conn.commit()

    with ResultsDatabase(db_path) as db:
        row = db.all_results()[0]
        assert row["peak_pa"] is None and row["impulse_pa_ms"] is None
        assert row["peak_dba"] is None
        assert row["leq10ms_pa"] == pytest.approx(150.0)  # own window, untouched
        assert row["liaeq_pa"] == pytest.approx(150.0)
        assert db._schema_version() == 3


# --- containment round-trip ------------------------------------------------- #


def test_full_tree_round_trip(repo, batch):
    combination_id, batch_id, cluster_id = batch

    shot_id = repo.add_unmarked_shot(
        "SUP-1_AR15_01_003.dxd",
        suppressor_sku="SUP-1",
        test_platform="AR15",
        cluster_index=1,
        shot_order=3,
    )
    repo.mark_shot(
        shot_id,
        cluster_id=cluster_id,
        ammo="M855",
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
        captured_at="2026-07-15T09:30:15",
    )
    repo.set_shot_channels(shot_id, se_channel="AI 2", ml_channel="AI 1")
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0, "AI 2"))
    repo.save_channel_metric(shot_id, MicPosition.ML, _metric(150.0, "AI 1"))

    # Combination / batch / cluster read back.
    combination = repo.get_combination(combination_id)
    assert combination.label == "SUP-1 / AR15 / M855"
    batch_row = repo.get_batch(batch_id)
    assert batch_row.combination_id == combination_id and batch_row.closed is False
    cluster = repo.get_cluster(cluster_id)
    assert cluster.cluster_index == 1 and cluster.batch_id == batch_id

    # Shot read back with marking metadata.
    shot = repo.get_shot(shot_id)
    assert shot.marked is True
    assert shot.ammo == "M855" and shot.cluster_id == cluster_id
    assert shot.wind_speed == 5.0 and shot.temp == 72.0 and shot.relative_humidity == 40.0
    assert shot.se_channel == "AI 2" and shot.ml_channel == "AI 1"
    assert shot.captured_at == "2026-07-15T09:30:15"
    assert shot.shot_order == 3 and shot.cluster_index == 1  # preserved from ingest
    # Marking never brings a shot forward on its own.
    assert shot.included is False

    # Two mic metric rows, one per position.
    metrics = repo.metrics_for_shot(shot_id)
    assert {m["mic_position"] for m in metrics} == {"SE", "ML"}

    # Shot no longer unmarked; it shows up under its cluster and its batch.
    assert repo.unmarked_shots() == []
    assert [s.id for s in repo.shots_by_cluster(cluster_id)] == [shot_id]
    assert [s.id for s in repo.shots_for_batch(batch_id)] == [shot_id]


def test_batch_session_metadata_round_trip(repo, batch):
    _combination_id, batch_id, _cluster_id = batch
    repo.update_batch(
        batch_id,
        label="Morning string",
        session_date="2026-07-22",
        wind_speed=4.0,
        temp=88.0,
        relative_humidity=35.0,
        notes="clear, light crosswind",
    )
    b = repo.get_batch(batch_id)
    assert b.label == "Morning string" and b.session_date == "2026-07-22"
    assert b.wind_speed == 4.0 and b.temp == 88.0 and b.relative_humidity == 35.0
    assert b.notes == "clear, light crosswind"

    # A full-form write: fields left unset are cleared, not preserved.
    repo.update_batch(batch_id, label="Afternoon")
    b = repo.get_batch(batch_id)
    assert b.label == "Afternoon"
    assert b.session_date is None and b.wind_speed is None and b.notes is None


def test_update_batch_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.update_batch(9999, label="nope")


def test_remark_shot_preserves_unsupplied_fields(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_01_002.dxd", "SUP-1", "AR15", 1, 2)
    repo.mark_shot(
        shot_id,
        cluster_id=cluster_id,
        ammo="M855",
        shot_order=2,
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
        captured_at="2026-07-15T09:30:15",
    )
    repo.set_shot_channels(shot_id, se_channel="AI 2", ml_channel="AI 1")

    # Re-mark to correct only the ammo; omit environment and order.
    repo.mark_shot(shot_id, cluster_id=cluster_id, ammo="M193")

    shot = repo.get_shot(shot_id)
    assert shot.ammo == "M193" and shot.cluster_id == cluster_id
    # Previously stored values survive the partial re-mark.
    assert shot.wind_speed == 5.0 and shot.temp == 72.0 and shot.relative_humidity == 40.0
    assert shot.se_channel == "AI 2" and shot.ml_channel == "AI 1"
    assert shot.captured_at == "2026-07-15T09:30:15"
    assert shot.shot_order == 2


def test_remark_shot_replace_optional_blanks_cleared_fields(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    shot_id = _placed_shot(
        repo,
        cluster_id,
        order=2,
        wind_speed=5.0,
        temp=72.0,
        relative_humidity=40.0,
        captured_at="2026-07-15T09:30:15",
    )
    repo.set_shot_channels(shot_id, se_channel="AI 2", ml_channel="AI 1")

    # A full-form edit re-mark that clears the optional fields (passes None):
    # replace_optional writes them exactly, so they are blanked, not preserved.
    repo.mark_shot(shot_id, cluster_id=cluster_id, ammo="M855", replace_optional=True)

    shot = repo.get_shot(shot_id)
    assert shot.shot_order is None
    assert shot.wind_speed is None and shot.temp is None and shot.relative_humidity is None
    # Channels and captured_at are not governed by replace_optional; they persist.
    assert shot.se_channel == "AI 2" and shot.ml_channel == "AI 1"
    assert shot.captured_at == "2026-07-15T09:30:15"


def test_remark_never_changes_inclusion(repo, batch):
    # Marking is about test context; the included flag only moves through an
    # explicit bring-forward, so a re-mark must not silently drop a shot out of
    # (or into) an average.
    _combination_id, _batch_id, cluster_id = batch
    shot_id = _placed_shot(repo, cluster_id, order=1)
    repo.set_shot_included(shot_id, True)

    repo.mark_shot(shot_id, cluster_id=cluster_id, ammo="M193", replace_optional=True)
    assert repo.get_shot(shot_id).included is True


def test_add_unmarked_shot_is_idempotent(repo):
    a = repo.add_unmarked_shot("SUP-1_AR15_01_001.dxd", "SUP-1", "AR15", 1, 1)
    b = repo.add_unmarked_shot("SUP-1_AR15_01_001.dxd", "SUP-1", "AR15", 1, 1)
    assert a == b
    assert len(repo.unmarked_shots()) == 1


def test_upsert_combination_returns_same_id(repo):
    c1 = repo.upsert_combination("SUP-1", "AR15", "M855")
    c2 = repo.upsert_combination("SUP-1", "AR15", "M855")
    c3 = repo.upsert_combination("SUP-1", "AR15", "M193")  # different ammo -> new
    c4 = repo.upsert_combination("SUP-1", "MK18", "M855")  # different platform -> new
    assert c1 == c2
    assert len({c1, c3, c4}) == 3


def test_upsert_cluster_returns_same_id_and_scopes_to_its_batch(repo):
    combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
    batch_a = repo.create_batch(combination_id)
    batch_b = repo.create_batch(combination_id)

    a1 = repo.upsert_cluster(batch_a, 1)
    a1_again = repo.upsert_cluster(batch_a, 1)
    a2 = repo.upsert_cluster(batch_a, 2)
    b1 = repo.upsert_cluster(batch_b, 1)

    assert a1 == a1_again
    assert a2 != a1
    # Cluster 1 of one session is a different string of fire from cluster 1 of
    # the next: the index is scoped to its batch, not global.
    assert b1 != a1


def test_close_batch(repo, batch):
    combination_id, batch_id, _cluster_id = batch
    repo.close_batch(batch_id)
    assert repo.get_batch(batch_id).closed is True
    # No open session remains, so the next shot starts a new one.
    assert repo.open_batch_for_combination(combination_id) is None


def test_close_batch_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.close_batch(9999)


def test_mark_shot_unknown_id_raises(repo, batch):
    _combination_id, _batch_id, cluster_id = batch
    with pytest.raises(LookupError):
        repo.mark_shot(9999, cluster_id=cluster_id, ammo="M855")


# --- inclusion -------------------------------------------------------------- #


def test_set_shot_included_records_and_clears_the_reason(repo, batch):
    _combination_id, _batch_id, cluster_id = batch
    shot_id = _placed_shot(repo, cluster_id, order=1)

    repo.set_shot_included(shot_id, False, exclusion_reason="high winds")
    shot = repo.get_shot(shot_id)
    assert shot.included is False and shot.exclusion_reason == "high winds"

    # A reason for leaving a shot out is meaningless once it is in.
    repo.set_shot_included(shot_id, True)
    shot = repo.get_shot(shot_id)
    assert shot.included is True and shot.exclusion_reason is None


def test_set_shot_included_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.set_shot_included(9999, True)


def test_set_cluster_included_fans_out_over_its_shots(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    for order in (1, 2, 3):
        _placed_shot(repo, cluster_id, order=order)

    assert repo.set_cluster_included(cluster_id, True) == 3
    assert all(s.included for s in repo.shots_by_cluster(cluster_id))

    # The flag still lives on each shot, so individual ones can be dropped back
    # out afterwards — which is how a batch lands on an exact count.
    shots = repo.shots_by_cluster(cluster_id)
    repo.set_shot_included(shots[-1].id, False, exclusion_reason="ambient noise")
    assert [s.included for s in repo.shots_by_cluster(cluster_id)] == [True, True, False]


def test_set_cluster_included_counts_coverage_not_a_delta(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    for order in (1, 2, 3):
        _placed_shot(repo, cluster_id, order=order)

    # The count answers "how many shots does this cluster contribute", so a
    # repeat call reports the same number rather than 0 — the CLI prints it as
    # the cluster's standing contribution, not as work done.
    assert repo.set_cluster_included(cluster_id, True) == 3
    assert repo.set_cluster_included(cluster_id, True) == 3


def test_set_cluster_included_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.set_cluster_included(9999, True)


def test_shots_for_batch_included_only_filters_the_data_bank(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    ids = [_placed_shot(repo, cluster_id, order=o) for o in (1, 2, 3)]
    repo.set_shot_included(ids[0], True)

    # The data bank keeps everything; nothing is deleted for being left out.
    assert len(repo.shots_for_batch(batch_id)) == 3
    assert [s.id for s in repo.shots_for_batch(batch_id, included_only=True)] == [ids[0]]


def test_count_shots_in_batch_spans_clusters(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    second_cluster = repo.upsert_cluster(batch_id, 2)
    for order in (1, 2, 3):
        _placed_shot(repo, cluster_id, order=order)
    excluded = _placed_shot(repo, second_cluster, order=1, name="SUP-1_AR15_02_0001.dxd")
    repo.set_shot_included(excluded, False)

    # The count is the data-bank total, matching shots_for_batch's population:
    # every cluster, inclusion flag irrelevant.
    assert repo.count_shots_in_batch(batch_id) == len(repo.shots_for_batch(batch_id)) == 4


def test_count_shots_in_batch_unknown_id_is_zero(repo):
    assert repo.count_shots_in_batch(9999) == 0


def test_inclusion_counts_by_role(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    ids = [_placed_shot(repo, cluster_id, order=o) for o in (0, 1, 2)]
    for shot_id in ids:
        repo.set_shot_included(shot_id, True)

    counts = repo.inclusion_counts(batch_id)
    # Exactly one FRP by construction (order 0); the rest are regular.
    assert counts == {ShotRole.FRP: 1, ShotRole.REGULAR: 2}


def test_inclusion_counts_reports_zero_for_an_empty_role(repo, batch):
    _combination_id, batch_id, _cluster_id = batch
    assert repo.inclusion_counts(batch_id) == {ShotRole.FRP: 0, ShotRole.REGULAR: 0}


# --- cleanup ---------------------------------------------------------------- #


def test_delete_cluster_if_empty_removes_only_empty_clusters(repo, batch):
    _combination_id, batch_id, occupied = batch
    empty = repo.upsert_cluster(batch_id, 2)
    _placed_shot(repo, occupied, order=1)

    assert repo.delete_cluster_if_empty(empty) is True
    assert repo.get_cluster(empty) is None
    # The index is now free to be re-created without colliding on the unique key.
    assert repo.upsert_cluster(batch_id, 2) != empty

    # A cluster that still holds a shot is left untouched.
    assert repo.delete_cluster_if_empty(occupied) is False
    assert repo.get_cluster(occupied) is not None


def test_delete_empty_clusters_sweeps_all_shot_less_clusters(repo, batch):
    _combination_id, batch_id, occupied = batch
    empty_a = repo.upsert_cluster(batch_id, 2)
    empty_b = repo.upsert_cluster(batch_id, 3)
    _placed_shot(repo, occupied, order=1)

    assert repo.delete_empty_clusters() == 2
    assert repo.get_cluster(empty_a) is None and repo.get_cluster(empty_b) is None
    assert repo.get_cluster(occupied) is not None
    # Idempotent: a second sweep with nothing empty removes nothing.
    assert repo.delete_empty_clusters() == 0


def test_delete_batch_if_empty_removes_only_cluster_less_batches(repo):
    combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
    empty = repo.create_batch(combination_id)
    occupied = repo.create_batch(combination_id)
    repo.upsert_cluster(occupied, 1)

    assert repo.delete_batch_if_empty(empty) is True
    assert repo.get_batch(empty) is None
    assert repo.delete_batch_if_empty(occupied) is False
    assert repo.get_batch(occupied) is not None


def test_delete_empty_batches_sweeps_all_cluster_less_batches(repo):
    combination_id = repo.upsert_combination("SUP-1", "AR15", "M855")
    empty_a = repo.create_batch(combination_id)
    empty_b = repo.create_batch(combination_id)
    occupied = repo.create_batch(combination_id)
    repo.upsert_cluster(occupied, 1)

    assert repo.delete_empty_batches() == 2
    assert repo.get_batch(empty_a) is None and repo.get_batch(empty_b) is None
    assert repo.get_batch(occupied) is not None
    assert repo.delete_empty_batches() == 0


def test_delete_combination_if_empty_removes_only_batch_less_combinations(repo):
    empty = repo.upsert_combination("SUP-1", "AR15", "M193")
    occupied = repo.upsert_combination("SUP-1", "AR15", "M855")
    repo.create_batch(occupied)

    assert repo.delete_combination_if_empty(empty) is True
    assert repo.get_combination(empty) is None
    assert repo.delete_combination_if_empty(occupied) is False
    assert repo.get_combination(occupied) is not None


def test_delete_empty_combinations_sweeps_all_batch_less_combinations(repo):
    empty_a = repo.upsert_combination("SUP-1", "AR15", "M193")
    empty_b = repo.upsert_combination("SUP-2", "MK18", "M855")
    occupied = repo.upsert_combination("SUP-1", "AR15", "M855")
    repo.create_batch(occupied)

    assert repo.delete_empty_combinations() == 2
    assert repo.get_combination(empty_a) is None and repo.get_combination(empty_b) is None
    assert repo.get_combination(occupied) is not None
    assert repo.delete_empty_combinations() == 0


# --- roll-up ---------------------------------------------------------------- #


def test_batch_averages_produce_four_position_x_role_slots(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    # FRP (order 0): SE 150, ML 140. Regulars (orders 1-2): SE 160/170 -> 165,
    # ML 150/160 -> 155. Positions and roles never mixed.
    for order, (se, ml) in enumerate([(150.0, 140.0), (160.0, 150.0), (170.0, 160.0)]):
        shot_id = _placed_shot(repo, cluster_id, order=order)
        repo.set_shot_included(shot_id, True)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(se))
        repo.save_channel_metric(shot_id, MicPosition.ML, _metric(ml))

    averages = repo.batch_averages(batch_id)
    assert set(averages) == {
        (MicPosition.SE, ShotRole.FRP),
        (MicPosition.SE, ShotRole.REGULAR),
        (MicPosition.ML, ShotRole.FRP),
        (MicPosition.ML, ShotRole.REGULAR),
    }
    frp_se = averages[(MicPosition.SE, ShotRole.FRP)]
    assert frp_se["n"] == 1 and frp_se["peak_pa"] == pytest.approx(150.0)
    reg_se = averages[(MicPosition.SE, ShotRole.REGULAR)]
    # Linear-Pa mean (165), then converted to dB — not a mean of the dB values.
    assert reg_se["n"] == 2
    assert reg_se["peak_pa"] == pytest.approx(165.0)
    assert reg_se["peak_db"] == pytest.approx(pa_to_db(165.0))
    reg_ml = averages[(MicPosition.ML, ShotRole.REGULAR)]
    assert reg_ml["peak_pa"] == pytest.approx(155.0)
    assert reg_ml["peak_db"] == pytest.approx(pa_to_db(155.0))
    # Every metric averages its linear magnitude, then converts once.
    for lin, db in (
        ("peak_a_pa", "peak_dba"),
        ("impulse_pa_ms", "peak_impulse_db"),
        ("leq10ms_pa", "leq10ms_db"),
        ("liaeq_pa", "liaeq_100ms_db"),
    ):
        assert reg_se[lin] == pytest.approx(165.0)
        assert reg_se[db] == pytest.approx(pa_to_db(165.0))


def test_batch_averages_only_count_included_shots(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    for order, pa in enumerate([150.0, 160.0, 999.0]):
        shot_id = _placed_shot(repo, cluster_id, order=order)
        # Leave the third shot idle; its wild value must not reach the average.
        repo.set_shot_included(shot_id, order != 2)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(pa))

    reg = repo.batch_averages(batch_id)[(MicPosition.SE, ShotRole.REGULAR)]
    assert reg["n"] == 1 and reg["peak_pa"] == pytest.approx(160.0)


def test_batch_averages_omit_slots_with_nothing_included(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    shot_id = _placed_shot(repo, cluster_id, order=0)
    repo.set_shot_included(shot_id, True)
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))

    # Only the SE FRP slot exists: a single-mic FRP-only batch yields one entry,
    # not four empty ones.
    assert set(repo.batch_averages(batch_id)) == {(MicPosition.SE, ShotRole.FRP)}


def test_batch_averages_skip_shots_with_no_order(repo, batch):
    # An unordered shot has no derivable role, so it cannot be placed in a slot.
    _combination_id, batch_id, cluster_id = batch
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_01_009.dxd", "SUP-1", "AR15", 1, 9)
    repo.mark_shot(shot_id, cluster_id=cluster_id, ammo="M855", replace_optional=True)
    repo.set_shot_included(shot_id, True)
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))

    assert repo.batch_averages(batch_id) == {}
    assert repo.inclusion_counts(batch_id) == {ShotRole.FRP: 0, ShotRole.REGULAR: 0}


def test_batch_averages_span_multiple_clusters(repo, batch):
    # The uneven-cluster case from the directive: a 3-shot cluster contributes two
    # regulars and a 4-shot cluster three, so five regulars come from two
    # clusters and two FRPs (the 0000 of each) come along with them.
    _combination_id, batch_id, cluster_one = batch
    cluster_two = repo.upsert_cluster(batch_id, 2)
    for cluster_id, cluster_no, n in ((cluster_one, 1, 3), (cluster_two, 2, 4)):
        for order in range(n):
            shot_id = repo.add_unmarked_shot(
                f"SUP-1_AR15_{cluster_no:02d}_{order:04d}.dxd", "SUP-1", "AR15", cluster_no, order
            )
            repo.mark_shot(shot_id, cluster_id=cluster_id, ammo="M855", shot_order=order)
            repo.set_shot_included(shot_id, True)
            repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))

    assert repo.inclusion_counts(batch_id) == {ShotRole.FRP: 2, ShotRole.REGULAR: 5}
    averages = repo.batch_averages(batch_id)
    assert averages[(MicPosition.SE, ShotRole.FRP)]["n"] == 2
    assert averages[(MicPosition.SE, ShotRole.REGULAR)]["n"] == 5


def test_shot_metrics_for_batch_drills_into_the_same_slots(repo, batch):
    _combination_id, batch_id, cluster_id = batch
    for order in (0, 1, 2):
        shot_id = _placed_shot(repo, cluster_id, order=order)
        repo.set_shot_included(shot_id, order != 2)
        repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))

    included = repo.shot_metrics_for_batch(batch_id)
    # Keys match the averages exactly, so the drill-down explains the numbers.
    assert set(included) == set(repo.batch_averages(batch_id))
    assert len(included[(MicPosition.SE, ShotRole.REGULAR)]) == 1
    rows = included[(MicPosition.SE, ShotRole.REGULAR)]
    assert rows[0]["cluster_index"] == 1 and rows[0]["included"] is True

    # The data-bank flavour keeps the idle shot alongside the included one.
    everything = repo.shot_metrics_for_batch(batch_id, included_only=False)
    regulars = everything[(MicPosition.SE, ShotRole.REGULAR)]
    assert [r["included"] for r in regulars] == [True, False]


def test_save_channel_metric_returns_updated_row_id_on_upsert(repo):
    """Re-saving a position must return that position's row id, not the last insert's."""
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_01_001.dxd", "SUP-1", "AR15", 1, 1)

    se_id = repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))
    ml_id = repo.save_channel_metric(shot_id, MicPosition.ML, _metric(150.0))
    assert se_id != ml_id

    # Conflict path (UPDATE of the SE row) must return the SE row id again.
    se_id_again = repo.save_channel_metric(shot_id, MicPosition.SE, _metric(165.0))
    assert se_id_again == se_id


def test_set_shot_channels_clears_unsupplied_tag(repo, batch):
    _combination_id, _batch_id, cluster_id = batch
    shot_id = _placed_shot(repo, cluster_id, order=1)
    repo.set_shot_channels(shot_id, se_channel="AI 2", ml_channel="AI 1")

    # Unlike mark_shot, a None tag is written through, not preserved.
    repo.set_shot_channels(shot_id, se_channel="AI 2", ml_channel=None)

    shot = repo.get_shot(shot_id)
    assert (shot.se_channel, shot.ml_channel) == ("AI 2", None)


def test_set_shot_channels_unknown_id_raises(repo):
    with pytest.raises(LookupError):
        repo.set_shot_channels(9999, se_channel="AI 1", ml_channel=None)


def test_delete_channel_metrics_except(repo):
    shot_id = repo.add_unmarked_shot("SUP-1_AR15_01_001.dxd", "SUP-1", "AR15", 1, 1)
    repo.save_channel_metric(shot_id, MicPosition.SE, _metric(160.0))
    repo.save_channel_metric(shot_id, MicPosition.ML, _metric(150.0))

    repo.delete_channel_metrics_except(shot_id, [MicPosition.SE])
    assert {m["mic_position"] for m in repo.metrics_for_shot(shot_id)} == {"SE"}

    # An empty keep set clears every row for the shot.
    repo.delete_channel_metrics_except(shot_id, [])
    assert repo.metrics_for_shot(shot_id) == []
