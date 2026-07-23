"""SQLite persistence for the Combination -> Batch -> Cluster -> Shot -> channel
containment tree.

This is the workflow store described in the README's "Data Model & Workflow".
It lives alongside the legacy flat :class:`~sound_metric_app.storage.database.ResultsDatabase`
(the single-file CLI path) in the same database file, and only ever creates its
own five tables, so the two can share a ``.db`` without interfering.

The tree is pure containment. Two separate concerns ride on top of it:

* **Roles** are derived, not stored: a shot with ``shot_order = 0`` is its
  cluster's FRP and every other shot is regular (see :data:`_ROLE_CASE`), so a
  cluster has exactly one FRP by construction and re-ordering a shot cannot
  leave a stale role behind.
* **Inclusion** (``shots.included``) is what moves a shot from the data bank
  into the batch average. Nothing is ever deleted for being left out.

Standard-library ``sqlite3`` only. Foreign keys are enforced per connection.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from ..config import P_REF
from ..models import (
    FRP_SHOT_ORDER,
    Batch,
    Cluster,
    Combination,
    MetricResult,
    MicPosition,
    Shot,
    ShotRole,
)
from ._base import _SqliteStore
from .database import _METRIC_COLUMNS, _PEAK_WINDOW_COLUMNS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS combinations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sku         TEXT NOT NULL,
    platform    TEXT NOT NULL,
    ammo        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (sku, platform, ammo)
);

CREATE TABLE IF NOT EXISTS batches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    combination_id    INTEGER NOT NULL REFERENCES combinations(id) ON DELETE CASCADE,
    label             TEXT,      -- user's name for the session
    session_date      TEXT,      -- ISO-8601 date the session was fired
    wind_speed        REAL,      -- typical for the session, mph
    temp              REAL,      -- typical for the session, degrees Fahrenheit
    relative_humidity REAL,      -- typical for the session, percent
    notes             TEXT,
    closed            INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at         TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    cluster_index INTEGER NOT NULL,   -- 1-based, from the filename
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (batch_id, cluster_index)
);

CREATE TABLE IF NOT EXISTS shots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file       TEXT NOT NULL UNIQUE,
    suppressor_sku    TEXT,      -- provisional combination key from filename
    test_platform     TEXT,      -- provisional combination key from filename
    ammo              TEXT,      -- set at marking; completes the combination key
    cluster_index     INTEGER,   -- provisional cluster key from filename
    shot_order        INTEGER,   -- position within its cluster; 0 == FRP
    wind_speed        REAL,      -- this shot's specific weather, mph
    temp              REAL,      -- degrees Fahrenheit
    relative_humidity REAL,      -- percent
    se_channel        TEXT,      -- raw channel name tagged SE
    ml_channel        TEXT,      -- raw channel name tagged ML
    marked            INTEGER NOT NULL DEFAULT 0,
    included          INTEGER NOT NULL DEFAULT 0,  -- feeds the batch average
    exclusion_reason  TEXT,      -- why it was left idle
    cluster_id        INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    captured_at       TEXT       -- when the shot was fired (Dewesoft start_store_time), ISO-8601
);

CREATE TABLE IF NOT EXISTS channel_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id         INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    mic_position    TEXT NOT NULL,   -- 'SE' | 'ML'
    channel         TEXT NOT NULL,   -- raw source channel name
    sample_rate     REAL NOT NULL,
    n_samples       INTEGER NOT NULL,
    peak_pa         REAL,
    peak_db         REAL,
    peak_a_pa       REAL,
    peak_dba        REAL,
    impulse_pa_ms   REAL,
    peak_impulse_db REAL,
    leq10ms_pa      REAL,
    leq10ms_db      REAL,
    liaeq_pa        REAL,
    liaeq_100ms_db  REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (shot_id, mic_position)
);
"""

#: SQL that derives a shot's role from its order. Kept as one constant so every
#: roll-up query agrees on the definition (order 0 == FRP, everything else
#: regular) and an unordered shot is excluded rather than silently counted.
_ROLE_CASE = (
    f"CASE WHEN s.shot_order = {FRP_SHOT_ORDER} THEN '{ShotRole.FRP.value}' "
    f"ELSE '{ShotRole.REGULAR.value}' END"
)

#: Every stored metric column (linear magnitude + its dB level), shared with the
#: flat ResultsDatabase so both stores' schemas stay in lockstep.
_METRIC_FIELDS = _METRIC_COLUMNS

#: Linear-magnitude column -> the dB column derived from its batch-average. Batch
#: aggregation averages the linear magnitude, then converts once (MATH.md §9).
_LINEAR_TO_DB = {
    "peak_pa": "peak_db",
    "peak_a_pa": "peak_dba",
    "impulse_pa_ms": "peak_impulse_db",
    "leq10ms_pa": "leq10ms_db",
    "liaeq_pa": "liaeq_100ms_db",
}


def _pa_to_db(magnitude: float | None) -> float | None:
    """dB level of a batch-averaged linear magnitude (Pa or Pa·ms), or None.

    Returns ``None`` when the average is unavailable (no non-NULL rows) and
    ``-inf`` for a non-positive average, so a silent slot never taints the log.
    """
    if magnitude is None:
        return None
    if magnitude <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(magnitude / P_REF)


class WorkflowRepository(_SqliteStore):
    """Data-management wrapper over the containment tree in a local SQLite file."""

    _SCHEMA = _SCHEMA
    _PRAGMAS = ("PRAGMA foreign_keys = ON",)

    def _migrate(self) -> None:
        # captured_at was added after the shots table shipped; back-fill the
        # column on databases created before it existed.
        self._add_column_if_missing("shots", "captured_at", "TEXT")

        # The linear-magnitude / new-metric columns were added after
        # channel_metrics first shipped; back-fill them on older databases.
        for column in _METRIC_COLUMNS:
            self._add_column_if_missing("channel_metrics", column, "REAL")

        if self._schema_version() < 1:
            # peak_impulse_db used to be a plain Impulse level [dB]; a later
            # revision made it dB*ms. Older rows cannot be converted, so blank them.
            self._conn.execute("UPDATE channel_metrics SET peak_impulse_db = NULL")
            self._set_schema_version(1)
        if self._schema_version() < 2:
            # Metrics were realigned to TBAC's onset-anchored definitions and now
            # store a linear magnitude per metric (MATH.md §6/§7/§9). Old rows hold
            # values under the previous whole-frame definitions with no linear
            # companion, so blank every metric column; re-marking a shot
            # re-processes the capture and repopulates them.
            cols = ", ".join(f"{c} = NULL" for c in _METRIC_COLUMNS)
            self._conn.execute(f"UPDATE channel_metrics SET {cols}")
            self._set_schema_version(2)
        if self._schema_version() < 3:
            # PEAK_WINDOW_MS widened from 75 ms to 100 ms (MATH.md §2.8), so peak,
            # peak dBA and the impulse integral are no longer comparable with rows
            # computed under the old window — averaging the two together would
            # quietly mix definitions. Blank them; re-marking a shot re-processes
            # the capture and repopulates them. The Leq/LIAeq columns have their
            # own windows and are left alone.
            cols = ", ".join(f"{c} = NULL" for c in _PEAK_WINDOW_COLUMNS)
            self._conn.execute(f"UPDATE channel_metrics SET {cols}")
            self._set_schema_version(3)
        if self._schema_version() < 4:
            # The muzzle mic moved from the right of the barrel to the left, so
            # the MR position became ML. The measurements themselves are still
            # valid, so rename in place rather than blanking: the old column
            # keeps its channel tags and the old rows keep their metrics.
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(shots)")}
            if "mr_channel" in columns and "ml_channel" not in columns:
                self._conn.execute("ALTER TABLE shots RENAME COLUMN mr_channel TO ml_channel")
            self._conn.execute(
                "UPDATE channel_metrics SET mic_position = 'ML' WHERE mic_position = 'MR'"
            )
            self._set_schema_version(4)
        if self._schema_version() < 5:
            # The tree was restructured. The old shape nested Group (platform +
            # ammo) under Batch (a SKU); the new one makes SKU + Platform + Ammo a
            # single *combination*, a batch one test *session* beneath it, and
            # inserts a *cluster* (string of fire) between batch and shot.
            #
            # That is a change of meaning, not of column names: an old "batch" is
            # not a new batch, and there is no cluster to infer, so old rows
            # cannot be reshaped without inventing session boundaries. The
            # hierarchy tables are therefore dropped and recreated empty and the
            # captures re-ingested. Only this store's tables are touched — the
            # flat ``results`` table the ``sma-analyze`` path writes is untouched.
            #
            # Runs last so the version-guarded blocks above still see their own
            # data: a pre-v5 database passes through them (harmlessly, since its
            # rows are about to be dropped) rather than being skipped by a stamp
            # that had already jumped ahead to 5.
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(shots)")}
            # The schema script has already run, so `shots` always exists: on a
            # fresh database with the new columns (nothing to do), or on an old
            # one still carrying the pre-v5 columns, since CREATE TABLE IF NOT
            # EXISTS left it untouched. `group_id` is the tell-tale for the latter.
            if "group_id" in columns:
                for table in ("channel_metrics", "shots", "groups", "batches"):
                    self._conn.execute(f"DROP TABLE IF EXISTS {table}")
                self._conn.executescript(_SCHEMA)
            self._set_schema_version(5)

    #: True while a :meth:`transaction` block is active, so the individual
    #: mutating methods defer their commit to the enclosing block.
    _in_transaction: bool = False

    # ---- transactions --------------------------------------------------- #

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group multiple writes into one atomic unit.

        While the block is active, the mutating methods skip their per-call
        commit; the block commits once on success or rolls back every pending
        write on any exception. Callers that must not leave a half-applied state
        (e.g. marking a shot and storing its metrics) wrap their writes here.

        Not reentrant — SQLite has a single connection-level transaction.
        """
        if self._in_transaction:
            raise RuntimeError("WorkflowRepository.transaction() is not reentrant")
        self._in_transaction = True
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()
        finally:
            self._in_transaction = False

    def _commit(self) -> None:
        """Commit now, unless inside a :meth:`transaction` block (defer to it)."""
        if not self._in_transaction:
            self._conn.commit()

    # ---- combinations --------------------------------------------------- #

    def upsert_combination(self, sku: str, platform: str, ammo: str) -> int:
        """Return the id of the (SKU, platform, ammo) combination, creating it if new."""
        self._conn.execute(
            """
            INSERT INTO combinations (sku, platform, ammo) VALUES (?, ?, ?)
            ON CONFLICT (sku, platform, ammo) DO NOTHING
            """,
            (sku, platform, ammo),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT id FROM combinations WHERE sku = ? AND platform = ? AND ammo = ?",
            (sku, platform, ammo),
        ).fetchone()
        return int(row["id"])

    def get_combination(self, combination_id: int) -> Combination | None:
        row = self._conn.execute(
            "SELECT * FROM combinations WHERE id = ?", (combination_id,)
        ).fetchone()
        return _row_to_combination(row) if row else None

    def find_combination(self, sku: str, platform: str, ammo: str) -> Combination | None:
        """The combination for a test context, or ``None`` if it has never been used."""
        row = self._conn.execute(
            "SELECT * FROM combinations WHERE sku = ? AND platform = ? AND ammo = ?",
            (sku, platform, ammo),
        ).fetchone()
        return _row_to_combination(row) if row else None

    def all_combinations(self) -> list[Combination]:
        """Every combination, in SKU -> platform -> ammo order (the tree's order)."""
        cur = self._conn.execute("SELECT * FROM combinations ORDER BY sku, platform, ammo")
        return [_row_to_combination(r) for r in cur.fetchall()]

    # ---- batches -------------------------------------------------------- #

    def create_batch(
        self,
        combination_id: int,
        *,
        label: str | None = None,
        session_date: str | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        notes: str | None = None,
    ) -> int:
        """Open a new test session under a combination and return its id.

        Every field but the combination is optional: a batch is created the
        moment its first shot is marked, before the user has necessarily filled
        in the session's date, typical weather, or notes. Those are edited later
        through :meth:`update_batch`.
        """
        cur = self._conn.execute(
            """
            INSERT INTO batches
                (combination_id, label, session_date, wind_speed, temp,
                 relative_humidity, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (combination_id, label, session_date, wind_speed, temp, relative_humidity, notes),
        )
        self._commit()
        return int(cur.lastrowid)

    def update_batch(
        self,
        batch_id: int,
        *,
        label: str | None = None,
        session_date: str | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        notes: str | None = None,
    ) -> None:
        """Write a batch's session metadata exactly, blanking any field left unset.

        Unlike :meth:`mark_shot`'s partial-update mode, this is always a
        full-form write: the batch-edit dialog supplies the complete intended
        state, so an omitted (``None``) field means *clear it*, not *keep it*.
        Placement fields (``combination_id``, ``closed``) are untouched — moving
        or closing a batch are separate operations.

        Raises ``LookupError`` if the batch id is unknown.
        """
        cur = self._conn.execute(
            """
            UPDATE batches SET
                label = ?, session_date = ?, wind_speed = ?, temp = ?,
                relative_humidity = ?, notes = ?
            WHERE id = ?
            """,
            (label, session_date, wind_speed, temp, relative_humidity, notes, batch_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No batch with id {batch_id}")
        self._commit()

    def close_batch(self, batch_id: int) -> None:
        """Mark a batch closed. Idempotent. Raises ``LookupError`` if unknown."""
        cur = self._conn.execute(
            "UPDATE batches SET closed = 1, closed_at = datetime('now') WHERE id = ?",
            (batch_id,),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No batch with id {batch_id}")
        self._commit()

    def get_batch(self, batch_id: int) -> Batch | None:
        row = self._conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        return _row_to_batch(row) if row else None

    def open_batch_for_combination(self, combination_id: int) -> Batch | None:
        """Newest non-closed session for a combination, if any (helper for clustering)."""
        row = self._conn.execute(
            "SELECT * FROM batches WHERE combination_id = ? AND closed = 0 "
            "ORDER BY id DESC LIMIT 1",
            (combination_id,),
        ).fetchone()
        return _row_to_batch(row) if row else None

    def batches_for_combination(self, combination_id: int) -> list[Batch]:
        cur = self._conn.execute(
            "SELECT * FROM batches WHERE combination_id = ? ORDER BY id", (combination_id,)
        )
        return [_row_to_batch(r) for r in cur.fetchall()]

    def all_batches(self) -> list[Batch]:
        """Every batch, oldest first (for the CLI/GUI ``list batches`` view)."""
        cur = self._conn.execute("SELECT * FROM batches ORDER BY id")
        return [_row_to_batch(r) for r in cur.fetchall()]

    # ---- clusters ------------------------------------------------------- #

    def upsert_cluster(self, batch_id: int, cluster_index: int) -> int:
        """Return the id of the (batch, cluster index) string of fire, creating it if new."""
        self._conn.execute(
            """
            INSERT INTO clusters (batch_id, cluster_index) VALUES (?, ?)
            ON CONFLICT (batch_id, cluster_index) DO NOTHING
            """,
            (batch_id, cluster_index),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT id FROM clusters WHERE batch_id = ? AND cluster_index = ?",
            (batch_id, cluster_index),
        ).fetchone()
        return int(row["id"])

    def get_cluster(self, cluster_id: int) -> Cluster | None:
        row = self._conn.execute("SELECT * FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
        return _row_to_cluster(row) if row else None

    def clusters_for_batch(self, batch_id: int) -> list[Cluster]:
        """A batch's strings of fire, in firing order."""
        cur = self._conn.execute(
            "SELECT * FROM clusters WHERE batch_id = ? ORDER BY cluster_index, id", (batch_id,)
        )
        return [_row_to_cluster(r) for r in cur.fetchall()]

    def count_shots_in_cluster(self, cluster_id: int) -> int:
        """Number of shots in a cluster, without materializing their rows."""
        cur = self._conn.execute("SELECT COUNT(*) FROM shots WHERE cluster_id = ?", (cluster_id,))
        return int(cur.fetchone()[0])

    # ---- shots ---------------------------------------------------------- #

    def add_unmarked_shot(
        self,
        source_file: str,
        suppressor_sku: str | None = None,
        test_platform: str | None = None,
        cluster_index: int | None = None,
        shot_order: int | None = None,
    ) -> int:
        """Record a capture file as an Unmarked Data Set. Idempotent by source_file.

        Re-adding an already-ingested file returns the existing shot id without
        overwriting it, so re-scanning the input folder never duplicates shots.
        """
        self._conn.execute(
            """
            INSERT INTO shots
                (source_file, suppressor_sku, test_platform, cluster_index, shot_order, marked)
            VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT (source_file) DO NOTHING
            """,
            (source_file, suppressor_sku, test_platform, cluster_index, shot_order),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT id FROM shots WHERE source_file = ?", (source_file,)
        ).fetchone()
        return int(row["id"])

    def get_shot(self, shot_id: int) -> Shot | None:
        row = self._conn.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        return _row_to_shot(row) if row else None

    def get_shot_by_source(self, source_file: str) -> Shot | None:
        row = self._conn.execute(
            "SELECT * FROM shots WHERE source_file = ?", (source_file,)
        ).fetchone()
        return _row_to_shot(row) if row else None

    def unmarked_shots(self) -> list[Shot]:
        cur = self._conn.execute("SELECT * FROM shots WHERE marked = 0 ORDER BY id")
        return [_row_to_shot(r) for r in cur.fetchall()]

    def mark_shot(
        self,
        shot_id: int,
        *,
        cluster_id: int,
        ammo: str,
        cluster_index: int | None = None,
        shot_order: int | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        captured_at: str | None = None,
        replace_optional: bool = False,
    ) -> None:
        """Apply marking metadata, link to a cluster, and flag the shot marked.

        ``cluster_id`` and ``ammo`` are always written. By default every optional
        field (``cluster_index``, ``shot_order``, the environment columns, and
        ``captured_at``) is preserved when left unset, so re-marking to correct
        one field does not clobber the others: a value is only overwritten when
        explicitly supplied. This suits a partial re-mark (e.g. the CLI setting a
        single field).

        Pass ``replace_optional=True`` for a full-form edit, where the caller
        supplies the complete intended state and an unset user field means
        *blank it*: the four user-editable optional fields (``shot_order``,
        ``wind_speed``, ``temp``, ``relative_humidity``) are then written exactly,
        so passing ``None`` clears them. ``cluster_index`` mirrors the resolved
        cluster rather than a user field and is never blanked here; the channel
        and ``captured_at`` columns are likewise unaffected — the service sets
        channels definitively via :meth:`set_shot_channels` and always
        re-supplies ``captured_at`` from the capture file.

        Inclusion is deliberately untouched: marking is about test context, and a
        shot's ``included`` flag only ever moves through an explicit bring-forward
        (:meth:`set_shot_included`), so re-marking never silently changes which
        shots feed an average.

        Raises ``LookupError`` if ``shot_id`` matches no shot.
        """

        # Column names are hard-coded literals, so this f-string carries no
        # injection surface; user values still bind through placeholders.
        def _opt(column: str) -> str:
            return "?" if replace_optional else f"COALESCE(?, {column})"

        cur = self._conn.execute(
            f"""
            UPDATE shots SET
                cluster_id = ?,
                ammo = ?,
                cluster_index = COALESCE(?, cluster_index),
                shot_order = {_opt("shot_order")},
                wind_speed = {_opt("wind_speed")},
                temp = {_opt("temp")},
                relative_humidity = {_opt("relative_humidity")},
                captured_at = COALESCE(?, captured_at),
                marked = 1
            WHERE id = ?
            """,
            (
                cluster_id,
                ammo,
                cluster_index,
                shot_order,
                wind_speed,
                temp,
                relative_humidity,
                captured_at,
                shot_id,
            ),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No shot with id {shot_id}")
        self._commit()

    def set_shot_channels(
        self, shot_id: int, *, se_channel: str | None, ml_channel: str | None
    ) -> None:
        """Set a shot's SE/ML channel tags exactly, clearing either when ``None``.

        Unlike :meth:`mark_shot` — which preserves an unsupplied channel tag —
        this overwrites both columns unconditionally, so re-marking a shot with
        fewer mics drops the tag for the mic that is no longer present.

        Raises ``LookupError`` if ``shot_id`` matches no shot.
        """
        cur = self._conn.execute(
            "UPDATE shots SET se_channel = ?, ml_channel = ? WHERE id = ?",
            (se_channel, ml_channel, shot_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No shot with id {shot_id}")
        self._commit()

    def shots_by_cluster(self, cluster_id: int) -> list[Shot]:
        """Shots in a cluster, in firing order (then id for stability)."""
        cur = self._conn.execute(
            "SELECT * FROM shots WHERE cluster_id = ? ORDER BY shot_order, id",
            (cluster_id,),
        )
        return [_row_to_shot(r) for r in cur.fetchall()]

    def shots_for_batch(self, batch_id: int, *, included_only: bool = False) -> list[Shot]:
        """Every shot in a batch across all its clusters, in firing order.

        The flat read behind the data-bank view. ``included_only`` narrows it to
        the shots brought forward into the batch average; left false it returns
        the complete archive, idle shots included — nothing is ever hidden for
        having been left out.
        """
        clause = " AND s.included = 1" if included_only else ""
        cur = self._conn.execute(
            f"""
            SELECT s.* FROM shots s
            JOIN clusters c ON c.id = s.cluster_id
            WHERE c.batch_id = ?{clause}
            ORDER BY c.cluster_index, c.id, s.shot_order, s.id
            """,
            (batch_id,),
        )
        return [_row_to_shot(r) for r in cur.fetchall()]

    # ---- inclusion ------------------------------------------------------ #

    def set_shot_included(
        self, shot_id: int, included: bool, *, exclusion_reason: str | None = None
    ) -> None:
        """Bring a shot forward into the batch average, or return it to idle.

        The shot is the source of truth for inclusion. ``exclusion_reason`` is
        recorded alongside an *exclusion* (high winds, ambient noise, ...) and is
        cleared whenever a shot is included, since a reason for leaving a shot
        out is meaningless once it is in.

        Raises ``LookupError`` if ``shot_id`` matches no shot.
        """
        cur = self._conn.execute(
            "UPDATE shots SET included = ?, exclusion_reason = ? WHERE id = ?",
            (1 if included else 0, None if included else exclusion_reason, shot_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No shot with id {shot_id}")
        self._commit()

    def set_cluster_included(
        self, cluster_id: int, included: bool, *, exclusion_reason: str | None = None
    ) -> int:
        """Set the inclusion flag on every shot in a cluster; return how many it covered.

        The "bring cluster forward" convenience action. It is pure fan-out over
        :meth:`set_shot_included` — the flag still lives on each shot, so the
        user can afterwards drop individual shots to land on an exact count.

        The count is how many shots the cluster holds, not a delta: shots already
        carrying the flag are counted too, so calling this twice returns the same
        number both times.

        Raises ``LookupError`` if ``cluster_id`` matches no cluster.
        """
        if self.get_cluster(cluster_id) is None:
            raise LookupError(f"No cluster with id {cluster_id}")
        cur = self._conn.execute(
            "UPDATE shots SET included = ?, exclusion_reason = ? WHERE cluster_id = ?",
            (1 if included else 0, None if included else exclusion_reason, cluster_id),
        )
        self._commit()
        return cur.rowcount

    # ---- cleanup -------------------------------------------------------- #

    def delete_empty_clusters(self) -> int:
        """Delete every cluster that holds no shots; return how many were removed."""
        cur = self._conn.execute(
            """
            DELETE FROM clusters
            WHERE NOT EXISTS (SELECT 1 FROM shots WHERE shots.cluster_id = clusters.id)
            """
        )
        self._commit()
        return cur.rowcount

    def delete_cluster_if_empty(self, cluster_id: int) -> bool:
        """Delete a cluster only if it holds no shots; return whether it was deleted.

        Called after re-marking moves a shot out of its former cluster: if that
        leaves the cluster empty, drop the row so the tree does not accrue empty
        clusters and its index is free to be re-used. The ``WHERE NOT EXISTS``
        guard makes this a no-op for a cluster that still has shots, so a stale
        ``cluster_id`` can never orphan live shots.
        """
        cur = self._conn.execute(
            """
            DELETE FROM clusters
            WHERE id = ?
              AND NOT EXISTS (SELECT 1 FROM shots WHERE shots.cluster_id = clusters.id)
            """,
            (cluster_id,),
        )
        self._commit()
        return cur.rowcount > 0

    def delete_empty_batches(self) -> int:
        """Delete every batch that holds no clusters; return how many were removed."""
        cur = self._conn.execute(
            """
            DELETE FROM batches
            WHERE NOT EXISTS (SELECT 1 FROM clusters WHERE clusters.batch_id = batches.id)
            """
        )
        self._commit()
        return cur.rowcount

    def delete_batch_if_empty(self, batch_id: int) -> bool:
        """Delete a batch only if it holds no clusters; return whether it was deleted."""
        cur = self._conn.execute(
            """
            DELETE FROM batches
            WHERE id = ?
              AND NOT EXISTS (SELECT 1 FROM clusters WHERE clusters.batch_id = batches.id)
            """,
            (batch_id,),
        )
        self._commit()
        return cur.rowcount > 0

    def delete_empty_combinations(self) -> int:
        """Delete every combination that holds no batches; return how many were removed.

        The last step of the sweep: pruning a combination's final empty batch
        leaves the SKU/platform/ammo path itself an empty shell, which this
        removes so the tree does not accrete dead combinations over time.
        """
        cur = self._conn.execute(
            """
            DELETE FROM combinations
            WHERE NOT EXISTS (
                SELECT 1 FROM batches WHERE batches.combination_id = combinations.id
            )
            """
        )
        self._commit()
        return cur.rowcount

    def delete_combination_if_empty(self, combination_id: int) -> bool:
        """Delete a combination only if it holds no batches; return whether it was deleted."""
        cur = self._conn.execute(
            """
            DELETE FROM combinations
            WHERE id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM batches WHERE batches.combination_id = combinations.id
              )
            """,
            (combination_id,),
        )
        self._commit()
        return cur.rowcount > 0

    # ---- channel metrics ------------------------------------------------ #

    def save_channel_metric(
        self, shot_id: int, mic_position: MicPosition, result: MetricResult
    ) -> int:
        """Persist one mic's metrics for a shot. Upserts on (shot, position)."""
        row = result.as_row()
        metric_cols = ", ".join(_METRIC_COLUMNS)
        placeholders = ", ".join("?" * len(_METRIC_COLUMNS))
        updates = ", ".join(f"{c} = excluded.{c}" for c in _METRIC_COLUMNS)
        cur = self._conn.execute(
            f"""
            INSERT INTO channel_metrics
                (shot_id, mic_position, channel, sample_rate, n_samples,
                 {metric_cols})
            VALUES (?, ?, ?, ?, ?, {placeholders})
            ON CONFLICT (shot_id, mic_position) DO UPDATE SET
                channel = excluded.channel,
                sample_rate = excluded.sample_rate,
                n_samples = excluded.n_samples,
                {updates}
            RETURNING id
            """,
            (
                shot_id,
                mic_position.value,
                result.channel,
                result.sample_rate,
                result.n_samples,
                *(row[c] for c in _METRIC_COLUMNS),
            ),
        )
        row_id = int(cur.fetchone()[0])
        self._commit()
        return row_id

    def delete_channel_metrics_except(self, shot_id: int, keep: Iterable[MicPosition]) -> None:
        """Delete a shot's ``channel_metrics`` rows whose position is not in ``keep``.

        Called when re-marking drops a previously tagged mic, so aggregation
        does not keep averaging the orphaned row.
        """
        kept = [p.value for p in keep]
        if kept:
            placeholders = ",".join("?" * len(kept))
            self._conn.execute(
                "DELETE FROM channel_metrics "
                f"WHERE shot_id = ? AND mic_position NOT IN ({placeholders})",
                (shot_id, *kept),
            )
        else:
            self._conn.execute("DELETE FROM channel_metrics WHERE shot_id = ?", (shot_id,))
        self._commit()

    def metrics_for_shot(self, shot_id: int) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM channel_metrics WHERE shot_id = ? ORDER BY mic_position",
            (shot_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ---- aggregation ---------------------------------------------------- #

    def batch_averages(self, batch_id: int) -> dict[tuple[MicPosition, ShotRole], dict]:
        """Average each metric over a batch's **included** shots, per position x role.

        This is the batch-average view: the filter where ``included`` is true,
        grouped by mic position crossed with derived role, producing up to four
        output slots per batch —

        .. code-block:: text

           muzzle_left  (ML) . FRP        muzzle_left  (ML) . regular
           shooters_ear (SE) . FRP        shooters_ear (SE) . regular

        Each slot maps to a dict carrying every metric's averaged linear
        magnitude and dB level plus ``n``. Slots with no included shots are
        omitted, so a batch that has only had its FRPs brought forward yields two
        entries rather than four empty ones.

        Averaging is done in the **linear domain** (MATH.md §9): each metric's
        per-shot linear magnitude (Pa, or Pa·ms for the impulse) is meaned, then
        that mean is converted once to its dB level — not a mean of the dB
        values. Positions and roles are never mixed: the 3-FRP / 5-regular target
        applies per position, so each channel averages the same underlying
        selected shots on its own axis.

        Shots with no ``shot_order`` have no derivable role and are excluded — an
        unordered shot cannot be placed in either slot.

        ``n`` is ``COUNT(*)`` — the shot count for the slot — while each ``AVG``
        divides by the count of *non-NULL* values. In normal operation every
        shot's metrics are populated, so the two coincide (MATH.md §10). They
        diverge only in edge states: a partially re-processed batch (some shots
        still blanked by a migration) or a NaN-contaminated metric (stored as
        NULL, hence skipped by ``AVG`` and shown as "—" per shot). In those cases
        a metric's average is a mean over the available shots, not all ``n``.
        """
        linear_avgs = ", ".join(f"AVG(cm.{c}) AS {c}" for c in _LINEAR_TO_DB)
        cur = self._conn.execute(
            f"""
            SELECT cm.mic_position AS pos,
                   {_ROLE_CASE} AS role,
                   {linear_avgs},
                   COUNT(*) AS n
            FROM channel_metrics cm
            JOIN shots s    ON s.id = cm.shot_id
            JOIN clusters c ON c.id = s.cluster_id
            WHERE c.batch_id = ? AND s.included = 1 AND s.shot_order IS NOT NULL
            GROUP BY cm.mic_position, role
            """,
            (batch_id,),
        )
        out: dict[tuple[MicPosition, ShotRole], dict] = {}
        for r in cur.fetchall():
            entry: dict = {"n": int(r["n"])}
            for linear, db in _LINEAR_TO_DB.items():
                entry[linear] = r[linear]
                entry[db] = _pa_to_db(r[linear])
            out[(MicPosition(r["pos"]), ShotRole(r["role"]))] = entry
        return out

    def shot_metrics_for_batch(
        self, batch_id: int, *, included_only: bool = True
    ) -> dict[tuple[MicPosition, ShotRole], list[dict]]:
        """Per-shot metric rows for a batch, keyed by the same position x role slots.

        The un-averaged counterpart to :meth:`batch_averages`: instead of one
        mean per slot, returns every contributing shot's own metrics so a report
        can drill down from a slot's average into the individual shots behind it.
        Each slot maps to a list of ``{shot_id, cluster_index, shot_order,
        included, source_file, <every metric column>}``, in firing order.

        ``included_only`` defaults to true so the drill-down matches the averages
        exactly; pass false for the data-bank view, where idle shots are listed
        alongside the ones brought forward.
        """
        clause = " AND s.included = 1" if included_only else ""
        metric_select = ", ".join(f"cm.{c}" for c in _METRIC_COLUMNS)
        cur = self._conn.execute(
            f"""
            SELECT cm.mic_position   AS pos,
                   {_ROLE_CASE}      AS role,
                   s.id              AS shot_id,
                   c.cluster_index   AS cluster_index,
                   s.shot_order      AS shot_order,
                   s.included        AS included,
                   s.source_file     AS source_file,
                   {metric_select}
            FROM channel_metrics cm
            JOIN shots s    ON s.id = cm.shot_id
            JOIN clusters c ON c.id = s.cluster_id
            WHERE c.batch_id = ? AND s.shot_order IS NOT NULL{clause}
            ORDER BY cm.mic_position, role, c.cluster_index, s.shot_order, s.id
            """,
            (batch_id,),
        )
        out: dict[tuple[MicPosition, ShotRole], list[dict]] = {}
        for r in cur.fetchall():
            key = (MicPosition(r["pos"]), ShotRole(r["role"]))
            out.setdefault(key, []).append(
                {
                    "shot_id": int(r["shot_id"]),
                    "cluster_index": r["cluster_index"],
                    "shot_order": r["shot_order"],
                    "included": bool(r["included"]),
                    "source_file": r["source_file"],
                    **{k: r[k] for k in _METRIC_FIELDS},
                }
            )
        return out

    def inclusion_counts(self, batch_id: int) -> dict[ShotRole, int]:
        """How many shots of each role a batch currently has included.

        Counts *shots*, not channel rows, so a two-mic shot counts once — this is
        what the 3-FRP / 5-regular target is measured against. Roles with nothing
        included report 0 rather than being omitted, so a caller can render
        progress for both without special-casing an empty batch.
        """
        cur = self._conn.execute(
            f"""
            SELECT {_ROLE_CASE} AS role, COUNT(*) AS n
            FROM shots s
            JOIN clusters c ON c.id = s.cluster_id
            WHERE c.batch_id = ? AND s.included = 1 AND s.shot_order IS NOT NULL
            GROUP BY role
            """,
            (batch_id,),
        )
        counts = {role: 0 for role in ShotRole}
        for r in cur.fetchall():
            counts[ShotRole(r["role"])] = int(r["n"])
        return counts


# --------------------------------------------------------------------------- #
# Row -> dataclass mappers
# --------------------------------------------------------------------------- #


def _row_to_combination(row: sqlite3.Row) -> Combination:
    return Combination(
        id=row["id"],
        sku=row["sku"],
        platform=row["platform"],
        ammo=row["ammo"],
        created_at=row["created_at"],
    )


def _row_to_batch(row: sqlite3.Row) -> Batch:
    return Batch(
        id=row["id"],
        combination_id=row["combination_id"],
        label=row["label"],
        session_date=row["session_date"],
        wind_speed=row["wind_speed"],
        temp=row["temp"],
        relative_humidity=row["relative_humidity"],
        notes=row["notes"],
        closed=bool(row["closed"]),
        created_at=row["created_at"],
        closed_at=row["closed_at"],
    )


def _row_to_cluster(row: sqlite3.Row) -> Cluster:
    return Cluster(
        id=row["id"],
        batch_id=row["batch_id"],
        cluster_index=row["cluster_index"],
        created_at=row["created_at"],
    )


def _row_to_shot(row: sqlite3.Row) -> Shot:
    return Shot(
        id=row["id"],
        source_file=row["source_file"],
        suppressor_sku=row["suppressor_sku"],
        test_platform=row["test_platform"],
        ammo=row["ammo"],
        cluster_index=row["cluster_index"],
        shot_order=row["shot_order"],
        wind_speed=row["wind_speed"],
        temp=row["temp"],
        relative_humidity=row["relative_humidity"],
        se_channel=row["se_channel"],
        ml_channel=row["ml_channel"],
        marked=bool(row["marked"]),
        included=bool(row["included"]),
        exclusion_reason=row["exclusion_reason"],
        cluster_id=row["cluster_id"],
        created_at=row["created_at"],
        captured_at=row["captured_at"],
    )
