"""SQLite persistence for the Batch -> Group -> Shot -> channel-metrics hierarchy.

This is the workflow store described in the README's "Data Model & Workflow".
It lives alongside the legacy flat :class:`~sound_metric_app.storage.database.ResultsDatabase`
(the single-file CLI path) in the same database file, and only ever creates its
own four tables, so the two can share a ``.db`` without interfering.

Standard-library ``sqlite3`` only. Foreign keys are enforced per connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from ..models import Batch, Group, MetricResult, MicPosition, Shot
from ._base import _SqliteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sku         TEXT NOT NULL,
    closed      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at   TEXT
);

CREATE TABLE IF NOT EXISTS groups (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    test_platform TEXT NOT NULL,
    ammo          TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (batch_id, test_platform, ammo)
);

CREATE TABLE IF NOT EXISTS shots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file       TEXT NOT NULL UNIQUE,
    suppressor_sku    TEXT,      -- provisional batch key from filename
    test_platform     TEXT,      -- provisional group key from filename
    ammo              TEXT,      -- set at marking
    shot_order        INTEGER,
    wind_speed        REAL,      -- mph
    temp              REAL,      -- degrees Fahrenheit
    relative_humidity REAL,      -- percent
    se_channel        TEXT,      -- raw channel name tagged SE
    mr_channel        TEXT,      -- raw channel name tagged MR
    marked            INTEGER NOT NULL DEFAULT 0,
    group_id          INTEGER REFERENCES groups(id) ON DELETE SET NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    captured_at       TEXT       -- when the shot was fired (Dewesoft start_store_time), ISO-8601
);

CREATE TABLE IF NOT EXISTS channel_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id         INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    mic_position    TEXT NOT NULL,   -- 'SE' | 'MR'
    channel         TEXT NOT NULL,   -- raw source channel name
    sample_rate     REAL NOT NULL,
    n_samples       INTEGER NOT NULL,
    peak_db         REAL,
    peak_dba        REAL,
    peak_impulse_db REAL,
    laimax_db       REAL,
    liaeq_100ms_db  REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (shot_id, mic_position)
);
"""

_METRIC_FIELDS = ("peak_db", "peak_dba", "peak_impulse_db", "laimax_db", "liaeq_100ms_db")


class WorkflowRepository(_SqliteStore):
    """Data-management wrapper over the hierarchy tables in a local SQLite file."""

    _SCHEMA = _SCHEMA
    _PRAGMAS = ("PRAGMA foreign_keys = ON",)

    def _migrate(self) -> None:
        # captured_at was added after the shots table shipped; back-fill the
        # column on databases created before it existed.
        self._add_column_if_missing("shots", "captured_at", "TEXT")

        # laimax_db (LAImax) was added to channel_metrics after it shipped;
        # back-fill it likewise. Pre-existing rows read back NULL (unknown), so
        # AVG() skips them and the report renders them as an em-dash until the
        # source files are re-processed under the new metric set.
        self._add_column_if_missing("channel_metrics", "laimax_db", "REAL")

        if self._schema_version() < 1:
            # peak_impulse_db used to be the maximum Impulse level [dB]; it is
            # now that level integrated over the frame [dB*ms] (MATH.md §6).
            # Older rows cannot be converted -- the integral needs the waveform,
            # not the stored scalar -- so blank them rather than let AVG() mix
            # the two unit systems into a number that looks plausible and means
            # nothing. NULL reads as "unknown" and AVG() skips it. Re-processing
            # the source files repopulates these rows in the new units.
            self._conn.execute("UPDATE channel_metrics SET peak_impulse_db = NULL")
            self._set_schema_version(1)

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

    # ---- batches -------------------------------------------------------- #

    def create_batch(self, sku: str) -> int:
        """Create a new (open) batch for a SKU and return its id."""
        cur = self._conn.execute("INSERT INTO batches (sku) VALUES (?)", (sku,))
        self._commit()
        return int(cur.lastrowid)

    def rename_batch_sku(self, batch_id: int, sku: str) -> None:
        """Change a batch's SKU (e.g. to fix a mistyped one). Idempotent.

        Only rewrites the ``sku`` column; the batch's groups and shots stay put,
        so this corrects the label without re-clustering any shot. Raises
        ``LookupError`` if the batch id is unknown.
        """
        cur = self._conn.execute("UPDATE batches SET sku = ? WHERE id = ?", (sku, batch_id))
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

    def open_batch_for_sku(self, sku: str) -> Batch | None:
        """Newest non-closed batch for a SKU, if any (helper for clustering)."""
        row = self._conn.execute(
            "SELECT * FROM batches WHERE sku = ? AND closed = 0 ORDER BY id DESC LIMIT 1",
            (sku,),
        ).fetchone()
        return _row_to_batch(row) if row else None

    def all_batches(self) -> list[Batch]:
        """Every batch, oldest first (for the CLI/GUI ``list batches`` view)."""
        cur = self._conn.execute("SELECT * FROM batches ORDER BY id")
        return [_row_to_batch(r) for r in cur.fetchall()]

    # ---- groups --------------------------------------------------------- #

    def upsert_group(self, batch_id: int, test_platform: str, ammo: str) -> int:
        """Return the id of the (batch, platform, ammo) group, creating it if new."""
        self._conn.execute(
            """
            INSERT INTO groups (batch_id, test_platform, ammo) VALUES (?, ?, ?)
            ON CONFLICT (batch_id, test_platform, ammo) DO NOTHING
            """,
            (batch_id, test_platform, ammo),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT id FROM groups WHERE batch_id = ? AND test_platform = ? AND ammo = ?",
            (batch_id, test_platform, ammo),
        ).fetchone()
        return int(row["id"])

    def get_group(self, group_id: int) -> Group | None:
        row = self._conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        return _row_to_group(row) if row else None

    def groups_for_batch(self, batch_id: int) -> list[Group]:
        cur = self._conn.execute("SELECT * FROM groups WHERE batch_id = ? ORDER BY id", (batch_id,))
        return [_row_to_group(r) for r in cur.fetchall()]

    # ---- shots ---------------------------------------------------------- #

    def add_unmarked_shot(
        self,
        source_file: str,
        suppressor_sku: str | None = None,
        test_platform: str | None = None,
        shot_order: int | None = None,
    ) -> int:
        """Record a capture file as an Unmarked Data Set. Idempotent by source_file.

        Re-adding an already-ingested file returns the existing shot id without
        overwriting it, so re-scanning the input folder never duplicates shots.
        """
        self._conn.execute(
            """
            INSERT INTO shots (source_file, suppressor_sku, test_platform, shot_order, marked)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT (source_file) DO NOTHING
            """,
            (source_file, suppressor_sku, test_platform, shot_order),
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
        group_id: int,
        ammo: str,
        shot_order: int | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        se_channel: str | None = None,
        mr_channel: str | None = None,
        captured_at: str | None = None,
        replace_optional: bool = False,
    ) -> None:
        """Apply marking metadata, link to a group, and flag the shot marked.

        ``group_id`` and ``ammo`` are always written. By default every optional
        field (``shot_order``, the environment/channel columns, and
        ``captured_at``) is preserved when left unset, so re-marking to correct
        one field does not clobber the others: a value is only overwritten when
        explicitly supplied. This suits a partial re-mark (e.g. the CLI setting a
        single field).

        Pass ``replace_optional=True`` for a full-form edit, where the caller
        supplies the complete intended state and an unset user field means
        *blank it*: the four user-editable optional fields (``shot_order``,
        ``wind_speed``, ``temp``, ``relative_humidity``) are then written exactly,
        so passing ``None`` clears them. The channel and ``captured_at`` columns
        are unaffected — the service sets channels definitively via
        :meth:`set_shot_channels` and always re-supplies ``captured_at`` from the
        capture file.

        Raises ``LookupError`` if ``shot_id`` matches no shot.
        """

        # Column names are hard-coded literals, so this f-string carries no
        # injection surface; user values still bind through placeholders.
        def _opt(column: str) -> str:
            return "?" if replace_optional else f"COALESCE(?, {column})"

        cur = self._conn.execute(
            f"""
            UPDATE shots SET
                group_id = ?,
                ammo = ?,
                shot_order = {_opt("shot_order")},
                wind_speed = {_opt("wind_speed")},
                temp = {_opt("temp")},
                relative_humidity = {_opt("relative_humidity")},
                se_channel = COALESCE(?, se_channel),
                mr_channel = COALESCE(?, mr_channel),
                captured_at = COALESCE(?, captured_at),
                marked = 1
            WHERE id = ?
            """,
            (
                group_id,
                ammo,
                shot_order,
                wind_speed,
                temp,
                relative_humidity,
                se_channel,
                mr_channel,
                captured_at,
                shot_id,
            ),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No shot with id {shot_id}")
        self._commit()

    def set_shot_channels(
        self, shot_id: int, *, se_channel: str | None, mr_channel: str | None
    ) -> None:
        """Set a shot's SE/MR channel tags exactly, clearing either when ``None``.

        Unlike :meth:`mark_shot` — which preserves an unsupplied channel tag —
        this overwrites both columns unconditionally, so re-marking a shot with
        fewer mics drops the tag for the mic that is no longer present.

        Raises ``LookupError`` if ``shot_id`` matches no shot.
        """
        cur = self._conn.execute(
            "UPDATE shots SET se_channel = ?, mr_channel = ? WHERE id = ?",
            (se_channel, mr_channel, shot_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"No shot with id {shot_id}")
        self._commit()

    def shots_by_group(self, group_id: int) -> list[Shot]:
        """Shots in a group, ordered by shot order (then id for stability)."""
        cur = self._conn.execute(
            "SELECT * FROM shots WHERE group_id = ? ORDER BY shot_order, id",
            (group_id,),
        )
        return [_row_to_shot(r) for r in cur.fetchall()]

    def count_shots_in_group(self, group_id: int) -> int:
        """Number of shots in a group, without materializing their rows."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM shots WHERE group_id = ?",
            (group_id,),
        )
        return int(cur.fetchone()[0])

    def delete_empty_groups(self) -> int:
        """Delete every group that holds no shots; return how many were removed.

        A shot-less group can be left behind by edits made before per-re-mark
        cleanup existed, or by any path that empties a group without pruning it.
        The batch tree calls this on load so refreshing sweeps such stragglers,
        keeping the view uncluttered and freeing their (batch, platform, ammo)
        names for re-use.
        """
        cur = self._conn.execute(
            """
            DELETE FROM groups
            WHERE NOT EXISTS (SELECT 1 FROM shots WHERE shots.group_id = groups.id)
            """
        )
        self._commit()
        return cur.rowcount

    def delete_group_if_empty(self, group_id: int) -> bool:
        """Delete a group only if it holds no shots; return whether it was deleted.

        Called after re-marking moves a shot out of its former group: if that
        leaves the group empty, drop the row so the batch tree does not accrue
        empty groups and its (batch, platform, ammo) name is free to be re-used.
        The ``WHERE NOT EXISTS`` guard makes this a no-op for a group that still
        has shots, so a stale ``group_id`` can never orphan live shots.
        """
        cur = self._conn.execute(
            """
            DELETE FROM groups
            WHERE id = ?
              AND NOT EXISTS (SELECT 1 FROM shots WHERE shots.group_id = groups.id)
            """,
            (group_id,),
        )
        self._commit()
        return cur.rowcount > 0

    def delete_empty_batches(self) -> int:
        """Delete every batch that holds no groups; return how many were removed.

        Runs after :meth:`delete_empty_groups` in a sweep: pruning a batch's
        last shot-less group leaves the batch itself an empty shell (a re-marked
        or closed batch that no longer holds anything), which this removes so the
        tree does not accrete empty batches over time.
        """
        cur = self._conn.execute(
            """
            DELETE FROM batches
            WHERE NOT EXISTS (SELECT 1 FROM groups WHERE groups.batch_id = batches.id)
            """
        )
        self._commit()
        return cur.rowcount

    def delete_batch_if_empty(self, batch_id: int) -> bool:
        """Delete a batch only if it holds no groups; return whether it was deleted.

        Called after re-marking moves a shot's former group out from under it: if
        that was the batch's last group, drop the now-empty batch so re-marking
        the sole shot out of a (typically closed) batch does not leave it behind
        as a shell. The ``WHERE NOT EXISTS`` guard makes this a no-op for a batch
        that still has groups.
        """
        cur = self._conn.execute(
            """
            DELETE FROM batches
            WHERE id = ?
              AND NOT EXISTS (SELECT 1 FROM groups WHERE groups.batch_id = batches.id)
            """,
            (batch_id,),
        )
        self._commit()
        return cur.rowcount > 0

    # ---- channel metrics ------------------------------------------------ #

    def save_channel_metric(
        self, shot_id: int, mic_position: MicPosition, result: MetricResult
    ) -> int:
        """Persist one mic's metrics for a shot. Upserts on (shot, position)."""
        cur = self._conn.execute(
            """
            INSERT INTO channel_metrics
                (shot_id, mic_position, channel, sample_rate, n_samples,
                 peak_db, peak_dba, peak_impulse_db, laimax_db, liaeq_100ms_db)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (shot_id, mic_position) DO UPDATE SET
                channel = excluded.channel,
                sample_rate = excluded.sample_rate,
                n_samples = excluded.n_samples,
                peak_db = excluded.peak_db,
                peak_dba = excluded.peak_dba,
                peak_impulse_db = excluded.peak_impulse_db,
                laimax_db = excluded.laimax_db,
                liaeq_100ms_db = excluded.liaeq_100ms_db
            RETURNING id
            """,
            (
                shot_id,
                mic_position.value,
                result.channel,
                result.sample_rate,
                result.n_samples,
                result.peak_db,
                result.peak_dba,
                result.peak_impulse_db,
                result.laimax_db,
                result.liaeq_100ms_db,
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

    def group_averages(self, group_id: int) -> dict[MicPosition, dict]:
        """Average each metric across a group's shots, separately per mic position.

        SE and MR are never mixed: the result maps each present
        :class:`MicPosition` to ``{peak_db, peak_dba, peak_impulse_db,
        laimax_db, liaeq_100ms_db, n}``. Positions absent from the group are
        omitted.
        """
        cur = self._conn.execute(
            """
            SELECT cm.mic_position AS pos,
                   AVG(cm.peak_db)         AS peak_db,
                   AVG(cm.peak_dba)        AS peak_dba,
                   AVG(cm.peak_impulse_db) AS peak_impulse_db,
                   AVG(cm.laimax_db)       AS laimax_db,
                   AVG(cm.liaeq_100ms_db)  AS liaeq_100ms_db,
                   COUNT(*)                AS n
            FROM channel_metrics cm
            JOIN shots s ON s.id = cm.shot_id
            WHERE s.group_id = ?
            GROUP BY cm.mic_position
            """,
            (group_id,),
        )
        out: dict[MicPosition, dict] = {}
        for r in cur.fetchall():
            out[MicPosition(r["pos"])] = {
                **{k: r[k] for k in _METRIC_FIELDS},
                "n": int(r["n"]),
            }
        return out

    def shot_metrics_for_group(self, group_id: int) -> dict[MicPosition, list[dict]]:
        """Per-shot metric rows for a group, grouped by mic position.

        The un-averaged counterpart to :meth:`group_averages`: instead of one
        mean per position, returns every contributing shot's own metrics so a
        report can drill down from a group's SE/MR average into the individual
        shots behind it. Maps each present :class:`MicPosition` to a list of
        ``{shot_id, shot_order, source_file, peak_db, peak_dba,
        peak_impulse_db, laimax_db, liaeq_100ms_db}``, ordered by shot order then id.
        Positions absent from the group are omitted.
        """
        cur = self._conn.execute(
            """
            SELECT cm.mic_position       AS pos,
                   s.id                  AS shot_id,
                   s.shot_order          AS shot_order,
                   s.source_file         AS source_file,
                   cm.peak_db, cm.peak_dba, cm.peak_impulse_db, cm.laimax_db, cm.liaeq_100ms_db
            FROM channel_metrics cm
            JOIN shots s ON s.id = cm.shot_id
            WHERE s.group_id = ?
            ORDER BY cm.mic_position, s.shot_order, s.id
            """,
            (group_id,),
        )
        out: dict[MicPosition, list[dict]] = {}
        for r in cur.fetchall():
            out.setdefault(MicPosition(r["pos"]), []).append(
                {
                    "shot_id": int(r["shot_id"]),
                    "shot_order": r["shot_order"],
                    "source_file": r["source_file"],
                    **{k: r[k] for k in _METRIC_FIELDS},
                }
            )
        return out


# --------------------------------------------------------------------------- #
# Row -> dataclass mappers
# --------------------------------------------------------------------------- #


def _row_to_batch(row: sqlite3.Row) -> Batch:
    return Batch(
        id=row["id"],
        sku=row["sku"],
        closed=bool(row["closed"]),
        created_at=row["created_at"],
        closed_at=row["closed_at"],
    )


def _row_to_group(row: sqlite3.Row) -> Group:
    return Group(
        id=row["id"],
        batch_id=row["batch_id"],
        test_platform=row["test_platform"],
        ammo=row["ammo"],
        created_at=row["created_at"],
    )


def _row_to_shot(row: sqlite3.Row) -> Shot:
    return Shot(
        id=row["id"],
        source_file=row["source_file"],
        suppressor_sku=row["suppressor_sku"],
        test_platform=row["test_platform"],
        ammo=row["ammo"],
        shot_order=row["shot_order"],
        wind_speed=row["wind_speed"],
        temp=row["temp"],
        relative_humidity=row["relative_humidity"],
        se_channel=row["se_channel"],
        mr_channel=row["mr_channel"],
        marked=bool(row["marked"]),
        group_id=row["group_id"],
        created_at=row["created_at"],
        captured_at=row["captured_at"],
    )
