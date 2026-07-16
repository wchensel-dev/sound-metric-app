"""SQLite persistence for the Batch -> Group -> Shot -> channel-metrics hierarchy.

This is the workflow store described in the README's "Data Model & Workflow".
It lives alongside the legacy flat :class:`~sound_metric_app.storage.database.ResultsDatabase`
(the single-file CLI path) in the same database file, and only ever creates its
own four tables, so the two can share a ``.db`` without interfering.

Standard-library ``sqlite3`` only. Foreign keys are enforced per connection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import DEFAULT_DB_PATH
from ..models import Batch, Group, MetricResult, MicPosition, Shot

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
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
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
    liaeq_100ms_db  REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (shot_id, mic_position)
);
"""

_METRIC_FIELDS = ("peak_db", "peak_dba", "peak_impulse_db", "liaeq_100ms_db")


class WorkflowRepository:
    """Data-management wrapper over the hierarchy tables in a local SQLite file."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- batches -------------------------------------------------------- #

    def create_batch(self, sku: str) -> int:
        """Create a new (open) batch for a SKU and return its id."""
        cur = self._conn.execute("INSERT INTO batches (sku) VALUES (?)", (sku,))
        self._conn.commit()
        return int(cur.lastrowid)

    def close_batch(self, batch_id: int) -> None:
        """Mark a batch closed. Idempotent."""
        self._conn.execute(
            "UPDATE batches SET closed = 1, closed_at = datetime('now') WHERE id = ?",
            (batch_id,),
        )
        self._conn.commit()

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
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM groups WHERE batch_id = ? AND test_platform = ? AND ammo = ?",
            (batch_id, test_platform, ammo),
        ).fetchone()
        return int(row["id"])

    def get_group(self, group_id: int) -> Group | None:
        row = self._conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
        return _row_to_group(row) if row else None

    def groups_for_batch(self, batch_id: int) -> list[Group]:
        cur = self._conn.execute(
            "SELECT * FROM groups WHERE batch_id = ? ORDER BY id", (batch_id,)
        )
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
        self._conn.commit()
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
    ) -> None:
        """Apply marking metadata, link to a group, and flag the shot marked.

        ``shot_order`` overrides the provisional value from the filename when
        given; otherwise the existing value is kept.
        """
        self._conn.execute(
            """
            UPDATE shots SET
                group_id = ?,
                ammo = ?,
                shot_order = COALESCE(?, shot_order),
                wind_speed = ?,
                temp = ?,
                relative_humidity = ?,
                se_channel = ?,
                mr_channel = ?,
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
                shot_id,
            ),
        )
        self._conn.commit()

    def shots_by_group(self, group_id: int) -> list[Shot]:
        """Shots in a group, ordered by shot order (then id for stability)."""
        cur = self._conn.execute(
            "SELECT * FROM shots WHERE group_id = ? ORDER BY shot_order, id",
            (group_id,),
        )
        return [_row_to_shot(r) for r in cur.fetchall()]

    # ---- channel metrics ------------------------------------------------ #

    def save_channel_metric(
        self, shot_id: int, mic_position: MicPosition, result: MetricResult
    ) -> int:
        """Persist one mic's metrics for a shot. Upserts on (shot, position)."""
        cur = self._conn.execute(
            """
            INSERT INTO channel_metrics
                (shot_id, mic_position, channel, sample_rate, n_samples,
                 peak_db, peak_dba, peak_impulse_db, liaeq_100ms_db)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (shot_id, mic_position) DO UPDATE SET
                channel = excluded.channel,
                sample_rate = excluded.sample_rate,
                n_samples = excluded.n_samples,
                peak_db = excluded.peak_db,
                peak_dba = excluded.peak_dba,
                peak_impulse_db = excluded.peak_impulse_db,
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
                result.liaeq_100ms_db,
            ),
        )
        row_id = int(cur.fetchone()[0])
        self._conn.commit()
        return row_id

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
        liaeq_100ms_db, n}``. Positions absent from the group are omitted.
        """
        cur = self._conn.execute(
            """
            SELECT cm.mic_position AS pos,
                   AVG(cm.peak_db)         AS peak_db,
                   AVG(cm.peak_dba)        AS peak_dba,
                   AVG(cm.peak_impulse_db) AS peak_impulse_db,
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

    # ---- lifecycle ------------------------------------------------------ #

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> WorkflowRepository:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


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
    )
