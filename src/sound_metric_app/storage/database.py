"""SQLite persistence for measurement results (standard-library only)."""

from __future__ import annotations

from ..models import MetricResult
from ._base import _SqliteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file     TEXT NOT NULL,
    channel         TEXT NOT NULL,
    timestamp       TEXT,
    sample_rate     REAL NOT NULL,
    n_samples       INTEGER NOT NULL,
    peak_db         REAL,
    peak_dba        REAL,
    peak_impulse_db REAL,
    liaeq_100ms_db  REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class ResultsDatabase(_SqliteStore):
    """Thin data-management wrapper over a local SQLite file."""

    _SCHEMA = _SCHEMA

    def add_result(self, result: MetricResult) -> int:
        row = result.as_row()
        cur = self._conn.execute(
            """
            INSERT INTO results
                (source_file, channel, timestamp, sample_rate, n_samples,
                 peak_db, peak_dba, peak_impulse_db, liaeq_100ms_db)
            VALUES
                (:source_file, :channel, :timestamp, :sample_rate, :n_samples,
                 :peak_db, :peak_dba, :peak_impulse_db, :liaeq_100ms_db)
            """,
            row,
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def all_results(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM results ORDER BY id DESC")
        return [dict(r) for r in cur.fetchall()]

    def delete_result(self, result_id: int) -> None:
        self._conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
        self._conn.commit()
