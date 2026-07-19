"""SQLite persistence for measurement results (standard-library only)."""

from __future__ import annotations

from ..models import MetricResult
from ._base import _SqliteStore

#: Metric columns each store carries: a linear magnitude (Pa / Pa·ms) and its dB
#: level. Kept in one place so the two stores and their migrations stay in sync.
_METRIC_COLUMNS = (
    "peak_pa", "peak_db",
    "peak_a_pa", "peak_dba",
    "impulse_pa_ms", "peak_impulse_db",
    "leq10ms_pa", "leq10ms_db",
    "liaeq_pa", "liaeq_100ms_db",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file     TEXT NOT NULL,
    channel         TEXT NOT NULL,
    timestamp       TEXT,
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
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class ResultsDatabase(_SqliteStore):
    """Thin data-management wrapper over a local SQLite file."""

    _SCHEMA = _SCHEMA

    def _migrate(self) -> None:
        # peak_pa and the linear-magnitude / new-metric columns were added after
        # the results table first shipped; back-fill them on older databases.
        for column in _METRIC_COLUMNS:
            self._add_column_if_missing("results", column, "REAL")

        if self._schema_version() < 1:
            # Rows written before peak_impulse_db became dB*ms hold a plain dB
            # level that cannot be converted after the fact. Blank them.
            self._conn.execute("UPDATE results SET peak_impulse_db = NULL")
            self._set_schema_version(1)
        if self._schema_version() < 2:
            # Metrics were realigned to TBAC's onset-anchored definitions and now
            # store a linear magnitude per metric (MATH.md §6/§7/§9). Old rows hold
            # values under the previous whole-frame definitions with no linear
            # companion, so blank every metric column; re-processing the source
            # file repopulates them under the new definitions.
            cols = ", ".join(f"{c} = NULL" for c in _METRIC_COLUMNS)
            self._conn.execute(f"UPDATE results SET {cols}")
            self._set_schema_version(2)

    def add_result(self, result: MetricResult) -> int:
        row = result.as_row()
        metric_cols = ", ".join(_METRIC_COLUMNS)
        metric_vals = ", ".join(f":{c}" for c in _METRIC_COLUMNS)
        cur = self._conn.execute(
            f"""
            INSERT INTO results
                (source_file, channel, timestamp, sample_rate, n_samples,
                 {metric_cols})
            VALUES
                (:source_file, :channel, :timestamp, :sample_rate, :n_samples,
                 {metric_vals})
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
