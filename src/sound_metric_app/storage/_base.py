"""Shared SQLite connection lifecycle for the storage-layer wrappers.

Both :class:`~sound_metric_app.storage.database.ResultsDatabase` and
:class:`~sound_metric_app.storage.repository.WorkflowRepository` open a
standard-library ``sqlite3`` connection, install their schema, and act as
context managers. That boilerplate lives here so the two stores stay in sync.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Self

from ..config import DEFAULT_DB_PATH


class _SqliteStore:
    """Base for the local-SQLite data wrappers.

    Subclasses set :attr:`_SCHEMA` (run via ``executescript`` on connect) and,
    if needed, :attr:`_PRAGMAS` (per-connection ``PRAGMA`` statements, e.g.
    ``PRAGMA foreign_keys = ON``). Rows come back as :class:`sqlite3.Row`.
    """

    #: ``CREATE TABLE IF NOT EXISTS ...`` script applied on connect.
    _SCHEMA: str = ""
    #: Per-connection PRAGMA statements applied before the schema.
    _PRAGMAS: tuple[str, ...] = ()

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        for pragma in self._PRAGMAS:
            self._conn.execute(pragma)
        self._conn.executescript(self._SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Bring an already-existing database up to the current schema.

        The schema script only ever *creates* missing tables, so columns added
        to a table after it was first created never reach a database that already
        has that table. Subclasses override this to apply idempotent
        ``ALTER TABLE ... ADD COLUMN`` migrations; the default is a no-op.
        """

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        """Add ``column`` to ``table`` if it is not already present. Idempotent."""
        existing = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
