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
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
