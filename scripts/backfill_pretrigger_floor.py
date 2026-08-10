"""Fill in ``channel_metrics.pretrigger_floor_pa`` for already-ingested shots.

The pre-trigger floor is a diagnostic added after these rows were written, so
every existing channel row holds NULL for it and renders as "—" in the Batch
average tree. Re-marking each shot would recompute it, but that also rewrites
every metric — this script touches the one diagnostic column and nothing else,
so a backfill can never move a number anybody has already reported.

It re-reads each row's source capture, so it needs the ``.dxd`` files still to
be where ``shots.source_file`` says they are. A row whose file is missing or
unreadable is left NULL and reported at the end; running again after restoring
the file picks it up.

Usage::

    python scripts/backfill_pretrigger_floor.py [--db sound_metrics.db] [--dry-run]

Rows that already hold a value are skipped unless ``--all`` is passed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sound_metric_app.config import DEFAULT_DB_PATH  # noqa: E402
from sound_metric_app.dsp.metrics import pretrigger_floor_pa  # noqa: E402
from sound_metric_app.ingestion.dewesoft_reader import read_frame  # noqa: E402
from sound_metric_app.storage.repository import WorkflowRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument(
        "--all",
        action="store_true",
        help="recompute rows that already hold a floor, not just the NULL ones",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )
    args = parser.parse_args()

    # Open the repository once first: the column is added by its migration, and
    # on a database last touched by the previous build it does not exist yet, so
    # the query below would fail before writing anything.
    with WorkflowRepository(args.db):
        pass

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    where = "" if args.all else " AND cm.pretrigger_floor_pa IS NULL"
    rows = conn.execute(
        f"""
        SELECT cm.id, cm.channel, cm.mic_position, s.source_file
        FROM channel_metrics cm
        JOIN shots s ON s.id = cm.shot_id
        WHERE s.source_file IS NOT NULL{where}
        ORDER BY cm.id
        """
    ).fetchall()

    if not rows:
        print("Nothing to backfill.")
        return 0

    # One capture usually feeds two channel rows (SE and ML); cache per file so a
    # two-mic shot is read from disk once, not twice.
    cache: dict[tuple[str, str], float] = {}
    updated = 0
    failures: list[tuple[str, str, str]] = []

    for row in rows:
        key = (row["source_file"], row["channel"])
        if key not in cache:
            try:
                frame = read_frame(row["source_file"], row["channel"])
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((row["source_file"], row["channel"], str(exc)))
                continue
            cache[key] = pretrigger_floor_pa(frame.samples)
        floor = cache[key]
        print(f"  {Path(row['source_file']).name:40s} {row['mic_position']:2s}  {floor:+.3f} Pa")
        if not args.dry_run:
            conn.execute(
                "UPDATE channel_metrics SET pretrigger_floor_pa = ? WHERE id = ?",
                (floor, row["id"]),
            )
        updated += 1

    if not args.dry_run:
        conn.commit()

    verb = "Would update" if args.dry_run else "Updated"
    print(f"\n{verb} {updated} of {len(rows)} channel rows.")
    if failures:
        print(f"\n{len(failures)} row(s) left NULL — capture unreadable:")
        for source_file, channel, exc in failures:
            print(f"  {source_file} [{channel}]: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
