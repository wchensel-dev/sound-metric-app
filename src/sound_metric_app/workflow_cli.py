"""Workflow command line — drive the whole ingest -> mark -> cluster -> report
pipeline over the Phase B services (BUILD_PLAN Task 8).

Exposed as the ``sma`` console script. The legacy single-file analyzer keeps its
own ``sma-analyze`` entry point (:mod:`sound_metric_app.cli`); this command adds
the batch/group/shot workflow on top of the same local SQLite database.

Subcommands
-----------
``ingest``       Scan the input folder for new captures -> Unmarked Data Sets.
``mark``         Annotate an unmarked shot, tag SE/MR, compute + store metrics.
``list``         Show unmarked shots, batches, or a batch's groups.
``close-batch``  Close a batch so further testing starts a new one.
``report``       Per-group SE/MR metric averages for a batch or single group.
``config``       Show or set persisted settings (e.g. the input folder).

Every subcommand takes ``--db PATH`` (defaults to
:data:`~sound_metric_app.config.DEFAULT_DB_PATH`). The channel/capture readers
are module-level so tests can substitute fakes without a real ``.dxd`` file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .ingestion import list_channels, read_capture
from .models import MicPosition
from .services import (
    AggregationService,
    ClosedBatchError,
    ClusteringService,
    IngestionService,
    MarkingService,
)
from .storage import WorkflowRepository

#: Readers used by the workflow services. Kept at module scope so tests can
#: monkeypatch them with fakes instead of opening real DewesoftX files.
_channel_reader = list_channels
_capture_reader = read_capture

#: Exceptions the services raise for bad user input / state, reported as a clean
#: one-line CLI error (exit 2) rather than a traceback.
_USER_ERRORS = (
    LookupError,
    ValueError,
    ClosedBatchError,
    FileNotFoundError,
    NotADirectoryError,
)


# --------------------------------------------------------------------------- #
# Subcommand handlers  (each takes parsed args + an open repo, returns exit code)
# --------------------------------------------------------------------------- #


def _cmd_ingest(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    folder = args.folder or config.get_input_folder()
    if not folder:
        print(
            "No input folder given and none configured. Pass one, or set a "
            "default with:  sma config set-input-folder <path>",
            file=sys.stderr,
        )
        return 2

    svc = IngestionService(repo, reader=_channel_reader)
    report = svc.scan(folder, validate=not args.no_validate)

    print(f"Scanned {folder}")
    print(f"  ingested       : {report.n_ingested}")
    for shot in report.ingested:
        print(f"    #{shot.id}  {Path(shot.source_file).name}")
    print(f"  already present : {len(report.already_present)}")
    if report.malformed:
        print(f"  malformed       : {len(report.malformed)}")
        for path, reason in report.malformed:
            print(f"    {Path(path).name}: {reason}")
    if report.unreadable:
        print(f"  unreadable      : {len(report.unreadable)}")
        for path, reason in report.unreadable:
            print(f"    {Path(path).name}: {reason}")
    return 0


def _cmd_mark(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    if args.se and args.mr and args.se == args.mr:
        print("--se and --mr cannot be the same channel.", file=sys.stderr)
        return 2
    channel_map: dict[str, MicPosition] = {}
    if args.se:
        channel_map[args.se] = MicPosition.SE
    if args.mr:
        channel_map[args.mr] = MicPosition.MR
    if not channel_map:
        print("Tag at least one mic channel with --se and/or --mr.", file=sys.stderr)
        return 2

    svc = MarkingService(repo, ClusteringService(repo), reader=_capture_reader)
    marked = svc.mark(
        args.shot_id,
        ammo=args.ammo,
        channel_map=channel_map,
        suppressor_sku=args.sku,
        test_platform=args.platform,
        shot_order=args.shot_order,
        wind_speed=args.wind_speed,
        temp=args.temp,
        relative_humidity=args.rh,
    )

    shot = marked.shot
    print(f"Marked shot #{shot.id}  ({Path(shot.source_file).name})")
    print(f"  batch  : #{marked.batch.id}  SKU {marked.batch.sku}")
    print(f"  group  : #{marked.group_id}  {shot.test_platform} / {shot.ammo}")
    if shot.captured_at:
        print(f"  fired  : {shot.captured_at}")
    for position in (MicPosition.SE, MicPosition.MR):
        result = marked.metrics.get(position)
        if result is None:
            continue
        print(
            f"  {position.value}: peak {result.peak_db:7.2f} dB   "
            f"peakA {result.peak_dba:7.2f} dBA   "
            f"impulse {result.peak_impulse_db:9.2f} dB*ms   "
            f"LAImax {result.laimax_db:7.2f} dBA   "
            f"LIAeq {result.liaeq_100ms_db:7.2f} dBA"
        )
    return 0


def _cmd_list(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    if args.target == "unmarked":
        shots = repo.unmarked_shots()
        if not shots:
            print("No unmarked shots.")
            return 0
        print(f"Unmarked shots ({len(shots)}):")
        for s in shots:
            keys = " / ".join(str(k) for k in (s.suppressor_sku, s.test_platform, s.shot_order))
            print(f"  #{s.id}  {Path(s.source_file).name}  [{keys}]")
        return 0

    if args.target == "batches":
        batches = repo.all_batches()
        if not batches:
            print("No batches.")
            return 0
        print(f"Batches ({len(batches)}):")
        for b in batches:
            state = "closed" if b.closed else "open"
            print(f"  #{b.id}  SKU {b.sku}  [{state}]")
        return 0

    # target == "groups"
    if args.batch is None:
        print("`list groups` requires --batch <id>.", file=sys.stderr)
        return 2
    if repo.get_batch(args.batch) is None:
        raise LookupError(f"No batch with id {args.batch}")
    groups = repo.groups_for_batch(args.batch)
    if not groups:
        print(f"Batch #{args.batch} has no groups.")
        return 0
    print(f"Groups in batch #{args.batch} ({len(groups)}):")
    for g in groups:
        n = repo.count_shots_in_group(g.id)
        print(f"  #{g.id}  {g.test_platform} / {g.ammo}  ({n} shot(s))")
    return 0


def _cmd_close_batch(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    ClusteringService(repo).close_batch(args.batch_id)
    print(f"Closed batch #{args.batch_id}.")
    return 0


def _cmd_report(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    agg = AggregationService(repo)
    if args.group is not None:
        _print_group_averages(agg.group_averages(args.group))
        return 0

    report = agg.batch_report(args.batch)
    state = "closed" if report.batch.closed else "open"
    print(f"Report — batch #{report.batch.id}  SKU {report.batch.sku}  [{state}]")
    if not report.groups:
        print("  (no groups)")
        return 0
    for group_avg in report.groups:
        _print_group_averages(group_avg, indent="  ")
    return 0


def _print_group_averages(group_avg, *, indent: str = "") -> None:
    g = group_avg.group
    print(f"{indent}Group #{g.id}  {g.test_platform} / {g.ammo}  ({group_avg.n_shots} shot(s))")
    if not group_avg.averages:
        print(f"{indent}  (no metrics)")
        return
    for position in (MicPosition.SE, MicPosition.MR):
        avg = group_avg.averages.get(position)
        if avg is None:
            continue
        print(
            f"{indent}  {position.value} (n={avg['n']}): "
            f"peak {avg['peak_db']:7.2f} dB   "
            f"peakA {avg['peak_dba']:7.2f} dBA   "
            f"impulse {avg['peak_impulse_db']:9.2f} dB*ms   "
            f"LAImax {avg['laimax_db']:7.2f} dBA   "
            f"LIAeq {avg['liaeq_100ms_db']:7.2f} dBA"
        )


def _cmd_config(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    if args.config_action == "set-input-folder":
        resolved = config.set_input_folder(args.path)
        print(f"Input folder set to {resolved}")
        return 0

    # config_action == "show"
    print(f"Settings file : {config.config_path()}")
    folder = config.get_input_folder()
    print(f"Input folder  : {folder if folder else '(unset)'}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sma",
        description="Sound Metric App workflow: ingest, mark, cluster, and report shots.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_db(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--db",
            default=config.DEFAULT_DB_PATH,
            metavar="PATH",
            help=f"SQLite workflow database (default: {config.DEFAULT_DB_PATH})",
        )

    # ingest
    p_ingest = sub.add_parser("ingest", help="scan the input folder for new captures")
    p_ingest.add_argument(
        "folder",
        nargs="?",
        help="folder to scan (defaults to the configured input folder)",
    )
    p_ingest.add_argument(
        "--no-validate",
        action="store_true",
        help="skip opening each new file to confirm it is readable",
    )
    add_db(p_ingest)
    p_ingest.set_defaults(func=_cmd_ingest)

    # mark
    p_mark = sub.add_parser("mark", help="annotate an unmarked shot and compute metrics")
    p_mark.add_argument("shot_id", type=int, help="id of the unmarked shot (see `list unmarked`)")
    p_mark.add_argument("--ammo", required=True, help="ammunition / load identifier (group key)")
    p_mark.add_argument("--se", metavar="CHANNEL", help="raw channel name to tag as SE")
    p_mark.add_argument("--mr", metavar="CHANNEL", help="raw channel name to tag as MR")
    p_mark.add_argument("--sku", help="override the provisional suppressor SKU (batch key)")
    p_mark.add_argument("--platform", help="override the provisional test platform (group key)")
    p_mark.add_argument("--shot-order", type=int, dest="shot_order", help="shot order within group")
    p_mark.add_argument("--wind-speed", type=float, dest="wind_speed", help="wind speed (mph)")
    p_mark.add_argument("--temp", type=float, help="ambient temperature (deg F)")
    p_mark.add_argument("--rh", type=float, help="relative humidity (percent)")
    add_db(p_mark)
    p_mark.set_defaults(func=_cmd_mark)

    # list
    p_list = sub.add_parser("list", help="list unmarked shots, batches, or groups")
    p_list.add_argument("target", choices=["unmarked", "batches", "groups"])
    p_list.add_argument("--batch", type=int, help="batch id (required for `list groups`)")
    add_db(p_list)
    p_list.set_defaults(func=_cmd_list)

    # close-batch
    p_close = sub.add_parser("close-batch", help="close a batch")
    p_close.add_argument("batch_id", type=int)
    add_db(p_close)
    p_close.set_defaults(func=_cmd_close_batch)

    # report
    p_report = sub.add_parser("report", help="per-group SE/MR averages")
    target = p_report.add_mutually_exclusive_group(required=True)
    target.add_argument("--batch", type=int, help="report every group in this batch")
    target.add_argument("--group", type=int, help="report a single group")
    add_db(p_report)
    p_report.set_defaults(func=_cmd_report)

    # config
    p_config = sub.add_parser("config", help="show or set persisted settings")
    config_sub = p_config.add_subparsers(dest="config_action", required=True)
    config_sub.add_parser("show", help="show current settings")
    p_set_folder = config_sub.add_parser("set-input-folder", help="set the default input folder")
    p_set_folder.add_argument("path", help="folder to scan by default on `ingest`")
    p_config.set_defaults(func=_cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # `config` never touches the database; run it without opening one.
    if args.command == "config":
        try:
            return args.func(args, None)
        except _USER_ERRORS as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        with WorkflowRepository(args.db) as repo:
            return args.func(args, repo)
    except _USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
