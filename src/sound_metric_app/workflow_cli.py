"""Workflow command line — drive the whole ingest -> mark -> include -> report
pipeline over the Phase B services (BUILD_PLAN Task 8).

Exposed as the ``sma`` console script. The legacy single-file analyzer keeps its
own ``sma-analyze`` entry point (:mod:`sound_metric_app.cli`); this command adds
the containment-tree workflow on top of the same local SQLite database.

Subcommands
-----------
``ingest``       Scan the input folder for new captures -> Unmarked Data Sets.
``mark``         Annotate an unmarked shot, tag ML/SE, compute + store metrics.
``list``         Show unmarked shots, combinations, batches, or clusters.
``bank``         Data-bank view: every cluster and shot in a batch, idle or not.
``include``      Bring a shot or cluster forward into the batch average.
``exclude``      Return a shot or cluster to idle, with an optional reason.
``batch``        Update a batch's session metadata (date, typical weather, notes).
``close-batch``  Close a batch so further testing starts a new session.
``report``       Batch-average view: the four position x role output slots.
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
    AVERAGE_SLOTS,
    AggregationService,
    ClosedBatchError,
    ClusteringService,
    InclusionService,
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
        print(
            f"    #{shot.id}  {Path(shot.source_file).name}"
            f"  [cluster {shot.cluster_index}, shot {shot.shot_order}]"
        )
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
    if args.se and args.ml and args.se == args.ml:
        print("--se and --ml cannot be the same channel.", file=sys.stderr)
        return 2
    # An explicit map overrides the AI 1 / AI 2 auto-tagging; passing neither
    # flag leaves it None so the service auto-tags from the DAQ convention.
    channel_map: dict[str, MicPosition] | None = None
    if args.se or args.ml:
        channel_map = {}
        if args.se:
            channel_map[args.se] = MicPosition.SE
        if args.ml:
            channel_map[args.ml] = MicPosition.ML

    svc = MarkingService(repo, ClusteringService(repo), reader=_capture_reader)
    marked = svc.mark(
        args.shot_id,
        ammo=args.ammo,
        channel_map=channel_map,
        suppressor_sku=args.sku,
        test_platform=args.platform,
        cluster_index=args.cluster,
        shot_order=args.shot_order,
        wind_speed=args.wind_speed,
        temp=args.temp,
        relative_humidity=args.rh,
    )

    shot = marked.shot
    role = shot.role.label if shot.role else "—"
    print(f"Marked shot #{shot.id}  ({Path(shot.source_file).name})")
    print(f"  combination : #{marked.combination.id}  {marked.combination.label}")
    print(f"  batch       : #{marked.batch.id}  {marked.batch.title}")
    print(f"  cluster     : #{marked.cluster.id}  {marked.cluster.label}")
    print(f"  shot        : order {shot.shot_order}  ({role})")
    if shot.captured_at:
        print(f"  fired       : {shot.captured_at}")
    print("  status      : idle — bring it forward with `sma include shot "
          f"{shot.id}` to feed the batch average")
    for position in (MicPosition.ML, MicPosition.SE):
        result = marked.metrics.get(position)
        if result is None:
            continue
        print(
            f"  {position.value}: peakPa {result.peak_pa:8.2f} Pa   "
            f"peak {result.peak_db:7.2f} dB   "
            f"peakA {result.peak_dba:7.2f} dBA   "
            f"impulse {result.impulse_pa_ms:7.2f} Pa*ms ({result.peak_impulse_db:6.2f} dB*ms)   "
            f"Leq10ms {result.leq10ms_db:7.2f} dBA   "
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
            keys = " / ".join(
                str(k) for k in (s.suppressor_sku, s.test_platform, s.cluster_index, s.shot_order)
            )
            print(f"  #{s.id}  {Path(s.source_file).name}  [{keys}]")
        return 0

    if args.target == "combinations":
        combos = repo.all_combinations()
        if not combos:
            print("No combinations.")
            return 0
        print(f"Combinations ({len(combos)}):")
        for c in combos:
            n = len(repo.batches_for_combination(c.id))
            print(f"  #{c.id}  {c.label}  ({n} batch(es))")
        return 0

    if args.target == "batches":
        batches = (
            repo.batches_for_combination(args.combination)
            if args.combination is not None
            else repo.all_batches()
        )
        if not batches:
            print("No batches.")
            return 0
        print(f"Batches ({len(batches)}):")
        inclusion = InclusionService(repo)
        for b in batches:
            combo = repo.get_combination(b.combination_id)
            state = "closed" if b.closed else "open"
            n = repo.count_shots_in_batch(b.id)
            print(
                f"  #{b.id}  {combo.label if combo else '?'}  {b.title}  "
                f"[{state}]  {n} shot(s)  {inclusion.status(b.id).summary()}"
            )
        return 0

    # target == "clusters"
    if args.batch is None:
        print("`list clusters` requires --batch <id>.", file=sys.stderr)
        return 2
    if repo.get_batch(args.batch) is None:
        raise LookupError(f"No batch with id {args.batch}")
    clusters = repo.clusters_for_batch(args.batch)
    if not clusters:
        print(f"Batch #{args.batch} has no clusters.")
        return 0
    print(f"Clusters in batch #{args.batch} ({len(clusters)}):")
    for c in clusters:
        shots = repo.shots_by_cluster(c.id)
        n_in = sum(1 for s in shots if s.included)
        print(f"  #{c.id}  {c.label}  ({len(shots)} shot(s), {n_in} included)")
    return 0


def _cmd_bank(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    """The data-bank view: every cluster and shot in a batch, included or idle."""
    batch = repo.get_batch(args.batch)
    if batch is None:
        raise LookupError(f"No batch with id {args.batch}")
    combo = repo.get_combination(batch.combination_id)
    state = "closed" if batch.closed else "open"
    print(
        f"Data bank — batch #{batch.id}  {combo.label if combo else '?'}  "
        f"{batch.title}  [{state}]"
    )
    clusters = repo.clusters_for_batch(batch.id)
    if not clusters:
        print("  (no clusters)")
        return 0
    for cluster in clusters:
        shots = repo.shots_by_cluster(cluster.id)
        print(f"  {cluster.label}  (#{cluster.id}, {len(shots)} shot(s))")
        for s in shots:
            flag = "[x]" if s.included else "[ ]"
            role = s.role.label if s.role else "—"
            reason = f"  — {s.exclusion_reason}" if s.exclusion_reason else ""
            print(
                f"    {flag} #{s.id}  order {s.shot_order}  {role:8s}"
                f"  {Path(s.source_file).name}{reason}"
            )
    print(f"  {InclusionService(repo).status(batch.id).summary()}")
    return 0


def _cmd_include(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    return _set_inclusion(args, repo, included=True)


def _cmd_exclude(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    return _set_inclusion(args, repo, included=False)


def _set_inclusion(
    args: argparse.Namespace, repo: WorkflowRepository, *, included: bool
) -> int:
    """Shared body of ``include`` / ``exclude``: flip the flag, then report progress."""
    svc = InclusionService(repo)
    reason = getattr(args, "reason", None)
    verb = "Included" if included else "Idled"

    if args.target == "shot":
        svc.include_shot(args.id, included, reason=reason)
        shot = repo.get_shot(args.id)
        batch_id = _batch_of_shot(repo, shot)
        print(f"{verb} shot #{args.id} ({Path(shot.source_file).name}).")
    else:  # target == "cluster"
        n = svc.include_cluster(args.id, included, reason=reason)
        cluster = repo.get_cluster(args.id)
        batch_id = cluster.batch_id
        print(f"{verb} {n} shot(s) in cluster #{args.id} ({cluster.label}).")

    if batch_id is not None:
        print(f"  batch #{batch_id}: {svc.status(batch_id).summary()}")
    return 0


#: ``sma batch`` field name -> (argparse dest, :class:`~.models.Batch` attribute).
#: Drives both the read-modify-write merge and ``--clear``.
_BATCH_FIELDS = {
    "label": ("label", "label"),
    "date": ("date", "session_date"),
    "wind-speed": ("wind_speed", "wind_speed"),
    "temp": ("temp", "temp"),
    "rh": ("rh", "relative_humidity"),
    "notes": ("notes", "notes"),
}


def _cmd_batch(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    """Update a batch's session metadata; a flag left off keeps its stored value."""
    batch = repo.get_batch(args.batch_id)
    if batch is None:
        raise LookupError(f"No batch with id {args.batch_id}")

    # `repo.update_batch` is a full-form write — an unset field is blanked —
    # because the GUI dialog always supplies the complete intended state. Here an
    # absent flag means "leave it alone", so fill the gaps from the stored row and
    # blank only what `--clear` names.
    cleared = set(args.clear or ())
    values: dict[str, object] = {}
    for field, (dest, attr) in _BATCH_FIELDS.items():
        given = getattr(args, dest)
        if field in cleared:
            if given is not None:
                print(
                    f"--{field} and --clear {field} contradict each other.",
                    file=sys.stderr,
                )
                return 2
            values[attr] = None
        else:
            values[attr] = given if given is not None else getattr(batch, attr)

    repo.update_batch(args.batch_id, **values)
    batch = repo.get_batch(args.batch_id)
    print(f"Updated batch #{batch.id}  {batch.title}")
    print(f"  typical weather : {batch.weather_summary or '(none)'}")
    print(f"  notes           : {batch.notes or '(none)'}")
    return 0


def _cmd_close_batch(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    ClusteringService(repo).close_batch(args.batch_id)
    print(f"Closed batch #{args.batch_id}.")
    return 0


def _cmd_report(args: argparse.Namespace, repo: WorkflowRepository) -> int:
    agg = AggregationService(repo)
    if args.batch is not None:
        _print_batch_averages(agg.batch_averages(args.batch))
        return 0

    report = agg.combination_report(args.combination)
    print(f"Report — combination #{report.combination.id}  {report.combination.label}")
    if not report.batches:
        print("  (no batches)")
        return 0
    for batch_avg in report.batches:
        _print_batch_averages(batch_avg, indent="  ")
    return 0


def _fmt(value: float | None, width: int = 7) -> str:
    """Fixed-width metric value; a blanked (None) average renders as an em-dash.

    Batch averages come back ``None`` for any metric whose column is all-NULL over
    the slot — e.g. after a migration blanks a legacy database's metrics but
    before the shots are re-marked. Mirrors the GUI's ``_format_metric`` so the CLI
    report degrades to "—" instead of crashing on ``f"{None:.2f}"``.
    """
    return f"{'—':>{width}}" if value is None else f"{value:{width}.2f}"


def _print_batch_averages(batch_avg, *, indent: str = "") -> None:
    """Render one batch's four position x role output slots."""
    b = batch_avg.batch
    combo = batch_avg.combination
    state = "closed" if b.closed else "open"
    print(
        f"{indent}Batch #{b.id}  {combo.label if combo else '?'}  {b.title}  [{state}]"
    )
    print(
        f"{indent}  {batch_avg.n_included} of {batch_avg.n_shots} shot(s) included   "
        f"{batch_avg.status.summary()}"
    )
    if not batch_avg.averages:
        print(f"{indent}  (nothing brought forward yet)")
        return
    for position, role in AVERAGE_SLOTS:
        avg = batch_avg.averages.get((position, role))
        slot = f"{position.label} · {role.label}"
        if avg is None:
            print(f"{indent}  {slot:26s} (none included)")
            continue
        print(
            f"{indent}  {slot:26s} (n={avg['n']}): "
            f"peakPa {_fmt(avg['peak_pa'], 8)} Pa   "
            f"peak {_fmt(avg['peak_db'])} dB   "
            f"peakA {_fmt(avg['peak_dba'])} dBA   "
            f"impulse {_fmt(avg['impulse_pa_ms'])} Pa*ms ({_fmt(avg['peak_impulse_db'], 6)} dB*ms)   "
            f"Leq10ms {_fmt(avg['leq10ms_db'])} dBA   "
            f"LIAeq {_fmt(avg['liaeq_100ms_db'])} dBA"
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
    print(f"Targets       : FRP {config.TARGET_FRP_SHOTS}, regular {config.TARGET_REGULAR_SHOTS}")
    return 0


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _batch_of_shot(repo: WorkflowRepository, shot) -> int | None:
    """The batch id a shot sits under, or ``None`` if it is not yet placed."""
    if shot is None or shot.cluster_id is None:
        return None
    cluster = repo.get_cluster(shot.cluster_id)
    return cluster.batch_id if cluster else None


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sma",
        description=(
            "Sound Metric App workflow: ingest, mark, bring forward, and report shots."
        ),
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
    p_mark.add_argument("--ammo", required=True, help="ammunition / load identifier")
    p_mark.add_argument(
        "--se",
        metavar="CHANNEL",
        help="raw channel to tag as shooter's ear (default: auto-tag AI 2)",
    )
    p_mark.add_argument(
        "--ml",
        metavar="CHANNEL",
        help="raw channel to tag as muzzle left (default: auto-tag AI 1)",
    )
    p_mark.add_argument("--sku", help="override the provisional suppressor SKU")
    p_mark.add_argument("--platform", help="override the provisional test platform")
    p_mark.add_argument("--cluster", type=int, help="override the cluster index from the filename")
    p_mark.add_argument(
        "--shot-order",
        type=int,
        dest="shot_order",
        help="position within the cluster; 0 is the FRP",
    )
    p_mark.add_argument("--wind-speed", type=float, dest="wind_speed", help="wind speed (mph)")
    p_mark.add_argument("--temp", type=float, help="ambient temperature (deg F)")
    p_mark.add_argument("--rh", type=float, help="relative humidity (percent)")
    add_db(p_mark)
    p_mark.set_defaults(func=_cmd_mark)

    # list
    p_list = sub.add_parser("list", help="list unmarked shots, combinations, batches, or clusters")
    p_list.add_argument("target", choices=["unmarked", "combinations", "batches", "clusters"])
    p_list.add_argument("--combination", type=int, help="narrow `list batches` to one combination")
    p_list.add_argument("--batch", type=int, help="batch id (required for `list clusters`)")
    add_db(p_list)
    p_list.set_defaults(func=_cmd_list)

    # bank
    p_bank = sub.add_parser("bank", help="data-bank view of a batch: every cluster and shot")
    p_bank.add_argument("batch", type=int, help="batch id")
    add_db(p_bank)
    p_bank.set_defaults(func=_cmd_bank)

    # include / exclude
    for name, handler, helptext in (
        ("include", _cmd_include, "bring a shot or cluster forward into the batch average"),
        ("exclude", _cmd_exclude, "return a shot or cluster to idle"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("target", choices=["shot", "cluster"])
        p.add_argument("id", type=int, help="shot or cluster id")
        if name == "exclude":
            p.add_argument("--reason", help="why it is being left out (e.g. 'high winds')")
        add_db(p)
        p.set_defaults(func=handler)

    # batch
    p_batch = sub.add_parser(
        "batch",
        help="update a batch's session metadata",
        description=(
            "Update a batch's session metadata. Only the fields you pass change; "
            "everything else keeps its stored value. Use --clear to blank a field."
        ),
    )
    p_batch.add_argument("batch_id", type=int)
    p_batch.add_argument("--label", help="name for the session")
    p_batch.add_argument("--date", help="session date (ISO-8601, e.g. 2026-07-22)")
    p_batch.add_argument(
        "--wind-speed", type=float, dest="wind_speed", help="typical wind speed (mph)"
    )
    p_batch.add_argument("--temp", type=float, help="typical temperature (deg F)")
    p_batch.add_argument("--rh", type=float, help="typical relative humidity (percent)")
    p_batch.add_argument("--notes", help="free-form session notes")
    p_batch.add_argument(
        "--clear",
        action="append",
        choices=sorted(_BATCH_FIELDS),
        metavar="FIELD",
        help=(
            "blank a stored field; repeatable. One of: "
            + ", ".join(sorted(_BATCH_FIELDS))
        ),
    )
    add_db(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    # close-batch
    p_close = sub.add_parser("close-batch", help="close a batch")
    p_close.add_argument("batch_id", type=int)
    add_db(p_close)
    p_close.set_defaults(func=_cmd_close_batch)

    # report
    p_report = sub.add_parser("report", help="batch-average view: four position x role slots")
    target = p_report.add_mutually_exclusive_group(required=True)
    target.add_argument("--batch", type=int, help="report a single batch")
    target.add_argument(
        "--combination", type=int, help="report every batch in this combination"
    )
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
