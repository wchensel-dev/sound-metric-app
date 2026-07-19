"""Workflow view-model: the service calls the GUI makes, with no Qt dependency.

The GUI (``main_window``) is a thin front-end over these methods, exactly as the
``sma`` CLI is a thin front-end over the same Phase B services. Keeping the logic
here — free of any ``QWidget``/``QApplication`` — means the whole ingest -> mark
-> close -> report flow is testable without a live GUI (see ``tests/test_controller``).

Connection model
----------------
Every method opens a short-lived :class:`WorkflowRepository` on the configured
``db_path`` and closes it before returning, mirroring how each CLI subcommand
opens its own repo. Because no ``sqlite3`` connection outlives a call, these
methods are safe to invoke from a Qt worker thread (SQLite connections are bound
to the thread that created them); the returned objects are detached dataclasses,
safe to hand back to the UI thread.

The channel and capture readers are injected so tests can substitute fakes
without a real ``.dxd`` file, matching ``workflow_cli``'s module-level readers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .. import config
from ..dsp import SMOOTHING_INSTANT, MetricTrace, build_metric_trace
from ..ingestion import ChannelInfo, list_channels, read_capture
from ..models import Batch, Group, MicPosition, Shot
from ..services import (
    AggregationService,
    BatchReport,
    ClosedBatchError,
    ClusteringService,
    GroupAverages,
    IngestionService,
    IngestReport,
    MarkedShot,
    MarkingService,
)
from ..storage import WorkflowRepository

#: Exceptions the services raise for bad user input / state. The GUI turns these
#: (and any other error) into a dialog; kept here so the widget layer and any
#: test can share the same catch set the CLI uses (``workflow_cli._USER_ERRORS``).
USER_ERRORS = (
    LookupError,
    ValueError,
    ClosedBatchError,
    FileNotFoundError,
    NotADirectoryError,
)


@dataclass
class GroupNode:
    """A group with its shots already materialized."""

    group: Group
    shots: list[Shot]


@dataclass
class BatchNode:
    """A batch with its groups (each carrying its shots)."""

    batch: Batch
    groups: list[GroupNode]


class WorkflowController:
    """Headless driver for the GUI: ingest, mark, close, and report."""

    def __init__(
        self,
        db_path: str | Path = config.DEFAULT_DB_PATH,
        *,
        channel_reader=list_channels,
        capture_reader=read_capture,
    ):
        self._db_path = str(db_path)
        self._channel_reader = channel_reader
        self._capture_reader = capture_reader
        #: Cached ammo presets. The GUI re-reads these on every view refresh
        #: (each tab switch / mark / ingest / close), but the list only changes
        #: through :meth:`set_ammo_definitions`, so we parse the settings file
        #: once and invalidate on write instead of hitting disk each refresh.
        self._ammo_definitions_cache: list[str] | None = None

    @contextmanager
    def _repo(self) -> Iterator[WorkflowRepository]:
        with WorkflowRepository(self._db_path) as repo:
            yield repo

    # ---- config / input folder ----------------------------------------- #

    def input_folder(self) -> str | None:
        """The configured default input folder, or ``None`` if unset."""
        return config.get_input_folder()

    def set_input_folder(self, folder: str | Path) -> Path:
        """Persist the default input folder and return the resolved path."""
        return config.set_input_folder(folder)

    def ammo_definitions(self) -> list[str]:
        """The ammo presets offered when marking a shot (built-in defaults if unset).

        Cached after the first read (see ``_ammo_definitions_cache``): a malformed
        settings file still raises ``ValueError`` on every call until fixed, since
        only a successful read is cached. Returns a fresh copy so callers can't
        mutate the cached list.
        """
        if self._ammo_definitions_cache is None:
            self._ammo_definitions_cache = config.get_ammo_definitions()
        return list(self._ammo_definitions_cache)

    def set_ammo_definitions(self, definitions: list[str]) -> list[str]:
        """Persist the ammo presets (normalized), refresh the cache, and return them."""
        stored = config.set_ammo_definitions(definitions)
        self._ammo_definitions_cache = list(stored)
        return stored

    # ---- ingest --------------------------------------------------------- #

    def ingest(self, folder: str | Path | None = None, *, validate: bool = True) -> IngestReport:
        """Scan ``folder`` (or the configured input folder) for new captures.

        Raises ``ValueError`` if no folder is given and none is configured, so
        the caller can prompt the user to pick one.
        """
        folder = folder or config.get_input_folder()
        if not folder:
            raise ValueError(
                "No input folder configured. Choose one before ingesting."
            )
        with self._repo() as repo:
            return IngestionService(repo, reader=self._channel_reader).scan(
                folder, validate=validate
            )

    def unmarked_shots(self) -> list[Shot]:
        with self._repo() as repo:
            return repo.unmarked_shots()

    def get_shot(self, shot_id: int) -> Shot | None:
        with self._repo() as repo:
            return repo.get_shot(shot_id)

    # ---- mark ----------------------------------------------------------- #

    def channels_for(self, source_file: str) -> list[ChannelInfo]:
        """Raw channels in a capture, to offer as SE/MR choices in the mark form."""
        return self._channel_reader(source_file)

    def mark(
        self,
        shot_id: int,
        *,
        ammo: str,
        channel_map: dict[str, MicPosition],
        suppressor_sku: str | None = None,
        test_platform: str | None = None,
        shot_order: int | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        replace_optional: bool = False,
    ) -> MarkedShot:
        """Annotate a shot, tag SE/MR, and compute + store its metrics.

        ``replace_optional=True`` writes the optional per-shot fields (shot order
        and environment) exactly, so a cleared field blanks the stored value —
        used by the full-form edit dialog. Leave ``False`` for a partial re-mark.
        """
        with self._repo() as repo:
            svc = MarkingService(repo, ClusteringService(repo), reader=self._capture_reader)
            return svc.mark(
                shot_id,
                ammo=ammo,
                channel_map=channel_map,
                suppressor_sku=suppressor_sku,
                test_platform=test_platform,
                shot_order=shot_order,
                wind_speed=wind_speed,
                temp=temp,
                relative_humidity=relative_humidity,
                replace_optional=replace_optional,
            )

    # ---- batches / groups / shots (read) -------------------------------- #

    def batches(self) -> list[Batch]:
        with self._repo() as repo:
            return repo.all_batches()

    def get_batch(self, batch_id: int) -> Batch | None:
        with self._repo() as repo:
            return repo.get_batch(batch_id)

    def rename_batch(self, batch_id: int, sku: str) -> None:
        """Correct a batch's SKU in place, keeping all its groups and shots.

        Guards the "at most one open batch per SKU" invariant: renaming an *open*
        batch onto a SKU that already has a different open batch is rejected, so
        future marking never has two open batches to choose between. Renaming a
        closed batch, or renaming to its own current SKU, is always allowed.

        Raises ``ValueError`` on an empty SKU or such a collision, ``LookupError``
        if the batch id is unknown.
        """
        sku = sku.strip()
        if not sku:
            raise ValueError("SKU cannot be empty.")
        with self._repo() as repo:
            batch = repo.get_batch(batch_id)
            if batch is None:
                raise LookupError(f"No batch with id {batch_id}")
            if not batch.closed:
                other = repo.open_batch_for_sku(sku)
                if other is not None and other.id != batch_id:
                    raise ValueError(
                        f"SKU {sku!r} already has an open batch (#{other.id}). "
                        "Close it first, or pick a different SKU."
                    )
            repo.rename_batch_sku(batch_id, sku)

    def groups_for_batch(self, batch_id: int) -> list[Group]:
        with self._repo() as repo:
            return repo.groups_for_batch(batch_id)

    def shots_by_group(self, group_id: int) -> list[Shot]:
        with self._repo() as repo:
            return repo.shots_by_group(group_id)

    def sweep_empty(self) -> None:
        """Drop any shot-less groups and the batches their removal leaves empty.

        An explicit maintenance pass the GUI runs on refresh, kept out of the
        read accessors so loading the tree never mutates the DB. Empty groups are
        swept first, then any batch left group-less by that sweep — cleaning up
        containers left behind by an edit or by data predating per-re-mark
        cleanup (see :meth:`WorkflowRepository.delete_empty_groups` and
        :meth:`WorkflowRepository.delete_empty_batches`).
        """
        with self._repo() as repo:
            repo.delete_empty_groups()
            repo.delete_empty_batches()

    def batch_tree(self) -> list[BatchNode]:
        """The whole batch -> group -> shot tree, over a single connection.

        The GUI's batch tree renders all three levels at once; loading them here
        opens one repo instead of a connection per batch/group, and shot counts
        come from ``len(node.shots)`` rather than a separate COUNT query. This is
        a pure read; call :meth:`sweep_empty` first to prune empty containers.
        """
        with self._repo() as repo:
            return [
                BatchNode(
                    batch=batch,
                    groups=[
                        GroupNode(group=group, shots=repo.shots_by_group(group.id))
                        for group in repo.groups_for_batch(batch.id)
                    ],
                )
                for batch in repo.all_batches()
            ]

    # ---- close ---------------------------------------------------------- #

    def close_batch(self, batch_id: int) -> None:
        with self._repo() as repo:
            ClusteringService(repo).close_batch(batch_id)

    # ---- report --------------------------------------------------------- #

    def batch_report(self, batch_id: int) -> BatchReport:
        with self._repo() as repo:
            return AggregationService(repo).batch_report(batch_id)

    def group_averages(self, group_id: int) -> GroupAverages:
        with self._repo() as repo:
            return AggregationService(repo).group_averages(group_id)

    # ---- report graph --------------------------------------------------- #

    def metric_trace(
        self,
        shot_id: int,
        position: MicPosition,
        metric_key: str,
        smoothing: str = SMOOTHING_INSTANT,
    ) -> MetricTrace:
        """The time-series a Report graph draws for one shot/mic/metric.

        Re-reads the shot's capture (the raw samples are not stored, only the
        scalar metrics are) and builds the metric's curve. ``smoothing`` picks
        how the SPL-over-time curve is drawn (instantaneous vs Fast/Slow
        time-weighted); see :func:`~sound_metric_app.dsp.build_metric_trace`. The
        DB read and the capture read + DSP are both done here so the whole thing
        can run on a worker thread. Raises ``LookupError`` if the shot is gone and
        ``ValueError`` if the mic position has no channel or the channel is
        missing from the capture.
        """
        with self._repo() as repo:
            shot = repo.get_shot(shot_id)
        if shot is None:
            raise LookupError(f"No shot with id {shot_id}")

        channel = shot.se_channel if position is MicPosition.SE else shot.mr_channel
        if not channel:
            raise ValueError(f"Shot #{shot_id} has no {position.value} channel to graph.")

        frames = self._capture_reader(shot.source_file)
        frame = next((f for f in frames if f.channel == channel), None)
        if frame is None:
            raise ValueError(
                f"Channel {channel!r} not found in {Path(shot.source_file).name}."
            )
        return build_metric_trace(frame, metric_key, smoothing)
