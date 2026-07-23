"""Workflow view-model: the service calls the GUI makes, with no Qt dependency.

The GUI (``main_window``) is a thin front-end over these methods, exactly as the
``sma`` CLI is a thin front-end over the same Phase B services. Keeping the logic
here — free of any ``QWidget``/``QApplication`` — means the whole ingest -> mark
-> include -> report flow is testable without a live GUI (see
``tests/test_controller``).

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
from ..ingestion import ChannelInfo, autotag_map, list_channels, read_capture
from ..models import Batch, Cluster, Combination, MicPosition, Shot
from ..services import (
    AggregationService,
    BatchAverages,
    ClosedBatchError,
    ClusteringService,
    CombinationReport,
    InclusionService,
    InclusionStatus,
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
class ClusterNode:
    """A cluster with its shots already materialized, in firing order."""

    cluster: Cluster
    shots: list[Shot]

    @property
    def n_included(self) -> int:
        return sum(1 for s in self.shots if s.included)


@dataclass
class BatchNode:
    """A batch (test session) with its clusters and its inclusion progress."""

    batch: Batch
    clusters: list[ClusterNode]
    status: InclusionStatus

    @property
    def n_shots(self) -> int:
        return sum(len(c.shots) for c in self.clusters)


@dataclass
class CombinationNode:
    """A SKU / Platform / Ammo combination with the batches fired under it."""

    combination: Combination
    batches: list[BatchNode]


class WorkflowController:
    """Headless driver for the GUI: ingest, mark, include, close, and report."""

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
            raise ValueError("No input folder configured. Choose one before ingesting.")
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
        """Raw channels in a capture, to offer as ML/SE choices in the mark form."""
        return self._channel_reader(source_file)

    def suggested_channel_map(self, source_file: str) -> dict[str, MicPosition]:
        """The DAQ-convention channel tagging to pre-fill the mark form with.

        Applies the AI 1 = muzzle left / AI 2 = shooter's ear mapping. A capture
        that does not follow the convention yields a partial or empty map, which
        the form shows as un-tagged dropdowns for the user to set by hand.
        """
        return autotag_map(self._channel_reader(source_file))

    def mark(
        self,
        shot_id: int,
        *,
        ammo: str,
        channel_map: dict[str, MicPosition] | None = None,
        suppressor_sku: str | None = None,
        test_platform: str | None = None,
        cluster_index: int | None = None,
        shot_order: int | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        replace_optional: bool = False,
    ) -> MarkedShot:
        """Annotate a shot, tag ML/SE, and compute + store its metrics.

        ``channel_map=None`` auto-tags from the DAQ convention. The shot lands in
        the data bank idle — marking never sets ``included``.

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
                cluster_index=cluster_index,
                shot_order=shot_order,
                wind_speed=wind_speed,
                temp=temp,
                relative_humidity=relative_humidity,
                replace_optional=replace_optional,
            )

    # ---- tree reads ----------------------------------------------------- #

    def combinations(self) -> list[Combination]:
        with self._repo() as repo:
            return repo.all_combinations()

    def batches(self) -> list[Batch]:
        with self._repo() as repo:
            return repo.all_batches()

    def get_batch(self, batch_id: int) -> Batch | None:
        with self._repo() as repo:
            return repo.get_batch(batch_id)

    def get_cluster(self, cluster_id: int) -> Cluster | None:
        with self._repo() as repo:
            return repo.get_cluster(cluster_id)

    def get_combination(self, combination_id: int) -> Combination | None:
        with self._repo() as repo:
            return repo.get_combination(combination_id)

    def update_batch(
        self,
        batch_id: int,
        *,
        label: str | None = None,
        session_date: str | None = None,
        wind_speed: float | None = None,
        temp: float | None = None,
        relative_humidity: float | None = None,
        notes: str | None = None,
    ) -> None:
        """Write a batch's session metadata. Always a full-form write: unset fields clear.

        Raises ``LookupError`` if the batch id is unknown.
        """
        with self._repo() as repo:
            repo.update_batch(
                batch_id,
                label=(label or "").strip() or None,
                session_date=(session_date or "").strip() or None,
                wind_speed=wind_speed,
                temp=temp,
                relative_humidity=relative_humidity,
                notes=(notes or "").strip() or None,
            )

    def shots_by_cluster(self, cluster_id: int) -> list[Shot]:
        with self._repo() as repo:
            return repo.shots_by_cluster(cluster_id)

    def shots_for_batch(self, batch_id: int, *, included_only: bool = False) -> list[Shot]:
        """Every shot in a batch, in firing order — the flat data-bank read."""
        with self._repo() as repo:
            return repo.shots_for_batch(batch_id, included_only=included_only)

    def sweep_empty(self) -> None:
        """Drop shot-less clusters, then the batches and combinations that leaves empty.

        An explicit maintenance pass the GUI runs on refresh, kept out of the
        read accessors so loading the tree never mutates the DB. The sweep walks
        the tree bottom-up — clusters, then batches, then combinations — so a
        container emptied by the previous step is caught in the same pass.
        """
        with self._repo() as repo:
            repo.delete_empty_clusters()
            repo.delete_empty_batches()
            repo.delete_empty_combinations()

    def data_bank(self) -> list[CombinationNode]:
        """The whole Combination -> Batch -> Cluster -> Shot tree, over one connection.

        This is the **data bank view**: every cluster and every shot, included or
        idle. Nothing is filtered out — a shot left out of an average is still
        part of the complete archive. Loading all four levels here opens one repo
        instead of a connection per node. This is a pure read; call
        :meth:`sweep_empty` first to prune empty containers.
        """
        with self._repo() as repo:
            inclusion = InclusionService(repo)
            return [
                CombinationNode(
                    combination=combination,
                    batches=[
                        BatchNode(
                            batch=batch,
                            clusters=[
                                ClusterNode(
                                    cluster=cluster, shots=repo.shots_by_cluster(cluster.id)
                                )
                                for cluster in repo.clusters_for_batch(batch.id)
                            ],
                            status=inclusion.status(batch.id),
                        )
                        for batch in repo.batches_for_combination(combination.id)
                    ],
                )
                for combination in repo.all_combinations()
            ]

    # ---- inclusion ------------------------------------------------------ #

    def include_shot(
        self, shot_id: int, included: bool = True, *, reason: str | None = None
    ) -> None:
        """Bring one shot forward into its batch average, or return it to idle."""
        with self._repo() as repo:
            InclusionService(repo).include_shot(shot_id, included, reason=reason)

    def include_cluster(
        self, cluster_id: int, included: bool = True, *, reason: str | None = None
    ) -> int:
        """Bring a whole cluster forward, or idle it. Returns how many shots it covered."""
        with self._repo() as repo:
            return InclusionService(repo).include_cluster(cluster_id, included, reason=reason)

    def inclusion_status(self, batch_id: int) -> InclusionStatus:
        """A batch's included counts against the soft 3-FRP / 5-regular targets."""
        with self._repo() as repo:
            return InclusionService(repo).status(batch_id)

    # ---- close ---------------------------------------------------------- #

    def close_batch(self, batch_id: int) -> None:
        with self._repo() as repo:
            ClusteringService(repo).close_batch(batch_id)

    # ---- report --------------------------------------------------------- #

    def batch_averages(self, batch_id: int) -> BatchAverages:
        """The four position x role output slots for one batch."""
        with self._repo() as repo:
            return AggregationService(repo).batch_averages(batch_id)

    def combination_report(self, combination_id: int) -> CombinationReport:
        with self._repo() as repo:
            return AggregationService(repo).combination_report(combination_id)

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

        channel = shot.se_channel if position is MicPosition.SE else shot.ml_channel
        if not channel:
            raise ValueError(f"Shot #{shot_id} has no {position.label} channel to graph.")

        frames = self._capture_reader(shot.source_file)
        frame = next((f for f in frames if f.channel == channel), None)
        if frame is None:
            raise ValueError(f"Channel {channel!r} not found in {Path(shot.source_file).name}.")
        return build_metric_trace(frame, metric_key, smoothing)
