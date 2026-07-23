"""Marking service (BUILD_PLAN Task 5).

Turns an Unmarked Data Set into a marked shot: applies the user's metadata, tags
which raw channel is ML and which is SE, runs the DSP
:class:`~sound_metric_app.dsp.MetricsProcessor` per tagged channel, and persists
one per-mic ``channel_metrics`` row for each.

Placement in the containment tree is resolved automatically through the
:class:`~sound_metric_app.services.clustering_service.ClusteringService`, so a
caller marks a shot with plain test context (SKU, platform, ammo, cluster index)
and never has to hand-manage combination, batch, or cluster ids.

Channel tagging defaults to the DAQ convention (AI 1 = muzzle left, AI 2 =
shooter's ear) via :func:`~sound_metric_app.ingestion.autotag_map`, and an
explicit ``channel_map`` overrides it for a capture that breaks convention.

Marking never touches a shot's ``included`` flag: a freshly marked shot sits idle
in the data bank until it is explicitly brought forward, and re-marking one that
is already included leaves it included.

The capture reader and DSP processor are injected for unit-testing without a
real ``.dxd`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..dsp import MetricsProcessor
from ..ingestion import autotag_map, read_capture, tag_channels
from ..models import Batch, Cluster, Combination, Frame, MetricResult, MicPosition, Shot
from ..storage import WorkflowRepository
from .clustering_service import ClusteringService

#: Signature of the capture reader: path -> one Frame per mic channel.
CaptureReader = Callable[[str], list[Frame]]


@dataclass
class MarkedShot:
    """Result of :meth:`MarkingService.mark`: the marked shot and where it landed."""

    shot: Shot
    combination: Combination
    batch: Batch
    cluster: Cluster
    #: Computed metrics per tagged mic position (ML and/or SE).
    metrics: dict[MicPosition, MetricResult]


class MarkingService:
    """Annotate an unmarked shot, tag its mics, and compute + store metrics."""

    def __init__(
        self,
        repo: WorkflowRepository,
        clustering: ClusteringService,
        *,
        reader: CaptureReader = read_capture,
        processor: MetricsProcessor | None = None,
    ):
        self._repo = repo
        self._clustering = clustering
        self._reader = reader
        self._processor = processor or MetricsProcessor()

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
        """Mark ``shot_id`` and compute its per-mic metrics.

        Parameters
        ----------
        ammo:
            Ammunition / load identifier — the third leg of the combination key,
            not encoded in the filename, so always required.
        channel_map:
            Maps a raw channel name (as it appears in the capture) to the
            :class:`~sound_metric_app.models.MicPosition` it represents. Leave
            ``None`` to auto-tag from the DAQ convention (AI 1 = ML, AI 2 = SE);
            pass a map to override it for a non-conforming capture. Tag one
            channel (single-mic shot) or two; each position at most once.
        suppressor_sku, test_platform, cluster_index:
            Placement keys. Default to the shot's provisional values parsed from
            its filename at ingest; pass them to override a mis-named file.
        shot_order, wind_speed, temp, relative_humidity:
            Per-shot fields. ``shot_order`` is the position within the cluster
            and determines the shot's FRP / regular role. By default each is
            preserved if omitted (the store only overwrites values explicitly
            supplied). Pass ``replace_optional=True`` for a full-form edit, where
            these are written exactly and an omitted (``None``) field blanks the
            stored value instead of preserving it.
        replace_optional:
            See ``shot_order`` et al. above. Leave ``False`` for a partial
            re-mark (e.g. the CLI); set ``True`` when the caller supplies the
            complete intended state, such as the GUI edit dialog.

        Returns
        -------
        MarkedShot
            The reloaded marked shot, the combination / open batch / cluster it
            landed in, and the metrics computed per tagged mic.

        Raises
        ------
        LookupError
            If ``shot_id`` matches no shot.
        ValueError
            If SKU / platform / cluster index cannot be determined, the capture's
            channels cannot be auto-tagged and no map was given, or the channel
            map names a channel absent from the capture / reuses a mic position.
        """
        shot = self._repo.get_shot(shot_id)
        if shot is None:
            raise LookupError(f"No shot with id {shot_id}")
        previous = self._previous_placement(shot)

        sku = suppressor_sku or shot.suppressor_sku
        platform = test_platform or shot.test_platform
        cluster_no = cluster_index if cluster_index is not None else shot.cluster_index
        if not sku or not platform or cluster_no is None:
            raise ValueError(
                "Suppressor SKU, Test Platform and cluster are required to mark a shot; "
                "the shot has no provisional value from its filename, so pass them explicitly."
            )

        # Read the capture and validate the channel tagging before touching the DB.
        frames = self._reader(shot.source_file)
        mapping = channel_map if channel_map is not None else autotag_map(frames)
        if not mapping:
            available = sorted(f.channel for f in frames)
            raise ValueError(
                f"Could not auto-tag the mics in {shot.source_file!r}: no channel matched "
                f"the AI 1 / AI 2 convention (found {available}). Tag them explicitly."
            )
        tagged = tag_channels(frames, mapping)
        # Every frame from one capture shares the file's start-store time; record
        # it as the shot's fired-at timestamp (None if the file carried none).
        captured_at = _capture_timestamp(frames)
        se_channel = _channel_for(mapping, MicPosition.SE)
        ml_channel = _channel_for(mapping, MicPosition.ML)

        # Resolve (and open, if needed) the combination/batch/cluster for this context.
        placement = self._clustering.resolve_placement(sku, platform, ammo, cluster_no)

        # Run all DSP before any DB write, so a processing failure never leaves
        # a persisted mark behind.
        metrics: dict[MicPosition, MetricResult] = {
            mic.position: self._processor.process(mic.frame) for mic in tagged
        }

        # Marking and metric storage must be atomic: a shot is either fully
        # marked with all its metrics or left untouched. A partial commit would
        # drop the shot from the unmarked list while missing metrics.
        with self._repo.transaction():
            self._repo.mark_shot(
                shot_id,
                cluster_id=placement.cluster_id,
                ammo=ammo,
                cluster_index=cluster_no,
                shot_order=shot_order,
                wind_speed=wind_speed,
                temp=temp,
                relative_humidity=relative_humidity,
                captured_at=captured_at,
                replace_optional=replace_optional,
            )
            # The mapping fully defines this shot's tagging, so set the tags
            # definitively (clearing a mic dropped on re-mark) rather than letting
            # mark_shot's preserve-on-None semantics keep a stale tag.
            self._repo.set_shot_channels(shot_id, se_channel=se_channel, ml_channel=ml_channel)
            for position, result in metrics.items():
                self._repo.save_channel_metric(shot_id, position, result)
            # Re-marking may drop a previously tagged mic; remove its now-stale
            # metric row so aggregation stops averaging orphaned data.
            self._repo.delete_channel_metrics_except(shot_id, metrics)
            self._prune_previous(previous, placement.cluster_id)

        return MarkedShot(
            shot=self._repo.get_shot(shot_id),
            combination=placement.combination,
            batch=placement.batch,
            cluster=placement.cluster,
            metrics=metrics,
        )

    # ---- re-mark cleanup ------------------------------------------------ #

    def _previous_placement(self, shot: Shot) -> tuple[int | None, int | None, int | None]:
        """The (cluster, batch, combination) ids a shot sits in before a re-mark.

        Captured up front so :meth:`_prune_previous` can walk back up the tree
        afterwards: emptying a cluster can leave its batch empty, which can in
        turn leave the combination an empty shell.
        """
        cluster_id = shot.cluster_id
        if cluster_id is None:
            return (None, None, None)
        cluster = self._repo.get_cluster(cluster_id)
        if cluster is None:
            return (cluster_id, None, None)
        batch = self._repo.get_batch(cluster.batch_id)
        return (cluster_id, cluster.batch_id, batch.combination_id if batch else None)

    def _prune_previous(
        self, previous: tuple[int | None, int | None, int | None], new_cluster_id: int
    ) -> None:
        """Drop the containers a re-mark just emptied, walking cluster -> batch -> combination.

        Each step is guarded by a "only if empty" delete, so a container that
        still holds live rows is left alone and the walk stops at the first level
        that survives.
        """
        cluster_id, batch_id, combination_id = previous
        if cluster_id is None or cluster_id == new_cluster_id:
            return
        if not self._repo.delete_cluster_if_empty(cluster_id):
            return
        if batch_id is None or not self._repo.delete_batch_if_empty(batch_id):
            return
        if combination_id is not None:
            self._repo.delete_combination_if_empty(combination_id)


def _channel_for(channel_map: dict[str, MicPosition], position: MicPosition) -> str | None:
    """Raw channel name tagged as ``position`` in the map, or ``None`` if unmapped."""
    for name, pos in channel_map.items():
        if pos is position:
            return name
    return None


def _capture_timestamp(frames: list[Frame]) -> str | None:
    """The capture's fired-at time as an ISO-8601 string, or ``None``.

    Every frame from one file carries the same ``start_store_time``; take it from
    the first frame that has one so a missing timestamp on one channel does not
    lose it. Returns ``None`` when no frame carried a timestamp.
    """
    for frame in frames:
        if frame.timestamp is not None:
            return frame.timestamp.isoformat()
    return None
