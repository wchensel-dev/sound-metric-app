"""Marking service (BUILD_PLAN Task 5).

Turns an Unmarked Data Set into a marked shot: applies the user's metadata,
tags which raw channel is SE and which is MR, runs the DSP
:class:`~sound_metric_app.dsp.MetricsProcessor` per tagged channel, and persists
one per-mic ``channel_metrics`` row for each.

Batch/group placement is resolved automatically through the
:class:`~sound_metric_app.services.clustering_service.ClusteringService`, so a
caller marks a shot with plain test context (SKU, platform, ammo) and never has
to hand-manage batch or group ids. The capture reader and DSP processor are
injected for unit-testing without a real ``.dxd`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..dsp import MetricsProcessor
from ..ingestion import read_capture, tag_channels
from ..models import Batch, Frame, MetricResult, MicPosition, Shot
from ..storage import WorkflowRepository
from .clustering_service import ClusteringService

#: Signature of the capture reader: path -> one Frame per mic channel.
CaptureReader = Callable[[str], list[Frame]]


@dataclass
class MarkedShot:
    """Result of :meth:`MarkingService.mark`: the marked shot and what it produced."""

    shot: Shot
    batch: Batch
    group_id: int
    #: Computed metrics per tagged mic position (SE and/or MR).
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
        channel_map: dict[str, MicPosition],
        suppressor_sku: str | None = None,
        test_platform: str | None = None,
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
            Ammunition / load identifier — part of the group key, not in the
            filename, so always required.
        channel_map:
            Maps a raw channel name (as it appears in the capture) to the
            :class:`~sound_metric_app.models.MicPosition` it represents. Tag one
            channel (single-mic shot) or two; each position at most once.
        suppressor_sku, test_platform:
            Batch/group keys. Default to the shot's provisional values parsed
            from its filename at ingest; pass them to override a mis-named file.
        shot_order, wind_speed, temp, relative_humidity:
            Per-shot fields. By default each is preserved if omitted (the store
            only overwrites values explicitly supplied). Pass
            ``replace_optional=True`` for a full-form edit, where these are
            written exactly and an omitted (``None``) field blanks the stored
            value instead of preserving it.
        replace_optional:
            See ``shot_order`` et al. above. Leave ``False`` for a partial
            re-mark (e.g. the CLI); set ``True`` when the caller supplies the
            complete intended state, such as the GUI edit dialog.

        Returns
        -------
        MarkedShot
            The reloaded marked shot, its (open) batch, group id, and the metrics
            computed per tagged mic.

        Raises
        ------
        LookupError
            If ``shot_id`` matches no shot.
        ValueError
            If SKU/platform cannot be determined, or the channel map names a
            channel absent from the capture / reuses a mic position.
        """
        shot = self._repo.get_shot(shot_id)
        if shot is None:
            raise LookupError(f"No shot with id {shot_id}")
        previous_group_id = shot.group_id
        # Remember the group's batch too: pruning an emptied group can leave its
        # batch empty (e.g. re-marking the sole shot out of a closed batch).
        previous_batch_id = None
        if previous_group_id is not None:
            previous_group = self._repo.get_group(previous_group_id)
            previous_batch_id = previous_group.batch_id if previous_group else None

        sku = suppressor_sku or shot.suppressor_sku
        platform = test_platform or shot.test_platform
        if not sku or not platform:
            raise ValueError(
                "Suppressor SKU and Test Platform are required to mark a shot; "
                "the shot has no provisional value from its filename, so pass them explicitly."
            )

        # Read the capture and validate the channel tagging before touching the DB.
        frames = self._reader(shot.source_file)
        tagged = tag_channels(frames, channel_map)
        # Every frame from one capture shares the file's start-store time; record
        # it as the shot's fired-at timestamp (None if the file carried none).
        captured_at = _capture_timestamp(frames)
        se_channel = _channel_for(channel_map, MicPosition.SE)
        mr_channel = _channel_for(channel_map, MicPosition.MR)

        # Resolve (and open, if needed) the batch/group for this test context.
        resolved = self._clustering.resolve_group(sku, platform, ammo)

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
                group_id=resolved.group_id,
                ammo=ammo,
                shot_order=shot_order,
                wind_speed=wind_speed,
                temp=temp,
                relative_humidity=relative_humidity,
                captured_at=captured_at,
                replace_optional=replace_optional,
            )
            # channel_map fully defines this shot's tagging, so set the tags
            # definitively (clearing a mic dropped on re-mark) rather than letting
            # mark_shot's preserve-on-None semantics keep a stale tag.
            self._repo.set_shot_channels(shot_id, se_channel=se_channel, mr_channel=mr_channel)
            for position, result in metrics.items():
                self._repo.save_channel_metric(shot_id, position, result)
            # Re-marking may drop a previously tagged mic; remove its now-stale
            # metric row so aggregation stops averaging orphaned data.
            self._repo.delete_channel_metrics_except(shot_id, metrics)
            # Re-marking into a different group may leave the former group empty;
            # drop it so the batch tree stays uncluttered and its name is re-usable.
            # If that group was its batch's last, the batch is now an empty shell
            # (the closed-batch re-mark flow), so prune it too.
            if previous_group_id is not None and previous_group_id != resolved.group_id:
                if (
                    self._repo.delete_group_if_empty(previous_group_id)
                    and previous_batch_id is not None
                ):
                    self._repo.delete_batch_if_empty(previous_batch_id)

        return MarkedShot(
            shot=self._repo.get_shot(shot_id),
            batch=resolved.batch,
            group_id=resolved.group_id,
            metrics=metrics,
        )


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
