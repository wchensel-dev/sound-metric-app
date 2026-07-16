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
            Per-shot fields; each is preserved if omitted (the store only
            overwrites values explicitly supplied).

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
        se_channel = _channel_for(channel_map, MicPosition.SE)
        mr_channel = _channel_for(channel_map, MicPosition.MR)

        # Resolve (and open, if needed) the batch/group for this test context.
        resolved = self._clustering.resolve_group(sku, platform, ammo)

        self._repo.mark_shot(
            shot_id,
            group_id=resolved.group_id,
            ammo=ammo,
            shot_order=shot_order,
            wind_speed=wind_speed,
            temp=temp,
            relative_humidity=relative_humidity,
            se_channel=se_channel,
            mr_channel=mr_channel,
        )

        metrics: dict[MicPosition, MetricResult] = {}
        for mic in tagged:
            result = self._processor.process(mic.frame)
            self._repo.save_channel_metric(shot_id, mic.position, result)
            metrics[mic.position] = result

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
