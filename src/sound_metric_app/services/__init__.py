"""Headless workflow services orchestrating ingestion, marking, clustering, and
aggregation over the storage/ingestion/DSP layers.

These are pure logic services with no UI, reused by both the CLI and the GUI:

* :class:`IngestionService` — input folder -> Unmarked Data Sets.
* :class:`MarkingService` — annotate a shot, tag SE/MR, compute + store metrics.
* :class:`ClusteringService` — assign shots to batch/group; batch lifecycle.
* :class:`AggregationService` — per-group and per-batch SE/MR averages.
"""

from .aggregation_service import AggregationService, BatchReport, GroupAverages
from .clustering_service import ClosedBatchError, ClusteringService, ResolvedGroup
from .ingestion_service import IngestionService, IngestReport
from .marking_service import MarkedShot, MarkingService

__all__ = [
    "IngestionService",
    "IngestReport",
    "MarkingService",
    "MarkedShot",
    "ClusteringService",
    "ResolvedGroup",
    "ClosedBatchError",
    "AggregationService",
    "GroupAverages",
    "BatchReport",
]
