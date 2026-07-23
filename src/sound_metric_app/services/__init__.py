"""Headless workflow services orchestrating ingestion, marking, clustering,
inclusion, and aggregation over the storage/ingestion/DSP layers.

These are pure logic services with no UI, reused by both the CLI and the GUI:

* :class:`IngestionService` — input folder -> Unmarked Data Sets.
* :class:`MarkingService` — annotate a shot, tag ML/SE, compute + store metrics.
* :class:`ClusteringService` — place shots in the combination -> batch -> cluster
  tree; batch lifecycle.
* :class:`InclusionService` — the data-bank -> batch-average gate: bring shots or
  whole clusters forward, and track progress against the 3-FRP / 5-regular
  targets.
* :class:`AggregationService` — per-batch roll-up into the four position x role
  output slots.
"""

from .aggregation_service import (
    AVERAGE_SLOTS,
    AggregationService,
    BatchAverages,
    CombinationReport,
)
from .clustering_service import ClosedBatchError, ClusteringService, ResolvedPlacement
from .inclusion_service import (
    ROLE_TARGETS,
    InclusionService,
    InclusionStatus,
    RoleProgress,
)
from .ingestion_service import IngestionService, IngestReport
from .marking_service import MarkedShot, MarkingService

__all__ = [
    "IngestionService",
    "IngestReport",
    "MarkingService",
    "MarkedShot",
    "ClusteringService",
    "ResolvedPlacement",
    "ClosedBatchError",
    "InclusionService",
    "InclusionStatus",
    "RoleProgress",
    "ROLE_TARGETS",
    "AggregationService",
    "BatchAverages",
    "CombinationReport",
    "AVERAGE_SLOTS",
]
