"""Clustering & batch lifecycle (BUILD_PLAN Task 6).

Resolves a shot's human test context — Suppressor SKU, Test Platform, Ammo, and
the cluster index carried by its filename — into the persisted rows it belongs
to, and owns the batch lifecycle:

* A **combination** is one SKU + Platform + Ammo path, created on demand. It is
  the "test combination" batches hang from.
* A **batch** is one test *session* under a combination. There is at most one
  *open* batch per combination; :meth:`ClusteringService.resolve_placement`
  reuses it or opens one if none is.
* A **cluster** is one string of fire within a batch, addressed by the 1-based
  index in the capture's filename, created on demand.
* **Closing** a batch is an explicit user action. Once closed, the next shot for
  that combination starts a *new* batch rather than reopening the old one —
  which falls out naturally because a closed batch is no longer the
  combination's open batch, so the next resolve creates a fresh one.

Note the cluster index is scoped to its *batch*, not globally: cluster 1 of one
session and cluster 1 of the next are different strings of fire, and land in
different batches because the sessions are separated by a close.

The closed-batch guard lives here (:meth:`ensure_open`): callers that hold a
specific batch id — rather than going through :meth:`resolve_placement` — can
reject marking into a batch the user has already closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Batch, Cluster, Combination
from ..storage import WorkflowRepository


class ClosedBatchError(RuntimeError):
    """Raised when an operation would add shots to a closed batch."""


@dataclass
class ResolvedPlacement:
    """Where a shot's test context landed: its combination, open batch, and cluster."""

    combination: Combination
    batch: Batch
    cluster: Cluster

    @property
    def cluster_id(self) -> int:
        return self.cluster.id


class ClusteringService:
    """Assign shots into the containment tree and manage the batch lifecycle."""

    def __init__(self, repo: WorkflowRepository):
        self._repo = repo

    def resolve_placement(
        self,
        suppressor_sku: str,
        test_platform: str,
        ammo: str,
        cluster_index: int,
    ) -> ResolvedPlacement:
        """Return the combination, open batch, and cluster for a shot, creating as needed.

        Upserts the (SKU, platform, ammo) combination, finds that combination's
        open batch (opening a new session if none is), then upserts the string of
        fire at ``cluster_index`` within it. Because a closed batch is never an
        "open batch", the first shot for a combination after its batch was closed
        transparently opens a new one.

        Raises ``ValueError`` if ``cluster_index`` is below 1 — cluster indices
        are 1-based, matching the filename convention.
        """
        if cluster_index < 1:
            raise ValueError(
                f"Cluster index must be 1 or greater, got {cluster_index}."
            )
        combination_id = self._repo.upsert_combination(suppressor_sku, test_platform, ammo)
        combination = self._repo.get_combination(combination_id)
        batch = self._open_batch(combination_id)
        cluster_id = self._repo.upsert_cluster(batch.id, cluster_index)
        return ResolvedPlacement(
            combination=combination,
            batch=batch,
            cluster=self._repo.get_cluster(cluster_id),
        )

    def open_batch_for(self, suppressor_sku: str, test_platform: str, ammo: str) -> Batch:
        """The combination's current open session, creating one if none is open."""
        combination_id = self._repo.upsert_combination(suppressor_sku, test_platform, ammo)
        return self._open_batch(combination_id)

    def close_batch(self, batch_id: int) -> None:
        """Close a batch (explicit user action). Idempotent; unknown id raises ``LookupError``."""
        self._repo.close_batch(batch_id)

    def ensure_open(self, batch_id: int) -> Batch:
        """Return the batch if open; raise if it is closed or unknown.

        The guard for callers that mark into an explicit batch instead of going
        through :meth:`resolve_placement`.
        """
        batch = self._repo.get_batch(batch_id)
        if batch is None:
            raise LookupError(f"No batch with id {batch_id}")
        if batch.closed:
            raise ClosedBatchError(
                f"Batch #{batch_id} is closed; start a new session instead."
            )
        return batch

    # ---- internal ------------------------------------------------------- #

    def _open_batch(self, combination_id: int) -> Batch:
        batch = self._repo.open_batch_for_combination(combination_id)
        if batch is None:
            batch_id = self._repo.create_batch(combination_id)
            batch = self._repo.get_batch(batch_id)
            if batch is None:
                raise LookupError(
                    f"Batch {batch_id} vanished immediately after creation "
                    f"(combination {combination_id})"
                )
        return batch
