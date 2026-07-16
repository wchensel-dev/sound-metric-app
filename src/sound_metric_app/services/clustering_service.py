"""Clustering & batch lifecycle (BUILD_PLAN Task 6).

Resolves a shot's human test context — Suppressor SKU, Test Platform, Ammo —
into the persisted Batch and Group it belongs to, and owns the batch lifecycle:

* A **batch** is one Suppressor SKU. There is at most one *open* batch per SKU;
  :meth:`ClusteringService.resolve_group` reuses it or creates one if none is open.
* A **group** is a (Test Platform + Ammo) within a batch, created on demand.
* **Closing** a batch is an explicit user action (README §3). Once closed, the
  next shot for that SKU starts a *new* batch rather than reopening the old one —
  which falls out naturally because a closed batch is no longer the SKU's open
  batch, so the next resolve creates a fresh one.

The closed-batch guard lives here (:meth:`ensure_open`): callers that hold a
specific batch id — rather than going through :meth:`resolve_group` — can reject
marking into a batch the user has already closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Batch
from ..storage import WorkflowRepository


class ClosedBatchError(RuntimeError):
    """Raised when an operation would add shots to a closed batch."""


@dataclass
class ResolvedGroup:
    """Where a shot's (SKU, platform, ammo) landed: its open batch and group id."""

    batch: Batch
    group_id: int


class ClusteringService:
    """Assign shots to their batch/group and manage the batch lifecycle."""

    def __init__(self, repo: WorkflowRepository):
        self._repo = repo

    def resolve_group(self, suppressor_sku: str, test_platform: str, ammo: str) -> ResolvedGroup:
        """Return the open batch + group for a shot's test context, creating as needed.

        Finds the SKU's open batch (opening a new one if none is open), then
        upserts the (Test Platform + Ammo) group within it. Because a closed
        batch is never an "open batch", the first shot for a SKU after its batch
        was closed transparently opens a new batch.
        """
        batch = self._open_batch(suppressor_sku)
        group_id = self._repo.upsert_group(batch.id, test_platform, ammo)
        return ResolvedGroup(batch=batch, group_id=group_id)

    def open_batch_for(self, suppressor_sku: str) -> Batch:
        """The SKU's current open batch, creating one if none is open."""
        return self._open_batch(suppressor_sku)

    def close_batch(self, batch_id: int) -> None:
        """Close a batch (explicit user action). Idempotent; unknown id raises ``LookupError``."""
        self._repo.close_batch(batch_id)

    def ensure_open(self, batch_id: int) -> Batch:
        """Return the batch if open; raise if it is closed or unknown.

        The guard for callers that mark into an explicit batch instead of going
        through :meth:`resolve_group`.
        """
        batch = self._repo.get_batch(batch_id)
        if batch is None:
            raise LookupError(f"No batch with id {batch_id}")
        if batch.closed:
            raise ClosedBatchError(
                f"Batch {batch_id} (SKU {batch.sku!r}) is closed; start a new batch instead."
            )
        return batch

    # ---- internal ------------------------------------------------------- #

    def _open_batch(self, suppressor_sku: str) -> Batch:
        batch = self._repo.open_batch_for_sku(suppressor_sku)
        if batch is None:
            batch_id = self._repo.create_batch(suppressor_sku)
            batch = self._repo.get_batch(batch_id)
            if batch is None:
                raise LookupError(
                    f"Batch {batch_id} vanished immediately after creation "
                    f"(SKU {suppressor_sku!r})"
                )
        return batch
