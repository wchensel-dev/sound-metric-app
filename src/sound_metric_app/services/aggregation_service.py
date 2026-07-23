"""Aggregation service (BUILD_PLAN Task 7).

Computes the **batch average view**: the filter where a shot's ``included`` flag
is true, grouped by mic position crossed with derived role, producing four output
slots per batch —

.. code-block:: text

   muzzle_left  (ML) . FRP        muzzle_left  (ML) . regular
   shooters_ear (SE) . FRP        shooters_ear (SE) . regular

Positions are never mixed and roles are never mixed. The 3-FRP / 5-regular
target applies per position, so each channel averages the same underlying
selected shots on its own axis.

Averaging is done in the **linear domain** (MATH.md §9), matching TBAC: each
metric's per-shot linear magnitude (Pa, or Pa·ms for the impulse) is meaned and
the mean converted once to its dB level — not a mean of the dB values. The
underlying store does this; the service wraps it per batch and rolls a
combination's batches up for the report views.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Batch, Combination, MicPosition, ShotRole
from ..storage import WorkflowRepository
from .inclusion_service import InclusionService, InclusionStatus

#: The four output slots, in report order: position major, role minor.
AVERAGE_SLOTS: tuple[tuple[MicPosition, ShotRole], ...] = tuple(
    (position, role) for position in (MicPosition.ML, MicPosition.SE) for role in ShotRole
)


@dataclass
class BatchAverages:
    """One batch's four output slots, plus the shots behind them.

    ``averages`` maps each populated ``(position, role)`` slot to a dict carrying
    every metric's averaged linear magnitude and dB level (``peak_pa``/``peak_db``,
    ``peak_a_pa``/``peak_dba``, ``impulse_pa_ms``/``peak_impulse_db``,
    ``leq10ms_pa``/``leq10ms_db``, ``liaeq_pa``/``liaeq_100ms_db``) plus ``n``,
    the number of included shots feeding that slot. Slots with nothing included
    are omitted, so a batch whose regulars have not been brought forward yet
    yields two entries rather than four empty ones.

    ``shots`` is the un-averaged drill-down behind those averages: each populated
    slot maps to the list of individual shot metric rows the average was taken
    over. Its keys always match ``averages``; it is a required field so this
    invariant cannot be broken by omitting it at construction.

    ``n_shots`` counts every shot in the batch — the data-bank total, not the
    included subset — so a report can show "5 of 27 brought forward". ``status``
    carries the per-role progress against the soft 3 / 5 targets.
    """

    batch: Batch
    combination: Combination
    n_shots: int
    averages: dict[tuple[MicPosition, ShotRole], dict]
    shots: dict[tuple[MicPosition, ShotRole], list[dict]]
    status: InclusionStatus

    @property
    def n_included(self) -> int:
        """Included shots across both roles (counted once, not per channel)."""
        return sum(p.included for p in self.status.progress.values())


@dataclass
class CombinationReport:
    """A test combination's batches, each with its four output slots."""

    combination: Combination
    batches: list[BatchAverages]


class AggregationService:
    """Per-batch roll-up into the four position x role output slots."""

    def __init__(self, repo: WorkflowRepository):
        self._repo = repo
        self._inclusion = InclusionService(repo)

    def batch_averages(self, batch_id: int) -> BatchAverages:
        """The four output slots for one batch. Raises ``LookupError`` if unknown."""
        batch = self._repo.get_batch(batch_id)
        if batch is None:
            raise LookupError(f"No batch with id {batch_id}")
        return self._averages_for(batch)

    def _averages_for(self, batch: Batch) -> BatchAverages:
        """Slots for an already-loaded batch, skipping the ``get_batch`` re-fetch."""
        return BatchAverages(
            batch=batch,
            combination=self._repo.get_combination(batch.combination_id),
            n_shots=self._repo.count_shots_in_batch(batch.id),
            averages=self._repo.batch_averages(batch.id),
            shots=self._repo.shot_metrics_for_batch(batch.id),
            status=self._inclusion.status(batch.id),
        )

    def combination_report(self, combination_id: int) -> CombinationReport:
        """Every batch under a combination with its four slots.

        Raises ``LookupError`` for an unknown combination.
        """
        combination = self._repo.get_combination(combination_id)
        if combination is None:
            raise LookupError(f"No combination with id {combination_id}")
        return CombinationReport(
            combination=combination,
            batches=[
                self._averages_for(batch)
                for batch in self._repo.batches_for_combination(combination_id)
            ],
        )
