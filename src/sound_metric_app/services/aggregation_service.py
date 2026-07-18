"""Aggregation service (BUILD_PLAN Task 7).

Computes per-group averages of the four metrics, **separately for SE and MR**
(README §4: positions are never mixed), and rolls the groups of a batch up into
one report ready for the CLI/GUI report views.

Averaging in dB is done as a plain arithmetic mean of the per-shot metric values
— the same convention the underlying store uses — so a group's SE average is the
mean of its shots' SE values and likewise for MR.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Batch, Group, MicPosition
from ..storage import WorkflowRepository


@dataclass
class GroupAverages:
    """Per-mic averages for one group, plus how many shots fed them.

    ``averages`` maps each present :class:`~sound_metric_app.models.MicPosition`
    to ``{peak_db, peak_dba, peak_impulse_db, liaeq_100ms_db, n}`` where ``n`` is
    the number of shots contributing that position. Positions absent from the
    group are omitted, so a single-mic group yields a single entry.

    ``shots`` is the un-averaged drill-down behind those averages: each present
    position maps to a list of the individual shot metric rows the average was
    taken over (see
    :meth:`~sound_metric_app.storage.WorkflowRepository.shot_metrics_for_group`).
    Its keys always match ``averages``; it is a required field so this invariant
    cannot be broken by omitting it at construction.
    """

    group: Group
    n_shots: int
    averages: dict[MicPosition, dict]
    shots: dict[MicPosition, list[dict]]


@dataclass
class BatchReport:
    """A batch's groups, each with its SE/MR averages."""

    batch: Batch
    groups: list[GroupAverages]


class AggregationService:
    """Per-group and per-batch metric averaging, SE and MR kept separate."""

    def __init__(self, repo: WorkflowRepository):
        self._repo = repo

    def group_averages(self, group_id: int) -> GroupAverages:
        """Averages for one group. Raises ``LookupError`` for an unknown group."""
        group = self._repo.get_group(group_id)
        if group is None:
            raise LookupError(f"No group with id {group_id}")
        return self._averages_for(group)

    def _averages_for(self, group: Group) -> GroupAverages:
        """Averages for an already-loaded group, skipping the ``get_group`` re-fetch."""
        return GroupAverages(
            group=group,
            n_shots=self._repo.count_shots_in_group(group.id),
            averages=self._repo.group_averages(group.id),
            shots=self._repo.shot_metrics_for_group(group.id),
        )

    def batch_report(self, batch_id: int) -> BatchReport:
        """Every group in a batch with its SE/MR averages.

        Raises ``LookupError`` for an unknown batch.
        """
        batch = self._repo.get_batch(batch_id)
        if batch is None:
            raise LookupError(f"No batch with id {batch_id}")
        groups = [
            self._averages_for(group) for group in self._repo.groups_for_batch(batch_id)
        ]
        return BatchReport(batch=batch, groups=groups)
