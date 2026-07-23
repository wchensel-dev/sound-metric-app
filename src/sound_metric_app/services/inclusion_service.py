"""Inclusion service — the gate between the data bank and the batch average.

Marking a shot files it in the **data bank**: every cluster and shot the app has
ever seen, included or idle, kept as the complete archive. Nothing is ever
deleted for being left out of an average. What moves a shot into the **batch
average view** is its ``included`` flag, and this service owns flipping it.

Two grains, one source of truth:

* :meth:`InclusionService.include_shot` sets the flag on a single shot. The shot
  is the source of truth, which is what lets a batch land on exactly 3 FRPs and
  5 regulars — two independent counts drawn from however many clusters it takes.
* :meth:`InclusionService.include_cluster` is the "bring cluster forward"
  convenience: it fans the same flag out over a cluster's shots, under the same
  rules — a shot with no ``shot_order`` is skipped rather than flagged into an
  average it could never reach. It cannot replace shot-level control, because
  regulars arrive in uneven cluster sizes (a 3-shot cluster contributes two, a
  4-shot cluster three), so whole clusters never sum cleanly to 5.

Exclusion reasons (high winds, ambient noise, ...) ride along with an exclusion
and are cleared on inclusion — a reason for leaving a shot out is meaningless
once it is in.

The 3 / 5 targets are **soft**: :meth:`InclusionService.status` reports progress
against them so a batch that is short or over is visible, but nothing here
refuses an inclusion that overshoots.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TARGET_FRP_SHOTS, TARGET_REGULAR_SHOTS
from ..models import ShotRole
from ..storage import WorkflowRepository

#: Target included-shot count per role, per position. Soft targets (README).
ROLE_TARGETS: dict[ShotRole, int] = {
    ShotRole.FRP: TARGET_FRP_SHOTS,
    ShotRole.REGULAR: TARGET_REGULAR_SHOTS,
}


@dataclass
class RoleProgress:
    """How one role's included count compares with its target."""

    role: ShotRole
    included: int
    target: int

    @property
    def remaining(self) -> int:
        """Shots still needed to reach the target; 0 once met or exceeded."""
        return max(0, self.target - self.included)

    @property
    def met(self) -> bool:
        return self.included >= self.target

    @property
    def over(self) -> bool:
        """True when more shots are included than the target calls for."""
        return self.included > self.target

    def __str__(self) -> str:
        return f"{self.role.label}: {self.included}/{self.target}"


@dataclass
class InclusionStatus:
    """A batch's progress toward its 3-FRP / 5-regular targets."""

    batch_id: int
    progress: dict[ShotRole, RoleProgress]

    @property
    def complete(self) -> bool:
        """True once every role has met its target."""
        return all(p.met for p in self.progress.values())

    def summary(self) -> str:
        """One-line ``FRP: 2/3   Regular: 5/5`` rendering for CLI/status bars."""
        return "   ".join(str(self.progress[role]) for role in ShotRole)


class InclusionService:
    """Bring shots and clusters forward into a batch average, or return them to idle."""

    def __init__(self, repo: WorkflowRepository):
        self._repo = repo

    def include_shot(
        self, shot_id: int, included: bool = True, *, reason: str | None = None
    ) -> None:
        """Include or idle one shot. ``reason`` records why an excluded shot was left out.

        Raises ``LookupError`` if the shot is unknown, and ``ValueError`` if the
        shot has no ``shot_order`` — without one it has no derivable role, so it
        could not be placed in an FRP or regular slot and would silently vanish
        from the average it was just added to.
        """
        shot = self._repo.get_shot(shot_id)
        if shot is None:
            raise LookupError(f"No shot with id {shot_id}")
        if included and shot.shot_order is None:
            raise ValueError(
                f"Shot #{shot_id} has no shot order, so it has no FRP/regular role "
                "and cannot be brought forward. Set its shot order first."
            )
        self._repo.set_shot_included(shot_id, included, exclusion_reason=reason)

    def include_cluster(
        self, cluster_id: int, included: bool = True, *, reason: str | None = None
    ) -> int:
        """Include or idle every shot in a cluster; return how many rows changed.

        Fan-out over :meth:`include_shot`, and it carries that method's guard with
        it: shots without a ``shot_order`` are **skipped** on inclusion, not
        brought forward, since they have no derivable role and the roll-ups would
        drop them anyway. The return count reports the shots actually included, so
        a 3-shot cluster holding one unordered shot returns 2. Idling still covers
        the whole cluster — returning a shot to idle is safe whatever its order.

        Raises ``LookupError`` if the cluster is unknown.
        """
        if not included:
            return self._repo.set_cluster_included(cluster_id, included, exclusion_reason=reason)
        if self._repo.get_cluster(cluster_id) is None:
            raise LookupError(f"No cluster with id {cluster_id}")
        orderable = [s for s in self._repo.shots_by_cluster(cluster_id) if s.shot_order is not None]
        with self._repo.transaction():
            for shot in orderable:
                self.include_shot(shot.id, True)
        return len(orderable)

    def status(self, batch_id: int) -> InclusionStatus:
        """A batch's included-shot counts against the soft targets.

        Raises ``LookupError`` if the batch is unknown.
        """
        if self._repo.get_batch(batch_id) is None:
            raise LookupError(f"No batch with id {batch_id}")
        counts = self._repo.inclusion_counts(batch_id)
        return InclusionStatus(
            batch_id=batch_id,
            progress={
                role: RoleProgress(role=role, included=counts[role], target=target)
                for role, target in ROLE_TARGETS.items()
            },
        )
