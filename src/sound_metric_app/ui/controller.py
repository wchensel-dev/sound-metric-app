"""Workflow view-model: the service calls the GUI makes, with no Qt dependency.

The GUI (``main_window``) is a thin front-end over these methods, exactly as the
``sma`` CLI is a thin front-end over the same Phase B services. Keeping the logic
here — free of any ``QWidget``/``QApplication`` — means the whole ingest -> mark
-> close -> report flow is testable without a live GUI (see ``tests/test_controller``).

Connection model
----------------
Every method opens a short-lived :class:`WorkflowRepository` on the configured
``db_path`` and closes it before returning, mirroring how each CLI subcommand
opens its own repo. Because no ``sqlite3`` connection outlives a call, these
methods are safe to invoke from a Qt worker thread (SQLite connections are bound
to the thread that created them); the returned objects are detached dataclasses,
safe to hand back to the UI thread.

The channel and capture readers are injected so tests can substitute fakes
without a real ``.dxd`` file, matching ``workflow_cli``'s module-level readers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .. import config
from ..ingestion import ChannelInfo, list_channels, read_capture
from ..models import Batch, Group, MicPosition, Shot
from ..services import (
    AggregationService,
    BatchReport,
    ClosedBatchError,
    ClusteringService,
    GroupAverages,
    IngestionService,
    IngestReport,
    MarkedShot,
    MarkingService,
)
from ..storage import WorkflowRepository

#: Exceptions the services raise for bad user input / state. The GUI turns these
#: (and any other error) into a dialog; kept here so the widget layer and any
#: test can share the same catch set the CLI uses (``workflow_cli._USER_ERRORS``).
USER_ERRORS = (
    LookupError,
    ValueError,
    ClosedBatchError,
    FileNotFoundError,
    NotADirectoryError,
)


@dataclass
class GroupNode:
    """A group with its shots already materialized."""

    group: Group
    shots: list[Shot]


@dataclass
class BatchNode:
    """A batch with its groups (each carrying its shots)."""

    batch: Batch
    groups: list[GroupNode]


class WorkflowController:
    """Headless driver for the GUI: ingest, mark, close, and report."""

    def __init__(
        self,
        db_path: str | Path = config.DEFAULT_DB_PATH,
        *,
        channel_reader=list_channels,
        capture_reader=read_capture,
    ):
        self._db_path = str(db_path)
        self._channel_reader = channel_reader
        self._capture_reader = capture_reader

    @contextmanager
    def _repo(self) -> Iterator[WorkflowRepository]:
        with WorkflowRepository(self._db_path) as repo:
            yield repo

    # ---- config / input folder ----------------------------------------- #

    def input_folder(self) -> str | None:
        """The configured default input folder, or ``None`` if unset."""
        return config.get_input_folder()

    def set_input_folder(self, folder: str | Path) -> Path:
        """Persist the default input folder and return the resolved path."""
        return config.set_input_folder(folder)

    # ---- ingest --------------------------------------------------------- #

    def ingest(self, folder: str | Path | None = None, *, validate: bool = True) -> IngestReport:
        """Scan ``folder`` (or the configured input folder) for new captures.

        Raises ``ValueError`` if no folder is given and none is configured, so
        the caller can prompt the user to pick one.
        """
        folder = folder or config.get_input_folder()
        if not folder:
            raise ValueError(
                "No input folder configured. Choose one before ingesting."
            )
        with self._repo() as repo:
            return IngestionService(repo, reader=self._channel_reader).scan(
                folder, validate=validate
            )

    def unmarked_shots(self) -> list[Shot]:
        with self._repo() as repo:
            return repo.unmarked_shots()

    def get_shot(self, shot_id: int) -> Shot | None:
        with self._repo() as repo:
            return repo.get_shot(shot_id)

    # ---- mark ----------------------------------------------------------- #

    def channels_for(self, source_file: str) -> list[ChannelInfo]:
        """Raw channels in a capture, to offer as SE/MR choices in the mark form."""
        return self._channel_reader(source_file)

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
        """Annotate a shot, tag SE/MR, and compute + store its metrics."""
        with self._repo() as repo:
            svc = MarkingService(repo, ClusteringService(repo), reader=self._capture_reader)
            return svc.mark(
                shot_id,
                ammo=ammo,
                channel_map=channel_map,
                suppressor_sku=suppressor_sku,
                test_platform=test_platform,
                shot_order=shot_order,
                wind_speed=wind_speed,
                temp=temp,
                relative_humidity=relative_humidity,
            )

    # ---- batches / groups / shots (read) -------------------------------- #

    def batches(self) -> list[Batch]:
        with self._repo() as repo:
            return repo.all_batches()

    def get_batch(self, batch_id: int) -> Batch | None:
        with self._repo() as repo:
            return repo.get_batch(batch_id)

    def rename_batch(self, batch_id: int, sku: str) -> None:
        """Correct a batch's SKU in place, keeping all its groups and shots.

        Guards the "at most one open batch per SKU" invariant: renaming an *open*
        batch onto a SKU that already has a different open batch is rejected, so
        future marking never has two open batches to choose between. Renaming a
        closed batch, or renaming to its own current SKU, is always allowed.

        Raises ``ValueError`` on an empty SKU or such a collision, ``LookupError``
        if the batch id is unknown.
        """
        sku = sku.strip()
        if not sku:
            raise ValueError("SKU cannot be empty.")
        with self._repo() as repo:
            batch = repo.get_batch(batch_id)
            if batch is None:
                raise LookupError(f"No batch with id {batch_id}")
            if not batch.closed:
                other = repo.open_batch_for_sku(sku)
                if other is not None and other.id != batch_id:
                    raise ValueError(
                        f"SKU {sku!r} already has an open batch (#{other.id}). "
                        "Close it first, or pick a different SKU."
                    )
            repo.rename_batch_sku(batch_id, sku)

    def groups_for_batch(self, batch_id: int) -> list[Group]:
        with self._repo() as repo:
            return repo.groups_for_batch(batch_id)

    def shots_by_group(self, group_id: int) -> list[Shot]:
        with self._repo() as repo:
            return repo.shots_by_group(group_id)

    def batch_tree(self) -> list[BatchNode]:
        """The whole batch -> group -> shot tree, over a single connection.

        The GUI's batch tree renders all three levels at once; loading them here
        opens one repo instead of a connection per batch/group, and shot counts
        come from ``len(node.shots)`` rather than a separate COUNT query.

        Empty groups are swept before the tree is built, so a refresh cleans up
        any group left shot-less by an earlier edit (see
        :meth:`WorkflowRepository.delete_empty_groups`).
        """
        with self._repo() as repo:
            repo.delete_empty_groups()
            return [
                BatchNode(
                    batch=batch,
                    groups=[
                        GroupNode(group=group, shots=repo.shots_by_group(group.id))
                        for group in repo.groups_for_batch(batch.id)
                    ],
                )
                for batch in repo.all_batches()
            ]

    # ---- close ---------------------------------------------------------- #

    def close_batch(self, batch_id: int) -> None:
        with self._repo() as repo:
            ClusteringService(repo).close_batch(batch_id)

    # ---- report --------------------------------------------------------- #

    def batch_report(self, batch_id: int) -> BatchReport:
        with self._repo() as repo:
            return AggregationService(repo).batch_report(batch_id)

    def group_averages(self, group_id: int) -> GroupAverages:
        with self._repo() as repo:
            return AggregationService(repo).group_averages(group_id)
