"""Ingestion service — turn the input-folder drop target into Unmarked Data Sets.

This is the first headless workflow service (BUILD_PLAN Task 4). It scans a
configured input folder, parses each capture filename into its provisional
placement keys — SKU, platform, cluster index, and shot order
(:func:`~sound_metric_app.models.parse_capture_filename`) — and records each file
as an *Unmarked Data Set*: a shot with ``marked = False`` and no test context
beyond what its name carried.

Because the cluster is encoded in the filename, a shot arrives already knowing
which string of fire it belongs to and its position within it, so its FRP /
regular role is determined from the moment it lands.

Design notes:

* **User-actuated.** Ingest is an explicit :meth:`IngestionService.scan` call,
  not an auto-watcher (README design principle). A folder watcher can wrap this
  later.
* **Idempotent.** ``source_file`` is stored as a resolved absolute path and the
  underlying :meth:`~sound_metric_app.storage.repository.WorkflowRepository.add_unmarked_shot`
  is a no-op for files already ingested, so re-scanning the same folder adds zero
  duplicates.
* **Validating.** By default each *new* file is opened once to confirm it is a
  readable capture before it is recorded; files that fail to open are reported,
  not ingested. Already-ingested files are never re-read, keeping re-scans cheap.
  The channel reader is injected so the service is unit-testable without a real
  ``.dxd`` file present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..ingestion import ChannelInfo, list_channels
from ..models import CAPTURE_EXTENSIONS, Shot, parse_capture_filename
from ..storage import WorkflowRepository

#: Signature of the channel reader used to validate that a file opens.
ChannelReader = Callable[[str], list[ChannelInfo]]


@dataclass
class IngestReport:
    """Outcome of one :meth:`IngestionService.scan`.

    ``ingested`` holds the freshly created unmarked shots; the remaining lists
    account for every other capture-extension file the scan saw, so the sum of
    all four equals the number of candidate files considered.
    """

    ingested: list[Shot] = field(default_factory=list)
    #: source paths skipped because they were already ingested.
    already_present: list[str] = field(default_factory=list)
    #: (path, reason) for files whose name did not fit the naming convention.
    malformed: list[tuple[str, str]] = field(default_factory=list)
    #: (path, reason) for files that could not be opened/read.
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_ingested(self) -> int:
        return len(self.ingested)


class IngestionService:
    """Scan an input folder and record new captures as unmarked shots."""

    def __init__(self, repo: WorkflowRepository, *, reader: ChannelReader = list_channels):
        self._repo = repo
        self._reader = reader

    def scan(self, input_dir: str | Path, *, validate: bool = True) -> IngestReport:
        """Ingest every capture file in ``input_dir`` as an Unmarked Data Set.

        Parameters
        ----------
        input_dir:
            Folder to scan (non-recursive). Missing folders raise
            ``FileNotFoundError``; a non-directory path raises ``NotADirectoryError``.
        validate:
            When true (default), each new file is opened once via the injected
            channel reader to confirm it is a readable capture before it is
            recorded. Set false to skip the read (e.g. filename-only ingest).

        Returns
        -------
        IngestReport
            New shots plus the files skipped as already-present, malformed, or
            unreadable.
        """
        folder = Path(input_dir)
        if not folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Input path is not a folder: {folder}")

        report = IngestReport()
        for path in self._capture_files(folder):
            source_file = str(path.resolve())

            try:
                parsed = parse_capture_filename(path.name)
            except ValueError as exc:
                report.malformed.append((source_file, str(exc)))
                continue

            if self._repo.get_shot_by_source(source_file) is not None:
                report.already_present.append(source_file)
                continue

            if validate:
                try:
                    self._reader(source_file)
                except Exception as exc:  # noqa: BLE001 — reader failures are reported, not raised
                    report.unreadable.append((source_file, str(exc)))
                    continue

            shot_id = self._repo.add_unmarked_shot(
                source_file,
                suppressor_sku=parsed.suppressor_sku,
                test_platform=parsed.test_platform,
                cluster_index=parsed.cluster_index,
                shot_order=parsed.shot_order,
            )
            report.ingested.append(self._repo.get_shot(shot_id))

        return report

    @staticmethod
    def _capture_files(folder: Path) -> list[Path]:
        """Capture-extension files directly in ``folder``, in stable name order."""
        files = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in CAPTURE_EXTENSIONS
        ]
        return sorted(files, key=lambda p: p.name)
