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

One file can hold several shots: when the recorder is left armed, DewesoftX
appends each further trigger to the same file. Each new file is opened once to
count its stored trigger blocks, and each block becomes its own shot, keyed
``<path>|<k>`` (:func:`~sound_metric_app.models.shot_source`). A one-shot file
keeps its plain path, exactly as before.

Design notes:

* **User-actuated.** Ingest is an explicit :meth:`IngestionService.scan` call,
  not an auto-watcher (README design principle). A folder watcher can wrap this
  later.
* **Idempotent.** ``source_file`` is stored as a resolved absolute path and a
  file that already has any shot recorded is skipped without being opened, so
  re-scanning the same folder adds zero duplicates, and files ingested before
  multi-shot support are never re-split.
* **Validating.** By default each *new* file is opened once, to count its shots
  and so confirm it is a readable capture, before it is recorded; files that
  fail to open are reported, not ingested. Already-ingested files are never
  re-read, keeping re-scans cheap. The shot counter is injected so the service
  is unit-testable without a real ``.dxd`` file present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..ingestion import count_shots
from ..models import CAPTURE_EXTENSIONS, Shot, parse_capture_filename, shot_source
from ..storage import WorkflowRepository

#: Signature of the shot counter: path -> how many stored trigger blocks it holds.
#: Opening the file to count them is also what validates that it is readable.
ShotCounter = Callable[[str], int]


@dataclass
class IngestReport:
    """Outcome of one :meth:`IngestionService.scan`.

    ``ingested`` holds the freshly created unmarked shots, one per shot (so a
    multi-shot file contributes several); the remaining lists account for every
    other capture-extension file the scan saw.
    """

    ingested: list[Shot] = field(default_factory=list)
    #: path -> shot count, for each newly ingested file that held more than one.
    multi_shot_files: dict[str, int] = field(default_factory=dict)
    #: source paths skipped because they were already ingested.
    already_present: list[str] = field(default_factory=list)
    #: (path, reason) for files whose name did not fit the naming convention.
    malformed: list[tuple[str, str]] = field(default_factory=list)
    #: (path, reason) for files that could not be opened/read.
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    #: source paths skipped because the operator previously discarded them —
    #: see :meth:`~sound_metric_app.storage.repository.WorkflowRepository.discard_file`.
    discarded: list[str] = field(default_factory=list)

    @property
    def n_ingested(self) -> int:
        return len(self.ingested)


class IngestionService:
    """Scan an input folder and record new captures as unmarked shots."""

    def __init__(self, repo: WorkflowRepository, *, reader: ShotCounter = count_shots):
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
            shot counter, to count its shots and confirm it is a readable
            capture, before it is recorded. Set false to skip the read
            (filename-only ingest), which records every file as one shot.

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
        discarded = self._repo.discarded_source_files()
        ingested = self._repo.ingested_files()
        for path in self._capture_files(folder):
            source_file = str(path.resolve())

            if source_file in discarded:
                report.discarded.append(source_file)
                continue

            try:
                parsed = parse_capture_filename(path.name)
            except ValueError as exc:
                report.malformed.append((source_file, str(exc)))
                continue

            if source_file in ingested:
                report.already_present.append(source_file)
                continue

            n_shots = 1
            if validate:
                try:
                    n_shots = self._reader(source_file)
                except Exception as exc:  # noqa: BLE001 — reader failures are reported, not raised
                    # dwdatareader's DWError carries its real DLL message on
                    # `.message`; str(exc) alone degrades to a bare status code
                    # (e.g. "2") because DWStatus is an IntEnum.
                    reason = getattr(exc, "message", None) or str(exc)
                    report.unreadable.append((source_file, reason))
                    continue
                if n_shots < 1:
                    report.unreadable.append((source_file, "No stored shots in file."))
                    continue

            # A one-shot file keeps its plain path as its key; only a file the
            # recorder stayed armed into gets one suffixed key per shot.
            keys = (
                [source_file]
                if n_shots == 1
                else [shot_source(source_file, k) for k in range(n_shots)]
            )
            with self._repo.transaction():
                shot_ids = [
                    self._repo.add_unmarked_shot(
                        key,
                        suppressor_sku=parsed.suppressor_sku,
                        test_platform=parsed.test_platform,
                        cluster_index=parsed.cluster_index,
                        shot_order=parsed.shot_order,
                    )
                    for key in keys
                ]
            if n_shots > 1:
                report.multi_shot_files[source_file] = n_shots
            report.ingested.extend(self._repo.get_shot(i) for i in shot_ids)

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
