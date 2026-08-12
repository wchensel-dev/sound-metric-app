"""Tab 1 — Ingest: scan the input folder and list Unmarked Data Sets.

Three lists over one folder: the unmarked shots a scan turned up, the
malformed/unreadable files it could not use (this scan only), and the paths
previously discarded, which is database truth rather than scan-scoped. Discard
and Restore move a file between the last two; **Mark selected shot** hands a row
to the Marking tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

from PySide6 import QtWidgets

from ...models import Shot
from ..controller import WorkflowController
from ..format import _EMPTY
from .base import _View

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from ..main_window import MainWindow


class IngestView(_View):
    _COLUMNS = ["ID", "File", "SKU", "Platform", "Cluster", "Shot #", "Role", ""]
    _DISCARD_COL = len(_COLUMNS) - 1

    #: Malformed/unreadable rows from the latest scan, one Discard button each.
    _BAD_COLUMNS = ["File", "Kind", "Reason", ""]
    _BAD_ACTION_COL = len(_BAD_COLUMNS) - 1
    #: Previously discarded paths (DB truth, not scan-scoped), one Restore button each.
    _DISCARDED_COLUMNS = ["File", "Reason", "Discarded", ""]
    _DISCARDED_ACTION_COL = len(_DISCARDED_COLUMNS) - 1

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        #: The most recent scan's report, kept so a Discard click can drop its
        #: row from the tree immediately instead of waiting on a re-scan.
        self._last_report = None
        layout = QtWidgets.QVBoxLayout(self)

        folder_row = QtWidgets.QHBoxLayout()
        self.folder_label = QtWidgets.QLabel()
        self.folder_label.setWordWrap(True)
        change_btn = QtWidgets.QPushButton("Change…")
        change_btn.clicked.connect(self._change_folder)
        folder_row.addWidget(QtWidgets.QLabel("Input folder:"))
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(change_btn)
        layout.addLayout(folder_row)

        action_row = QtWidgets.QHBoxLayout()
        self.ingest_btn = QtWidgets.QPushButton("Ingest")
        self.ingest_btn.clicked.connect(self._ingest)
        self.mark_btn = QtWidgets.QPushButton("Mark selected shot →")
        self.mark_btn.clicked.connect(self._mark_selected)
        action_row.addWidget(self.ingest_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.mark_btn)
        layout.addLayout(action_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QtWidgets.QLabel("Needs attention (this scan):"))
        self.bad_files_tree = self._build_action_tree(
            self._BAD_COLUMNS, self._BAD_ACTION_COL, ("Discard",)
        )
        # Short and scrollable rather than tall: these are exceptions to
        # triage, not the view's primary content.
        self.bad_files_tree.setMaximumHeight(120)
        layout.addWidget(self.bad_files_tree)

        layout.addWidget(QtWidgets.QLabel("Discarded files:"))
        self.discarded_tree = self._build_action_tree(
            self._DISCARDED_COLUMNS, self._DISCARDED_ACTION_COL, ("Restore",)
        )
        self.discarded_tree.setMaximumHeight(120)
        layout.addWidget(self.discarded_tree)

        layout.addWidget(QtWidgets.QLabel("Unmarked data sets:"))
        self.table = QtWidgets.QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # The trailing Discard column holds a button, not text — stretching it
        # like the old last-section default would leave a wide empty strip
        # beside a left-hung button, so File (the column worth the extra room)
        # stretches instead and Discard is sized from the button itself.
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(self._DISCARD_COL, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(
            self._DISCARD_COL, QtWidgets.QPushButton("Discard").sizeHint().width()
        )
        self.table.doubleClicked.connect(lambda *_: self._mark_selected())
        layout.addWidget(self.table)

        self._update_folder_label()

    def refresh(self) -> None:
        self._update_folder_label()
        shots = self.controller.unmarked_shots()
        self.table.setRowCount(len(shots))
        for row, s in enumerate(shots):
            # Role reads straight off the filename's shot order — a shot knows
            # whether it is its cluster's FRP before anyone marks it.
            role = s.role
            values = [
                str(s.id),
                Path(s.source_file).name,
                s.suppressor_sku or _EMPTY,
                s.test_platform or _EMPTY,
                _EMPTY if s.cluster_index is None else str(s.cluster_index),
                _EMPTY if s.shot_order is None else str(s.shot_order),
                role.label if role else _EMPTY,
            ]
            for col, text in enumerate(values):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
            self._add_discard_shot_button(row, s)
        self.table.resizeColumnsToContents()
        self._draw_discarded()

    def _add_discard_shot_button(self, row: int, shot: Shot) -> None:
        btn = QtWidgets.QPushButton("Discard")
        btn.setToolTip("Remove this shot and ignore its file on future scans.")
        shot_id, name = shot.id, Path(shot.source_file).name
        btn.clicked.connect(lambda: self._defer(lambda: self._prompt_discard_shot(shot_id, name)))
        self.table.setCellWidget(row, self._DISCARD_COL, btn)

    def _prompt_discard_shot(self, shot_id: int, filename: str) -> None:
        reason = self._prompt_discard_reason(
            "Discard shot",
            f"Discard {filename!r}? It will be removed from Unmarked data sets "
            "and ignored on future ingest scans. Optional reason:",
        )
        if reason is None:
            return
        self._run_async(
            lambda: self.controller.discard_shot(shot_id, reason=reason or None),
            lambda _: self.main.notify_changed(),
        )

    @staticmethod
    def _build_action_tree(
        columns: list[str], action_col: int, button_labels: tuple[str, ...]
    ) -> QtWidgets.QTreeWidget:
        """A flat, headed tree whose last column hosts one button per row.

        Shared shape for the bad-files and discarded-files lists: real column
        headers (unlike CompareView's headerless pinned list) since these rows
        carry a reason string worth a labeled column, not just a swatch and a
        name. The action column is sized from the button's own sizeHint, since
        resizeColumnToContents sees no text behind a setItemWidget button.
        """
        tree = QtWidgets.QTreeWidget()
        tree.setColumnCount(len(columns))
        tree.setHeaderLabels(columns)
        tree.setRootIsDecorated(False)
        tree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        width = max(QtWidgets.QPushButton(text).sizeHint().width() for text in button_labels)
        header.setSectionResizeMode(action_col, QtWidgets.QHeaderView.Fixed)
        tree.setColumnWidth(action_col, width)
        return tree

    def _draw_bad_files(self, report) -> None:
        self.bad_files_tree.clear()
        rows = [(p, "malformed", r) for p, r in report.malformed]
        rows += [(p, "unreadable", r) for p, r in report.unreadable]
        for source_file, kind, reason in rows:
            item = QtWidgets.QTreeWidgetItem([Path(source_file).name, kind, reason])
            item.setToolTip(2, reason)
            self.bad_files_tree.addTopLevelItem(item)
            self._add_discard_button(item, source_file, reason)

    def _add_discard_button(
        self, item: QtWidgets.QTreeWidgetItem, source_file: str, reason: str
    ) -> None:
        btn = QtWidgets.QPushButton("Discard")
        btn.setToolTip("Ignore this file on every future scan until restored.")
        btn.clicked.connect(lambda: self._defer(lambda: self._prompt_discard(source_file, reason)))
        self.bad_files_tree.setItemWidget(item, self._BAD_ACTION_COL, btn)

    def _prompt_discard(self, source_file: str, reason: str) -> None:
        """Confirm (and let the operator edit) the reason before it's persisted.

        Pre-filled with the scan's own failure reason so the common case is one
        click through; the field stays editable for a truer note (e.g. "known
        bad export, re-shooting Tuesday").
        """
        new_reason = self._prompt_discard_reason(
            "Discard file", f"Why discard {Path(source_file).name!r}?", reason
        )
        if new_reason is None:
            return
        self._discard(source_file, new_reason or reason)

    def _discard(self, source_file: str, reason: str) -> None:
        self._run_async(
            lambda: self.controller.discard_file(source_file, reason=reason),
            lambda _: self._after_discard(source_file),
        )

    def _after_discard(self, source_file: str) -> None:
        """Drop the just-discarded row from the live report and redraw both trees.

        No re-scan needed: the report already in memory just loses this entry,
        while the Discarded panel (DB truth) picks up the new row.
        """
        if self._last_report is not None:
            self._last_report.malformed = [
                (p, r) for p, r in self._last_report.malformed if p != source_file
            ]
            self._last_report.unreadable = [
                (p, r) for p, r in self._last_report.unreadable if p != source_file
            ]
            self._draw_bad_files(self._last_report)
        self._draw_discarded()

    def _draw_discarded(self) -> None:
        self.discarded_tree.clear()
        for d in self.controller.discarded_files():
            item = QtWidgets.QTreeWidgetItem(
                [Path(d.source_file).name, d.reason or _EMPTY, d.discarded_at or _EMPTY]
            )
            item.setToolTip(1, d.reason or "")
            self.discarded_tree.addTopLevelItem(item)
            self._add_restore_button(item, d.source_file)

    def _add_restore_button(self, item: QtWidgets.QTreeWidgetItem, source_file: str) -> None:
        btn = QtWidgets.QPushButton("Restore")
        btn.setToolTip("Stop ignoring this file; it will be re-evaluated on the next scan.")
        btn.clicked.connect(lambda: self._defer(lambda: self._restore(source_file)))
        self.discarded_tree.setItemWidget(item, self._DISCARDED_ACTION_COL, btn)

    def _restore(self, source_file: str) -> None:
        self._run_async(
            lambda: self.controller.restore_file(source_file),
            lambda _: self._draw_discarded(),
        )

    def _update_folder_label(self) -> None:
        folder = self.controller.input_folder()
        self.folder_label.setText(folder if folder else "(unset)")

    def _change_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose input folder")
        if not folder:
            return
        self.controller.set_input_folder(folder)
        self._update_folder_label()

    def _ingest(self) -> None:
        self.status_label.setText("Ingesting…")
        self._run_async(
            self.controller.ingest,
            self._on_ingested,
            busy=(self.ingest_btn,),
        )

    def _on_ingested(self, report) -> None:
        self._last_report = report
        self.status_label.setText(
            f"Ingested {report.n_ingested}, "
            f"already present {len(report.already_present)}, "
            f"malformed {len(report.malformed)}, "
            f"unreadable {len(report.unreadable)}, "
            f"discarded {len(report.discarded)}."
        )
        self._draw_bad_files(report)
        self.main.notify_changed()

    def _selected_shot_id(self) -> int | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return int(item.text()) if item else None

    def _mark_selected(self) -> None:
        shot_id = self._selected_shot_id()
        if shot_id is None:
            QtWidgets.QMessageBox.information(
                self, "No selection", "Select an unmarked shot to mark."
            )
            return
        self.main.open_marking_for(shot_id)
