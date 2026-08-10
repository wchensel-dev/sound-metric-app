"""PySide6 desktop app for the ingest -> mark -> bring-forward -> report workflow.

Five views over the same Phase B services the ``sma`` CLI drives, wired through
:class:`~sound_metric_app.ui.controller.WorkflowController`:

1. **Ingest / Unmarked** — scan the input folder, list Unmarked Data Sets.
2. **Mark** — annotate a shot, confirm its ML/SE channel tags, compute + store
   metrics. The shot lands in the data bank idle.
3. **Data bank** — the Combination -> Batch -> Cluster -> Shot tree: every shot
   the app has seen, included or idle, with the bring-forward actions that decide
   which ones feed an average, plus session editing and Close batch.
4. **Batch average** — the four position x role output slots per batch
   (muzzle-left / shooter's-ear crossed with FRP / regular), positions and roles
   never mixed, each averaged row ending in a Copy button that yields the
   SilencerScout ``SSR1`` paste string for that slot.
5. **Compare** — one metric's curve for any number of shots overlaid on a single
   graph, pinned there by a Compare button on a Batch average shot row or a
   Data bank shot row — the former one button per slot's mic, the latter one
   button per marked channel, since a data-bank row is not scoped to a
   position. Either source can reach an idle shot's curve just as well as an
   included one. Shots from different batches, SKUs, and mics all coexist
   here, and pinning the same shot/mic from both tabs is a no-op the second
   time (:class:`CompareSeries` keys on shot id + position, not on which tab
   sent it) — the batch-average and data-bank tabs are where shots are
   chosen, this one is where they are read against each other.

The split between tabs 3 and 4 is the directive's two views: the data bank is the
complete archive where nothing is deleted for being left out, and the batch
average is the filter over ``included``.

Ingest, mark, include, and close are explicit buttons (README user-actuated
principle). The two file-reading operations (ingest, mark) run on a worker thread
so a large capture never freezes the window; every service error surfaces as a
dialog.

Run with:  python -m sound_metric_app.ui.main_window   (needs the 'gui' extra)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..dsp import SMOOTHING_FAST, SMOOTHING_INSTANT, SMOOTHING_SLOW, MetricTrace
from ..models import MicPosition, Shot, ShotRole, role_for_order
from ..services import AVERAGE_SLOTS, BatchAverages, slot_line
from .controller import WorkflowController

_NONE_LABEL = "(none)"

_LOADING_LABEL = "loading…"
#: Shown where a mic position has no channel tagged / a value is missing.
_EMPTY = "—"

#: Sentinel row in a SKU-filter dropdown that clears the filter (data ``None``).
_ALL_SKUS_LABEL = "All SKUs"

#: Text on an averaged row's Scout-paste button, and the confirmation it flashes
#: after a click (a clipboard write is otherwise invisible) with how long it
#: holds that text.
_COPY_LABEL = "Copy"
_COPIED_LABEL = "Copied ✓"
_COPIED_FLASH_MS = 1200

#: Text on a shot row's Compare button, and the two confirmations it flashes:
#: pinning happens on another tab, so the button is the only feedback there is.
#: "already" is its own message rather than a silent repeat of "added" — the
#: same shot/mic can only be pinned once, and a click that changed nothing
#: should say so.
_COMPARE_LABEL = "Compare"
_COMPARE_ADDED_LABEL = "Added ✓"
_COMPARE_ALREADY_LABEL = "Pinned ✓"

#: The Compare tab's title, which grows a count of what is pinned to it (see
#: :meth:`MainWindow.update_compare_count`).
_COMPARE_TAB_LABEL = "Compare"

#: The report's metric columns, in display order: (header label, stored metric
#: key). Shared by the Batch average tree (which spends them as columns) and the
#: Compare tab's metric picker (which spends them as dropdown rows), so the two
#: can only ever offer the same metrics under the same names. Every key is one
#: :func:`~sound_metric_app.dsp.build_metric_trace` accepts.
_REPORT_METRICS = (
    ("Peak Pa", "peak_pa"),
    ("Peak dB", "peak_db"),
    ("Peak dBA", "peak_dba"),
    ("Impulse Pa·ms", "impulse_pa_ms"),
    ("Impulse dB·ms", "peak_impulse_db"),
    ("Peak Leq10ms dBA", "leq10ms_db"),
    ("LIAeq,100ms dBA", "liaeq_100ms_db"),
)

#: Header for the pre-trigger floor wherever it is shown. Named once because two
#: trees carry the column with different bodies — Batch average one mic per row,
#: the data bank both mics in one cell — and a label that drifted between them
#: would read as two different diagnostics.
_PRETRIGGER_FLOOR_LABEL = "Pre-trig floor Pa"

#: Per-shot diagnostic columns: (header label, stored key). Deliberately *not*
#: part of :data:`_REPORT_METRICS` — every key there must be one
#: :func:`~sound_metric_app.dsp.build_metric_trace` accepts (the Compare picker
#: and the click-to-graph handler both assume it), and these have no curve. They
#: are QC readouts on the capture, not measurements of the shot, so they trail
#: the metrics and only shot rows carry them. Named for the report the way
#: :data:`_REPORT_METRICS` is, distinct from the storage layer's
#: ``_DIAGNOSTIC_COLUMNS``, which lists database column names.
_REPORT_DIAGNOSTICS = (
    (_PRETRIGGER_FLOOR_LABEL, "pretrigger_floor_pa"),
)

#: Mid-grey for a Compare row that is not on the graph: the text of a hidden
#: one (set aside, but still pinned) and the outline of the empty swatch an
#: unloadable one gets. A *hidden* row keeps its colour swatch at full strength
#: — that is what its curve comes back as.
_MUTED_INK = QtGui.QColor(140, 140, 148)

#: Tint behind the Batch average tree's slot rows, so an averaged row is legible
#: at a glance against the individual shots nested under it. Deliberately
#: low-alpha: it composites over whichever base/alternate-base colour the active
#: theme paints, so the same amber reads as a soft wash in light and dark alike.
_AVERAGE_ROW_TINT = QtGui.QColor(255, 176, 32, 64)


class _TopLevelRowTint(QtWidgets.QStyledItemDelegate):
    """Wash a colour behind a tree's top-level rows, leaving children plain.

    ``QTreeWidgetItem.setBackground`` cannot do this job here: once a stylesheet
    applies to the view (see ``_style_grid_tree``), Qt paints the row background
    from the stylesheet and drops the item's own brush, so the tint never
    appears. Painting it ourselves sidesteps that. The fill goes *under*
    ``super().paint()``, so the stylesheet's grid lines, the selection
    highlight, and the text all still draw over it at full strength.
    """

    def __init__(self, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self._color = color

    def paint(self, painter, option, index) -> None:
        if not index.parent().isValid():
            painter.fillRect(option.rect, self._color)
        super().paint(painter, option, index)


def _repopulate_sku_filter(combo: QtWidgets.QComboBox, skus: list[str]) -> None:
    """Refill a SKU-filter dropdown, preserving the current selection.

    Row 0 is the ``All SKUs`` sentinel (data ``None``, meaning "no filter"); each
    remaining row carries a SKU string as both its label and data. Signals are
    blocked over the rebuild so it does not fire the combo's
    ``currentIndexChanged`` — the caller re-reads the selection and refreshes
    explicitly. A previously selected SKU that no longer exists (its last
    combination was swept) falls back to ``All SKUs``.
    """
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(_ALL_SKUS_LABEL, None)
    for sku in skus:
        combo.addItem(sku, sku)
    index = combo.findData(current)
    combo.setCurrentIndex(index if index >= 0 else 0)
    combo.blockSignals(False)


def _style_grid_tree(tree: QtWidgets.QTreeWidget) -> None:
    """Give a ``QTreeWidget`` visible column/row grid lines.

    A tree has no built-in grid, so we draw one: per-item borders supply the
    column and row rules and alternating row colours make wide numeric rows
    easier to scan. Colours come from ``palette(...)`` so the grid tracks the
    active light/dark theme instead of clashing with it. Shared by the Report
    and Batches trees so both read the same way.
    """
    tree.setAlternatingRowColors(True)
    tree.header().setSectionsMovable(False)
    tree.setStyleSheet(
        "QTreeWidget {"
        " alternate-background-color: palette(alternate-base);"
        " background: palette(base); }"
        "QTreeWidget::item {"
        " border-right: 1px solid palette(mid);"
        " border-bottom: 1px solid palette(mid);"
        " padding: 2px 4px; }"
        "QTreeWidget::item:selected {"
        " background: palette(highlight);"
        " color: palette(highlighted-text); }"
    )


def _flash_button(button: QtWidgets.QPushButton, message: str, revert_to: str) -> None:
    """Briefly swap a button's text to confirm an action that leaves no mark.

    Both of the Batch average tree's row buttons act somewhere the operator is
    not looking — the clipboard, and the Compare tab — so the button itself has
    to acknowledge the click. The revert timer is anchored to the button: a
    rebuild of the tree deletes it, and an anchored ``singleShot`` is dropped
    rather than firing into a deleted widget.
    """
    button.setText(message)
    QtCore.QTimer.singleShot(
        _COPIED_FLASH_MS, button, lambda: button.setText(revert_to)
    )


def _compare_series(
    *,
    shot_id: int,
    position: MicPosition,
    sku: str,
    where: str,
    combo_label: str,
    batch_id: int,
    note: str = "",
) -> CompareSeries:
    """Name one shot/mic as a Compare-tab series.

    Both the Data bank and the Batch average tabs pin the same kind of thing --
    one shot's one mic -- to the same Compare tab, and a curve has to read
    identically no matter which one sent it (``CompareView.add_series`` dedupes
    on ``(shot_id, position)`` across tabs). ``where`` is the caller's
    cluster/shot locator (``"C{cluster}·S{order}"``, or ``"S{order}"`` with no
    cluster); ``note`` is appended to the detail verbatim, letting the Data
    bank tab mark an idle shot without the Batch average tab -- which never
    sees idle shots -- carrying dead code for it.
    """
    return CompareSeries(
        shot_id=shot_id,
        position=position,
        label=f"#{shot_id} · {sku} · {where} · {position.value}",
        detail=f"{combo_label}\nBatch #{batch_id} · {where} · {position.label}{note}",
    )


def _pin_compare_series(
    main: "MainWindow",
    button: QtWidgets.QPushButton,
    series: CompareSeries,
    revert_label: str,
) -> None:
    """Pin one shot/mic to the Compare tab and flash the button to confirm.

    Shared by the Data bank and Batch average tabs' own ``_pin_for_compare``
    methods, which differ only in what the button reverts to afterwards (a mic
    label there, the constant "Compare" here).
    """
    added = main.add_to_compare(series)
    _flash_button(
        button, _COMPARE_ADDED_LABEL if added else _COMPARE_ALREADY_LABEL, revert_label
    )


def _color_swatch(color: tuple[int, int, int] | None) -> QtGui.QIcon:
    """A small filled square in ``color`` — the Compare list's legend key.

    ``None`` (a series that could not be loaded) yields a hollow outline, so an
    undrawn row is visibly not claiming one of the graph's colours.
    """
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtGui.QColor(*color) if color else QtCore.Qt.transparent)
    if color is None:
        painter = QtGui.QPainter(pixmap)
        painter.setPen(_MUTED_INK)
        painter.drawRect(0, 0, 11, 11)
        painter.end()
    return QtGui.QIcon(pixmap)


def _tree_items(tree: QtWidgets.QTreeWidget):
    """Yield every item in ``tree``, parents before their children."""

    def walk(item):
        yield item
        for i in range(item.childCount()):
            yield from walk(item.child(i))

    for i in range(tree.topLevelItemCount()):
        yield from walk(tree.topLevelItem(i))


def _expanded_keys(tree: QtWidgets.QTreeWidget, key) -> set:
    """Snapshot which branches are open, keyed by ``key(item)``.

    Both archive trees are rebuilt from scratch on every refresh, so the
    ``QTreeWidgetItem`` an expansion belongs to is gone by the time the new one
    exists. Keying by row *identity* rather than position lets the state survive
    that: a batch that gained a cluster, or moved because another one was swept,
    still reopens. Rows whose ``key`` is ``None`` are skipped.
    """
    return {
        k
        for item in _tree_items(tree)
        if item.isExpanded() and (k := key(item)) is not None
    }


def _restore_expanded(tree: QtWidgets.QTreeWidget, keys: set, key) -> None:
    """Reopen the branches named by ``keys`` (the inverse of _expanded_keys)."""
    for item in _tree_items(tree):
        if key(item) in keys:
            item.setExpanded(True)


# --------------------------------------------------------------------------- #
# Off-thread task runner
# --------------------------------------------------------------------------- #


class _Task(QtCore.QThread):
    """Run a no-arg callable on a worker thread; emit its result or exception.

    The controller opens its own SQLite connection per call, so running one of
    its methods here is thread-safe: nothing touches a connection owned by the
    UI thread. Widgets are never touched from ``run``; results come back via the
    queued-connection signals.
    """

    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # executed on the worker thread
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — reported to the UI as a dialog
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)


class _View(QtWidgets.QWidget):
    """Base view holding the controller, coordinator, and the async helper."""

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__()
        self.controller = controller
        self.main = main
        self._tasks: set[_Task] = set()

    def refresh(self) -> None:  # overridden by views that show live data
        """Reload this view's data from the controller."""

    def _add_sku_filter(self, row: QtWidgets.QHBoxLayout) -> QtWidgets.QComboBox:
        """Build the shared SKU-filter dropdown into ``row`` and return it.

        Both archive views (data bank, batch average) front their tree with the
        same filter: a ``SKU:`` label and a combo whose ``currentIndexChanged``
        re-runs this view's ``refresh``. The rows are (re)filled at refresh time
        by :func:`_repopulate_sku_filter`, and the caller reads the choice back
        with ``currentData()``. Kept here so the wiring lives in one place and
        the two views can't drift.
        """
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(self.refresh)
        row.addWidget(QtWidgets.QLabel("SKU:"))
        row.addWidget(combo)
        return combo

    def _prompt_discard_reason(self, title: str, message: str, default: str = "") -> str | None:
        """Ask for an optional multi-line reason before a discard; ``None`` on Cancel.

        Shared by the Ingest table, the Ingest bad-files row, and the Marking
        tab so their discard-confirmation copy and validation can't drift
        apart between the three.
        """
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, title, message, default)
        if not ok:
            return None
        return text.strip()

    def _defer(self, fn) -> None:
        """Run ``fn`` from the event loop once the current signal has unwound.

        The escape hatch for a handler that rebuilds the very widget whose
        signal invoked it: a refresh clears its tree, and freeing the row Qt is
        still emitting for is a use-after-free that takes the app down. Passing
        ``self`` as the context object drops the call if this view is destroyed
        before it fires.
        """
        QtCore.QTimer.singleShot(0, self, fn)

    def _run_async(self, fn, on_success, *, busy=()) -> None:
        """Run ``fn`` off the UI thread; call ``on_success(result)`` when done.

        ``busy`` widgets are disabled and a wait cursor shown for the duration.
        Any exception becomes a critical dialog instead of a crash.
        """
        for w in busy:
            w.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        task = _Task(fn, self)
        self._tasks.add(task)

        def cleanup() -> None:
            QtWidgets.QApplication.restoreOverrideCursor()
            for w in busy:
                w.setEnabled(True)
            self._tasks.discard(task)

        def handle_success(result) -> None:
            cleanup()
            on_success(result)

        def handle_failure(exc) -> None:
            cleanup()
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))

        task.succeeded.connect(handle_success)
        task.failed.connect(handle_failure)
        task.finished.connect(task.deleteLater)
        task.start()


# --------------------------------------------------------------------------- #
# 1. Ingest / Unmarked view
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# 2. Marking view
# --------------------------------------------------------------------------- #


class MarkingView(_View):
    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        #: bumped on each shot switch so a slow channel load for a previous shot
        #: is ignored when it finally returns.
        self._channel_token = 0

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.shot_combo = QtWidgets.QComboBox()
        self.shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        form.addRow("Unmarked shot:", self.shot_combo)

        # Pre-filled from the AI 1 / AI 2 DAQ convention; still editable so a
        # capture that breaks the convention can be tagged by hand.
        self.ml_combo = QtWidgets.QComboBox()
        self.se_combo = QtWidgets.QComboBox()
        form.addRow("Muzzle Left channel:", self.ml_combo)
        form.addRow("Shooter's Ear channel:", self.se_combo)

        self.ammo_combo = QtWidgets.QComboBox()
        # Editable so a one-off ammo can still be typed, but the configured
        # presets (Settings ▸ Ammo definitions) are one click away.
        self.ammo_combo.setEditable(True)
        self.ammo_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        form.addRow("Ammo *:", self.ammo_combo)
        self.sku_edit = QtWidgets.QLineEdit()
        form.addRow("SKU override:", self.sku_edit)
        self.platform_edit = QtWidgets.QLineEdit()
        form.addRow("Platform override:", self.platform_edit)
        self.cluster_edit = QtWidgets.QLineEdit()
        form.addRow("Cluster override:", self.cluster_edit)
        self.shot_order_edit = QtWidgets.QLineEdit()
        # Role is derived, never entered: echo it live so the user can see which
        # shot of the string this is about to become.
        self.shot_order_edit.textChanged.connect(self._update_role_preview)
        form.addRow("Shot order:", self.shot_order_edit)
        self.role_label = QtWidgets.QLabel(_EMPTY)
        form.addRow("Role (derived):", self.role_label)
        self.wind_edit = QtWidgets.QLineEdit()
        form.addRow("Wind speed (mph):", self.wind_edit)
        self.temp_edit = QtWidgets.QLineEdit()
        form.addRow("Temp (°F):", self.temp_edit)
        self.rh_edit = QtWidgets.QLineEdit()
        form.addRow("Relative humidity (%):", self.rh_edit)

        layout.addLayout(form)

        self.mark_btn = QtWidgets.QPushButton("Mark")
        self.mark_btn.clicked.connect(self._mark)
        self.discard_btn = QtWidgets.QPushButton("Discard")
        self.discard_btn.setToolTip("Remove this shot and ignore its file on future scans.")
        self.discard_btn.clicked.connect(self._discard_current)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.mark_btn)
        button_row.addWidget(self.discard_btn)
        layout.addLayout(button_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    # ---- population ----------------------------------------------------- #

    def refresh(self) -> None:
        """Reload the unmarked-shot picker, preserving the current selection."""
        self._populate_ammo()
        current = self._current_shot_id()
        shots = self.controller.unmarked_shots()
        self.shot_combo.blockSignals(True)
        self.shot_combo.clear()
        for s in shots:
            self.shot_combo.addItem(f"#{s.id}  {Path(s.source_file).name}", s)
        self.shot_combo.blockSignals(False)

        index = self._index_of_shot(current)
        if index is None:
            self.shot_combo.setCurrentIndex(0 if shots else -1)
            self._on_shot_changed()
        else:
            self.shot_combo.setCurrentIndex(index)

    def _populate_ammo(self) -> None:
        """Reload the ammo preset list, keeping whatever the user has typed/chosen."""
        # This runs synchronously from refresh() (including at launch, via
        # MainWindow.notify_changed), so a malformed ammo_definitions setting must
        # surface as a dialog rather than escaping as an unhandled crash — the
        # same treatment the async config read paths get from _run_async.
        try:
            presets = self.controller.ammo_definitions()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            presets = []
        current = self.ammo_combo.currentText()
        self.ammo_combo.blockSignals(True)
        self.ammo_combo.clear()
        self.ammo_combo.addItems(presets)
        # Leave the field blank rather than silently defaulting to the first
        # preset — ammo is required, so the user must pick or type it.
        self.ammo_combo.setCurrentText(current)
        if not current:
            self.ammo_combo.setCurrentIndex(-1)
        self.ammo_combo.blockSignals(False)

    def select_shot(self, shot_id: int) -> None:
        """Focus the picker on ``shot_id`` (called from the Ingest view)."""
        self.refresh()
        index = self._index_of_shot(shot_id)
        if index is not None:
            self.shot_combo.setCurrentIndex(index)

    def _current_shot(self) -> Shot | None:
        data = self.shot_combo.currentData()
        return data if isinstance(data, Shot) else None

    def _current_shot_id(self) -> int | None:
        shot = self._current_shot()
        return shot.id if shot else None

    def _index_of_shot(self, shot_id: int | None) -> int | None:
        if shot_id is None:
            return None
        for i in range(self.shot_combo.count()):
            data = self.shot_combo.itemData(i)
            if isinstance(data, Shot) and data.id == shot_id:
                return i
        return None

    def _update_role_preview(self, *_args) -> None:
        """Echo the FRP / Regular role implied by the entered shot order.

        Falls back to the shot's own order when the box is blank, since an empty
        field means "keep what the filename gave it", not "no order".
        """
        text = self.shot_order_edit.text().strip()
        if text:
            try:
                order = int(text)
            except ValueError:
                self.role_label.setText(_EMPTY)
                return
        else:
            shot = self._current_shot()
            order = shot.shot_order if shot else None
        role = role_for_order(order)
        self.role_label.setText(role.label if role else _EMPTY)

    def _on_shot_changed(self, *_args) -> None:
        self._channel_token += 1
        token = self._channel_token
        shot = self._current_shot()

        # Prefill override placeholders from the shot's provisional filename keys.
        self.sku_edit.setPlaceholderText(shot.suppressor_sku or "" if shot else "")
        self.platform_edit.setPlaceholderText(shot.test_platform or "" if shot else "")
        self.cluster_edit.setPlaceholderText(
            _str_or_empty(shot.cluster_index) if shot else ""
        )
        self.shot_order_edit.setPlaceholderText(_str_or_empty(shot.shot_order) if shot else "")
        self._update_role_preview()

        self._set_channel_choices([], loading=True)
        if shot is None:
            self._set_channel_choices([])
            return

        def load():
            # Fetch the names and the DAQ-convention tagging in one worker hop,
            # so the form opens already tagged for a conforming capture.
            channels = self.controller.channels_for(shot.source_file)
            return [c.name for c in channels], self.controller.suggested_channel_map(
                shot.source_file
            )

        def done(result):
            if token != self._channel_token:
                return  # a newer shot was selected; ignore this stale result
            names, suggested = result
            self._set_channel_choices(names, suggested=suggested)

        self._run_async(load, done)

    def _set_channel_choices(
        self,
        names: list[str],
        *,
        loading: bool = False,
        suggested: dict[str, MicPosition] | None = None,
    ) -> None:
        """Repopulate both channel combos, preselecting the auto-tagged mapping.

        ``suggested`` comes from the AI 1 / AI 2 convention. A channel it does not
        cover is left at ``(none)`` for the user to set, so a non-conforming
        capture degrades to manual tagging instead of being tagged wrongly.
        """
        for combo in (self.ml_combo, self.se_combo):
            combo.blockSignals(True)
            combo.clear()
            if loading:
                combo.addItem(_LOADING_LABEL)
                combo.setEnabled(False)
            else:
                combo.addItem(_NONE_LABEL)
                combo.addItems(names)
                combo.setEnabled(True)
            combo.blockSignals(False)
        if loading:
            return
        suggested = suggested or {}
        for position, combo in ((MicPosition.ML, self.ml_combo), (MicPosition.SE, self.se_combo)):
            name = next((n for n, p in suggested.items() if p is position), None)
            _select_channel(combo, name)

    # ---- mark ----------------------------------------------------------- #

    def _mark(self) -> None:
        shot = self._current_shot()
        if shot is None:
            QtWidgets.QMessageBox.information(self, "No shot", "No unmarked shot selected.")
            return

        ammo = self.ammo_combo.currentText().strip()
        if not ammo:
            QtWidgets.QMessageBox.warning(self, "Missing ammo", "Ammo is required to mark a shot.")
            return

        channel_map = _tagged_channel_map(self, self.ml_combo, self.se_combo)
        if channel_map is None:
            return

        try:
            cluster_index = _opt_int(self.cluster_edit.text())
            kwargs = dict(
                suppressor_sku=self.sku_edit.text().strip() or None,
                test_platform=self.platform_edit.text().strip() or None,
                cluster_index=cluster_index,
                shot_order=_opt_int(self.shot_order_edit.text()),
                wind_speed=_opt_float(self.wind_edit.text()),
                temp=_opt_float(self.temp_edit.text()),
                relative_humidity=_opt_float(self.rh_edit.text()),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        # Blank is allowed here — it falls back to the shot's filename cluster —
        # but an explicit override must be a real 1-based index. Catch it now,
        # not after the capture is read and the DSP has run on the worker thread.
        if cluster_index is not None and cluster_index < 1:
            QtWidgets.QMessageBox.warning(
                self, "Invalid cluster", "A cluster of 1 or greater is required."
            )
            return

        shot_id = shot.id
        self.status_label.setText("Marking…")
        self._run_async(
            lambda: self.controller.mark(shot_id, ammo=ammo, channel_map=channel_map, **kwargs),
            self._on_marked,
            busy=(self.mark_btn,),
        )

    def _on_marked(self, marked) -> None:
        shot = marked.shot
        role = shot.role.label if shot.role else _EMPTY
        parts = [
            f"Marked shot #{shot.id} — {marked.combination.label}, "
            f"batch #{marked.batch.id}, {marked.cluster.label}, "
            f"shot {shot.shot_order} ({role}).",
            "It is idle in the data bank; bring it forward there to feed the average.",
        ]
        for position in (MicPosition.ML, MicPosition.SE):
            result = marked.metrics.get(position)
            if result is not None:
                parts.append(
                    f"{position.label}: peak {result.peak_db:.2f} dB, "
                    f"LIAeq {result.liaeq_100ms_db:.2f} dBA"
                )
        self.status_label.setText("\n".join(parts))
        self.ammo_combo.setCurrentIndex(-1)
        self.ammo_combo.clearEditText()
        self.sku_edit.clear()
        self.platform_edit.clear()
        self.cluster_edit.clear()
        self.shot_order_edit.clear()
        self.wind_edit.clear()
        self.temp_edit.clear()
        self.rh_edit.clear()
        self.main.notify_changed()

    # ---- discard ---------------------------------------------------------- #

    def _discard_current(self) -> None:
        shot = self._current_shot()
        if shot is None:
            QtWidgets.QMessageBox.information(self, "No shot", "No unmarked shot selected.")
            return

        reason = self._prompt_discard_reason(
            "Discard shot",
            f"Discard {Path(shot.source_file).name!r}? It will be removed from "
            "Unmarked data sets and ignored on future ingest scans. Optional reason:",
        )
        if reason is None:
            return

        shot_id = shot.id
        self.status_label.setText("Discarding…")
        self._run_async(
            lambda: self.controller.discard_shot(shot_id, reason=reason or None),
            lambda _: self.main.notify_changed(),
            busy=(self.mark_btn, self.discard_btn),
        )


# --------------------------------------------------------------------------- #
# 3. Data bank: Combination -> Batch -> Cluster -> Shot tree
# --------------------------------------------------------------------------- #


class ShotEditDialog(QtWidgets.QDialog):
    """Correct a marked shot's fields, pre-filled from its current state.

    Purely a form: it validates and exposes the collected values via
    :meth:`values`; the caller re-marks the shot (which re-places it in the right
    combination/batch/cluster and recomputes metrics). SKU/platform/ammo default
    to the combination the shot was actually placed in — not the provisional
    filename keys — so an unchanged save is a true no-op.

    Inclusion is deliberately absent: bringing a shot forward is its own action
    in the tree, not something an edit can change by accident.
    """

    def __init__(
        self,
        shot: Shot,
        *,
        sku: str,
        platform: str,
        ammo: str,
        cluster_index: int | None,
        channel_names: list[str],
        ammo_definitions: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Edit shot #{shot.id}")
        self._shot = shot
        self._values: dict | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(Path(shot.source_file).name))
        form = QtWidgets.QFormLayout()

        self.ml_combo = QtWidgets.QComboBox()
        self.se_combo = QtWidgets.QComboBox()
        for combo in (self.ml_combo, self.se_combo):
            combo.addItem(_NONE_LABEL)
            combo.addItems(channel_names)
        _select_channel(self.ml_combo, shot.ml_channel)
        _select_channel(self.se_combo, shot.se_channel)
        form.addRow("Muzzle Left channel:", self.ml_combo)
        form.addRow("Shooter's Ear channel:", self.se_combo)

        self.ammo_combo = QtWidgets.QComboBox()
        self.ammo_combo.setEditable(True)
        self.ammo_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.ammo_combo.addItems(ammo_definitions or [])
        self.ammo_combo.setCurrentText(ammo or "")
        form.addRow("Ammo *:", self.ammo_combo)
        self.sku_edit = QtWidgets.QLineEdit(sku or "")
        form.addRow("SKU *:", self.sku_edit)
        self.platform_edit = QtWidgets.QLineEdit(platform or "")
        form.addRow("Platform *:", self.platform_edit)
        self.cluster_edit = QtWidgets.QLineEdit(_str_or_empty(cluster_index))
        form.addRow("Cluster *:", self.cluster_edit)
        self.shot_order_edit = QtWidgets.QLineEdit(_str_or_empty(shot.shot_order))
        self.shot_order_edit.textChanged.connect(self._update_role_preview)
        form.addRow("Shot order:", self.shot_order_edit)
        self.role_label = QtWidgets.QLabel(_EMPTY)
        form.addRow("Role (derived):", self.role_label)
        self.wind_edit = QtWidgets.QLineEdit(_str_or_empty(shot.wind_speed))
        form.addRow("Wind speed (mph):", self.wind_edit)
        self.temp_edit = QtWidgets.QLineEdit(_str_or_empty(shot.temp))
        form.addRow("Temp (°F):", self.temp_edit)
        self.rh_edit = QtWidgets.QLineEdit(_str_or_empty(shot.relative_humidity))
        form.addRow("Relative humidity (%):", self.rh_edit)
        # Read-only: the capture's fired-at time, pulled from the Dewesoft file at
        # marking. Shown for reference; not user-editable.
        form.addRow("Captured:", QtWidgets.QLabel(_format_captured_at(shot.captured_at)))
        layout.addLayout(form)
        self._update_role_preview()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_role_preview(self, *_args) -> None:
        """Echo the FRP / Regular role the entered order implies."""
        role = role_for_order(_safe_int(self.shot_order_edit.text()))
        self.role_label.setText(role.label if role else _EMPTY)

    def _on_accept(self) -> None:
        ammo = self.ammo_combo.currentText().strip()
        if not ammo:
            QtWidgets.QMessageBox.warning(self, "Missing ammo", "Ammo is required.")
            return
        sku = self.sku_edit.text().strip()
        platform = self.platform_edit.text().strip()
        if not sku or not platform:
            QtWidgets.QMessageBox.warning(
                self, "Missing key", "SKU and Platform are required to re-mark a shot."
            )
            return
        try:
            cluster_index = _opt_int(self.cluster_edit.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        if cluster_index is None or cluster_index < 1:
            QtWidgets.QMessageBox.warning(
                self, "Missing cluster", "A cluster of 1 or greater is required."
            )
            return

        channel_map = _tagged_channel_map(self, self.ml_combo, self.se_combo)
        if channel_map is None:
            return

        try:
            self._values = dict(
                ammo=ammo,
                channel_map=channel_map,
                suppressor_sku=sku,
                test_platform=platform,
                cluster_index=cluster_index,
                shot_order=_opt_int(self.shot_order_edit.text()),
                wind_speed=_opt_float(self.wind_edit.text()),
                temp=_opt_float(self.temp_edit.text()),
                relative_humidity=_opt_float(self.rh_edit.text()),
                # A full correction form: a cleared box means "blank this field",
                # not "leave it as it was", so write the optional fields exactly.
                replace_optional=True,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        self.accept()

    def values(self) -> dict:
        """The validated ``controller.mark`` kwargs. Valid only after Save."""
        assert self._values is not None, "values() called before an accepted Save"
        return self._values


class BatchEditDialog(QtWidgets.QDialog):
    """Edit a batch's session context: label, date, typical weather, notes.

    These are the *session*-level values. Each shot keeps its own specific
    weather, because conditions drift within a session; what is recorded here is
    what was typical for the day.

    A full-form write — a cleared box blanks the stored field.
    """

    def __init__(self, batch, *, combination_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit batch #{batch.id}")
        self._values: dict | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(combination_label))
        form = QtWidgets.QFormLayout()

        self.label_edit = QtWidgets.QLineEdit(batch.label or "")
        self.label_edit.setPlaceholderText("e.g. Morning string")
        form.addRow("Session label:", self.label_edit)
        self.date_edit = QtWidgets.QLineEdit(batch.session_date or "")
        self.date_edit.setPlaceholderText("YYYY-MM-DD")
        form.addRow("Session date:", self.date_edit)
        self.wind_edit = QtWidgets.QLineEdit(_str_or_empty(batch.wind_speed))
        form.addRow("Typical wind (mph):", self.wind_edit)
        self.temp_edit = QtWidgets.QLineEdit(_str_or_empty(batch.temp))
        form.addRow("Typical temp (°F):", self.temp_edit)
        self.rh_edit = QtWidgets.QLineEdit(_str_or_empty(batch.relative_humidity))
        form.addRow("Typical RH (%):", self.rh_edit)
        self.notes_edit = QtWidgets.QPlainTextEdit(batch.notes or "")
        self.notes_edit.setFixedHeight(80)
        form.addRow("Notes:", self.notes_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        date = self.date_edit.text().strip()
        if date:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid date", f"{date!r} is not a YYYY-MM-DD date."
                )
                return
        try:
            self._values = dict(
                label=self.label_edit.text().strip() or None,
                session_date=date or None,
                wind_speed=_opt_float(self.wind_edit.text()),
                temp=_opt_float(self.temp_edit.text()),
                relative_humidity=_opt_float(self.rh_edit.text()),
                notes=self.notes_edit.toPlainText().strip() or None,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        self.accept()

    def values(self) -> dict:
        """The validated ``controller.update_batch`` kwargs. Valid only after Save."""
        assert self._values is not None, "values() called before an accepted Save"
        return self._values


class DataBankView(_View):
    """The data bank: every combination, batch, cluster, and shot the app holds.

    Nothing is filtered here — a shot left out of an average is still part of the
    archive, shown idle. The bring-forward actions on this tree are what move a
    shot into its batch's average; the Batch average tab then shows the result.

    Inclusion is rendered as a checkbox on each shot row so the state is visible
    at a glance across a 50-cluster batch, and a cluster row offers the
    bring-whole-cluster-forward shortcut. Because the flag lives on the shot,
    tickng a cluster and then un-ticking two of its shots is exactly how a batch
    lands on 3 FRPs and 5 regulars.
    """

    _COLUMNS = [
        "Combination / Batch / Cluster / Shot",
        "Detail",
        "Role",
        "Timestamp",
        _PRETRIGGER_FLOOR_LABEL,
        "Compare",
    ]
    #: The pre-trigger floor diagnostic, filled on shot rows only (a cluster has
    #: no single baseline). Both mics share the cell — a shot row stands for the
    #: capture, not for one channel — in the same "ML … SE …" shape the Detail
    #: column already uses for the channel tags. Located rather than hard-coded,
    #: so inserting a column ahead of it cannot silently write into its neighbour.
    _FLOOR_COL = _COLUMNS.index(_PRETRIGGER_FLOOR_LABEL)
    #: The Compare column trails the rest; only shot rows fill it, with one
    #: button per marked mic (ML, SE, or both).
    _COMPARE_COL = len(_COLUMNS) - 1
    #: Wide enough for the ML and SE buttons side by side — resizeColumnToContents
    #: sees no text behind them (they are widgets, not cell text), so this column
    #: is sized by hand rather than measured, same as Batch average's button columns.
    _COMPARE_COL_WIDTH = 130

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        #: Set while refresh() repopulates the tree, so the itemChanged handler
        #: does not treat programmatic check-state writes as user clicks.
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)

        # SKU filter: work one SKU at a time so a large bank shows only the
        # combinations under it. Changing it re-runs refresh(), which rebuilds
        # both the dropdown and the filtered tree.
        filter_row = QtWidgets.QHBoxLayout()
        self.sku_combo = self._add_sku_filter(filter_row)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(self._COLUMNS)
        self.tree.itemSelectionChanged.connect(self._update_actions_enabled)
        # Double-click means "edit" here, so Qt's default expand/collapse on the
        # same gesture is off: a batch row is editable *and* has children, and
        # one double-click must not both toggle the branch and pop a modal
        # (cancelling the modal would leave the branch toggled anyway). Since
        # refresh() leaves the tree collapsed, expanding is the branch arrow and
        # the keyboard.
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)
        _style_grid_tree(self.tree)
        layout.addWidget(self.tree)

        button_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        self.include_btn = QtWidgets.QPushButton("Bring forward")
        self.include_btn.setEnabled(False)
        self.include_btn.clicked.connect(lambda: self._set_inclusion(True))
        self.exclude_btn = QtWidgets.QPushButton("Set idle…")
        self.exclude_btn.setEnabled(False)
        self.exclude_btn.clicked.connect(lambda: self._set_inclusion(False))
        self.edit_btn = QtWidgets.QPushButton("Edit…")
        self.edit_btn.setEnabled(False)
        self.edit_btn.clicked.connect(self._edit_selected)
        self.close_btn = QtWidgets.QPushButton("Close batch")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self._close_batch)
        button_row.addWidget(refresh_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.include_btn)
        button_row.addWidget(self.exclude_btn)
        button_row.addWidget(self.edit_btn)
        button_row.addWidget(self.close_btn)
        layout.addLayout(button_row)

    # ---- render ---------------------------------------------------------- #

    def refresh(self) -> None:
        # A pure read/render: rebuild the filter and tree from data_bank() without
        # touching the archive, so this stays safe on navigation (tab and SKU-filter
        # changes both land here). Pruning empty clusters/batches/combinations is a
        # destructive step, so it lives in notify_changed() — run after a mutation,
        # not on every refresh.
        #
        # Take the expansion snapshot before the rebuild: every edit, tick and
        # bring-forward lands here, and re-collapsing the archive under the user
        # each time would lose the place they were working in.
        open_rows = _expanded_keys(self.tree, self._row_key)
        self._loading = True
        try:
            # One read feeds both the filter and the tree: data_bank_view() returns
            # the full SKU list plus the tree already narrowed to the active SKU,
            # off a single all_combinations() scan. Read the current selection first,
            # then repopulate — a swept SKU may have vanished, and the helper (which
            # blocks the combo's own signal so the rebuild doesn't re-enter refresh)
            # falls back to "All SKUs" for it, matching the view's full-tree fallback.
            sku = self.sku_combo.currentData()
            skus, nodes = self.controller.data_bank_view(sku=sku)
            _repopulate_sku_filter(self.sku_combo, skus)
            self.tree.clear()
            for node in nodes:
                self.tree.addTopLevelItem(self._combination_item(node))
        finally:
            self._loading = False
        # Size the columns against every row, then collapse back to whatever was
        # open before. A first load has nothing to restore, so the tree opens on
        # the Combination rows only — a big archive is a short list rather than a
        # wall of shots. Sizing while fully expanded means the widths already fit
        # the deeper rows by the time one is opened.
        self.tree.expandAll()
        for col in range(len(self._COLUMNS)):
            if col == self._COMPARE_COL:
                # Contents-sizing measures cell text, and this column's cells
                # are empty behind their buttons; give it a hand-picked width
                # instead, same as Batch average's button columns.
                self.tree.setColumnWidth(col, self._COMPARE_COL_WIDTH)
                continue
            self.tree.resizeColumnToContents(col)
        self.tree.collapseAll()
        _restore_expanded(self.tree, open_rows, self._row_key)
        self._update_actions_enabled()

    @staticmethod
    def _row_key(item: QtWidgets.QTreeWidgetItem):
        """Identify a row by what it stands for, not where it sits.

        Rows carry their payload as ``(kind, obj, *context)``; the kind plus the
        record's primary key is stable across a rebuild. An unsaved record (no
        id) has nothing to match on, so it is left out.
        """
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload or payload[1].id is None:
            return None
        return (payload[0], payload[1].id)

    def _combination_item(self, node) -> QtWidgets.QTreeWidgetItem:
        combo = node.combination
        item = QtWidgets.QTreeWidgetItem(
            [combo.label, f"{len(node.batches)} batch(es)", "", ""]
        )
        item.setData(0, QtCore.Qt.UserRole, ("combination", combo))
        for b_node in node.batches:
            item.addChild(self._batch_item(b_node, combo))
        return item

    def _batch_item(self, node, combo) -> QtWidgets.QTreeWidgetItem:
        batch = node.batch
        state = "closed" if batch.closed else "open"
        detail = f"[{state}]  {node.n_shots} shot(s)  {node.status.summary()}"
        item = QtWidgets.QTreeWidgetItem(
            [f"Batch #{batch.id}  {batch.title}", detail, "", batch.weather_summary]
        )
        item.setData(0, QtCore.Qt.UserRole, ("batch", batch, combo))
        for c_node in node.clusters:
            item.addChild(self._cluster_item(c_node, batch, combo))
        return item

    def _cluster_item(self, node, batch, combo) -> QtWidgets.QTreeWidgetItem:
        cluster = node.cluster
        item = QtWidgets.QTreeWidgetItem(
            [
                f"{cluster.label}  (#{cluster.id})",
                f"{len(node.shots)} shot(s), {node.n_included} included",
                "",
                "",
            ]
        )
        item.setData(0, QtCore.Qt.UserRole, ("cluster", cluster, batch, combo))
        for shot in node.shots:
            shot_item = self._shot_item(shot, cluster, batch, combo, node.floors.get(shot.id))
            item.addChild(shot_item)
            # setItemWidget needs the row parented first, same reason as the
            # Batch average tab's Compare column (see BatchAverageView._load_report).
            self.tree.setItemWidget(
                shot_item, self._COMPARE_COL, self._compare_widget(shot, cluster, batch, combo)
            )
        return item

    def _shot_item(
        self, shot, cluster, batch, combo, floors: dict[MicPosition, float] | None = None
    ) -> QtWidgets.QTreeWidgetItem:
        tags = f"ML:{shot.ml_channel or _EMPTY}  SE:{shot.se_channel or _EMPTY}"
        if shot.exclusion_reason:
            tags = f"{tags}  — {shot.exclusion_reason}"
        role = shot.role
        item = QtWidgets.QTreeWidgetItem(
            [
                f"Shot #{shot.id}  order {_str_or_empty(shot.shot_order) or _EMPTY}"
                f"  {Path(shot.source_file).name}",
                tags,
                role.label if role else _EMPTY,
                _format_captured_at(shot.captured_at),
                _format_floor_pair(floors),
            ]
        )
        # The checkbox *is* the inclusion flag: the data bank's whole job is
        # showing which shots are carried forward and letting that be toggled.
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(
            0, QtCore.Qt.Checked if shot.included else QtCore.Qt.Unchecked
        )
        # Carry the shot's cluster/batch/combination so an edit can pre-fill the
        # context it was actually placed in (which may differ from its
        # provisional filename keys after an override).
        item.setData(0, QtCore.Qt.UserRole, ("shot", shot, cluster, batch, combo))
        return item

    # ---- selection ------------------------------------------------------- #

    def _selected_entry(self) -> tuple | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole)

    def _selected_batch(self) -> tuple | None:
        """The selected row's batch, whatever level it sits at (``None`` above batch)."""
        entry = self._selected_entry()
        if not entry:
            return None
        kind = entry[0]
        if kind == "batch":
            return entry[1]
        if kind in ("cluster", "shot"):
            return entry[-2]
        return None

    def _update_actions_enabled(self) -> None:
        entry = self._selected_entry()
        kind = entry[0] if entry else None
        # Inclusion applies to a shot or a whole cluster; combinations and
        # batches are containers, not roll-up units.
        self.include_btn.setEnabled(kind in ("shot", "cluster"))
        self.exclude_btn.setEnabled(kind in ("shot", "cluster"))
        # A batch (session metadata) or a shot (re-mark) can be edited.
        self.edit_btn.setEnabled(kind in ("batch", "shot"))
        batch = self._selected_batch() if kind == "batch" else None
        self.close_btn.setEnabled(batch is not None and not batch.closed)

    # ---- inclusion ------------------------------------------------------- #

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        """Persist a shot checkbox the user just toggled.

        Guarded by ``_loading`` so the check states written during a refresh do
        not each fire a write back to the database.
        """
        if self._loading or column != 0:
            return
        entry = item.data(0, QtCore.Qt.UserRole)
        if not entry or entry[0] != "shot":
            return
        shot = entry[1]
        included = item.checkState(0) == QtCore.Qt.Checked
        if included == shot.included:
            return
        # Qt is still mid-emission on `item` here, and the write ends in a
        # refresh that clears the tree `item` lives in — freeing it under the
        # emission still on the stack crashes Qt. So hand the work to the event
        # loop, carrying the shot id and state by value rather than the row,
        # and let the emission unwind before anything is rebuilt.
        self._defer(lambda: self._apply_checkbox(shot.id, included))

    def _apply_checkbox(self, shot_id: int, included: bool) -> None:
        """Write back a toggled checkbox, once the tree is safe to rebuild."""
        try:
            self.controller.include_shot(shot_id, included)
        except Exception as exc:  # noqa: BLE001 — surface to the user as a dialog
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
        # Refresh either way: on failure it snaps the checkbox back to the
        # stored flag rather than leaving the row lying about what was saved.
        self.main.notify_changed()

    def _set_inclusion(self, included: bool) -> None:
        """Bring the selected shot or cluster forward, or set it idle with a reason."""
        entry = self._selected_entry()
        if not entry or entry[0] not in ("shot", "cluster"):
            return
        kind, target = entry[0], entry[1]

        reason = None
        if not included:
            # Only an exclusion carries a reason; inclusion clears it.
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Set idle",
                "Reason (optional, e.g. high winds):",
                QtWidgets.QLineEdit.Normal,
                getattr(target, "exclusion_reason", "") or "",
            )
            if not ok:
                return
            reason = text.strip() or None

        try:
            if kind == "shot":
                self.controller.include_shot(target.id, included, reason=reason)
            else:
                self.controller.include_cluster(target.id, included, reason=reason)
        except Exception as exc:  # noqa: BLE001 — surface to the user as a dialog
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        self.main.notify_changed()

    # ---- edit ------------------------------------------------------------ #

    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        entry = item.data(0, QtCore.Qt.UserRole)
        if entry and entry[0] in ("shot", "batch"):
            # Deferred for the same reason as the checkbox: saving the edit
            # refreshes the tree, which would free this row mid-emission.
            self._defer(self._edit_selected)

    def _edit_selected(self) -> None:
        entry = self._selected_entry()
        if not entry:
            return
        if entry[0] == "batch":
            self._edit_batch(entry[1], entry[2])
        elif entry[0] == "shot":
            self._edit_shot(entry[1], entry[2], entry[3], entry[4])

    def _edit_batch(self, batch, combo) -> None:
        dialog = BatchEditDialog(batch, combination_label=combo.label, parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            self.controller.update_batch(batch.id, **dialog.values())
        except Exception as exc:  # noqa: BLE001 — surface to the user as a dialog
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        self.main.notify_changed()

    def _edit_shot(self, shot, cluster, batch, combo) -> None:
        # Re-marking a shot whose batch is closed re-places it in a *new* open
        # batch (a closed batch is never the combination's open batch), so warn.
        if batch.closed:
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Batch closed",
                f"Batch #{batch.id} is closed. Saving changes will move shot "
                f"#{shot.id} into a new open session. Continue?",
            )
            if confirm != QtWidgets.QMessageBox.Yes:
                return

        # Load the raw channels off the UI thread, then open the pre-filled dialog.
        self._run_async(
            lambda: self.controller.channels_for(shot.source_file),
            lambda channels: self._open_shot_dialog(shot, cluster, combo, channels),
            busy=(self.edit_btn,),
        )

    def _open_shot_dialog(self, shot, cluster, combo, channels) -> None:
        dialog = ShotEditDialog(
            shot,
            sku=combo.sku,
            platform=combo.platform,
            ammo=combo.ammo,
            cluster_index=cluster.cluster_index,
            channel_names=[c.name for c in channels],
            ammo_definitions=self.controller.ammo_definitions(),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.values()
        shot_id = shot.id
        self._run_async(
            lambda: self.controller.mark(shot_id, **values),
            lambda _result: self.main.notify_changed(),
            busy=(self.edit_btn,),
        )

    def _close_batch(self) -> None:
        batch = self._selected_batch()
        if batch is None:
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Close batch",
            f"Close batch #{batch.id}? Further testing for this combination "
            "starts a new session.",
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.controller.close_batch(batch.id)
        except Exception as exc:  # noqa: BLE001 — surface to the user as a dialog
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        self.main.notify_changed()

    # ---- compare ----------------------------------------------------------- #

    def _compare_widget(self, shot: Shot, cluster, batch, combo) -> QtWidgets.QWidget:
        """One shot row's Compare cell: a button per marked mic, ML then SE.

        Unlike the Batch average tab (where a shot row sits under one position's
        slot, so one button pins one curve), a data-bank shot row is not scoped
        to a mic — both channels live on the same row. An unmarked shot has
        neither channel set and gets no button at all, since there is no curve
        to pin yet.
        """
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(4)
        for position, channel in (
            (MicPosition.ML, shot.ml_channel),
            (MicPosition.SE, shot.se_channel),
        ):
            if not channel:
                continue
            row.addWidget(self._compare_button(shot, position, cluster, batch, combo))
        row.addStretch(1)
        return container

    def _series_for(
        self, shot: Shot, position: MicPosition, cluster, batch, combo
    ) -> CompareSeries:
        """Name one data-bank shot/mic as a Compare-tab series.

        Builds off the shared :func:`_compare_series`, adding the one thing
        that is specific to this tab: idle shots (which Batch average never
        shows) get a note in the detail saying so.
        """
        where = (
            f"C{shot.cluster_index}·S{shot.shot_order}"
            if shot.cluster_index
            else f"S{shot.shot_order}"
        )
        return _compare_series(
            shot_id=shot.id,
            position=position,
            sku=combo.sku if combo else "?",
            where=where,
            combo_label=combo.label if combo else "?",
            batch_id=batch.id,
            note="" if shot.included else "  (idle — not brought forward)",
        )

    def _compare_button(
        self, shot: Shot, position: MicPosition, cluster, batch, combo
    ) -> QtWidgets.QPushButton:
        """A small ML/SE button that pins one channel — the row has room for two."""
        series = self._series_for(shot, position, cluster, batch, combo)
        button = QtWidgets.QPushButton(position.value)
        button.setToolTip(
            f"Overlay this shot's {position.label} curve on the Compare tab.\n\n"
            f"{series.detail}"
        )
        button.clicked.connect(lambda: self._pin_for_compare(button, series))
        return button

    def _pin_for_compare(
        self, button: QtWidgets.QPushButton, series: CompareSeries
    ) -> None:
        """Pin one shot/mic to the Compare tab, without leaving the data bank.

        Uses the shared :func:`_pin_compare_series`, but reverts the button to
        its own label (ML or SE) rather than the other tab's "Compare" constant.
        """
        _pin_compare_series(self.main, button, series, series.position.value)


# --------------------------------------------------------------------------- #
# 4. Report view
# --------------------------------------------------------------------------- #


class MetricGraph(QtWidgets.QWidget):
    """One metric's time series, for one shot or for several overlaid.

    Used unchanged by both graphing tabs — the Batch average tab hands it a
    single trace, the Compare tab a list — so the framing buttons, the
    level-weighting dropdown, the point readout, and the theming are one
    implementation that cannot drift between the two. Overlaying is the only
    difference between the callers, and it is expressed as the *count* of series
    passed in: :meth:`show_trace` is :meth:`show_traces` with one.

    A thin wrapper over a :class:`pyqtgraph.PlotWidget`, with a header row above
    it: the graph title on the left and a level-weighting dropdown on the right.
    That dropdown chooses how the SPL-over-time curve is drawn — the raw
    per-sample level (a point cloud) or a Fast/Slow time-weighted RMS envelope (a
    continuous line). It emits :attr:`smoothingChanged` when the user switches so
    the owning view can re-request the trace(s). Colours track the active
    light/dark palette so the plot doesn't clash with the rest of the window.

    With more than one series, each curve takes the next colour from
    :data:`_SERIES_COLORS` and a legend keyed by the caller's labels appears; the
    annotations that bracket a single curve (calculation window, drawn extent)
    become the *union* over the series, so the framing buttons still land on a
    span that contains every curve's. A lone series keeps exactly the plain,
    legend-free graph it had.

    Clicking a point on a drawn curve snaps to the nearest sample and shows its
    value (with the trace's unit) and time in a small readout box at the bottom
    right, plus a highlight ring on the picked sample. The box has a Clear button
    that dismisses both.
    """

    #: Emitted when the user picks a different level-weighting from the dropdown.
    smoothingChanged = QtCore.Signal()

    #: Dropdown entries: (label, ``build_metric_trace`` smoothing mode).
    _SMOOTHING_OPTIONS = (
        ("Instantaneous", SMOOTHING_INSTANT),
        ("Fast (125 ms)", SMOOTHING_FAST),
        ("Slow (1 s)", SMOOTHING_SLOW),
    )

    #: Colour cycle for overlaid series, in the order they are handed over.
    #: Entry 0 is the blue a single-trace graph has always drawn, so the Batch
    #: average tab is unchanged by there being a cycle at all. Chosen to stay
    #: distinguishable against both the light and dark plot backgrounds.
    _SERIES_COLORS = (
        (66, 135, 245),   # blue
        (214, 90, 70),    # red
        (46, 160, 90),    # green
        (170, 90, 200),   # purple
        (230, 160, 30),   # amber
        (40, 180, 190),   # teal
        (200, 70, 140),   # magenta
        (120, 130, 145),  # slate
    )
    #: Colour of the peak marker / level line on a *single*-series graph, where
    #: the annotation reads best set apart from the curve. Overlaid series take
    #: their own colour for these instead — with several curves in the plot, the
    #: shared colour is the only thing tying a peak line to the curve it marks.
    _MARK_COLOR = (214, 90, 70)
    #: Yellow dotted verticals on the first/last sample — the drawn-curve extent.
    _BOUND_PEN = pg.mkPen((240, 200, 0), width=1, style=QtCore.Qt.DotLine)
    #: Dashed verticals on *both* edges of the metric's calculation window.
    #: Everything outside them is drawn for context and fed into no reported number.
    _WINDOW_PEN = pg.mkPen((150, 150, 160), width=1, style=QtCore.Qt.DashLine)
    #: Both window labels run vertically up their line and sit at the top of the
    #: view. ``rotateAxis`` is given in the *line's own* coordinates, and the line
    #: item already carries the 90° rotation that stands it upright -- so (1, 0),
    #: its local x-axis, is the direction along the line, and pyqtgraph then picks
    #: above/below anchors that flip the label to the inside near a view edge.
    #: ``position`` is measured along the line within the view, so ~1.0 pins the
    #: text to the top however the user pans or zooms (0.99 leaves a few px of
    #: inset). The explicit ``anchors`` are what make it *top*-aligned rather than
    #: top-centred: pyqtgraph's default pair for rotated text centres the label on
    #: ``position``, which hangs half of it above the view and clips it. Anchoring
    #: the text's far end (x = 1) instead pins its top edge and lets it hang down;
    #: the y component still flips the label to the line's other side near an edge.
    #: Vertical text keeps both labels legible when the window is narrow --
    #: Peak-10 ms-Leq's is only 25 ms wide.
    _WINDOW_LABEL_OPTS = {
        "position": 0.99,
        "rotateAxis": (1, 0),
        "anchors": [(1, 0), (1, 1)],
        "color": (150, 150, 160),
        "movable": False,
    }
    #: Highlight ring drawn on the sample the user clicks to read out.
    _PICK_BRUSH = pg.mkBrush(240, 200, 0)
    _PICK_PEN = pg.mkPen((30, 30, 30), width=1)
    #: How near (screen pixels) a click must land to a sample to select it.
    _PICK_TOLERANCE_PX = 20.0
    #: Left edge (ms) shared by every onset close-up: a fixed point on the
    #: capture's own time axis, just past the trigger, rather than wherever
    #: onset detection happened to fire. Fixed is the point -- the same slice of
    #: every shot lands in the same place, so close-ups compare shot to shot.
    _ONSET_ZOOM_START_MS = 10.5
    #: Widths (ms) of the onset close-ups -- ``_ONSET_ZOOM_START_MS`` to this far
    #: past it -- one button each, in the order they appear on the toolbar. 10 ms
    #: matches the Peak-10 ms-Leq integration length, so that close-up frames the
    #: same slice that metric reports on; 5 ms halves it for the rise itself.
    _ONSET_ZOOM_MS = (5.0, 10.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header: title on the left, level-weighting dropdown on the right.
        header = QtWidgets.QHBoxLayout()
        self._title_label = QtWidgets.QLabel("")
        self._title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("Level:"))
        self._smoothing_combo = QtWidgets.QComboBox()
        for label, mode in self._SMOOTHING_OPTIONS:
            self._smoothing_combo.addItem(label, mode)
        self._smoothing_combo.setToolTip(
            "How the SPL-over-time curve is drawn.\n"
            "Instantaneous: raw per-sample level (a point cloud).\n"
            "Fast / Slow: time-weighted RMS envelope (a smooth line)."
        )
        self._smoothing_combo.currentIndexChanged.connect(
            lambda *_: self.smoothingChanged.emit()
        )
        header.addWidget(self._smoothing_combo)
        layout.addLayout(header)

        self._plot = pg.PlotWidget()
        # Keep redraws cheap on a full 20k-sample frame: clip to the view and let
        # pyqtgraph peak-downsample when zoomed out.
        self._plot.setClipToView(True)
        self._plot.setDownsampling(auto=True, mode="peak")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "Time (ms)")
        # The legend is built once and refilled per render (pyqtgraph only ever
        # hands out one per plot). It stays hidden until there are at least two
        # curves to tell apart, so a single-trace graph is unchanged by it.
        self._legend = self._plot.addLegend(offset=(-12, 12))
        self._legend.setVisible(False)
        layout.addWidget(self._plot)

        # Bottom toolbar: ease-of-use view controls for the plot above, plus the
        # point-readout box pinned to the right.
        toolbar = QtWidgets.QHBoxLayout()
        self._auto_frame_btn = QtWidgets.QPushButton("Auto Frame")
        self._auto_frame_btn.setToolTip(
            "Snap the X range to the drawn curve's extent\n"
            "(first to last sample carrying a value)."
        )
        self._auto_frame_btn.setEnabled(False)
        self._auto_frame_btn.clicked.connect(self.auto_frame)
        toolbar.addWidget(self._auto_frame_btn)
        self._frame_window_btn = QtWidgets.QPushButton("Frame Calc Window")
        self._frame_window_btn.setToolTip(
            "Snap the X range to the calculation window\n"
            "(the samples this metric's number came from)."
        )
        self._frame_window_btn.setEnabled(False)
        self._frame_window_btn.clicked.connect(self.frame_calc_window)
        toolbar.addWidget(self._frame_window_btn)
        #: One onset close-up button per span in ``_ONSET_ZOOM_MS``, all enabled
        #: and disabled together (they share the one precondition: a drawn curve).
        self._frame_onset_btns: list[QtWidgets.QPushButton] = []
        for span_ms in self._ONSET_ZOOM_MS:
            btn = QtWidgets.QPushButton(f"+{span_ms:g} ms")
            btn.setToolTip(
                f"Snap the X range to {self._ONSET_ZOOM_START_MS:g}"
                f"–{self._ONSET_ZOOM_START_MS + span_ms:g} ms\n"
                "(the onset transient, in detail)."
            )
            btn.setEnabled(False)
            # Default-arg binding, not a bare closure over the loop variable,
            # which would leave every button framing the last span. ``*_`` eats
            # the ``checked`` flag ``clicked`` sends.
            btn.clicked.connect(lambda *_, ms=span_ms: self.frame_onset_zoom(ms))
            toolbar.addWidget(btn)
            self._frame_onset_btns.append(btn)
        toolbar.addStretch(1)

        # Readout box (bottom right): shows the clicked sample's value + time. The
        # label and its Clear button are hidden until a point is actually picked.
        self._readout_label = QtWidgets.QLabel("")
        self._readout_label.setStyleSheet(
            "QLabel {"
            " border: 1px solid palette(mid);"
            " border-radius: 3px;"
            " background: palette(base);"
            " padding: 2px 6px; }"
        )
        self._readout_label.setToolTip("Click a point on the graph to read its value.")
        self._readout_clear_btn = QtWidgets.QPushButton("Clear")
        self._readout_clear_btn.setToolTip("Dismiss the point readout.")
        self._readout_clear_btn.clicked.connect(self.clear_readout)
        toolbar.addWidget(self._readout_label)
        toolbar.addWidget(self._readout_clear_btn)
        layout.addLayout(toolbar)

        #: (x_first_ms, x_last_ms) of the current trace's finite (non-NaN) span, or
        #: None when no trace is shown. Drives both Auto Frame and the yellow curve-
        #: extent bound lines.
        self._x_bounds: tuple[float, float] | None = None
        #: (x_start_ms, x_end_ms) of the current trace's calculation window -- the
        #: same two times the dashed window lines mark -- or None when no trace is
        #: shown or the window has no width to frame. Drives Frame Calc Window.
        self._window_x_bounds: tuple[float, float] | None = None
        #: (y_min, y_max) covering every drawn curve (plus any level line, which
        #: autorange would also take in), or None when no trace is shown. Every
        #: framing button pins Y here, so the zoom levels stay comparable.
        self._y_bounds: tuple[float, float] | None = None
        #: The (label, trace, colour) series currently drawn, in the order they
        #: were handed over — which is also the legend's order. Kept so a plot
        #: click can find the sample it landed on. Empty whenever the plot shows
        #: a message rather than curves.
        self._series: list[tuple[str, MetricTrace, tuple[int, int, int]]] = []
        #: Scatter item marking the picked sample, or None when nothing is picked.
        self._pick_marker: pg.ScatterPlotItem | None = None

        # A click anywhere on the plot scene tries to select the nearest sample.
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._apply_theme()
        self.show_message("Click a metric cell on a shot row to graph it.")

    def current_smoothing(self) -> str:
        """The ``build_metric_trace`` smoothing mode currently selected."""
        return self._smoothing_combo.currentData()

    @classmethod
    def series_color(cls, index: int) -> tuple[int, int, int]:
        """The RGB an overlaid series at ``index`` is drawn in, cycling.

        Public so a view listing the same series (the Compare tab's swatches)
        keys off the graph's palette rather than keeping a second copy of it.
        """
        return cls._SERIES_COLORS[index % len(cls._SERIES_COLORS)]

    def _apply_theme(self) -> None:
        pal = self.palette()
        self._fg = pal.color(QtGui.QPalette.Text)
        self._plot.setBackground(pal.color(QtGui.QPalette.Base))
        for name in ("left", "bottom"):
            axis = self._plot.getAxis(name)
            axis.setPen(self._fg)
            axis.setTextPen(self._fg)
        # pyqtgraph draws legend text in its own default foreground, which is a
        # light grey that vanishes on a light theme's Base background. Track the
        # palette like the axes do, and float the labels on a Base-coloured
        # panel so they stay readable where curves run under them.
        self._legend.setLabelTextColor(self._fg)
        self._legend.setBrush(pg.mkBrush(pal.color(QtGui.QPalette.Base)))
        self._legend.setPen(pg.mkPen(pal.color(QtGui.QPalette.Mid)))

    def show_message(self, text: str) -> None:
        """Clear the plot and show a short prompt in place of a graph."""
        self._plot.clear()
        self._legend.clear()
        self._legend.setVisible(False)
        self._title_label.setText(text)
        self._plot.setLabel("left", "")
        self._x_bounds = None
        self._window_x_bounds = None
        self._y_bounds = None
        self._series = []
        self.clear_readout()
        self._auto_frame_btn.setEnabled(False)
        self._frame_window_btn.setEnabled(False)
        self._set_onset_btns_enabled(False)

    def _set_onset_btns_enabled(self, enabled: bool) -> None:
        """Enable/disable every onset close-up button at once."""
        for btn in self._frame_onset_btns:
            btn.setEnabled(enabled)

    def _frame_x(self, span: tuple[float, float] | None, padding: float = 0) -> None:
        """Snap the X range to ``span``, or do nothing when it is None.

        The one place the framing buttons meet. Y is pinned to the whole curve's
        vertical extent — the same range Auto Frame lands on — for *every* span,
        rather than autoranging to the framed slice. Letting Y refit meant a
        narrow close-up rescaled the axis to that slice's few dB, so the ticks
        collapsed to single-digit steps and the curve redrew at a wildly
        different height than the wider views: the zoom levels stopped being
        comparable. A fixed Y makes zooming in a purely horizontal move.
        """
        if span is None:
            return
        self._plot.setXRange(span[0], span[1], padding=padding)
        if self._y_bounds is None:
            self._plot.enableAutoRange(axis="y")
        else:
            self._plot.setYRange(*self._y_bounds)

    def auto_frame(self) -> None:
        """Snap the X range to the drawn curve's extent (first to last live sample).

        A no-op when no trace is shown.
        """
        self._frame_x(self._x_bounds)

    def frame_calc_window(self) -> None:
        """Snap the X range to the calculation window's start/end lines.

        The zoomed-in counterpart to :meth:`auto_frame`: same behaviour, but
        bracketing only the samples that fed the reported number. A little
        padding is kept so both window lines stay visible at the edges rather
        than sitting exactly on the frame. A no-op when the trace carries no
        window.
        """
        self._frame_x(self._window_x_bounds, padding=0.02)

    def frame_onset_zoom(self, span_ms: float) -> None:
        """Snap the X range to ``span_ms`` starting at :data:`_ONSET_ZOOM_START_MS`.

        The most zoomed-in of the framing actions: the onset transient itself.
        Unlike the other two, both edges are fixed times rather than fitted to
        the trace — that is the point, since the same slice of every shot then
        frames identically and close-ups can be compared across shots. A no-op
        when no trace is shown.
        """
        if self._x_bounds is None:
            return
        x0 = self._ONSET_ZOOM_START_MS
        self._frame_x((x0, x0 + span_ms), padding=0.02)

    def show_trace(self, trace, subtitle: str = "") -> None:
        """Render a :class:`~sound_metric_app.dsp.MetricTrace` as the sole graph.

        The one-series case of :meth:`show_traces`, spelled out because it is
        what the Batch average tab always wants: no legend, no label, and the
        annotations in their own colour rather than the curve's.
        """
        self.show_traces([("", trace, self.series_color(0))], subtitle)

    def show_traces(self, series, subtitle: str = "") -> None:
        """Draw ``series`` — ordered ``(label, MetricTrace, colour)`` — overlaid.

        Every trace is assumed to be the *same* metric over different shots, so
        they share one Y axis, one unit, and one set of framing bounds; the y
        label and the fallback title come from the first. An empty sequence
        leaves the prompt ``subtitle`` on an empty plot rather than an axis with
        nothing under it.

        Colour is the caller's to assign rather than this widget's to derive from
        position: a view that can hide a curve without unpinning it has to keep
        the survivors on the colours they already had, which position alone
        cannot express. :meth:`series_color` is the palette to spend.
        """
        series = [(label, trace, color) for label, trace, color in series]
        if not series:
            self.show_message(subtitle or "Nothing to graph.")
            return

        self._plot.clear()
        self._legend.clear()
        # One curve needs no key: the title already names it.
        self._legend.setVisible(len(series) > 1)
        # New curves invalidate any prior point pick (different samples/units).
        self._series = series
        self.clear_readout()
        first = series[0][1]
        self._title_label.setText(subtitle or first.title)
        self._plot.setLabel("left", first.y_label)
        multiple = len(series) > 1

        # Accumulated across the series, so every bound below brackets *all* of
        # them: the framing buttons must land somewhere that contains each curve
        # rather than whichever one happened to be drawn first.
        finite_x: list[float] = []
        finite_y: list[float] = []
        window_starts: list[float] = []
        window_ends: list[float] = []

        for label, trace, color in series:
            mark_color = color if multiple else self._MARK_COLOR
            # A label is only spent on a legend entry when there is a legend;
            # pyqtgraph adds one row per named curve regardless of visibility.
            self._draw_curve(trace, color, label if multiple else None)
            if trace.peak_index is not None:
                self._plot.addItem(
                    pg.InfiniteLine(
                        pos=float(trace.t_ms[trace.peak_index]), angle=90,
                        pen=pg.mkPen(mark_color, width=1),
                    )
                )
            if trace.level is not None:
                self._plot.addItem(
                    pg.InfiniteLine(
                        pos=trace.level, angle=0,
                        pen=pg.mkPen(mark_color, width=1, style=QtCore.Qt.DashLine),
                    )
                )
            if trace.window_start_index is not None:
                window_starts.append(float(trace.t_ms[trace.window_start_index]))
            if trace.window_end_index is not None:
                window_ends.append(float(trace.t_ms[trace.window_end_index]))
            # The drawn curve's extent: the finite (non-NaN) span rather than the
            # raw sample axis. The Impulse ∫p·dt curve is NaN before the onset
            # (the integral is undefined there), so framing to the full frame
            # would open with dead space at the left. Full-frame SPL traces are
            # all-finite, so their bounds are unchanged.
            finite = np.isfinite(trace.values)
            if not finite.any():
                continue
            xs = trace.t_ms[finite]
            finite_x += [float(xs[0]), float(xs[-1])]
            # Y extent of everything autorange would take in -- the curve plus
            # the horizontal level line -- so a framed slice keeps the Y range
            # the full view has. A non-finite level compares False against the
            # running min/max and drops out rather than poisoning the range,
            # which is why it is appended to an already-seeded list.
            finite_y += [
                float(np.min(trace.values[finite])),
                float(np.max(trace.values[finite])),
            ]
            if trace.level is not None:
                finite_y.append(float(trace.level))

        self._draw_window_markers(series, window_starts, window_ends)
        # Yellow dotted verticals bracket the drawn extent, so the data stays
        # visible however far the user pans or zooms. Note this is the *drawn*
        # extent, which runs to the end of the capture — the calculation
        # window's end is the separate dashed line above.
        if finite_x:
            x0, x1 = min(finite_x), max(finite_x)
            self._x_bounds = (x0, x1)
            y0, y1 = min(finite_y), max(finite_y)
            # A flat curve would give a zero-height range Qt cannot draw, so give
            # it a nominal 1 dB.
            if y1 <= y0:
                y0, y1 = y0 - 0.5, y0 + 0.5
            self._y_bounds = (y0, y1)
            for x in (x0, x1):
                self._plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=self._BOUND_PEN))
            self._auto_frame_btn.setEnabled(True)
            # The onset close-ups frame fixed times, so a drawn curve is their
            # only precondition -- they need no calculation window at all.
            self._set_onset_btns_enabled(True)
        else:
            self._x_bounds = None
            self._y_bounds = None
            self._auto_frame_btn.setEnabled(False)
            self._set_onset_btns_enabled(False)
        self._plot.enableAutoRange()

    def _draw_curve(self, trace, color: tuple[int, int, int], name: str | None) -> None:
        """Plot one trace's samples in ``color``, legending it as ``name`` if given."""
        if trace.connected:
            # Time-weighted envelope: a joined line reads as the continuous level
            # a meter shows. NaN samples break the line into gaps.
            self._plot.plot(
                trace.t_ms, trace.values, pen=pg.mkPen(color, width=1), name=name
            )
        else:
            # One dot per sample, no connecting line (pen=None). NaN samples
            # (silent Impulse tail) simply don't plot a point. pxMode keeps dots
            # a fixed screen size regardless of zoom.
            self._plot.plot(
                trace.t_ms,
                trace.values,
                pen=None,
                symbol="o",
                symbolSize=2,
                symbolPen=None,
                symbolBrush=pg.mkBrush(*color),
                pxMode=True,
                name=name,
            )

    def _draw_window_markers(
        self, series, starts: list[float], ends: list[float]
    ) -> None:
        """Bracket the calculation window and arm Frame Calc Window.

        The curves run straight through both edges so a pre-onset or late event
        stays visible, but only samples *between* them reached the reported
        numbers — the labels say so, since a curve that simply continues would
        otherwise read as all-included. The start also shows where onset
        detection fired — the same time for every metric bar Peak-10 ms-Leq,
        which opens its window a trailing-RMS length earlier so the bracket still
        contains every sample that fed the reported number. When nothing crossed
        the onset threshold the window falls back to the frame start, so the
        start label says that outright — otherwise a mis-triggered or silent
        capture reads as a confident onset at 0 ms.

        Overlaid series each detect their own onset, so one pair of lines is
        drawn at the *union* of their windows rather than a pair per curve: N
        labelled brackets would be unreadable, and the union keeps the reading
        true — outside it, no series' number saw a sample. (A superset, the same
        way Peak-10 ms-Leq's lookback already widens a single trace's bracket.)
        """
        start_text = "calc window starts"
        if not all(trace.onset_detected for _label, trace, _color in series):
            start_text += (
                " (no onset detected)"
                if len(series) == 1
                else " (a series had no onset)"
            )
        window_xs: list[float] = []
        for x, text in (
            (min(starts) if starts else None, start_text),
            (max(ends) if ends else None, "calc window ends"),
        ):
            if x is None:
                continue
            window_xs.append(x)
            self._plot.addItem(
                pg.InfiniteLine(
                    pos=x, angle=90, pen=self._WINDOW_PEN,
                    label=text,
                    labelOpts=self._WINDOW_LABEL_OPTS,
                )
            )
        # Frame Calc Window needs both edges to have a span to zoom to: a trace
        # with only one line (or a zero-width window) has nothing to frame.
        if len(window_xs) == 2 and window_xs[1] > window_xs[0]:
            self._window_x_bounds = (window_xs[0], window_xs[1])
            self._frame_window_btn.setEnabled(True)
        else:
            self._window_x_bounds = None
            self._frame_window_btn.setEnabled(False)

    # ---- point readout -------------------------------------------------- #

    def _on_plot_clicked(self, event) -> None:
        """Select the sample nearest the click and show its value + time.

        Considers the two samples bracketing the click in time on *every* drawn
        curve and keeps the one nearest in screen pixels — so with several curves
        overlaid the click reads the one it visually landed on, and a click on
        empty space (nothing within :data:`_PICK_TOLERANCE_PX`) leaves any
        current readout untouched. NaN samples are skipped rather than
        abandoning the pick: a gap in one curve is no reason to ignore another
        running through the same instant.
        """
        if not self._series:
            return
        scene_pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        vb = self._plot.getPlotItem().vb
        view_pos = vb.mapSceneToView(scene_pos)

        best: tuple[float, int, int, float] | None = None  # (px, series, idx, value)
        for s_index, (_label, trace, _color) in enumerate(self._series):
            t = trace.t_ms
            # t_ms is sorted ascending; the click falls between i-1 and i.
            i = int(np.searchsorted(t, view_pos.x()))
            for idx in (i - 1, i):
                if not 0 <= idx < t.size:
                    continue
                value = float(trace.values[idx])
                if not np.isfinite(value):
                    continue
                # Map the sample back to screen space so the comparison is the
                # true pixel distance, not just nearness in time.
                point = vb.mapViewToScene(QtCore.QPointF(float(t[idx]), value))
                distance = float(
                    np.hypot(point.x() - scene_pos.x(), point.y() - scene_pos.y())
                )
                if best is None or distance < best[0]:
                    best = (distance, s_index, idx, value)

        if best is None or best[0] > self._PICK_TOLERANCE_PX:
            return
        _distance, s_index, idx, value = best
        self._show_readout(idx, value, series_index=s_index)

    def _show_readout(self, idx: int, value: float, series_index: int = 0) -> None:
        """Mark sample ``idx`` of one series on the plot and fill the readout box.

        The series' label prefixes the value when there is one, since with
        curves overlaid the number alone does not say which shot it came from.
        """
        label, trace, _color = self._series[series_index]
        x = float(trace.t_ms[idx])
        if self._pick_marker is not None:
            self._plot.removeItem(self._pick_marker)
        self._pick_marker = pg.ScatterPlotItem(
            [x], [value], size=11, brush=self._PICK_BRUSH, pen=self._PICK_PEN, pxMode=True
        )
        self._plot.addItem(self._pick_marker)

        unit = _unit_of(trace.y_label)
        unit_suffix = f" {unit}" if unit else ""
        prefix = f"{label}:  " if label else ""
        self._readout_label.setText(f"{prefix}{value:.3f}{unit_suffix}  @ {x:.2f} ms")
        self._readout_label.setVisible(True)
        self._readout_clear_btn.setVisible(True)

    def clear_readout(self) -> None:
        """Remove the picked-point marker and hide the readout box."""
        if self._pick_marker is not None:
            self._plot.removeItem(self._pick_marker)
            self._pick_marker = None
        self._readout_label.clear()
        self._readout_label.setVisible(False)
        self._readout_clear_btn.setVisible(False)


class BatchAverageView(_View):
    """The batch-average view: a batch's four position x role output slots.

    Only shots whose ``included`` flag is set feed these numbers — this is the
    filtered counterpart to the Data bank tab. Every slot is listed even when
    empty, so a batch missing its shooter's-ear regulars reads as a gap rather
    than silently absent, and each populated slot expands into the individual
    shots averaged behind it.

    Positions and roles are never mixed: the 3-FRP / 5-regular target applies per
    position, so each channel averages the same selected shots on its own axis.

    Each populated slot row ends in a Copy button that puts that row's numbers on
    the clipboard as a SilencerScout ``SSR1`` paste string (see
    :mod:`~sound_metric_app.services.scout_paste`), ready to drop into the Scout
    Report editor's box for the matching test.

    Each *shot* row carries the mirror-image button one column earlier: Compare,
    which pins that shot's curve to the Compare tab. The two never appear on the
    same row, and for the same reason — a paste string is a batch average, so an
    individual shot has none; a curve is read off a capture, so an average has
    none. Which mic a pinned curve is of is the slot the shot row sits under, so
    choosing ML or SE needs no extra control: pin it from under Muzzle Left for
    ML, from under Shooter's Ear for SE, or from both to overlay the pair.
    """

    _METRIC_KEYS = tuple(key for _label, key in _REPORT_METRICS)
    _DIAGNOSTIC_KEYS = tuple(key for _label, key in _REPORT_DIAGNOSTICS)
    _COLUMNS = [
        "Slot / Shot", "n",
        *(label for label, _key in _REPORT_METRICS),
        *(label for label, _key in _REPORT_DIAGNOSTICS),
        "Compare", "Scout paste",
    ]
    #: Metric columns begin here; columns 0-1 are label / n.
    _FIRST_METRIC_COL = 2
    #: One past the last metric column — where a click stops being a graph request.
    #: The diagnostic columns sit past this bound precisely so a click on one is
    #: not read as a graph request: they have no trace to draw.
    _END_METRIC_COL = _FIRST_METRIC_COL + len(_METRIC_KEYS)
    #: The two trailing button columns, in the order the operator meets them:
    #: Compare (shot rows, see :meth:`_compare_button`) then Scout paste
    #: (averaged rows, see :meth:`_paste_button`).
    _COMPARE_COL = _END_METRIC_COL + len(_DIAGNOSTIC_KEYS)
    _PASTE_COL = _COMPARE_COL + 1
    #: Fixed width for both — Qt sizes a column to its item *text*, and these
    #: cells are empty text behind a button widget, so contents-sizing would
    #: collapse the button out of view.
    _BUTTON_COL_WIDTH = 90

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        #: Bumped on each graph request so a slow capture read for an earlier
        #: click is discarded when a newer cell is clicked.
        self._graph_token = 0
        #: The last graphed (shot_id, position, metric_key, subtitle), so the
        #: graph can be re-rendered when the level-weighting dropdown changes.
        self._current_request: tuple | None = None
        layout = QtWidgets.QVBoxLayout(self)

        picker_row = QtWidgets.QHBoxLayout()
        # SKU filter, left of the batch picker: narrows the batch list to one
        # SKU's sessions. Changing it re-runs refresh(), which rebuilds the
        # filtered batch list and reloads the report.
        self.sku_combo = self._add_sku_filter(picker_row)
        picker_row.addWidget(QtWidgets.QLabel("Batch:"))
        self.batch_combo = QtWidgets.QComboBox()
        self.batch_combo.currentIndexChanged.connect(self._load_report)
        picker_row.addWidget(self.batch_combo, 1)
        layout.addLayout(picker_row)

        # Progress against the soft 3-FRP / 5-regular targets, so a short or
        # over-filled batch is visible without reading the per-slot counts. Pin
        # its vertical size to the text height: otherwise a maximised window
        # hands the label a share of the extra vertical space, leaving a tall
        # empty band above the tree instead of feeding that room to the graph.
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        layout.addWidget(self.status_label)

        # Left half: the slot tree. Right half: the single-metric graph. A
        # splitter lets the user trade width between the two.
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # A tree, not a flat table: each slot's average is a top-level row that
        # expands to reveal the individual included shots it averages over.
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(len(self._COLUMNS))
        self.tree.setHeaderLabels(self._COLUMNS)
        self.tree.setRootIsDecorated(True)
        self.tree.itemClicked.connect(self._on_cell_clicked)
        _style_grid_tree(self.tree)
        # Amber wash behind the slot (averaging) rows, so each average reads
        # apart from the individual shots nested under it. Held on self: the
        # view does not take ownership of a delegate, and a garbage-collected
        # one leaves the tree painting through a dangling C++ pointer.
        self._row_tint = _TopLevelRowTint(_AVERAGE_ROW_TINT, self.tree)
        self.tree.setItemDelegate(self._row_tint)
        split.addWidget(self.tree)

        self.graph = MetricGraph()
        # Re-graph the same cell with the new weighting when the dropdown changes.
        self.graph.smoothingChanged.connect(self._render_current)
        split.addWidget(self.graph)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split)

    def refresh(self) -> None:
        # Rebuild the SKU filter first, then read its selection to narrow the
        # batch list below. The helper blocks the combo's own signal, so this
        # does not re-enter refresh().
        _repopulate_sku_filter(self.sku_combo, self.controller.skus())
        sku = self.sku_combo.currentData()

        current = self.batch_combo.currentData()
        # Keep signals blocked through setCurrentIndex so currentIndexChanged
        # doesn't fire _load_report; we call it once, explicitly, below.
        self.batch_combo.blockSignals(True)
        self.batch_combo.clear()
        combinations = {c.id: c for c in self.controller.combinations()}
        for batch in self.controller.batches():
            combo = combinations.get(batch.combination_id)
            # Skip batches outside the selected SKU (None == no filter). A batch
            # whose combination is missing is only shown when unfiltered.
            if sku is not None and (combo is None or combo.sku != sku):
                continue
            state = "closed" if batch.closed else "open"
            label = f"#{batch.id}  {combo.label if combo else '?'}  {batch.title}  [{state}]"
            self.batch_combo.addItem(label, batch.id)
        index = self.batch_combo.findData(current)
        self.batch_combo.setCurrentIndex(
            index if index >= 0 else (0 if self.batch_combo.count() else -1)
        )
        self.batch_combo.blockSignals(False)

        self._load_report()

    def _load_report(self, *_args) -> None:
        # Slot rows are the same four across every batch, so an open slot stays
        # open through a rebuild — whether that rebuild came from an edit or from
        # switching batches. Snapshot before clear().
        open_slots = _expanded_keys(self.tree, self._slot_key)
        self.tree.clear()
        self._graph_token += 1  # abandon any in-flight graph for the old report
        self._current_request = None
        self.graph.show_message("Click a metric cell on a shot row to graph it.")
        batch_id = self.batch_combo.currentData()
        if batch_id is None:
            self.status_label.setText("")
            return

        report = self.controller.batch_averages(batch_id)
        self.status_label.setText(
            f"{report.n_included} of {report.n_shots} shot(s) brought forward   —   "
            f"{report.status.summary()}"
        )
        # Names the batch on every Copy button's tooltip. The paste string itself
        # carries no test identity — the receiving row is chosen by the human —
        # so the button has to say which batch its numbers came from.
        combination = report.combination
        batch_label = f"Batch #{report.batch.id}  {combination.label if combination else '?'}"
        for position, role in AVERAGE_SLOTS:
            slot_label = f"{position.label} · {role.label}"
            avg = report.averages.get((position, role))
            if avg is None:
                # Keep the empty slot visible: a missing quadrant is information,
                # not something to hide. Pad so the row spans all _COLUMNS —
                # including the (buttonless) paste column, since there is nothing
                # to copy for a slot with no included shots.
                row = [slot_label, "0", "none included"]
                self._add_top([*row, *[""] * (len(self._COLUMNS) - len(row))])
                continue
            avg_item = self._add_top(
                [
                    slot_label,
                    str(avg["n"]),
                    *(_format_metric(avg[k]) for k in self._METRIC_KEYS),
                    # Diagnostics are per-capture, not aggregated: averaging the
                    # baselines of the shots in a slot would hide the one bad
                    # capture the column exists to expose, so a slot row leaves
                    # them blank and the shot rows beneath carry the numbers.
                    *("" for _ in self._DIAGNOSTIC_KEYS),
                    "",  # Compare column: an average has no single curve to pin
                    "",  # the paste column holds a button, not text
                ]
            )
            self.tree.setItemWidget(
                avg_item,
                self._PASTE_COL,
                self._paste_button(position, role, avg, batch_label, slot_label),
            )
            for shot in report.shots.get((position, role), ()):
                shot_item = self._shot_item(shot, position)
                # setItemWidget needs the row to be in the tree already, so the
                # Compare button goes on after the child is parented.
                avg_item.addChild(shot_item)
                self.tree.setItemWidget(
                    shot_item,
                    self._COMPARE_COL,
                    self._compare_button(self._series_for(shot, position, report)),
                )
        # Size against the shot rows, then collapse to the four slot averages --
        # the report's headline -- reopening whatever was open. The shots behind
        # a closed slot stay one arrow away.
        self.tree.expandAll()
        for col in range(len(self._COLUMNS)):
            if col in (self._COMPARE_COL, self._PASTE_COL):
                # Contents-sizing measures item text, and these columns' text is
                # empty behind their buttons; give them a width that fits one.
                self.tree.setColumnWidth(col, self._BUTTON_COL_WIDTH)
                continue
            self.tree.resizeColumnToContents(col)
        self.tree.collapseAll()
        _restore_expanded(self.tree, open_slots, self._slot_key)

    @staticmethod
    def _slot_key(item: QtWidgets.QTreeWidgetItem):
        """A slot row is identified by its "position · role" label."""
        return item.text(0) if item.parent() is None else None

    def _shot_item(self, shot: dict, position: MicPosition) -> QtWidgets.QTreeWidgetItem:
        cluster = shot.get("cluster_index")
        order = shot.get("shot_order")
        # Every shot here comes from shot_metrics_for_batch, which filters out
        # null shot_order, so order is always present.
        label = f"Cluster {cluster} · shot {order}" if cluster else f"Shot {order}"
        # Trailing pair: the Compare column (filled with a button once the row is
        # parented) and the paste column (empty on a shot row).
        item = QtWidgets.QTreeWidgetItem(
            [
                label, "",
                *(_format_metric(shot[k]) for k in self._METRIC_KEYS),
                *(_format_floor(shot.get(k)) for k in self._DIAGNOSTIC_KEYS),
                "", "",
            ]
        )
        # Carry the identity a graph request needs: the shot to re-read and which
        # mic's channel to pull. Only shot rows get this tag, so a click on an
        # average (top-level) row is easy to tell apart.
        item.setData(0, QtCore.Qt.UserRole, ("shot", shot["shot_id"], position))
        return item

    def _add_top(self, values: list[str]) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(values)
        self.tree.addTopLevelItem(item)
        return item

    # ---- scout paste ------------------------------------------------------ #

    def _paste_button(
        self,
        position: MicPosition,
        role: ShotRole,
        average: dict,
        batch_label: str,
        slot_label: str,
    ) -> QtWidgets.QPushButton:
        """The Copy button that ends one averaged row.

        The string is built once, here, rather than on click: the row's numbers
        are fixed for as long as the row exists, and holding the finished line
        lets the tooltip show exactly what the button will copy. Nothing can
        change those numbers without going through MainWindow.notify_changed(),
        which refreshes every view -- and this view's refresh clears the tree,
        destroying these buttons and rebuilding them off the new averages. So a
        captured line cannot outlive the numbers it was built from.

        The tooltip also names the batch and the slot, because the string itself
        does not: which test the numbers belong to is decided by the box they are
        pasted into, and a string dropped in the wrong row is accepted silently
        over there. Naming the source here is the only check there is.
        """
        line = slot_line(position, role, average)
        button = QtWidgets.QPushButton(_COPY_LABEL)
        button.setToolTip(
            f"Copy the SilencerScout paste string for\n{batch_label}\n{slot_label}\n\n{line}"
        )
        button.clicked.connect(lambda: self._copy_scout_line(button, line))
        return button

    def _copy_scout_line(self, button: QtWidgets.QPushButton, line: str) -> None:
        """Put one row's paste string on the clipboard and say so on the button."""
        QtWidgets.QApplication.clipboard().setText(line)
        _flash_button(button, _COPIED_LABEL, _COPY_LABEL)

    # ---- compare ---------------------------------------------------------- #

    def _series_for(
        self, shot: dict, position: MicPosition, report: BatchAverages
    ) -> CompareSeries:
        """Name one shot row as a Compare-tab series.

        The label is what the legend and the Compare list carry, so it has to
        separate this curve from any other the operator might pin — including
        the same cluster and shot number fired in a different session, or under
        a different SKU. The shot id leads because it is the only field that is
        unique on its own; the rest is there to be read, not to disambiguate.
        The longer identification (platform, ammo, batch) goes to the tooltip,
        where there is room for it. Built off the shared :func:`_compare_series`.
        """
        cluster = shot.get("cluster_index")
        order = shot.get("shot_order")
        where = f"C{cluster}·S{order}" if cluster else f"S{order}"
        combination = report.combination
        return _compare_series(
            shot_id=shot["shot_id"],
            position=position,
            sku=combination.sku if combination else "?",
            where=where,
            combo_label=combination.label if combination else "?",
            batch_id=report.batch.id,
        )

    def _compare_button(self, series: CompareSeries) -> QtWidgets.QPushButton:
        """The Compare button that sits on one shot row, before its Copy column."""
        button = QtWidgets.QPushButton(_COMPARE_LABEL)
        button.setToolTip(
            f"Overlay this shot's {series.position.label} curve on the Compare tab.\n\n"
            f"{series.detail}"
        )
        button.clicked.connect(lambda: self._pin_for_compare(button, series))
        return button

    def _pin_for_compare(
        self, button: QtWidgets.QPushButton, series: CompareSeries
    ) -> None:
        """Pin one shot/mic to the Compare tab, without leaving this one.

        Staying put is the point: comparing means picking several shots, often
        across batches, and a tab switch per pick would fight that. Uses the
        shared :func:`_pin_compare_series`; the button reverts to the constant
        "Compare" label, not a per-row one.
        """
        _pin_compare_series(self.main, button, series, _COMPARE_LABEL)

    # ---- graph ---------------------------------------------------------- #

    def _on_cell_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        """Graph the clicked metric for the clicked shot (one graph at a time)."""
        entry = item.data(0, QtCore.Qt.UserRole)
        if not entry or entry[0] != "shot":
            self._current_request = None
            self.graph.show_message("Select a metric cell on an individual shot row.")
            return
        # Columns either side of the metrics (label / n, and the trailing paste
        # column) have no curve behind them — and indexing _METRIC_KEYS past its
        # end would raise rather than just miss.
        if not self._FIRST_METRIC_COL <= column < self._END_METRIC_COL:
            self._current_request = None
            self.graph.show_message("Click a metric column (Peak dB, Peak dBA, …).")
            return

        _kind, shot_id, position = entry
        metric_key = self._METRIC_KEYS[column - self._FIRST_METRIC_COL]
        metric_label = self._COLUMNS[column]
        subtitle = f"Shot #{shot_id} · {position.label} · {metric_label}"
        self._current_request = (shot_id, position, metric_key, subtitle)
        self._render_current()

    def _render_current(self) -> None:
        """(Re)draw the last-clicked cell using the graph's current weighting.

        Called both on a fresh cell click and when the level-weighting dropdown
        changes; a no-op if no cell has been graphed yet.
        """
        if self._current_request is None:
            return
        shot_id, position, metric_key, subtitle = self._current_request

        self._graph_token += 1
        token = self._graph_token
        smoothing = self.graph.current_smoothing()
        self.graph.show_message("Loading…")

        def done(trace) -> None:
            if token != self._graph_token:
                return  # a newer request superseded this one; drop the stale trace
            self.graph.show_trace(trace, subtitle)

        self._run_async(
            lambda: self.controller.metric_trace(
                shot_id, position, metric_key, smoothing=smoothing
            ),
            done,
        )


# --------------------------------------------------------------------------- #
# 5. Compare view
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompareSeries:
    """One curve pinned to the Compare tab: a shot, one of its mics, a name.

    ``(shot_id, position)`` is the identity — the same shot can be pinned once
    per mic, so ML and SE overlay as two series, but neither can be pinned twice.
    ``label`` names the curve in the legend and the pinned list; ``detail`` is
    the longer identification its tooltip carries.
    """

    shot_id: int
    position: MicPosition
    label: str
    detail: str = ""

    @property
    def key(self) -> tuple[int, MicPosition]:
        """What makes this series the same one as another (see the class doc)."""
        return (self.shot_id, self.position)


class CompareView(_View):
    """The compare view: one metric's curve for any number of shots, overlaid.

    Shots arrive here from the Compare buttons on the Batch average and Data
    bank tabs and stay until removed, so a comparison can be assembled across
    batches, SKUs, both mics, and included/idle status alike — nothing about
    this view is scoped to one batch. The metric is
    picked once for the whole overlay (curves have to share a Y axis to be read
    against each other), and defaults to Impulse Pa·ms.

    The graph is the same :class:`MetricGraph` the Batch average tab draws into,
    not a copy of it: Auto Frame, Frame Calc Window, the onset close-ups, the
    level-weighting dropdown, and the click readout are one implementation used
    twice, so they cannot drift apart between the two tabs.

    Each pinned row carries its own Hide and Remove, because both act on that
    one shot and nothing else. Hide takes a curve off the graph but leaves it
    pinned — the way to read three of five without losing the other two — and
    the row keeps its colour while hidden, so unhiding puts it back exactly
    where the eye left it. Remove unpins outright.

    Traces are memoized per pinned series for as long as the metric and the
    level weighting hold still. Pinning the ninth shot then re-reads one capture
    rather than nine, which is what makes it reasonable to redraw on every
    change. Changing either control drops the whole cache — it also bounds it,
    since only one metric x weighting generation is ever held. A mutation
    elsewhere in the app drops it too (see :meth:`invalidate_traces`).
    """

    #: Pre-selected metric: the impulse curve is what these comparisons are for.
    _DEFAULT_METRIC = "impulse_pa_ms"

    _EMPTY_MESSAGE = (
        "Pin shots here with the Compare button on a Batch average or Data bank shot row."
    )

    #: Pinned-row columns: the shot (swatch + label), then its two own-row
    #: actions. The buttons are sized to their text and the label takes the rest.
    _LABEL_COL, _HIDE_COL, _REMOVE_COL = range(3)
    #: Room the label column needs for a full series name plus its swatch. Held
    #: as the pinned pane's minimum width (plus the buttons) so the splitter
    #: cannot open on a column narrow enough to elide every row to "#7 ·…" —
    #: rows the operator cannot tell apart are no index at all.
    _LABEL_COL_MIN_WIDTH = 190

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        #: Pinned series, in pin order — which is the order they are drawn,
        #: coloured, legended, and listed in. One tree row per entry, same
        #: index, so a row's position identifies its series.
        self._series: list[CompareSeries] = []
        #: Keys of the pinned series currently hidden from the graph. They keep
        #: their row, their colour, and their place in the order; only the curve
        #: goes. Kept as keys rather than indices so removing one series cannot
        #: silently hide its neighbour.
        self._hidden: set[tuple[int, MicPosition]] = set()
        #: Memoized curves and load failures for the current metric x weighting,
        #: both keyed by ``CompareSeries.key``.
        self._traces: dict[tuple[int, MicPosition], MetricTrace] = {}
        self._errors: dict[tuple[int, MicPosition], str] = {}
        #: The (metric, smoothing) the two dicts above were filled for.
        self._cache_key: tuple[str, str] | None = None
        #: Bumped on each load so a slow read for a superseded overlay is dropped.
        self._graph_token = 0

        layout = QtWidgets.QVBoxLayout(self)

        picker_row = QtWidgets.QHBoxLayout()
        picker_row.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_combo = QtWidgets.QComboBox()
        for label, key in _REPORT_METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.setCurrentIndex(self.metric_combo.findData(self._DEFAULT_METRIC))
        self.metric_combo.setToolTip(
            "Which metric's curve to overlay. All pinned shots are drawn with\n"
            "this one, so they share a Y axis and can be read against each other."
        )
        self.metric_combo.currentIndexChanged.connect(self._render)
        picker_row.addWidget(self.metric_combo)
        picker_row.addStretch(1)
        layout.addLayout(picker_row)

        # Left: what is pinned. Right: the shared metric graph.
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        pinned = QtWidgets.QWidget()
        pinned_layout = QtWidgets.QVBoxLayout(pinned)
        pinned_layout.setContentsMargins(0, 0, 0, 0)
        pinned_layout.addWidget(QtWidgets.QLabel("Overlaid shots:"))
        # A row per pinned shot: its swatch and label, then the two buttons that
        # act on it alone. Flat (no expanders) and headerless -- the buttons say
        # what they do, and three columns of headings over a narrow panel would
        # cost more width than they explain.
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        header = self.tree.header()
        # Spare width belongs to the label, not to the trailing button column a
        # tree header stretches by default -- which would leave every row elided
        # to "#7 ·…" beside a Remove button four times wider than its word.
        header.setStretchLastSection(False)
        header.setSectionResizeMode(self._LABEL_COL, QtWidgets.QHeaderView.Stretch)
        # Contents-sizing measures item *text*, and these cells are empty behind
        # their buttons — so the action columns are sized from what a button
        # actually needs at the current font and DPI, rather than a magic number
        # that a larger system font would spill out of.
        buttons_width = 0
        for col, labels in (
            (self._HIDE_COL, ("Hide", "Show")),
            (self._REMOVE_COL, ("Remove",)),
        ):
            width = max(QtWidgets.QPushButton(text).sizeHint().width() for text in labels)
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.Fixed)
            self.tree.setColumnWidth(col, width)
            buttons_width += width
        pinned_layout.addWidget(self.tree, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear all")
        self.clear_btn.setToolTip("Take every shot off the graph.")
        self.clear_btn.clicked.connect(self.clear)
        button_row.addWidget(self.clear_btn)
        button_row.addStretch(1)
        pinned_layout.addLayout(button_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        pinned_layout.addWidget(self.status_label)
        # Wide enough for a whole row -- label, both buttons, and a scrollbar --
        # so the pane never opens with its rows elided down to nothing.
        pinned.setMinimumWidth(
            self._LABEL_COL_MIN_WIDTH
            + buttons_width
            + self.tree.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
        )
        split.addWidget(pinned)

        self.graph = MetricGraph()
        # Re-draw with the new weighting when the dropdown changes; the cache is
        # keyed by it, so this reloads rather than redrawing stale curves.
        self.graph.smoothingChanged.connect(self._render)
        split.addWidget(self.graph)
        # The pinned rows are a narrow index; the graph is the view. Give it the
        # width.
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        layout.addWidget(split)

        self._render()

    # ---- pinned set ------------------------------------------------------- #

    def add_series(self, series: CompareSeries) -> bool:
        """Pin one shot/mic. Returns False if it was already pinned (a no-op).

        A second copy of a curve would draw exactly on top of the first and take
        a second legend row and colour to say nothing, so a repeat pin is
        refused rather than duplicated. The caller reports which happened.
        """
        if any(existing.key == series.key for existing in self._series):
            return False
        self._series.append(series)
        self._on_pinned_changed()
        return True

    def _remove(self, series: CompareSeries) -> None:
        """Unpin one series — the row's own Remove button."""
        self._series = [s for s in self._series if s.key != series.key]
        self._forget(series)
        self._on_pinned_changed()

    def _toggle_hidden(self, series: CompareSeries) -> None:
        """Take one series off the graph, or put it back — the row's Hide button.

        The series stays pinned, keeps its row, and keeps its colour, so this is
        purely about what the graph is crowded with. Hiding does not discard the
        memoized curve either: unhiding is then instant, and the round trip
        costs no capture read.
        """
        if series.key in self._hidden:
            self._hidden.discard(series.key)
        else:
            self._hidden.add(series.key)
        self._render()

    def clear(self) -> None:
        """Drop every pinned series."""
        if not self._series:
            return
        for series in self._series:
            self._forget(series)
        self._series = []
        self._on_pinned_changed()

    def _forget(self, series: CompareSeries) -> None:
        """Release a removed series' state, so nothing outlives its row."""
        self._traces.pop(series.key, None)
        self._errors.pop(series.key, None)
        self._hidden.discard(series.key)

    def _on_pinned_changed(self) -> None:
        self.main.update_compare_count(len(self._series))
        self._render()

    def invalidate_traces(self) -> None:
        """Drop the memoized curves, keeping what is pinned.

        Called for a mutation elsewhere in the app rather than on every refresh:
        a re-mark can retag which channel is ML, or repoint a shot at a
        different capture, and the curve drawn from it is then wrong in a way no
        amount of redrawing would fix. Navigation changes none of that, and a
        reload per tab visit would be paid every time for nothing.
        """
        self._traces.clear()
        self._errors.clear()
        self._cache_key = None

    def refresh(self) -> None:
        """Redraw only if the curves were invalidated; navigation leaves it alone.

        Nothing here is scoped to a batch or a SKU, so switching tabs has
        nothing to re-read — and a redraw would autorange away the zoom the
        operator had just framed, which on this tab is the whole point of the
        visit. A dropped cache (:meth:`invalidate_traces`, i.e. an actual
        mutation) is the one thing that forces the reload.
        """
        if self._cache_key is None and self._series:
            self._render()

    # ---- graph ------------------------------------------------------------ #

    def _render(self, *_args) -> None:
        """Load whatever the current overlay is missing, then draw it."""
        cache_key = (self.metric_combo.currentData(), self.graph.current_smoothing())
        if cache_key != self._cache_key:
            self._traces.clear()
            self._errors.clear()
            self._cache_key = cache_key

        self._graph_token += 1
        token = self._graph_token
        if not self._series:
            self.graph.show_message(self._EMPTY_MESSAGE)
            self.status_label.setText("")
            self.tree.clear()
            return

        metric_key, smoothing = cache_key
        # A hidden series is not drawn, so its capture is not worth reading --
        # unhiding is what asks for it (and finds it already cached if it was
        # hidden after being drawn).
        pending = [
            s
            for s in self._series
            if s.key not in self._hidden
            and s.key not in self._traces
            and s.key not in self._errors
        ]
        if not pending:
            self._draw()
            return

        self.graph.show_message("Loading…")

        def load():
            # One worker for the whole batch of missing curves, and one failure
            # bucket per series: a shot whose capture has moved must not take
            # the rest of the overlay down with it (nor pop a dialog, since a
            # redraw would pop it again). It is reported in the list instead.
            results = []
            for series in pending:
                try:
                    trace = self.controller.metric_trace(
                        series.shot_id, series.position, metric_key, smoothing=smoothing
                    )
                except Exception as exc:  # noqa: BLE001 — shown against its row
                    results.append((series, None, str(exc)))
                else:
                    results.append((series, trace, None))
            return results

        def done(results) -> None:
            if token != self._graph_token:
                return  # a newer overlay superseded this load; drop it
            for series, trace, error in results:
                if trace is None:
                    self._errors[series.key] = error
                else:
                    self._traces[series.key] = trace
            self._draw()

        self._run_async(load, done)

    def _draw(self) -> None:
        """Rebuild the pinned rows and the overlay from the memoized traces.

        Colour is taken from a series' position in the pinned order, not from
        its position among the drawn ones: hiding a curve must leave every other
        curve on the colour it already had, or the legend the operator has just
        learned reshuffles under them on every toggle.
        """
        self.tree.clear()
        drawn: list[tuple[str, MetricTrace, tuple[int, int, int]]] = []
        hidden = 0
        unavailable = 0
        for index, series in enumerate(self._series):
            color = MetricGraph.series_color(index)
            trace = self._traces.get(series.key)
            item = QtWidgets.QTreeWidgetItem([series.label, "", ""])
            item.setToolTip(self._LABEL_COL, series.detail)
            if series.key in self._hidden:
                hidden += 1
                # Keep the swatch: the colour is what the curve comes back as.
                item.setIcon(self._LABEL_COL, _color_swatch(color))
                item.setForeground(self._LABEL_COL, QtGui.QBrush(_MUTED_INK))
            elif trace is None:
                unavailable += 1
                error = self._errors.get(series.key, "not loaded")
                item.setText(self._LABEL_COL, f"{series.label}  — unavailable")
                item.setIcon(self._LABEL_COL, _color_swatch(None))
                item.setToolTip(self._LABEL_COL, f"{series.detail}\n\n{error}")
            else:
                item.setIcon(self._LABEL_COL, _color_swatch(color))
                drawn.append((series.label, trace, color))
            self.tree.addTopLevelItem(item)
            self._add_row_buttons(item, series)

        metric_label = self.metric_combo.currentText()
        self.graph.show_traces(drawn, f"{metric_label} — {len(drawn)} shot(s) overlaid")
        parts = [f"{len(drawn)} of {len(self._series)} drawn"]
        if hidden:
            parts.append(f"{hidden} hidden")
        if unavailable:
            parts.append(f"{unavailable} unavailable (hover for why)")
        self.status_label.setText("   —   ".join(parts))

    def _add_row_buttons(
        self, item: QtWidgets.QTreeWidgetItem, series: CompareSeries
    ) -> None:
        """Put one pinned row's own Hide and Remove buttons on it.

        Both handlers rebuild this very tree, which would free the button Qt is
        still emitting the click for, so both are deferred a turn of the event
        loop (see :meth:`_View._defer`).
        """
        is_hidden = series.key in self._hidden
        hide_btn = QtWidgets.QPushButton("Show" if is_hidden else "Hide")
        hide_btn.setToolTip(
            "Put this shot's curve back on the graph."
            if is_hidden
            else "Take this shot's curve off the graph, keeping it pinned here."
        )
        hide_btn.clicked.connect(lambda: self._defer(lambda: self._toggle_hidden(series)))
        self.tree.setItemWidget(item, self._HIDE_COL, hide_btn)

        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.setToolTip("Unpin this shot from the Compare tab.")
        remove_btn.clicked.connect(lambda: self._defer(lambda: self._remove(series)))
        self.tree.setItemWidget(item, self._REMOVE_COL, remove_btn)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


class AmmoDefinitionsDialog(QtWidgets.QDialog):
    """Manage the ammo presets offered when marking a shot.

    A plain list editor: add a typed ammo type, remove a selected one, then Save.
    The caller persists the collected list via the controller. Order is preserved
    and normalization (trim/de-dup) happens on save in :mod:`config`.
    """

    def __init__(self, definitions: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ammo definitions")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Ammo types offered when marking a shot:"))

        self.list = QtWidgets.QListWidget()
        self.list.addItems(definitions)
        layout.addWidget(self.list)

        entry_row = QtWidgets.QHBoxLayout()
        self.entry = QtWidgets.QLineEdit()
        self.entry.setPlaceholderText("e.g. LC M855 (5.56)")
        self.entry.returnPressed.connect(self._add)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(self._add)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove)
        entry_row.addWidget(self.entry, 1)
        entry_row.addWidget(add_btn)
        entry_row.addWidget(remove_btn)
        layout.addLayout(entry_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add(self) -> None:
        name = self.entry.text().strip()
        if not name:
            return
        if not self.list.findItems(name, QtCore.Qt.MatchExactly):
            self.list.addItem(name)
        self.entry.clear()
        self.entry.setFocus()

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def definitions(self) -> list[str]:
        """The ammo types currently listed, in display order."""
        return [self.list.item(i).text() for i in range(self.list.count())]


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: WorkflowController | None = None):
        super().__init__()
        self.setWindowTitle("Sound Metric App — Workflow")
        # Wider than the other tabs need: the Report tab splits into a tree on the
        # left and a metric graph on the right, so give both room by default.
        self.resize(1100, 620)
        self.controller = controller or WorkflowController()

        self.ingest_view = IngestView(self.controller, self)
        self.marking_view = MarkingView(self.controller, self)
        self.bank_view = DataBankView(self.controller, self)
        self.report_view = BatchAverageView(self.controller, self)
        self.compare_view = CompareView(self.controller, self)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.ingest_view, "Ingest")
        self.tabs.addTab(self.marking_view, "Mark")
        self.tabs.addTab(self.bank_view, "Data bank")
        self.tabs.addTab(self.report_view, "Batch average")
        self.tabs.addTab(self.compare_view, _COMPARE_TAB_LABEL)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._views = [
            self.ingest_view,
            self.marking_view,
            self.bank_view,
            self.report_view,
            self.compare_view,
        ]
        self._build_menus()
        self.notify_changed()

    def _build_menus(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")
        ammo_action = settings_menu.addAction("Ammo definitions…")
        ammo_action.triggered.connect(self._edit_ammo_definitions)

    def _edit_ammo_definitions(self) -> None:
        """Open the ammo-preset editor; on save, persist and refresh the mark form."""
        dialog = AmmoDefinitionsDialog(self.controller.ammo_definitions(), parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.controller.set_ammo_definitions(dialog.definitions())
        self.notify_changed()

    def _on_tab_changed(self, index: int) -> None:
        self.tabs.widget(index).refresh()

    def notify_changed(self) -> None:
        """Reload every view after a mutating action (ingest / mark / include / close)."""
        # A mutation is the only thing that orphans a cluster/batch/combination (a
        # re-mark or edit can empty the old container), so the destructive sweep is
        # tied to the mutation here rather than to every refresh() — pure navigation
        # must never delete rows.
        self.controller.sweep_empty()
        # Same reasoning for the Compare tab's memoized curves: a mutation can
        # change what a pinned curve is drawn *from* (a re-mark retags which
        # channel is ML), so they are dropped here and not on every refresh.
        self.compare_view.invalidate_traces()
        for view in self._views:
            view.refresh()

    def open_marking_for(self, shot_id: int) -> None:
        """Switch to the Mark tab focused on ``shot_id`` (from the Ingest view)."""
        self.marking_view.select_shot(shot_id)
        self.tabs.setCurrentWidget(self.marking_view)

    def add_to_compare(self, series: CompareSeries) -> bool:
        """Pin one shot/mic to the Compare tab. False if it was already there.

        Deliberately does *not* switch tabs: pinning is done from the Batch
        average tree, several rows at a time (see
        :meth:`BatchAverageView._pin_for_compare`).
        """
        return self.compare_view.add_series(series)

    def update_compare_count(self, count: int) -> None:
        """Carry how many shots are pinned on the Compare tab's own label.

        Pinning happens on another tab and draws nothing there, so the count is
        what tells the operator the click landed — and what the overlay is
        currently worth switching to.
        """
        index = self.tabs.indexOf(self.compare_view)
        self.tabs.setTabText(
            index, f"{_COMPARE_TAB_LABEL} ({count})" if count else _COMPARE_TAB_LABEL
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _str_or_empty(value) -> str:
    """Render an optional field for a pre-filled edit box (``None`` -> "")."""
    return "" if value is None else str(value)


def _safe_int(text: str) -> int | None:
    """Parse an int from a live-edited box, treating anything unparseable as ``None``.

    Unlike :func:`_opt_int` this never raises: it backs the as-you-type role
    preview, where a half-typed value is normal and must not pop a dialog.
    """
    try:
        return int(text.strip())
    except ValueError:
        return None


def _unit_of(y_label: str) -> str:
    """Pull the unit out of a trace's y-axis label for the point readout.

    Trace labels carry the unit in trailing parentheses — ``"SPL (dBA)"`` ->
    ``"dBA"``, ``"Pressure (Pa)"`` -> ``"Pa"``. Falls back to the whole label if
    it has no parenthesised unit, so the readout always shows something sensible.
    """
    start = y_label.rfind("(")
    end = y_label.rfind(")")
    if start != -1 and end > start:
        return y_label[start + 1 : end].strip()
    return y_label.strip()


def _format_metric(value) -> str:
    """Render a metric value for a report cell (``None`` -> "—").

    Metric columns are nullable REAL (and the schema-v1 migration blanks
    ``peak_impulse_db`` on pre-existing rows), so a value can arrive as ``None``.
    Show an em-dash for the one missing cell instead of letting ``f"{None:.2f}"``
    raise and abort the whole report render.
    """
    return "—" if value is None else f"{value:.2f}"


def _format_floor(value) -> str:
    """Render a pre-trigger floor for a report cell (``None`` -> "—").

    Signed and to three decimals, unlike :func:`_format_metric`: the sign says
    which way the baseline is displaced, and these values live in the tenths of
    a Pascal, where two decimals would round the differences between captures
    down to a couple of digits. ``None`` is a row stored before the column
    existed — re-marking the shot re-processes the capture and fills it in.
    """
    return "—" if value is None else f"{value:+.3f}"


def _format_floor_pair(floors) -> str:
    """Both mics' pre-trigger floors for one data-bank shot row.

    A shot row stands for the capture, not for a channel, so it shows ML and SE
    together — the same shape the Detail column uses for the channel tags, which
    keeps the two readable down the same row. A mic with no stored floor (never
    tagged, or a row predating the column) shows an em-dash in its half rather
    than dropping out, so the two positions stay in fixed places and a column of
    rows can be scanned straight down.

    ``None`` (no channel row at all for this shot) collapses to a single dash
    instead of a pair of them: there is nothing measured to line up.
    """
    if not floors:
        return _EMPTY
    return "  ".join(
        f"{position.value}:{_format_floor(floors.get(position))}"
        for position in (MicPosition.ML, MicPosition.SE)
    )


def _format_captured_at(captured_at: str | None) -> str:
    """Render a shot's ISO capture timestamp for display (``None`` -> "—").

    Falls back to the raw stored string if it does not parse as ISO-8601, so an
    unexpected format is still shown rather than hidden.
    """
    if not captured_at:
        return "—"
    try:
        return datetime.fromisoformat(captured_at).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return captured_at


def _select_channel(combo: QtWidgets.QComboBox, name: str | None) -> None:
    """Preselect ``name`` in a channel combo, falling back to ``(none)`` at index 0."""
    if name:
        index = combo.findText(name)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
    combo.setCurrentIndex(0)


def _selected_channel(combo: QtWidgets.QComboBox) -> str | None:
    """The channel a combo names, or ``None`` for the placeholder entries."""
    text = combo.currentText()
    return None if text in (_NONE_LABEL, _LOADING_LABEL, "") else text


def _tagged_channel_map(
    parent: QtWidgets.QWidget,
    ml_combo: QtWidgets.QComboBox,
    se_combo: QtWidgets.QComboBox,
) -> dict[str, MicPosition] | None:
    """The tagged channel map, or ``None`` after warning about a bad tagging.

    Returning ``None`` (rather than an empty map) keeps "the user needs to fix
    something" distinct from "nothing tagged" — the caller aborts either way,
    but the warning has already been shown here.
    """
    ml = _selected_channel(ml_combo)
    se = _selected_channel(se_combo)
    if not ml and not se:
        QtWidgets.QMessageBox.warning(
            parent,
            "No mic tagged",
            "Tag at least one channel as Muzzle Left or Shooter's Ear.",
        )
        return None
    if ml and se and ml == se:
        QtWidgets.QMessageBox.warning(
            parent,
            "Same channel",
            "Muzzle Left and Shooter's Ear cannot be the same channel.",
        )
        return None
    channel_map: dict[str, MicPosition] = {}
    if ml:
        channel_map[ml] = MicPosition.ML
    if se:
        channel_map[se] = MicPosition.SE
    return channel_map


def _opt_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a whole number.") from None


def _opt_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number.") from None


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
