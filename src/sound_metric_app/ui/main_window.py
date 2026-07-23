"""PySide6 desktop app for the ingest -> mark -> bring-forward -> report workflow.

Four views over the same Phase B services the ``sma`` CLI drives, wired through
:class:`~sound_metric_app.ui.controller.WorkflowController`:

1. **Ingest / Unmarked** — scan the input folder, list Unmarked Data Sets.
2. **Mark** — annotate a shot, confirm its ML/SE channel tags, compute + store
   metrics. The shot lands in the data bank idle.
3. **Data bank** — the Combination -> Batch -> Cluster -> Shot tree: every shot
   the app has seen, included or idle, with the bring-forward actions that decide
   which ones feed an average, plus session editing and Close batch.
4. **Batch average** — the four position x role output slots per batch
   (muzzle-left / shooter's-ear crossed with FRP / regular), positions and roles
   never mixed.

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
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..dsp import SMOOTHING_FAST, SMOOTHING_INSTANT, SMOOTHING_SLOW
from ..models import MicPosition, Shot, role_for_order
from ..services import AVERAGE_SLOTS
from .controller import WorkflowController

_NONE_LABEL = "(none)"
#: Shown where a mic position has no channel tagged / a value is missing.
_EMPTY = "—"


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
    _COLUMNS = ["ID", "File", "SKU", "Platform", "Cluster", "Shot #", "Role"]

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
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

        layout.addWidget(QtWidgets.QLabel("Unmarked data sets:"))
        self.table = QtWidgets.QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
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
        self.table.resizeColumnsToContents()

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
        lines = [
            f"Ingested {report.n_ingested}, "
            f"already present {len(report.already_present)}, "
            f"malformed {len(report.malformed)}, "
            f"unreadable {len(report.unreadable)}."
        ]
        for path, reason in report.malformed:
            lines.append(f"  malformed: {Path(path).name} — {reason}")
        for path, reason in report.unreadable:
            lines.append(f"  unreadable: {Path(path).name} — {reason}")
        self.status_label.setText("\n".join(lines))
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
        layout.addWidget(self.mark_btn)

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
                combo.addItem("loading…")
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

    def _selected_channel(self, combo: QtWidgets.QComboBox) -> str | None:
        text = combo.currentText()
        return None if text in (_NONE_LABEL, "loading…", "") else text

    def _channel_map(self) -> dict[str, MicPosition] | None:
        """The tagged channel map, or ``None`` after warning about a bad tagging.

        Returning ``None`` (rather than an empty map) keeps "the user needs to fix
        something" distinct from "nothing tagged" — the caller aborts either way,
        but the warning has already been shown here.
        """
        ml = self._selected_channel(self.ml_combo)
        se = self._selected_channel(self.se_combo)
        if not ml and not se:
            QtWidgets.QMessageBox.warning(
                self,
                "No mic tagged",
                "Tag at least one channel as Muzzle Left or Shooter's Ear.",
            )
            return None
        if ml and se and ml == se:
            QtWidgets.QMessageBox.warning(
                self,
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

    def _mark(self) -> None:
        shot = self._current_shot()
        if shot is None:
            QtWidgets.QMessageBox.information(self, "No shot", "No unmarked shot selected.")
            return

        ammo = self.ammo_combo.currentText().strip()
        if not ammo:
            QtWidgets.QMessageBox.warning(self, "Missing ammo", "Ammo is required to mark a shot.")
            return

        channel_map = self._channel_map()
        if channel_map is None:
            return

        try:
            kwargs = dict(
                suppressor_sku=self.sku_edit.text().strip() or None,
                test_platform=self.platform_edit.text().strip() or None,
                cluster_index=_opt_int(self.cluster_edit.text()),
                shot_order=_opt_int(self.shot_order_edit.text()),
                wind_speed=_opt_float(self.wind_edit.text()),
                temp=_opt_float(self.temp_edit.text()),
                relative_humidity=_opt_float(self.rh_edit.text()),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
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

    def _selected_channel(self, combo: QtWidgets.QComboBox) -> str | None:
        text = combo.currentText()
        return None if text in (_NONE_LABEL, "") else text

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

        ml = self._selected_channel(self.ml_combo)
        se = self._selected_channel(self.se_combo)
        if not ml and not se:
            QtWidgets.QMessageBox.warning(
                self,
                "No mic tagged",
                "Tag at least one channel as Muzzle Left or Shooter's Ear.",
            )
            return
        if ml and se and ml == se:
            QtWidgets.QMessageBox.warning(
                self,
                "Same channel",
                "Muzzle Left and Shooter's Ear cannot be the same channel.",
            )
            return
        channel_map: dict[str, MicPosition] = {}
        if ml:
            channel_map[ml] = MicPosition.ML
        if se:
            channel_map[se] = MicPosition.SE

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

    _COLUMNS = ["Combination / Batch / Cluster / Shot", "Detail", "Role", "Timestamp"]

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        #: Set while refresh() repopulates the tree, so the itemChanged handler
        #: does not treat programmatic check-state writes as user clicks.
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(self._COLUMNS)
        self.tree.itemSelectionChanged.connect(self._update_actions_enabled)
        # Double-click means "edit" here, so Qt's default expand/collapse on the
        # same gesture is off: a batch row is editable *and* has children, and
        # one double-click must not both toggle the branch and pop a modal
        # (cancelling the modal would leave the branch toggled anyway). Expanding
        # stays on the branch arrow, the keyboard, and expandAll() in refresh().
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
        # Prune empty clusters/batches/combinations before rendering; data_bank()
        # itself is a pure read, so the sweep is an explicit step on this path.
        self.controller.sweep_empty()
        self._loading = True
        try:
            self.tree.clear()
            for node in self.controller.data_bank():
                self.tree.addTopLevelItem(self._combination_item(node))
        finally:
            self._loading = False
        self.tree.expandAll()
        for col in range(len(self._COLUMNS)):
            self.tree.resizeColumnToContents(col)
        self._update_actions_enabled()

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
            item.addChild(self._shot_item(shot, cluster, batch, combo))
        return item

    def _shot_item(self, shot, cluster, batch, combo) -> QtWidgets.QTreeWidgetItem:
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


# --------------------------------------------------------------------------- #
# 4. Report view
# --------------------------------------------------------------------------- #


class MetricGraph(QtWidgets.QWidget):
    """The Report tab's right-hand pane: one metric's time series for one shot.

    A thin wrapper over a :class:`pyqtgraph.PlotWidget`, with a header row above
    it: the graph title on the left and a level-weighting dropdown on the right.
    That dropdown chooses how the SPL-over-time curve is drawn — the raw
    per-sample level (a point cloud) or a Fast/Slow time-weighted RMS envelope (a
    continuous line). It emits :attr:`smoothingChanged` when the user switches so
    the owning view can re-request the trace. Colours track the active light/dark
    palette so the plot doesn't clash with the rest of the window.

    Clicking a point on the drawn curve snaps to the nearest sample and shows its
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

    #: Small dots so the thousands of samples read as a point cloud, not a mass.
    _DOT_BRUSH = pg.mkBrush(66, 135, 245)
    #: A joined line for the time-weighted envelope (the smooth SLM-style curve).
    _LINE_PEN = pg.mkPen((66, 135, 245), width=1)
    _MARK_PEN = pg.mkPen((214, 90, 70), width=1)
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
        #: The trace currently drawn, kept so a plot click can find the sample it
        #: landed on. None whenever the plot shows a message rather than a curve.
        self._trace = None
        #: Scatter item marking the picked sample, or None when nothing is picked.
        self._pick_marker: pg.ScatterPlotItem | None = None

        # A click anywhere on the plot scene tries to select the nearest sample.
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._apply_theme()
        self.show_message("Click a metric cell on a shot row to graph it.")

    def current_smoothing(self) -> str:
        """The ``build_metric_trace`` smoothing mode currently selected."""
        return self._smoothing_combo.currentData()

    def _apply_theme(self) -> None:
        pal = self.palette()
        self._fg = pal.color(QtGui.QPalette.Text)
        self._plot.setBackground(pal.color(QtGui.QPalette.Base))
        for name in ("left", "bottom"):
            axis = self._plot.getAxis(name)
            axis.setPen(self._fg)
            axis.setTextPen(self._fg)

    def show_message(self, text: str) -> None:
        """Clear the plot and show a short prompt in place of a graph."""
        self._plot.clear()
        self._title_label.setText(text)
        self._plot.setLabel("left", "")
        self._x_bounds = None
        self._window_x_bounds = None
        self._trace = None
        self.clear_readout()
        self._auto_frame_btn.setEnabled(False)
        self._frame_window_btn.setEnabled(False)

    def auto_frame(self) -> None:
        """Snap the X range to the drawn curve's extent (first to last live sample).

        A no-op when no trace is shown. Y is left on autorange so the fitted
        width still shows the curve's full vertical extent.
        """
        if self._x_bounds is None:
            return
        x0, x1 = self._x_bounds
        self._plot.setXRange(x0, x1, padding=0)
        self._plot.enableAutoRange(axis="y")

    def frame_calc_window(self) -> None:
        """Snap the X range to the calculation window's start/end lines.

        The zoomed-in counterpart to :meth:`auto_frame`: same behaviour, but
        bracketing only the samples that fed the reported number. A little
        padding is kept so both window lines stay visible at the edges rather
        than sitting exactly on the frame. A no-op when the trace carries no
        window. Y is left on autorange, so the framed slice shows its own
        vertical extent rather than the whole curve's.
        """
        if self._window_x_bounds is None:
            return
        x0, x1 = self._window_x_bounds
        self._plot.setXRange(x0, x1, padding=0.02)
        self._plot.enableAutoRange(axis="y")

    def show_trace(self, trace, subtitle: str = "") -> None:
        """Render a :class:`~sound_metric_app.dsp.MetricTrace` as the sole graph."""
        self._plot.clear()
        self._title_label.setText(subtitle or trace.title)
        self._plot.setLabel("left", trace.y_label)
        # A new curve invalidates any prior point pick (different samples/units).
        self._trace = trace
        self.clear_readout()
        if trace.connected:
            # Time-weighted envelope: a joined line reads as the continuous level
            # a meter shows. NaN samples break the line into gaps.
            self._plot.plot(trace.t_ms, trace.values, pen=self._LINE_PEN)
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
                symbolBrush=self._DOT_BRUSH,
                pxMode=True,
            )
        if trace.peak_index is not None:
            x = float(trace.t_ms[trace.peak_index])
            self._plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=self._MARK_PEN))
        if trace.level is not None:
            self._plot.addItem(
                pg.InfiniteLine(
                    pos=trace.level, angle=0,
                    pen=pg.mkPen((214, 90, 70), width=1, style=QtCore.Qt.DashLine),
                )
            )
        # The calculation window's edges. The curve runs straight through both so
        # a pre-onset or late event stays visible, but only samples *between* them
        # reached the reported number — the labels say so, since a curve that
        # simply continues would otherwise read as all-included. The start also
        # shows where onset detection fired — the same time for every metric bar
        # Peak-10 ms-Leq, which opens its window a trailing-RMS length earlier so
        # the bracket still contains every sample that fed the reported number.
        # When nothing crossed the onset threshold the window falls back to the
        # frame start, so the start label says that outright — otherwise a
        # mis-triggered or silent capture reads as a confident onset at 0 ms.
        start_text = (
            "calc window starts"
            if trace.onset_detected
            else "calc window starts (no onset detected)"
        )
        window_xs: list[float] = []
        for index, text in (
            (trace.window_start_index, start_text),
            (trace.window_end_index, "calc window ends"),
        ):
            if index is None:
                continue
            x = float(trace.t_ms[index])
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
        # Yellow dotted verticals bracket the drawn curve's extent (first/last
        # sample that actually carries a value), so the data stays visible however
        # far the user pans or zooms. Use the finite (non-NaN) span rather than the
        # raw sample axis: the Impulse ∫p·dt curve is NaN before the onset (the
        # integral is undefined there), so framing to the full frame would open
        # with dead space at the left. Full-frame SPL traces are all-finite, so
        # their bounds are unchanged. Note this is the *drawn* extent, which now
        # runs to the end of the capture — the calculation window's end is the
        # separate dashed line above.
        finite = np.isfinite(trace.values)
        if finite.any():
            xs = trace.t_ms[finite]
            x0 = float(xs[0])
            x1 = float(xs[-1])
            self._x_bounds = (x0, x1)
            for x in (x0, x1):
                self._plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=self._BOUND_PEN))
            self._auto_frame_btn.setEnabled(True)
        else:
            self._x_bounds = None
            self._auto_frame_btn.setEnabled(False)
        self._plot.enableAutoRange()

    # ---- point readout -------------------------------------------------- #

    def _on_plot_clicked(self, event) -> None:
        """Select the sample nearest the click and show its value + time.

        Snaps to the nearest sample in time, then keeps the pick only if the
        click landed within :data:`_PICK_TOLERANCE_PX` screen pixels of that
        sample — so clicking empty space leaves any current readout untouched.
        """
        if self._trace is None or self._trace.t_ms.size == 0:
            return
        scene_pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            return
        vb = self._plot.getPlotItem().vb
        view_pos = vb.mapSceneToView(scene_pos)

        t = self._trace.t_ms
        # t_ms is sorted ascending; find the nearer of the two bracketing samples.
        i = int(np.searchsorted(t, view_pos.x()))
        candidates = [j for j in (i - 1, i) if 0 <= j < t.size]
        idx = min(candidates, key=lambda j: abs(t[j] - view_pos.x()))

        value = float(self._trace.values[idx])
        if not np.isfinite(value):
            return  # a silent/NaN gap (e.g. Impulse tail) has no level to show

        # Reject clicks that only landed near in time but far from the sample: map
        # the sample back to screen space and measure the true pixel distance.
        point_scene = vb.mapViewToScene(QtCore.QPointF(float(t[idx]), value))
        if np.hypot(point_scene.x() - scene_pos.x(), point_scene.y() - scene_pos.y()) > self._PICK_TOLERANCE_PX:
            return

        self._show_readout(idx, value)

    def _show_readout(self, idx: int, value: float) -> None:
        """Mark sample ``idx`` on the plot and fill the readout box."""
        x = float(self._trace.t_ms[idx])
        if self._pick_marker is not None:
            self._plot.removeItem(self._pick_marker)
        self._pick_marker = pg.ScatterPlotItem(
            [x], [value], size=11, brush=self._PICK_BRUSH, pen=self._PICK_PEN, pxMode=True
        )
        self._plot.addItem(self._pick_marker)

        unit = _unit_of(self._trace.y_label)
        unit_suffix = f" {unit}" if unit else ""
        self._readout_label.setText(f"{value:.3f}{unit_suffix}  @ {x:.2f} ms")
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
    """

    _COLUMNS = [
        "Slot / Shot", "n",
        "Peak Pa", "Peak dB", "Peak dBA",
        "Impulse Pa·ms", "Impulse dB·ms",
        "Peak Leq10ms dBA", "LIAeq,100ms dBA",
    ]
    _METRIC_KEYS = (
        "peak_pa", "peak_db", "peak_dba",
        "impulse_pa_ms", "peak_impulse_db",
        "leq10ms_db", "liaeq_100ms_db",
    )
    #: Metric columns begin here; columns 0-1 are label / n.
    _FIRST_METRIC_COL = 2

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
        picker_row.addWidget(QtWidgets.QLabel("Batch:"))
        self.batch_combo = QtWidgets.QComboBox()
        self.batch_combo.currentIndexChanged.connect(self._load_report)
        picker_row.addWidget(self.batch_combo, 1)
        layout.addLayout(picker_row)

        # Progress against the soft 3-FRP / 5-regular targets, so a short or
        # over-filled batch is visible without reading the per-slot counts.
        self.status_label = QtWidgets.QLabel("")
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
        split.addWidget(self.tree)

        self.graph = MetricGraph()
        # Re-graph the same cell with the new weighting when the dropdown changes.
        self.graph.smoothingChanged.connect(self._render_current)
        split.addWidget(self.graph)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split)

    def refresh(self) -> None:
        current = self.batch_combo.currentData()
        # Keep signals blocked through setCurrentIndex so currentIndexChanged
        # doesn't fire _load_report; we call it once, explicitly, below.
        self.batch_combo.blockSignals(True)
        self.batch_combo.clear()
        combinations = {c.id: c for c in self.controller.combinations()}
        for batch in self.controller.batches():
            combo = combinations.get(batch.combination_id)
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
        for position, role in AVERAGE_SLOTS:
            slot_label = f"{position.label} · {role.label}"
            avg = report.averages.get((position, role))
            if avg is None:
                # Keep the empty slot visible: a missing quadrant is information,
                # not something to hide. Pad so the row spans all _COLUMNS.
                self._add_top(
                    [slot_label, "0", "none included", *[""] * (len(self._METRIC_KEYS) - 1)]
                )
                continue
            avg_item = self._add_top(
                [
                    slot_label,
                    str(avg["n"]),
                    *(_format_metric(avg[k]) for k in self._METRIC_KEYS),
                ]
            )
            for shot in report.shots.get((position, role), ()):
                avg_item.addChild(self._shot_item(shot, position))
        self.tree.expandAll()
        for col in range(len(self._COLUMNS)):
            self.tree.resizeColumnToContents(col)

    def _shot_item(self, shot: dict, position: MicPosition) -> QtWidgets.QTreeWidgetItem:
        cluster = shot.get("cluster_index")
        order = shot.get("shot_order")
        label = f"Cluster {cluster} · shot {order}" if cluster else f"Shot {order}"
        if order is None:
            label = Path(shot["source_file"]).name
        item = QtWidgets.QTreeWidgetItem(
            [label, "", *(_format_metric(shot[k]) for k in self._METRIC_KEYS)]
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

    # ---- graph ---------------------------------------------------------- #

    def _on_cell_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        """Graph the clicked metric for the clicked shot (one graph at a time)."""
        entry = item.data(0, QtCore.Qt.UserRole)
        if not entry or entry[0] != "shot":
            self._current_request = None
            self.graph.show_message("Select a metric cell on an individual shot row.")
            return
        if column < self._FIRST_METRIC_COL:
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

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.ingest_view, "Ingest")
        self.tabs.addTab(self.marking_view, "Mark")
        self.tabs.addTab(self.bank_view, "Data bank")
        self.tabs.addTab(self.report_view, "Batch average")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._views = [self.ingest_view, self.marking_view, self.bank_view, self.report_view]
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
        for view in self._views:
            view.refresh()

    def open_marking_for(self, shot_id: int) -> None:
        """Switch to the Mark tab focused on ``shot_id`` (from the Ingest view)."""
        self.marking_view.select_shot(shot_id)
        self.tabs.setCurrentWidget(self.marking_view)


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
