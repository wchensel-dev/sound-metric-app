"""PySide6 desktop app for the ingest -> mark -> close -> report workflow.

Four views over the same Phase B services the ``sma`` CLI drives, wired through
:class:`~sound_metric_app.ui.controller.WorkflowController`:

1. **Ingest / Unmarked** — scan the input folder, list Unmarked Data Sets.
2. **Mark** — annotate a shot, tag SE/MR channels, compute + store metrics.
3. **Batches** — Batch -> Group -> Shot tree with a Close-batch action.
4. **Report** — per-group SE vs MR averages, positions never mixed.

Ingest, mark, and close are explicit buttons (README user-actuated principle).
The two file-reading operations (ingest, mark) run on a worker thread so a large
capture never freezes the window; every service error surfaces as a dialog.

Run with:  python -m sound_metric_app.ui.main_window   (needs the 'gui' extra)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..models import MicPosition, Shot
from .controller import WorkflowController

_NONE_LABEL = "(none)"


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
    _COLUMNS = ["ID", "File", "SKU", "Platform", "Shot #"]

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
            values = [
                str(s.id),
                Path(s.source_file).name,
                s.suppressor_sku or "—",
                s.test_platform or "—",
                "—" if s.shot_order is None else str(s.shot_order),
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

        self.se_combo = QtWidgets.QComboBox()
        self.mr_combo = QtWidgets.QComboBox()
        form.addRow("SE channel:", self.se_combo)
        form.addRow("MR channel:", self.mr_combo)

        self.ammo_edit = QtWidgets.QLineEdit()
        form.addRow("Ammo *:", self.ammo_edit)
        self.sku_edit = QtWidgets.QLineEdit()
        form.addRow("SKU override:", self.sku_edit)
        self.platform_edit = QtWidgets.QLineEdit()
        form.addRow("Platform override:", self.platform_edit)
        self.shot_order_edit = QtWidgets.QLineEdit()
        form.addRow("Shot order:", self.shot_order_edit)
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

    def _on_shot_changed(self, *_args) -> None:
        self._channel_token += 1
        token = self._channel_token
        shot = self._current_shot()

        # Prefill override placeholders from the shot's provisional filename keys.
        self.sku_edit.setPlaceholderText(shot.suppressor_sku or "" if shot else "")
        self.platform_edit.setPlaceholderText(shot.test_platform or "" if shot else "")

        self._set_channel_choices([], loading=True)
        if shot is None:
            self._set_channel_choices([])
            return

        def load():
            return self.controller.channels_for(shot.source_file)

        def done(channels):
            if token != self._channel_token:
                return  # a newer shot was selected; ignore this stale result
            self._set_channel_choices([c.name for c in channels])

        self._run_async(load, done)

    def _set_channel_choices(self, names: list[str], *, loading: bool = False) -> None:
        for combo in (self.se_combo, self.mr_combo):
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
        if not loading and names:
            # Default guess: first channel -> SE, second (if any) -> MR.
            self.se_combo.setCurrentIndex(1)
            if len(names) >= 2:
                self.mr_combo.setCurrentIndex(2)

    # ---- mark ----------------------------------------------------------- #

    def _selected_channel(self, combo: QtWidgets.QComboBox) -> str | None:
        text = combo.currentText()
        return None if text in (_NONE_LABEL, "loading…", "") else text

    def _mark(self) -> None:
        shot = self._current_shot()
        if shot is None:
            QtWidgets.QMessageBox.information(self, "No shot", "No unmarked shot selected.")
            return

        ammo = self.ammo_edit.text().strip()
        if not ammo:
            QtWidgets.QMessageBox.warning(self, "Missing ammo", "Ammo is required to mark a shot.")
            return

        channel_map: dict[str, MicPosition] = {}
        se = self._selected_channel(self.se_combo)
        mr = self._selected_channel(self.mr_combo)
        if se:
            channel_map[se] = MicPosition.SE
        if mr:
            channel_map[mr] = MicPosition.MR
        if not channel_map:
            QtWidgets.QMessageBox.warning(
                self, "No mic tagged", "Tag at least one channel as SE or MR."
            )
            return
        if se and mr and se == mr:
            QtWidgets.QMessageBox.warning(
                self, "Same channel", "SE and MR cannot be the same channel."
            )
            return

        try:
            kwargs = dict(
                suppressor_sku=self.sku_edit.text().strip() or None,
                test_platform=self.platform_edit.text().strip() or None,
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
        parts = [f"Marked shot #{shot.id} — batch #{marked.batch.id} (SKU {marked.batch.sku})."]
        for position in (MicPosition.SE, MicPosition.MR):
            result = marked.metrics.get(position)
            if result is not None:
                parts.append(
                    f"{position.value}: peak {result.peak_db:.2f} dB, "
                    f"LIAeq {result.liaeq_100ms_db:.2f} dBA"
                )
        self.status_label.setText("\n".join(parts))
        self.ammo_edit.clear()
        self.sku_edit.clear()
        self.platform_edit.clear()
        self.shot_order_edit.clear()
        self.wind_edit.clear()
        self.temp_edit.clear()
        self.rh_edit.clear()
        self.main.notify_changed()


# --------------------------------------------------------------------------- #
# 3. Batch -> Group -> Shot tree view
# --------------------------------------------------------------------------- #


class ShotEditDialog(QtWidgets.QDialog):
    """Correct a marked shot's fields, pre-filled from its current state.

    Purely a form: it validates and exposes the collected values via
    :meth:`values`; the caller re-marks the shot (which re-clusters it into the
    right batch/group and recomputes metrics). SKU/platform/ammo default to what
    the shot was actually clustered into — its batch and group — not the
    provisional filename keys, so an unchanged save is a true no-op.
    """

    def __init__(
        self,
        shot: Shot,
        *,
        sku: str,
        platform: str,
        ammo: str,
        channel_names: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Edit shot #{shot.id}")
        self._shot = shot
        self._values: dict | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(Path(shot.source_file).name))
        form = QtWidgets.QFormLayout()

        self.se_combo = QtWidgets.QComboBox()
        self.mr_combo = QtWidgets.QComboBox()
        for combo in (self.se_combo, self.mr_combo):
            combo.addItem(_NONE_LABEL)
            combo.addItems(channel_names)
        _select_channel(self.se_combo, shot.se_channel)
        _select_channel(self.mr_combo, shot.mr_channel)
        form.addRow("SE channel:", self.se_combo)
        form.addRow("MR channel:", self.mr_combo)

        self.ammo_edit = QtWidgets.QLineEdit(ammo or "")
        form.addRow("Ammo *:", self.ammo_edit)
        self.sku_edit = QtWidgets.QLineEdit(sku or "")
        form.addRow("SKU *:", self.sku_edit)
        self.platform_edit = QtWidgets.QLineEdit(platform or "")
        form.addRow("Platform *:", self.platform_edit)
        self.shot_order_edit = QtWidgets.QLineEdit(_str_or_empty(shot.shot_order))
        form.addRow("Shot order:", self.shot_order_edit)
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

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _selected_channel(self, combo: QtWidgets.QComboBox) -> str | None:
        text = combo.currentText()
        return None if text in (_NONE_LABEL, "") else text

    def _on_accept(self) -> None:
        ammo = self.ammo_edit.text().strip()
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

        channel_map: dict[str, MicPosition] = {}
        se = self._selected_channel(self.se_combo)
        mr = self._selected_channel(self.mr_combo)
        if se:
            channel_map[se] = MicPosition.SE
        if mr:
            channel_map[mr] = MicPosition.MR
        if not channel_map:
            QtWidgets.QMessageBox.warning(
                self, "No mic tagged", "Tag at least one channel as SE or MR."
            )
            return
        if se and mr and se == mr:
            QtWidgets.QMessageBox.warning(
                self, "Same channel", "SE and MR cannot be the same channel."
            )
            return

        try:
            self._values = dict(
                ammo=ammo,
                channel_map=channel_map,
                suppressor_sku=sku,
                test_platform=platform,
                shot_order=_opt_int(self.shot_order_edit.text()),
                wind_speed=_opt_float(self.wind_edit.text()),
                temp=_opt_float(self.temp_edit.text()),
                relative_humidity=_opt_float(self.rh_edit.text()),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return
        self.accept()

    def values(self) -> dict:
        """The validated ``controller.mark`` kwargs. Valid only after Save."""
        assert self._values is not None, "values() called before an accepted Save"
        return self._values


class BatchTreeView(_View):
    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        layout = QtWidgets.QVBoxLayout(self)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Batch / Group / Shot", "Detail", "Timestamp"])
        self.tree.itemSelectionChanged.connect(self._update_actions_enabled)
        # Only leaf shot rows edit on double-click; on a batch/group row that
        # gesture is Qt's expand/collapse and must not also pop an edit modal.
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)

        button_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        self.edit_btn = QtWidgets.QPushButton("Edit…")
        self.edit_btn.setEnabled(False)
        self.edit_btn.clicked.connect(self._edit_selected)
        self.close_btn = QtWidgets.QPushButton("Close batch")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self._close_batch)
        button_row.addWidget(refresh_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.edit_btn)
        button_row.addWidget(self.close_btn)
        layout.addLayout(button_row)

    def refresh(self) -> None:
        self.tree.clear()
        for node in self.controller.batch_tree():
            batch = node.batch
            state = "closed" if batch.closed else "open"
            b_item = QtWidgets.QTreeWidgetItem(
                [f"Batch #{batch.id}  SKU {batch.sku}", f"[{state}]", ""]
            )
            b_item.setData(0, QtCore.Qt.UserRole, ("batch", batch))
            self.tree.addTopLevelItem(b_item)
            for g_node in node.groups:
                group = g_node.group
                n = len(g_node.shots)
                g_item = QtWidgets.QTreeWidgetItem(
                    [f"Group #{group.id}  {group.test_platform} / {group.ammo}", f"{n} shot(s)", ""]
                )
                g_item.setData(0, QtCore.Qt.UserRole, ("group", group))
                b_item.addChild(g_item)
                for shot in g_node.shots:
                    tags = f"SE:{shot.se_channel or '—'}  MR:{shot.mr_channel or '—'}"
                    s_item = QtWidgets.QTreeWidgetItem(
                        [
                            f"Shot #{shot.id}  {Path(shot.source_file).name}",
                            tags,
                            _format_captured_at(shot.captured_at),
                        ]
                    )
                    # Carry the shot's group and batch so an edit can pre-fill the
                    # SKU/platform/ammo it was actually clustered into (which may
                    # differ from its provisional filename keys after an override).
                    s_item.setData(0, QtCore.Qt.UserRole, ("shot", shot, group, batch))
                    g_item.addChild(s_item)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)
        self._update_actions_enabled()

    def _selected_entry(self) -> tuple | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole)

    def _selected_batch(self) -> tuple[int, bool] | None:
        entry = self._selected_entry()
        if entry and entry[0] == "batch":
            batch = entry[1]
            return batch.id, batch.closed
        return None

    def _update_actions_enabled(self) -> None:
        entry = self._selected_entry()
        kind = entry[0] if entry else None
        # A batch (rename its SKU) or a shot (re-mark it) can be edited; a group
        # is renamed by editing its shots, so it has no direct edit action.
        self.edit_btn.setEnabled(kind in ("batch", "shot"))
        selected = self._selected_batch()
        self.close_btn.setEnabled(selected is not None and not selected[1])

    # ---- edit ----------------------------------------------------------- #

    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        entry = item.data(0, QtCore.Qt.UserRole)
        if entry and entry[0] == "shot":
            self._edit_selected()

    def _edit_selected(self) -> None:
        entry = self._selected_entry()
        if not entry:
            return
        if entry[0] == "batch":
            self._edit_batch(entry[1])
        elif entry[0] == "shot":
            self._edit_shot(entry[1], entry[2], entry[3])

    def _edit_batch(self, batch) -> None:
        new_sku, ok = QtWidgets.QInputDialog.getText(
            self,
            f"Edit batch #{batch.id}",
            "SKU:",
            QtWidgets.QLineEdit.Normal,
            batch.sku,
        )
        if not ok:
            return
        new_sku = new_sku.strip()
        if not new_sku or new_sku == batch.sku:
            return
        try:
            self.controller.rename_batch(batch.id, new_sku)
        except Exception as exc:  # noqa: BLE001 — surface to the user as a dialog
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        self.main.notify_changed()

    def _edit_shot(self, shot, group, batch) -> None:
        # Re-marking a shot whose batch is closed re-clusters it into a *new* open
        # batch (a closed batch is never the SKU's open batch), so warn first.
        if batch.closed:
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Batch closed",
                f"Batch #{batch.id} (SKU {batch.sku}) is closed. Saving changes will "
                f"move shot #{shot.id} into a new open batch. Continue?",
            )
            if confirm != QtWidgets.QMessageBox.Yes:
                return

        # Load the raw channels off the UI thread, then open the pre-filled dialog.
        self._run_async(
            lambda: self.controller.channels_for(shot.source_file),
            lambda channels: self._open_shot_dialog(shot, group, batch, channels),
            busy=(self.edit_btn,),
        )

    def _open_shot_dialog(self, shot, group, batch, channels) -> None:
        dialog = ShotEditDialog(
            shot,
            sku=batch.sku,
            platform=group.test_platform,
            ammo=group.ammo,
            channel_names=[c.name for c in channels],
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
        selected = self._selected_batch()
        if selected is None:
            return
        batch_id = selected[0]
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Close batch",
            f"Close batch #{batch_id}? Further testing for this SKU starts a new batch.",
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.controller.close_batch(batch_id)
        except Exception as exc:  # noqa: BLE001 — surface to the user as a dialog
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        self.main.notify_changed()


# --------------------------------------------------------------------------- #
# 4. Report view
# --------------------------------------------------------------------------- #


class ReportView(_View):
    _COLUMNS = ["Group", "Mic", "n", "Peak dB", "Peak dBA", "Peak Impulse dB", "LIAeq,100ms dBA"]

    def __init__(self, controller: WorkflowController, main: "MainWindow"):
        super().__init__(controller, main)
        layout = QtWidgets.QVBoxLayout(self)

        picker_row = QtWidgets.QHBoxLayout()
        picker_row.addWidget(QtWidgets.QLabel("Batch:"))
        self.batch_combo = QtWidgets.QComboBox()
        self.batch_combo.currentIndexChanged.connect(self._load_report)
        picker_row.addWidget(self.batch_combo, 1)
        layout.addLayout(picker_row)

        self.table = QtWidgets.QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def refresh(self) -> None:
        current = self.batch_combo.currentData()
        # Keep signals blocked through setCurrentIndex so currentIndexChanged
        # doesn't fire _load_report; we call it once, explicitly, below.
        self.batch_combo.blockSignals(True)
        self.batch_combo.clear()
        for batch in self.controller.batches():
            state = "closed" if batch.closed else "open"
            self.batch_combo.addItem(f"#{batch.id}  SKU {batch.sku}  [{state}]", batch.id)
        index = self.batch_combo.findData(current)
        self.batch_combo.setCurrentIndex(index if index >= 0 else (0 if self.batch_combo.count() else -1))
        self.batch_combo.blockSignals(False)

        self._load_report()

    def _load_report(self, *_args) -> None:
        self.table.setRowCount(0)
        batch_id = self.batch_combo.currentData()
        if batch_id is None:
            return
        report = self.controller.batch_report(batch_id)
        for group_avg in report.groups:
            g = group_avg.group
            group_label = f"#{g.id}  {g.test_platform} / {g.ammo}"
            if not group_avg.averages:
                self._add_row([group_label, "—", "0", "no metrics", "", "", ""])
                continue
            for position in (MicPosition.SE, MicPosition.MR):
                avg = group_avg.averages.get(position)
                if avg is None:
                    continue
                self._add_row(
                    [
                        group_label,
                        position.value,
                        str(avg["n"]),
                        f"{avg['peak_db']:.2f}",
                        f"{avg['peak_dba']:.2f}",
                        f"{avg['peak_impulse_db']:.2f}",
                        f"{avg['liaeq_100ms_db']:.2f}",
                    ]
                )
                group_label = ""  # only label the first mic row of each group
        self.table.resizeColumnsToContents()

    def _add_row(self, values: list[str]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, text in enumerate(values):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: WorkflowController | None = None):
        super().__init__()
        self.setWindowTitle("Sound Metric App — Workflow")
        self.resize(820, 560)
        self.controller = controller or WorkflowController()

        self.ingest_view = IngestView(self.controller, self)
        self.marking_view = MarkingView(self.controller, self)
        self.batch_view = BatchTreeView(self.controller, self)
        self.report_view = ReportView(self.controller, self)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.ingest_view, "Ingest")
        self.tabs.addTab(self.marking_view, "Mark")
        self.tabs.addTab(self.batch_view, "Batches")
        self.tabs.addTab(self.report_view, "Report")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self._views = [self.ingest_view, self.marking_view, self.batch_view, self.report_view]
        self.notify_changed()

    def _on_tab_changed(self, index: int) -> None:
        self.tabs.widget(index).refresh()

    def notify_changed(self) -> None:
        """Reload every view after a mutating action (ingest / mark / close)."""
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
