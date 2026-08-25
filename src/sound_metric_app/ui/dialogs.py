"""The workflow window's modal forms.

Three dialogs, all pure forms: they validate what was typed and expose the
collected values through a ``values()`` (or ``definitions()``) accessor, leaving
the caller to do the actual write through the controller. None of them touches
the database, so none of them can half-commit a correction.

* :class:`ShotEditDialog` — correct a marked shot; the caller re-marks it.
* :class:`BatchEditDialog` — a batch's session context (label, date, typical
  weather, notes).
* :class:`AmmoDefinitionsDialog` — the ammo presets offered when marking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import config
from ..models import Shot, role_for_order
from .format import (
    _EMPTY,
    _NONE_LABEL,
    _fmt_trigger,
    _format_captured_at,
    _opt_float,
    _opt_int,
    _safe_int,
    _select_channel,
    _str_or_empty,
    _tagged_channel_map,
)


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
        # Onset trigger this shot was recorded with. Pre-fill the stored value, or
        # the configured default for a legacy shot that never recorded one, so
        # re-marking it adopts the current trigger unless the operator says otherwise.
        trigger_default = (
            shot.trigger_pa if shot.trigger_pa is not None else config.get_default_trigger_pa()
        )
        self.trigger_edit = QtWidgets.QLineEdit(_fmt_trigger(trigger_default))
        form.addRow("Trigger (Pa):", self.trigger_edit)
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
                trigger_pa=_opt_float(self.trigger_edit.text()),
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
