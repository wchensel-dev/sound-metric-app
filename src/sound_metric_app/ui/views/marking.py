"""Tab 2 — Mark: annotate an unmarked shot and compute its metrics.

The form that turns an Unmarked Data Set into a marked shot. Channels arrive
pre-tagged from the AI 1 / AI 2 DAQ convention and stay editable; ammo is
required; the Role field echoes the FRP/Regular role the entered shot order
implies, since role is derived and never typed. Marking lands the shot in the
data bank **idle** — bringing it forward is the Data bank tab's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

from PySide6 import QtWidgets

from ... import config
from ...models import MicPosition, Shot, role_for_order
from ..controller import WorkflowController
from ..format import (
    _EMPTY,
    _LOADING_LABEL,
    _NONE_LABEL,
    _fmt_trigger,
    _opt_float,
    _opt_int,
    _select_channel,
    _str_or_empty,
    _tagged_channel_map,
)
from .base import _View

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from ..main_window import MainWindow


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
        # Onset trigger this shot was recorded with; pre-filled from the
        # configured default so a normal mark records the recorder's current
        # trigger, editable for a capture recorded at a different level.
        self.trigger_edit = QtWidgets.QLineEdit()
        self.trigger_edit.setPlaceholderText("Pa")
        form.addRow("Trigger (Pa):", self.trigger_edit)
        self._seed_trigger_default()

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

    def _seed_trigger_default(self) -> None:
        """Pre-fill the trigger field with the configured default (Pa)."""
        # Runs at construction (from __init__) and after each mark, so a corrupt
        # default_trigger_pa setting — which get_default_trigger_pa() now rejects
        # rather than silently accepts — must surface as a dialog rather than
        # escaping as an unhandled crash that stops launch, the same treatment
        # the malformed-ammo path gets in _populate_ammo.
        try:
            default_pa = config.get_default_trigger_pa()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            self.trigger_edit.clear()
            return
        self.trigger_edit.setText(_fmt_trigger(default_pa))

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
                trigger_pa=_opt_float(self.trigger_edit.text()),
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
        self._seed_trigger_default()
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
