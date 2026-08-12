"""Tab 3 — Data bank: the Combination -> Batch -> Cluster -> Shot archive.

The complete archive, unfiltered: every shot the app has seen, included or idle.
This is where inclusion is decided (the checkbox on a shot row, or the
bring-whole-cluster-forward buttons), where a batch is closed, and where a shot
or a session is corrected through the dialogs in
:mod:`~sound_metric_app.ui.dialogs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ...models import MicPosition, Shot
from ..compare_series import CompareSeries, _compare_series, _pin_compare_series
from ..controller import WorkflowController
from ..dialogs import BatchEditDialog, ShotEditDialog
from ..format import _EMPTY, _format_captured_at, _format_floor_pair, _str_or_empty
from ..metric_columns import _PRETRIGGER_FLOOR_LABEL
from ..qtwidgets import (
    _expanded_keys,
    _repopulate_sku_filter,
    _restore_expanded,
    _style_grid_tree,
)
from .base import _View

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from ..main_window import MainWindow


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
