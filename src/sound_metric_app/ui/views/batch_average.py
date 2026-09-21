"""Tab 4 — Batch average: a batch's four position x role output slots.

The filtered counterpart to the Data bank tab: only shots whose ``included`` flag
is set reach these numbers. A tree of four slot rows, each expanding into the
shots averaged behind it, beside the shared metric graph — clicking a metric cell
on a shot row draws that metric's curve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from ...models import MicPosition, ShotRole
from ...services import AVERAGE_SLOTS, BatchAverages, slot_line
from ..compare_series import (
    _COMPARE_LABEL,
    CompareSeries,
    _compare_series,
    _pin_compare_series,
)
from ..controller import WorkflowController
from ..format import _format_floor, _format_metric
from ..graph import MetricGraph
from ..metric_columns import _REPORT_DIAGNOSTICS, _REPORT_METRICS
from ..qtwidgets import (
    _AVERAGE_ROW_TINT,
    _TopLevelRowTint,
    _expanded_keys,
    _flash_button,
    _repopulate_sku_filter,
    _restore_expanded,
    _style_grid_tree,
)
from .base import _View

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from ..main_window import MainWindow

#: Text on an averaged row's Scout-paste button, and the confirmation it flashes
#: after a click (a clipboard write is otherwise invisible). How long it holds
#: that text is :data:`~sound_metric_app.ui.qtwidgets._COPIED_FLASH_MS`.
_COPY_LABEL = "Copy"
_COPIED_LABEL = "Copied ✓"


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
        # Re-graph the same cell with the new weighting when the dropdown changes,
        # or with the new signed/absolute presentation when the toggle flips.
        self.graph.smoothingChanged.connect(self._render_current)
        self.graph.absoluteChanged.connect(self._render_current)
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
        absolute = self.graph.absolute_value()
        self.graph.show_message("Loading…")

        def done(trace) -> None:
            if token != self._graph_token:
                return  # a newer request superseded this one; drop the stale trace
            self.graph.show_trace(trace, subtitle)

        self._run_async(
            lambda: self.controller.metric_trace(
                shot_id, position, metric_key, smoothing=smoothing, absolute=absolute
            ),
            done,
        )
