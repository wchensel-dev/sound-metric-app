"""Tab 5 — Compare: one metric's curve for any number of shots, overlaid.

Nothing here is scoped to a batch or a SKU. Shots are pinned from the Batch
average and Data bank tabs and stay until removed, so a comparison can mix
batches, SKUs, both mics, and included/idle status alike. One metric is picked
for the whole overlay, because curves have to share a Y axis to be read against
each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ...dsp import MetricTrace
from ...models import MicPosition
from ..compare_series import CompareSeries
from ..controller import WorkflowController
from ..graph import MetricGraph, _color_swatch, _MUTED_INK
from ..metric_columns import _REPORT_METRICS
from .base import _View

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from ..main_window import MainWindow


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
        #: The (metric, smoothing, absolute) the two dicts above were filled for.
        self._cache_key: tuple[str, str, bool] | None = None
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
        # Re-draw with the new weighting when the dropdown changes, or with the
        # new signed/absolute presentation when the toggle flips; the cache is
        # keyed by both, so this reloads rather than redrawing stale curves.
        self.graph.smoothingChanged.connect(self._render)
        self.graph.absoluteChanged.connect(self._render)
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
        cache_key = (
            self.metric_combo.currentData(),
            self.graph.current_smoothing(),
            self.graph.absolute_value(),
        )
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

        metric_key, smoothing, absolute = cache_key
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
                        series.shot_id,
                        series.position,
                        metric_key,
                        smoothing=smoothing,
                        absolute=absolute,
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
