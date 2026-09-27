"""Tab 5 — Compare: one metric's curve for any number of shots, overlaid.

Nothing here is scoped to a batch or a SKU. Shots are pinned from the Batch
average and Data bank tabs and stay until removed, so a comparison can mix
batches, SKUs, both mics, and included/idle status alike. One metric is picked
for the whole overlay, because curves have to share a Y axis to be read against
each other.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ...dsp import SIGNED_METRICS, MetricTrace
from ...dsp.quantile import QuantileCurve, fit_quantile_curve
from ...models import MicPosition
from ..compare_series import CompareSeries
from ..controller import WorkflowController
from ..graph import (
    QUANTILE_HIDDEN,
    MetricGraph,
    _color_swatch,
    _MUTED_INK,
    quantile_short_label,
)
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

    Every row but the first (the fixed anchor) has ``+`` / ``−`` / ``R`` slide
    controls that shift its drawn curve in 0.1 ms steps to line up fronts.
    Purely visual: the memoized trace and every reported number are untouched.

    On the LIAeq graph the quantile-curve button is offered; fits run on the
    loading worker, only once the button asks for them.

    Traces are memoized per pinned series for as long as the metric and the
    level weighting hold still. Pinning the ninth shot then re-reads one capture
    rather than nine, which is what makes it reasonable to redraw on every
    change. Changing either control drops the whole cache — it also bounds it,
    since only one metric x weighting generation is ever held. The weighting
    only counts for the signed peak metrics; elsewhere it draws no differently. A mutation
    elsewhere in the app drops it too (see :meth:`invalidate_traces`). Quantile
    fits are memoized alongside.
    """

    #: Pre-selected metric: the impulse curve is what these comparisons are for.
    _DEFAULT_METRIC = "impulse_pa_ms"

    #: The only metric drawn as a cloud with a spread to take a quantile of.
    _QUANTILE_METRIC = "liaeq_100ms_db"

    _EMPTY_MESSAGE = (
        "Pin shots here with the Compare button on a Batch average or Data bank shot row."
    )

    #: Pinned-row columns: shot label, slide controls, Hide, Remove.
    _LABEL_COL, _NUDGE_COL, _HIDE_COL, _REMOVE_COL = range(4)
    _NUDGE_STEP_MS = 0.1
    #: Pixels between the three slide buttons.
    _NUDGE_SPACING = 1
    #: Horizontal padding (px) added to a button's text width.
    _TEXT_BTN_PAD = 12
    _NUDGE_BTN_PAD = 8
    _COMPACT_BTN_QSS = "QPushButton { padding: 1px 3px; }"
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
        #: Per-series slide in :attr:`_NUDGE_STEP_MS` steps; drawn copy only.
        self._offsets: dict[tuple[int, MicPosition], int] = {}
        #: Memoized curves and load failures for the current metric x weighting,
        #: both keyed by ``CompareSeries.key``.
        self._traces: dict[tuple[int, MicPosition], MetricTrace] = {}
        self._errors: dict[tuple[int, MicPosition], str] = {}
        #: Memoized quantile fits, dropped with :attr:`_traces`. None = fit came
        #: back empty; missing = not yet attempted.
        self._quantiles: dict[tuple[int, MicPosition], QuantileCurve | None] = {}
        #: The (metric, absolute) the two dicts above were filled for.
        self._cache_key: tuple[str, bool] | None = None
        #: Bumped on each load so a slow read for a superseded overlay is dropped.
        self._graph_token = 0
        #: ``(x_range, y_range)`` captured by :meth:`_render` and restored at the
        #: end of :meth:`_draw`; None lets the plot autorange.
        self._held_view: tuple[tuple[float, float], tuple[float, float]] | None = None
        #: Set when the held view spans a metric/absolute change: its Y range is
        #: in the old units, so only X is restored and Y autoranges.
        self._refit_y = False

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
        self.tree.setColumnCount(4)
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
        # Label width plus a small pad, tighter than QPushButton's default.
        fm = self.tree.fontMetrics()
        buttons_width = 0
        self._btn_widths: dict[int, int] = {}
        for col, labels in (
            (self._HIDE_COL, ("Hide", "Show")),
            (self._REMOVE_COL, ("Remove",)),
        ):
            width = max(fm.horizontalAdvance(text) for text in labels) + self._TEXT_BTN_PAD
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.Fixed)
            self.tree.setColumnWidth(col, width)
            self._btn_widths[col] = width
            buttons_width += width
        self._nudge_btn_width = (
            max(fm.horizontalAdvance(text) for text in ("+", "−", "R"))
            + self._NUDGE_BTN_PAD
        )
        nudge_col_width = self._nudge_btn_width * 3 + self._NUDGE_SPACING * 2
        header.setSectionResizeMode(self._NUDGE_COL, QtWidgets.QHeaderView.Fixed)
        self.tree.setColumnWidth(self._NUDGE_COL, nudge_col_width)
        buttons_width += nudge_col_width
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

        self.graph = MetricGraph(quantile_curves=True)
        # Re-draw with the new signed/absolute presentation when the toggle
        # flips; the cache is keyed by it, so this reloads rather than redrawing
        # stale curves.
        self.graph.absoluteChanged.connect(self._render)
        self.graph.quantileModeChanged.connect(self._render)
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
        """Unpin one series — the row's own Remove button.

        Removing the anchor promotes the next row, whose slide the anchor
        position ignores; every survivor's slide is rebased onto it so the
        lined-up fronts stay lined up, and it loses its now-dead entry.
        """
        was_anchor = bool(self._series) and self._series[0].key == series.key
        self._series = [s for s in self._series if s.key != series.key]
        self._forget(series)
        if was_anchor and self._series:
            base = self._offsets.pop(self._series[0].key, 0)
            if base:
                self._offsets = {
                    key: steps - base
                    for key, steps in self._offsets.items()
                    if steps != base
                }
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
        """Drop every pinned series; the only action that does not keep the zoom."""
        if not self._series:
            return
        for series in self._series:
            self._forget(series)
        self._series = []
        self._on_pinned_changed(keep_view=False)

    def _forget(self, series: CompareSeries) -> None:
        """Release a removed series' state, so nothing outlives its row."""
        self._traces.pop(series.key, None)
        self._errors.pop(series.key, None)
        self._quantiles.pop(series.key, None)
        self._hidden.discard(series.key)
        self._offsets.pop(series.key, None)

    def _on_pinned_changed(self, keep_view: bool = True) -> None:
        self.main.update_compare_count(len(self._series))
        self._render(keep_view=keep_view)

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
        self._quantiles.clear()
        self._cache_key = None

    def refresh(self) -> None:
        """Redraw only if the curves were invalidated; navigation leaves it alone.

        Nothing here is scoped to a batch or a SKU, so switching tabs has
        nothing to re-read. A dropped cache (:meth:`invalidate_traces`) is the
        one thing that forces a reload, and that keeps the current zoom.
        """
        if self._cache_key is None and self._series:
            self._render()

    # ---- graph ------------------------------------------------------------ #

    def _render(self, *_args, keep_view: bool = True) -> None:
        """Load whatever the current overlay is missing, then draw it.

        ``keep_view`` holds the current zoom across the redraw (captured before
        any ``Loading…`` message blanks the plot). Only Clear All passes False.
        """
        metric_key = self.metric_combo.currentData()
        # Absolute changes only the signed curves; elsewhere it would drop
        # identical traces (and their fits) for nothing.
        cache_key = (
            metric_key,
            metric_key in SIGNED_METRICS and self.graph.absolute_value(),
        )
        if not keep_view:
            self._held_view = None
        elif self.graph.is_showing_curves():
            self._held_view = self.graph.current_view_bounds()
        # else "Loading…" is up: keep the zoom the superseded load was holding.
        self._refit_y = self._held_view is not None and (
            self._refit_y
            or (self._cache_key is not None and self._cache_key != cache_key)
        )
        if cache_key != self._cache_key:
            self._traces.clear()
            self._errors.clear()
            self._quantiles.clear()
            self._cache_key = cache_key
        # Before anything reads the mode: unavailable reads as Hidden.
        self.graph.set_quantile_available(cache_key[0] == self._QUANTILE_METRIC)

        self._graph_token += 1
        token = self._graph_token
        if not self._series:
            self._held_view = None
            self._refit_y = False
            self.graph.show_message(self._EMPTY_MESSAGE)
            self.status_label.setText("")
            self.tree.clear()
            return

        metric_key, absolute = cache_key
        # A hidden series is not drawn, so its capture is not worth reading --
        # unhiding is what asks for it (and finds it already cached if it was
        # hidden after being drawn).
        visible = [s for s in self._series if s.key not in self._hidden]
        pending = [
            s for s in visible if s.key not in self._traces and s.key not in self._errors
        ]
        # Fits for traces already in hand; traces loaded below are fitted after.
        want_fits = self.graph.quantile_mode() != QUANTILE_HIDDEN
        fit_pending = (
            [s for s in visible if s.key in self._traces and s.key not in self._quantiles]
            if want_fits
            else []
        )
        if not pending and not fit_pending:
            self._draw()
            return

        self.graph.show_message("Loading…")
        # Snapshot so the worker reads no dict this thread owns.
        in_hand = {s.key: self._traces[s.key] for s in fit_pending}

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
                        absolute=absolute,
                    )
                except Exception as exc:  # noqa: BLE001 — shown against its row
                    results.append((series, None, str(exc)))
                else:
                    results.append((series, trace, None))
            # Fits take a few hundred ms each; keep them off the UI thread.
            fits = {}
            if want_fits:
                for series in fit_pending:
                    fits[series.key] = self._fit_quantile(
                        in_hand[series.key], series, metric_key
                    )
                for series, trace, _error in results:
                    if trace is not None:
                        fits[series.key] = self._fit_quantile(trace, series, metric_key)
            return results, fits

        def done(payload) -> None:
            if token != self._graph_token:
                return  # a newer overlay superseded this load; drop it
            results, fits = payload
            for series, trace, error in results:
                if trace is None:
                    self._errors[series.key] = error
                else:
                    self._traces[series.key] = trace
            self._quantiles.update(fits)
            self._draw()

        self._run_async(load, done)

    def _fit_quantile(
        self, trace: MetricTrace, series: CompareSeries, metric_key: str
    ) -> QuantileCurve | None:
        """Fit one series' quantile curve on the worker; None on any failure.

        Anchored at the window start (the detected onset for this metric).
        """
        anchor = (
            float(trace.t_ms[trace.window_start_index])
            if trace.window_start_index is not None
            else None
        )
        try:
            return fit_quantile_curve(
                trace.t_ms,
                trace.values,
                anchor_ms=anchor,
                y_label=trace.y_label,
                source_metric=metric_key,
                label=series.label,
            )
        except Exception:  # noqa: BLE001 — an absent aid, never a broken overlay
            return None

    def _draw(self) -> None:
        """Rebuild the pinned rows and the overlay from the memoized traces.

        Colour is taken from a series' position in the pinned order, not from
        its position among the drawn ones: hiding a curve must leave every other
        curve on the colour it already had, or the legend the operator has just
        learned reshuffles under them on every toggle.
        """
        self.tree.clear()
        drawn: list[tuple[str, MetricTrace, tuple[int, int, int]]] = []
        # Parallel to ``drawn``; the graph pairs them by position.
        curves: list[QuantileCurve | None] = []
        hidden = 0
        unavailable = 0
        without_curve = 0
        quantile_on = self.graph.quantile_mode() != QUANTILE_HIDDEN
        for index, series in enumerate(self._series):
            color = MetricGraph.series_color(index)
            trace = self._traces.get(series.key)
            is_anchor = index == 0
            steps = 0 if is_anchor else self._offsets.get(series.key, 0)
            slid = f"(slid {steps * self._NUDGE_STEP_MS:+.1f} ms)" if steps else ""
            item = QtWidgets.QTreeWidgetItem(
                [f"{series.label}   {slid}" if slid else series.label, "", "", ""]
            )
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
                # Slide the drawn copy only; the memoized trace is untouched.
                draw_trace = (
                    replace(trace, t_ms=trace.t_ms + steps * self._NUDGE_STEP_MS)
                    if steps
                    else trace
                )
                # The graph's legend and click readout read times off the slid
                # copy, so carry the slide in the label they print.
                drawn.append(
                    (f"{series.label} {slid}" if slid else series.label, draw_trace, color)
                )
                # Translate the fit rather than refit; the anchor moves with it.
                curve = self._quantiles.get(series.key)
                if curve is not None and steps:
                    offset_ms = steps * self._NUDGE_STEP_MS
                    curve = replace(
                        curve,
                        t_ms=curve.t_ms + offset_ms,
                        anchor_ms=curve.anchor_ms + offset_ms,
                    )
                curves.append(curve)
                if curve is None and quantile_on:
                    without_curve += 1
            self.tree.addTopLevelItem(item)
            self._add_row_buttons(item, series, is_anchor)

        metric_label = self.metric_combo.currentText()
        self.graph.show_traces(
            drawn,
            f"{metric_label} — {len(drawn)} shot(s) overlaid",
            quantile_curves=curves,
        )
        # show_traces autoranges; restore the held zoom. Across a metric change
        # the held Y is in the old units, so keep X and leave Y autoranging.
        if self._held_view is not None:
            x_range, y_range = self._held_view
            self.graph.set_view(x_range, None if self._refit_y else y_range)
            self._held_view = None
        self._refit_y = False
        parts = [f"{len(drawn)} of {len(self._series)} drawn"]
        if hidden:
            parts.append(f"{hidden} hidden")
        if unavailable:
            parts.append(f"{unavailable} unavailable (hover for why)")
        if without_curve:
            parts.append(
                f"{without_curve} without a {quantile_short_label()} curve"
            )
        self.status_label.setText("   —   ".join(parts))

    def _add_row_buttons(
        self,
        item: QtWidgets.QTreeWidgetItem,
        series: CompareSeries,
        is_anchor: bool,
    ) -> None:
        """Put one pinned row's slide controls (if any), Hide, and Remove on it.

        Every handler rebuilds this very tree, which would free the button Qt is
        still emitting the click for, so each is deferred a turn of the event
        loop (see :meth:`_View._defer`). The anchor row gets no slide cell.
        """
        if not is_anchor:
            self.tree.setItemWidget(
                item, self._NUDGE_COL, self._make_nudge_widget(series)
            )

        is_hidden = series.key in self._hidden
        hide_btn = QtWidgets.QPushButton("Show" if is_hidden else "Hide")
        hide_btn.setToolTip(
            "Put this shot's curve back on the graph."
            if is_hidden
            else "Take this shot's curve off the graph, keeping it pinned here."
        )
        hide_btn.clicked.connect(lambda: self._defer(lambda: self._toggle_hidden(series)))
        hide_btn.setStyleSheet(self._COMPACT_BTN_QSS)
        hide_btn.setFixedWidth(self._btn_widths[self._HIDE_COL])
        self.tree.setItemWidget(item, self._HIDE_COL, hide_btn)

        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.setToolTip("Unpin this shot from the Compare tab.")
        remove_btn.clicked.connect(lambda: self._defer(lambda: self._remove(series)))
        remove_btn.setStyleSheet(self._COMPACT_BTN_QSS)
        remove_btn.setFixedWidth(self._btn_widths[self._REMOVE_COL])
        self.tree.setItemWidget(item, self._REMOVE_COL, remove_btn)

    def _make_nudge_widget(self, series: CompareSeries) -> QtWidgets.QWidget:
        """Build one row's ``+`` / ``−`` / ``R`` slide buttons into a cell."""
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self._NUDGE_SPACING)
        step = self._NUDGE_STEP_MS
        for text, tip, action in (
            (
                "+",
                f"Slide this shot {step:.1f} ms later (right) to line it up with the first shot.",
                lambda: self._nudge(series, +1),
            ),
            (
                "−",
                f"Slide this shot {step:.1f} ms earlier (left) to line it up with the first shot.",
                lambda: self._nudge(series, -1),
            ),
            (
                "R",
                "Reset this shot to its true recorded time (no slide).",
                lambda: self._reset_offset(series),
            ),
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tip)
            btn.setStyleSheet(self._COMPACT_BTN_QSS)
            btn.setFixedWidth(self._nudge_btn_width)
            # Bind `action` now rather than capturing the loop variable.
            btn.clicked.connect(lambda _=False, a=action: self._defer(a))
            row.addWidget(btn)
        return container

    def _nudge(self, series: CompareSeries, delta_steps: int) -> None:
        """Slide one shot's curve by ``delta_steps`` visual steps and redraw."""
        self._offsets[series.key] = self._offsets.get(series.key, 0) + delta_steps
        self._redraw_keeping_view()

    def _reset_offset(self, series: CompareSeries) -> None:
        """Return one shot's curve to its true time; a no-op if it never moved."""
        if self._offsets.pop(series.key, 0):
            self._redraw_keeping_view()

    def _redraw_keeping_view(self) -> None:
        """Redraw after a slide from cached traces, holding the current zoom."""
        if not self._series:
            return
        if not self.graph.is_showing_curves():
            # "Loading…" is up: there is no view to read, and a load is holding
            # the zoom. Supersede it so the redraw keeps that zoom and the slide.
            self._render()
            return
        x_range, y_range = self.graph.current_view_bounds()
        # Pinned below; drop any view held by an in-flight render.
        self._held_view = None
        self._refit_y = False
        self._draw()
        self.graph.set_view(x_range, y_range)
