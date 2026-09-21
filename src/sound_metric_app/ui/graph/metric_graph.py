"""The metric graph widget both graphing tabs draw into."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...dsp import SMOOTHING_FAST, SMOOTHING_INSTANT, SMOOTHING_SLOW, MetricTrace
from ..format import _unit_of
from .axis_bounds import AxisBoundsButton, AxisBoundsDialog
from .framing import _ONSET_ZOOM_MS, _ONSET_ZOOM_START_MS, FramingBar
from .palette import _MARK_COLOR, _SERIES_COLORS, series_color
from .readout import PointReadout
from .scale_drag_axis import ScaleDragAxis


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
    :data:`~sound_metric_app.ui.graph.palette._SERIES_COLORS` and a legend keyed
    by the caller's labels appears; the annotations that bracket a single curve
    (calculation window, drawn extent) become the *union* over the series, so the
    framing buttons still land on a span that contains every curve's. A lone
    series keeps exactly the plain, legend-free graph it had.

    Clicking a point on a drawn curve snaps to the nearest sample and shows its
    value (with the trace's unit) and time in a small readout box at the bottom
    right, plus a highlight ring on the picked sample. The box has a Clear button
    that dismisses both.

    Beside the framing buttons, Set Axis Bounds takes typed upper/lower bounds
    for both axes — the frames no snap can express, such as holding two shots to
    one scale or reading a curve against a spec limit. Typed bounds outlive a
    redraw of the same metric, so building an overlay up curve by curve doesn't
    spring the frame back; a typed Y also outlives the framing buttons, which
    otherwise pin Y to the curve's extent.

    The framing buttons, the axis-bounds form, and the readout box are built by
    :class:`~sound_metric_app.ui.graph.framing.FramingBar`,
    :class:`~sound_metric_app.ui.graph.axis_bounds.AxisBoundsButton`, and
    :class:`~sound_metric_app.ui.graph.readout.PointReadout`; the bounds those
    buttons frame to, and the hit-testing behind a pick, stay here — both need
    the drawn series this widget owns.
    """

    #: Emitted when the user picks a different level-weighting from the dropdown.
    smoothingChanged = QtCore.Signal()

    #: Emitted when the user toggles the absolute-value button. Like
    #: :attr:`smoothingChanged`, the owning view re-requests the trace(s) — the
    #: choice is baked into the curve by :func:`~sound_metric_app.dsp.build_metric_trace`,
    #: not applied to already-drawn values.
    absoluteChanged = QtCore.Signal()

    #: Dropdown entries: (label, ``build_metric_trace`` smoothing mode).
    _SMOOTHING_OPTIONS = (
        ("Instantaneous", SMOOTHING_INSTANT),
        ("Fast (125 ms)", SMOOTHING_FAST),
        ("Slow (1 s)", SMOOTHING_SLOW),
    )

    #: The graph's colour cycle, reachable through the widget because that is
    #: where callers meet it (see :meth:`series_color`). Defined in
    #: :mod:`~sound_metric_app.ui.graph.palette`.
    _SERIES_COLORS = _SERIES_COLORS
    _MARK_COLOR = _MARK_COLOR

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
    #: How near (screen pixels) a click must land to a sample to select it.
    _PICK_TOLERANCE_PX = 20.0
    #: Where the onset close-ups start and how wide each one is. Defined in
    #: :mod:`~sound_metric_app.ui.graph.framing`, which builds their buttons;
    #: :meth:`frame_onset_zoom` needs the start to compute the span it frames.
    _ONSET_ZOOM_START_MS = _ONSET_ZOOM_START_MS
    _ONSET_ZOOM_MS = _ONSET_ZOOM_MS

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
        # Absolute-value toggle: off draws the signed peak curve (whose high point
        # is the reported peak); on rectifies it to magnitude. Checkable so its
        # pressed state shows which presentation is live.
        self._absolute_button = QtWidgets.QPushButton("Absolute value")
        self._absolute_button.setCheckable(True)
        self._absolute_button.setToolTip(
            "How the peak Pa / dB / dBA curve is drawn.\n"
            "Off: signed — rarefactions dip below the line, so the curve's peak "
            "is the reported value.\n"
            "On: |value| — every excursion is drawn as a positive magnitude.\n"
            "The reported number and its marker never change."
        )
        self._absolute_button.toggled.connect(lambda *_: self.absoluteChanged.emit())
        header.addWidget(self._absolute_button)
        # Connect-points toggle: joins the instantaneous point cloud with a line
        # so the samples read as a curve. Purely a rendering choice over the same
        # data, so it restyles the drawn curves in place (see
        # :meth:`_apply_connect_style`) rather than re-requesting the trace — the
        # view and any picked point survive the toggle. No effect on the Fast/Slow
        # envelopes, which are already joined lines.
        self._connect_button = QtWidgets.QPushButton("Connect points")
        self._connect_button.setCheckable(True)
        self._connect_button.setToolTip(
            "Join the instantaneous samples with a line.\n"
            "Off: one dot per sample (a point cloud).\n"
            "On: the dots are connected so they read as a curve.\n"
            "The Fast/Slow envelopes are already lines and are unaffected."
        )
        self._connect_button.toggled.connect(self._on_connect_toggled)
        header.addWidget(self._connect_button)
        layout.addLayout(header)

        # Both axes carry the tick numbers as scale handles: dragging them
        # stretches or squeezes that axis (see :class:`ScaleDragAxis`), leaving
        # the plot body as the sole place a left-drag pans.
        self._plot = pg.PlotWidget(
            axisItems={
                "bottom": ScaleDragAxis(orientation="bottom"),
                "left": ScaleDragAxis(orientation="left"),
            }
        )
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
        self._framing = FramingBar(
            toolbar,
            on_auto=self.auto_frame,
            on_window=self.frame_calc_window,
            on_onset=self.frame_onset_zoom,
        )
        self._axis_bounds = AxisBoundsButton(toolbar, on_click=self.edit_axis_bounds)
        toolbar.addStretch(1)
        self._readout = PointReadout(toolbar, self._plot, on_clear=self.clear_readout)
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
        #: (y_min, y_max) the operator typed into the axis-bounds dialog, or None
        #: when Y is left to the drawn data. Overrides :attr:`_y_bounds` wherever
        #: that is spent, so a hand-set Y survives the framing buttons: someone
        #: who pinned Y to read two shots against the same scale means it to
        #: stay pinned while they zoom around in X.
        self._manual_y_bounds: tuple[float, float] | None = None
        #: (x_min, x_max) as typed, or None when X is left to the framing
        #: buttons. Unlike its Y counterpart this is *superseded* by them: a
        #: framing click is a fresh X decision, and re-imposing a typed X after
        #: it would make those buttons do nothing.
        self._manual_x_bounds: tuple[float, float] | None = None
        #: The ``y_label`` the typed bounds above were entered against, or None
        #: when nothing is pinned. It is what makes them survive a redraw: a
        #: hide/unhide, or a shot added to or dropped from an overlay, re-renders
        #: the *same* metric, and a frame the operator set by hand must not
        #: spring back on every toggle. A different y_label is a different
        #: quantity, whose numbers say nothing about the scale chosen for this
        #: one, so the bounds are dropped there.
        self._manual_bounds_metric: str | None = None
        #: The (label, trace, colour) series currently drawn, in the order they
        #: were handed over — which is also the legend's order. Kept so a plot
        #: click can find the sample it landed on. Empty whenever the plot shows
        #: a message rather than curves.
        self._series: list[tuple[str, MetricTrace, tuple[int, int, int]]] = []
        #: Whether the Connect-points toggle is on. Set here so :meth:`_draw_curve`
        #: reads a defined value on the very first render.
        self._connect_points = False
        #: The drawn curve items, one per series, as ``(item, colour, is_cloud)``.
        #: ``is_cloud`` is True for an instantaneous point cloud — the only kind
        #: the Connect-points toggle restyles; an envelope stays a line. Kept so
        #: the toggle can add or drop the connecting line without a full redraw.
        self._curve_items: list[tuple[object, tuple[int, int, int], bool]] = []

        # A click anywhere on the plot scene tries to select the nearest sample.
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._apply_theme()
        self.show_message("Click a metric cell on a shot row to graph it.")

    def current_smoothing(self) -> str:
        """The ``build_metric_trace`` smoothing mode currently selected."""
        return self._smoothing_combo.currentData()

    def absolute_value(self) -> bool:
        """Whether the absolute-value (magnitude) presentation is toggled on.

        Passed straight to ``build_metric_trace``'s ``absolute`` argument; see
        :attr:`absoluteChanged`.
        """
        return self._absolute_button.isChecked()

    def connect_points(self) -> bool:
        """Whether the Connect-points toggle is on (point clouds drawn joined)."""
        return self._connect_points

    def _on_connect_toggled(self, checked: bool) -> None:
        """Add or drop the connecting line on the drawn point clouds.

        Restyles the existing curve items in place: no data changes, so the
        view range and any picked-point readout are left exactly as they were.
        """
        self._connect_points = bool(checked)
        self._apply_connect_style()

    def _apply_connect_style(self) -> None:
        """Give each point-cloud curve a connecting pen, or none, per the toggle.

        Envelope curves (``is_cloud`` False) are already joined lines and keep
        their pen untouched, so switching to Fast/Slow and back is unaffected.
        """
        for item, color, is_cloud in self._curve_items:
            if is_cloud:
                item.setPen(pg.mkPen(color, width=1) if self._connect_points else None)

    @classmethod
    def series_color(cls, index: int) -> tuple[int, int, int]:
        """The RGB an overlaid series at ``index`` is drawn in, cycling.

        Public so a view listing the same series (the Compare tab's swatches)
        keys off the graph's palette rather than keeping a second copy of it.
        """
        return series_color(index)

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
        """Clear the plot and show a short prompt in place of a graph.

        Typed axis bounds deliberately outlive this: both views put "Loading…"
        up mid-redraw, so discarding them here would undo them on every
        re-render. :meth:`show_traces` drops them if what comes back is a
        different metric.
        """
        self._plot.clear()
        self._legend.clear()
        self._legend.setVisible(False)
        self._title_label.setText(text)
        self._plot.setLabel("left", "")
        self._x_bounds = None
        self._window_x_bounds = None
        self._y_bounds = None
        self._series = []
        self._curve_items = []
        self.clear_readout()
        self._framing.set_curve_enabled(False)
        self._framing.set_window_enabled(False)
        self._axis_bounds.set_curve_enabled(False)

    def _frame_x(self, span: tuple[float, float] | None, padding: float = 0) -> None:
        """Snap the X range to ``span``, or do nothing when it is None.

        The one place the framing buttons meet. Y is pinned to the whole curve's
        vertical extent — the same range Auto Frame lands on — for *every* span,
        rather than autoranging to the framed slice. Letting Y refit meant a
        narrow close-up rescaled the axis to that slice's few dB, so the ticks
        collapsed to single-digit steps and the curve redrew at a wildly
        different height than the wider views: the zoom levels stopped being
        comparable. A fixed Y makes zooming in a purely horizontal move.

        A Y range typed into the axis-bounds dialog wins over the curve's own
        extent, so framing stays a horizontal move against *that* scale too.
        """
        if span is None:
            return
        self._plot.setXRange(span[0], span[1], padding=padding)
        # A framing click is a fresh X decision and drops any typed one, or the
        # button would appear to do nothing the next time the plot redrew. Y is
        # untouched, by design: see :attr:`_manual_x_bounds`.
        self._manual_x_bounds = None
        if self._manual_y_bounds is not None:
            # Exactly as typed: pyqtgraph's default y padding would drift a
            # hand-set scale a few percent on every frame click, and a bound the
            # operator entered to read against is not ours to loosen.
            self._plot.setYRange(*self._manual_y_bounds, padding=0)
        elif self._y_bounds is None:
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

    def current_view_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The ``(x_range, y_range)`` currently in view.

        What the axis-bounds form opens on, so editing starts from what the
        operator is actually looking at rather than from the trace's extent —
        those differ the moment anyone pans or scroll-zooms.
        """
        x_range, y_range = self._plot.getViewBox().viewRange()
        return (float(x_range[0]), float(x_range[1])), (
            float(y_range[0]),
            float(y_range[1]),
        )

    def edit_axis_bounds(self) -> None:
        """Ask for X/Y bounds and frame the plot to them. A no-op when cancelled.

        The escape hatch from the framing buttons, which can only snap to spans
        the trace defines. A no-op when nothing is drawn — there is no view
        range to edit and nothing for the numbers to frame.
        """
        if not self._series:
            return
        x_range, y_range = self.current_view_bounds()
        dialog = AxisBoundsDialog(
            x_range=x_range,
            y_range=y_range,
            y_label=_unit_of(self._series[0][1].y_label),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.set_axis_bounds(*dialog.values())

    def set_axis_bounds(
        self,
        x_range: tuple[float, float] | None,
        y_range: tuple[float, float] | None,
    ) -> None:
        """Pin the axes to ``x_range`` / ``y_range``; None autoranges that axis.

        The bounds are recorded as well as applied: they have to outlive the
        framing buttons (Y) and a redraw of the same metric (both), so a
        hide/unhide or an added shot doesn't spring the frame back.
        """
        self._manual_x_bounds = x_range
        self._manual_y_bounds = y_range
        self._manual_bounds_metric = (
            self._series[0][1].y_label
            if self._series and (x_range is not None or y_range is not None)
            else None
        )
        for axis, bounds in (("x", x_range), ("y", y_range)):
            if bounds is None:
                self._plot.enableAutoRange(axis=axis)
        self._apply_manual_bounds()

    def _apply_manual_bounds(self) -> None:
        """Frame the plot at whichever axes are currently pinned by hand.

        No padding: an operator who typed 0–40 ms wants the frame to *be* 0–40
        ms, not the padded neighbourhood of it the framing buttons draw.
        """
        if self._manual_x_bounds is not None:
            self._plot.setXRange(*self._manual_x_bounds, padding=0)
        if self._manual_y_bounds is not None:
            self._plot.setYRange(*self._manual_y_bounds, padding=0)

    def clear_axis_bounds(self) -> None:
        """Forget the typed bounds, handing both axes back to automatic scaling.

        Only forgets them — the caller decides what the plot shows next, which
        is either a fresh autorange (a redraw) or nothing at all (a message).
        """
        self._manual_x_bounds = None
        self._manual_y_bounds = None
        self._manual_bounds_metric = None

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
        self._curve_items = []
        self.clear_readout()
        first = series[0][1]
        # Typed bounds are kept across a redraw of the same metric -- which is
        # what a hide/unhide or an added shot is -- and dropped when the
        # quantity on the axes changes. See :attr:`_manual_bounds_metric`.
        if self._manual_bounds_metric != first.y_label:
            self.clear_axis_bounds()
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
            item = self._draw_curve(trace, color, label if multiple else None)
            self._curve_items.append((item, color, not trace.connected))
            if trace.peak_index is not None:
                self._plot.addItem(
                    pg.InfiniteLine(
                        pos=float(trace.t_ms[trace.peak_index]), angle=90,
                        pen=pg.mkPen(mark_color, width=1),
                    )
                )
            if trace.level is not None:
                self._draw_level_line(trace, mark_color)
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
            # Auto Frame has the curve's extent, and the onset close-ups frame
            # fixed times, so a drawn curve is their only precondition -- they
            # need no calculation window at all.
            self._framing.set_curve_enabled(True)
        else:
            self._x_bounds = None
            self._y_bounds = None
            self._framing.set_curve_enabled(False)
        # Typed bounds share the framing buttons' one precondition -- something
        # drawn to frame, and therefore a view range to prefill the form with.
        self._axis_bounds.set_curve_enabled(bool(finite_x))
        self._plot.enableAutoRange()
        # Autorange first, then re-impose whatever was pinned: an axis the
        # operator left automatic refits to the new curves, the ones they typed
        # come back exactly where they put them.
        self._apply_manual_bounds()

    def _draw_curve(self, trace, color: tuple[int, int, int], name: str | None):
        """Plot one trace's samples in ``color``, legending it as ``name`` if given.

        Returns the created plot item so :meth:`_apply_connect_style` can restyle
        a point cloud in place when the Connect-points toggle flips.
        """
        if trace.connected:
            # Time-weighted envelope: a joined line reads as the continuous level
            # a meter shows. NaN samples break the line into gaps.
            return self._plot.plot(
                trace.t_ms, trace.values, pen=pg.mkPen(color, width=1), name=name
            )
        # One dot per sample. The Connect-points toggle adds a connecting line
        # (pen); off it is a bare point cloud (pen=None). NaN samples (silent
        # Impulse tail) simply don't plot a point, and break the line if drawn.
        # pxMode keeps dots a fixed screen size regardless of zoom.
        pen = pg.mkPen(color, width=1) if self._connect_points else None
        return self._plot.plot(
            trace.t_ms,
            trace.values,
            pen=pen,
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
            self._framing.set_window_enabled(True)
        else:
            self._window_x_bounds = None
            self._framing.set_window_enabled(False)

    def _draw_level_line(self, trace, color: tuple[int, int, int]) -> None:
        """Draw the energy-average marker: a dashed bar over the calc window.

        Unlike a peak metric's vertical, the energy-average metric (LIAeq) has no
        single sample to point at -- it reports one level integrated over its
        100 ms window. So the marker is a horizontal dashed segment spanning
        *exactly* that window rather than an infinite line across the whole plot,
        which ties the reported number to the time-bounded frame it was computed
        over; the value is printed in the trace's unit just above the bar's right
        end, so the number is readable without picking a point.

        Falls back to a full-width :class:`~pyqtgraph.InfiniteLine` when the trace
        carries no window to bracket (either edge outside the capture) -- there is
        then no span to draw the segment over, but the level is still worth showing.
        """
        pen = pg.mkPen(color, width=1, style=QtCore.Qt.DashLine)
        if trace.window_start_index is None or trace.window_end_index is None:
            self._plot.addItem(pg.InfiniteLine(pos=trace.level, angle=0, pen=pen))
            return
        x0 = float(trace.t_ms[trace.window_start_index])
        x1 = float(trace.t_ms[trace.window_end_index])
        self._plot.plot([x0, x1], [trace.level, trace.level], pen=pen)
        # Anchored bottom-right at the segment's right end, so the text sits just
        # above the bar and ends flush with the window close rather than centring
        # on it. Same unit the point readout uses (dBA for LIAeq).
        unit = _unit_of(trace.y_label)
        unit_suffix = f" {unit}" if unit else ""
        label = pg.TextItem(f"{trace.level:.2f}{unit_suffix}", color=color, anchor=(1, 1))
        label.setPos(x1, trace.level)
        self._plot.addItem(label)

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
        unit = _unit_of(trace.y_label)
        unit_suffix = f" {unit}" if unit else ""
        prefix = f"{label}:  " if label else ""
        self._readout.show(x, value, f"{prefix}{value:.3f}{unit_suffix}  @ {x:.2f} ms")

    def clear_readout(self) -> None:
        """Remove the picked-point marker and hide the readout box."""
        self._readout.clear()
