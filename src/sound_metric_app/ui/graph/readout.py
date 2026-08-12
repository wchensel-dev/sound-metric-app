"""The metric graph's click-to-read box and the marker that goes with it.

Clicking a drawn curve snaps to the nearest sample; this is what shows the
result — a small bordered label at the bottom right, a Clear button that dismisses
it, and a highlight ring on the picked sample. The three are one unit because
they appear and disappear together: there is no state where the ring is drawn and
the box is not.

*Which* sample was clicked is the graph's problem, not this module's — hit-testing
needs the drawn series and the plot's viewbox. This owns only the display of a
pick that has already been made.

Deliberately not a ``QWidget``, for the same reason as
:class:`~sound_metric_app.ui.graph.framing.FramingBar`: it builds into the
caller's toolbar layout so the widget tree, and therefore the spacing, is
unchanged.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtWidgets

#: Highlight ring drawn on the sample the user clicks to read out.
_PICK_BRUSH = pg.mkBrush(240, 200, 0)
_PICK_PEN = pg.mkPen((30, 30, 30), width=1)


class PointReadout:
    """Build the readout box into ``toolbar``, marking picks on ``plot``.

    Starts hidden: the label and its Clear button only appear once a point has
    actually been picked. ``on_clear`` is invoked by the Clear button — the
    graph routes it back to :meth:`clear` so both paths run the same teardown.
    """

    def __init__(self, toolbar: QtWidgets.QHBoxLayout, plot: pg.PlotWidget, *, on_clear):
        self._plot = plot
        #: Scatter item marking the picked sample, or None when nothing is picked.
        self._pick_marker: pg.ScatterPlotItem | None = None

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
        self._readout_clear_btn.clicked.connect(on_clear)
        toolbar.addWidget(self._readout_label)
        toolbar.addWidget(self._readout_clear_btn)

    def show(self, x: float, y: float, text: str) -> None:
        """Ring the sample at ``(x, y)`` and show ``text`` in the box."""
        if self._pick_marker is not None:
            self._plot.removeItem(self._pick_marker)
        self._pick_marker = pg.ScatterPlotItem(
            [x], [y], size=11, brush=_PICK_BRUSH, pen=_PICK_PEN, pxMode=True
        )
        self._plot.addItem(self._pick_marker)

        self._readout_label.setText(text)
        self._readout_label.setVisible(True)
        self._readout_clear_btn.setVisible(True)

    def clear(self) -> None:
        """Remove the picked-point marker and hide the readout box."""
        if self._pick_marker is not None:
            self._plot.removeItem(self._pick_marker)
            self._pick_marker = None
        self._readout_label.clear()
        self._readout_label.setVisible(False)
        self._readout_clear_btn.setVisible(False)
