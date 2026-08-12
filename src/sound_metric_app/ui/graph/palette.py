"""The colours the metric graph and its legends are drawn in.

Held apart from the widget because two things spend the same palette: the graph
assigns a colour per overlaid curve, and the Compare tab draws a matching swatch
beside each pinned row. A second copy of the cycle in the view would drift the
moment either side gained a colour, and the legend would stop matching the graph.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui

#: Colour cycle for overlaid series, in the order they are handed over.
#: Entry 0 is the blue a single-trace graph has always drawn, so the Batch
#: average tab is unchanged by there being a cycle at all. Chosen to stay
#: distinguishable against both the light and dark plot backgrounds.
_SERIES_COLORS = (
    (66, 135, 245),   # blue
    (214, 90, 70),    # red
    (46, 160, 90),    # green
    (170, 90, 200),   # purple
    (230, 160, 30),   # amber
    (40, 180, 190),   # teal
    (200, 70, 140),   # magenta
    (120, 130, 145),  # slate
)

#: Colour of the peak marker / level line on a *single*-series graph, where
#: the annotation reads best set apart from the curve. Overlaid series take
#: their own colour for these instead — with several curves in the plot, the
#: shared colour is the only thing tying a peak line to the curve it marks.
_MARK_COLOR = (214, 90, 70)

#: Mid-grey for a Compare row that is not on the graph: the text of a hidden
#: one (set aside, but still pinned) and the outline of the empty swatch an
#: unloadable one gets. A *hidden* row keeps its colour swatch at full strength
#: — that is what its curve comes back as.
_MUTED_INK = QtGui.QColor(140, 140, 148)


def series_color(index: int) -> tuple[int, int, int]:
    """The RGB an overlaid series at ``index`` is drawn in, cycling."""
    return _SERIES_COLORS[index % len(_SERIES_COLORS)]


def _color_swatch(color: tuple[int, int, int] | None) -> QtGui.QIcon:
    """A small filled square in ``color`` — the Compare list's legend key.

    ``None`` (a series that could not be loaded) yields a hollow outline, so an
    undrawn row is visibly not claiming one of the graph's colours.
    """
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtGui.QColor(*color) if color else QtCore.Qt.transparent)
    if color is None:
        painter = QtGui.QPainter(pixmap)
        painter.setPen(_MUTED_INK)
        painter.drawRect(0, 0, 11, 11)
        painter.end()
    return QtGui.QIcon(pixmap)
