"""The value pills printed on the energy-average (LIAeq) level bars.

Each overlaid series' level bar carries its reported value, so with several
shots in the plot the numbers land close together — often on top of one
another, since shots of one configuration report within a dB or two. Two things
keep them readable:

* each value sits on an opaque pill above everything else in the plot, so a
  point cloud running through it cannot swallow the text;
* :class:`LevelLabels` re-stacks the pills whenever the view moves: a pill that
  would overlap another on screen is lifted clear of it, with a leader line back
  down to the bar it belongs to.

Collision is a *screen* question — two levels 0.3 dB apart overlap when zoomed
out and separate when zoomed in — which is why the layout reruns on every view
change rather than once at draw time.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtGui

#: Pixel padding inside the pill, and the gap kept between the bar and the pill
#: (and between two stacked pills).
_PAD_X = 6.0
_PAD_Y = 2.0
_GAP = 2.0
#: Above the curves, markers and pick marker, so the value is never hidden.
_PILL_Z = 1000


class LevelPill(pg.GraphicsObject):
    """One level value on a rounded pill, right-aligned just above its bar end.

    Positioned in data coordinates at the bar's right end (``x``, ``level``) but
    drawn in screen pixels (``ItemIgnoresTransformations``), so it keeps its size
    at any zoom. :attr:`lift` raises it that many pixels off its natural spot
    (negative lowers it); when it leaves the bar, a leader line runs back to it.
    """

    def __init__(self, text: str, color: tuple[int, int, int], fill: QtGui.QColor,
                 x: float, level: float):
        super().__init__()
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(_PILL_Z)
        self._text = text
        self._color = QtGui.QColor(*color)
        self._fill = QtGui.QColor(fill)
        self._font = QtGui.QFont()
        metrics = QtGui.QFontMetricsF(self._font)
        self._w = metrics.horizontalAdvance(text) + 2 * _PAD_X
        self._h = metrics.height() + 2 * _PAD_Y
        self._lift = 0.0
        self.anchor = QtCore.QPointF(x, level)
        self.setPos(self.anchor)

    def text(self) -> str:
        return self._text

    def color(self) -> QtGui.QColor:
        return self._color

    def size(self) -> tuple[float, float]:
        return self._w, self._h

    @property
    def lift(self) -> float:
        return self._lift

    def set_lift(self, lift: float) -> None:
        if lift != self._lift:
            self.prepareGeometryChange()
            self._lift = lift
            self.update()

    def _pill_rect(self) -> QtCore.QRectF:
        # Local pixels, y down, origin at the bar's right end.
        return QtCore.QRectF(-self._w, -_GAP - self._h - self._lift, self._w, self._h)

    def boundingRect(self) -> QtCore.QRectF:
        # Include the leader to the bar (above or below), plus the pen's half-width.
        rect = self._pill_rect()
        rect.setTop(min(rect.top(), 0.0))
        rect.setBottom(max(rect.bottom(), 0.0))
        return rect.adjusted(-1, -1, 1, 1)

    def paint(self, painter, *_args) -> None:
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self._pill_rect()
        # A negative lift (pushed down from the view's top edge) can leave the
        # pill below its bar; the leader runs to whichever edge faces the bar.
        edge = rect.bottom() if rect.bottom() < 0 else rect.top() if rect.top() > 0 else None
        if edge is not None:
            painter.setPen(pg.mkPen(self._color, width=1))
            painter.drawLine(QtCore.QPointF(-1.0, 0.0), QtCore.QPointF(-1.0, edge))
        painter.setPen(pg.mkPen(self._color, width=1))
        painter.setBrush(pg.mkBrush(self._fill))
        radius = self._h / 2
        painter.drawRoundedRect(rect, radius, radius)
        painter.setFont(self._font)
        painter.drawText(rect, QtCore.Qt.AlignCenter, self._text)


class LevelLabels:
    """The pills on one plot, re-stacked so none overlap on screen.

    The graph adds a pill per level bar through :meth:`add` and drops them all
    through :meth:`clear` whenever it clears the plot. :meth:`add` does not
    stack -- the view is not yet ranged while a redraw is still adding pills --
    so the graph calls :meth:`relayout` once when it is done; after that the
    stacking follows the view box on its own.
    """

    def __init__(self, plot: pg.PlotWidget):
        self._plot = plot
        self._pills: list[LevelPill] = []
        view_box = plot.getViewBox()
        view_box.sigTransformChanged.connect(self.relayout)
        view_box.sigResized.connect(self.relayout)

    def pills(self) -> list[LevelPill]:
        return list(self._pills)

    def clear(self) -> None:
        """Forget the pills; the plot's own ``clear()`` removes the items."""
        self._pills = []

    def add(self, text: str, color: tuple[int, int, int], fill: QtGui.QColor,
            x: float, level: float) -> LevelPill:
        pill = LevelPill(text, color, fill, x, level)
        # Its bounding rect is in pixels, which autorange would read as data
        # units; the anchor lies on the level bar, which is already in range.
        self._plot.addItem(pill, ignoreBounds=True)
        self._pills.append(pill)
        return pill

    def screen_rect(self, pill: LevelPill) -> QtCore.QRectF:
        """The pill's on-screen rect (scene pixels) at its current lift."""
        return self._natural_rect(pill).translated(0.0, -pill.lift)

    def _natural_rect(self, pill: LevelPill) -> QtCore.QRectF:
        at = self._plot.getViewBox().mapViewToScene(pill.anchor)
        w, h = pill.size()
        return QtCore.QRectF(at.x() - w, at.y() - _GAP - h, w, h)

    def relayout(self, *_args) -> None:
        """Lift pills clear of each other, lowest bar first.

        The pill on the lowest bar keeps its natural spot; each one above is
        raised just past any already-placed pill it would overlap. Pills that
        don't share screen columns never move, so a slid shot's value stays put.

        A stack that would rise past the top of the view is then pushed back
        down inside it, so a cluster of levels near the top of the range stays
        readable instead of being clipped.
        """
        natural = {id(p): self._natural_rect(p) for p in self._pills}
        rects: dict[int, QtCore.QRectF] = {}
        placed: list[QtCore.QRectF] = []
        for pill in sorted(self._pills, key=lambda p: -natural[id(p)].bottom()):
            rect = QtCore.QRectF(natural[id(pill)])
            # Each pass only moves up, so it settles within len(placed) passes.
            for _ in range(len(placed) + 1):
                hits = [other for other in placed if rect.intersects(other)]
                if not hits:
                    break
                top = min(other.top() for other in hits)
                rect.moveBottom(top - _GAP)
            rects[id(pill)] = rect
            placed.append(rect)

        # Second pass, top down: clamp to the view's top edge and push anything
        # now overlapping further down. A pill whose own bar is above the view
        # is only held to its natural spot, never dragged into view.
        view_box = self._plot.getViewBox()
        view_top = view_box.sceneBoundingRect().top()
        placed = []
        for pill in sorted(self._pills, key=lambda p: rects[id(p)].top()):
            rect = rects[id(pill)]
            bar_y = view_box.mapViewToScene(pill.anchor).y()
            ceiling = view_top if bar_y >= view_top else natural[id(pill)].top()
            if rect.top() < ceiling:
                rect.moveTop(ceiling)
            # Each pass only moves down, so it settles within len(placed) passes.
            for _ in range(len(placed) + 1):
                hits = [other for other in placed if rect.intersects(other)]
                if not hits:
                    break
                rect.moveTop(max(other.bottom() for other in hits) + _GAP)
            pill.set_lift(natural[id(pill)].top() - rect.top())
            placed.append(rect)
