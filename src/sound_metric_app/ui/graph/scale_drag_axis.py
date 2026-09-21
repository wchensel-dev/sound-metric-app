"""An axis whose tick numbers scale the view instead of panning it."""

from __future__ import annotations

import pyqtgraph as pg
from pyqtgraph import Point
from PySide6 import QtCore


class ScaleDragAxis(pg.AxisItem):
    """A :class:`~pyqtgraph.AxisItem` that *scales* its own axis on a drag.

    pyqtgraph's stock axis forwards a left-drag on its tick numbers to the
    linked ViewBox as a *pan* restricted to that one axis -- grabbing the
    numbers slides the view along that axis, the same motion as grabbing the
    plot body but locked to one direction. Here the same gesture *scales* that
    axis instead: grab the numbers and stretch or squeeze the scale. Panning is
    unchanged -- it stays a left-drag on the body of the plot -- so the two
    motions no longer overlap.

    The value under the cursor when the drag begins is held fixed, so the axis
    expands and contracts around the tick the operator grabbed. The zoom rate
    matches the ViewBox's own right-drag scaling, so an axis drag and a
    right-drag on the body zoom at the same pace and in the same direction
    (drag toward higher numbers to zoom in). Right-drag and scroll-zoom are
    left to pyqtgraph untouched; only the left-drag on the axis changes meaning.
    """

    #: Scale change per screen pixel of drag, matching ViewBox's right-drag
    #: scaling so the two gestures zoom at the same rate.
    _SCALE_PER_PX = 0.02

    def mouseDragEvent(self, event):
        lv = self.linkedView()
        if lv is None:
            return
        # A drag that began inside the plot body is the ViewBox's pan; only one
        # that began on the axis itself is ours to reinterpret as a scale.
        if lv.sceneBoundingRect().contains(event.buttonDownScenePos()):
            event.ignore()
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            # Leave the right/middle-drag gestures to the stock axis behaviour.
            return super().mouseDragEvent(event)
        event.accept()

        is_y = self.orientation in ("left", "right")
        # Per-move screen delta on the axis's own direction. Screen y grows
        # downward, so a raw dy already reads "drag down = zoom out"; x is
        # negated to match ViewBox's right-drag, where dragging toward larger
        # numbers (rightward / upward) zooms in.
        delta = event.screenPos() - event.lastScreenPos()
        step = delta.y() if is_y else -delta.x()
        factor = (self._SCALE_PER_PX + 1.0) ** step

        # Hold the value under the initial grab fixed while the axis rescales.
        # The off-axis coordinate of this point is irrelevant -- only the axis
        # being scaled reads it -- so mapping the axis-side scene point straight
        # into view coordinates is enough.
        center = Point(lv.mapSceneToView(event.buttonDownScenePos()))
        lv._resetTarget()
        if is_y:
            lv.scaleBy(y=factor, center=center)
        else:
            lv.scaleBy(x=factor, center=center)
        lv.sigRangeChangedManually.emit(lv.state["mouseEnabled"])
