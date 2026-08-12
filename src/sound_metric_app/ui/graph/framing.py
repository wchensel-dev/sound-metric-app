"""The metric graph's framing buttons — everything that snaps the X range.

Four buttons over one plot: Auto Frame (the drawn curve's extent), Frame Calc
Window (the samples the reported number came from), and one onset close-up per
span in :data:`_ONSET_ZOOM_MS`. This module owns the buttons, their tooltips, and
which of them are currently usable; *what* each one frames stays with the graph,
which is where the bounds live.

Deliberately not a ``QWidget``: it builds its buttons straight into the caller's
toolbar layout, the way ``_View._add_sku_filter`` builds the SKU combo into a
caller's row. Wrapping them in a nested widget would add a layout level between
the buttons and the toolbar they share with the point readout, and change the
spacing the operator sees.
"""

from __future__ import annotations

from PySide6 import QtWidgets

#: Left edge (ms) shared by every onset close-up: a fixed point on the
#: capture's own time axis, just past the trigger, rather than wherever
#: onset detection happened to fire. Fixed is the point -- the same slice of
#: every shot lands in the same place, so close-ups compare shot to shot.
_ONSET_ZOOM_START_MS = 10.5

#: Widths (ms) of the onset close-ups -- ``_ONSET_ZOOM_START_MS`` to this far
#: past it -- one button each, in the order they appear on the toolbar. 10 ms
#: matches the Peak-10 ms-Leq integration length, so that close-up frames the
#: same slice that metric reports on; 5 ms halves it for the rise itself.
_ONSET_ZOOM_MS = (5.0, 10.0)


class FramingBar:
    """Build the framing buttons into ``toolbar`` and track what they can do.

    ``on_auto`` / ``on_window`` take no arguments; ``on_onset`` is called with
    the span (ms) of the button that was clicked. All three buttons start
    disabled, since nothing is drawn yet.
    """

    def __init__(self, toolbar: QtWidgets.QHBoxLayout, *, on_auto, on_window, on_onset):
        self._auto_frame_btn = QtWidgets.QPushButton("Auto Frame")
        self._auto_frame_btn.setToolTip(
            "Snap the X range to the drawn curve's extent\n"
            "(first to last sample carrying a value)."
        )
        self._auto_frame_btn.setEnabled(False)
        self._auto_frame_btn.clicked.connect(on_auto)
        toolbar.addWidget(self._auto_frame_btn)

        self._frame_window_btn = QtWidgets.QPushButton("Frame Calc Window")
        self._frame_window_btn.setToolTip(
            "Snap the X range to the calculation window\n"
            "(the samples this metric's number came from)."
        )
        self._frame_window_btn.setEnabled(False)
        self._frame_window_btn.clicked.connect(on_window)
        toolbar.addWidget(self._frame_window_btn)

        #: One onset close-up button per span in ``_ONSET_ZOOM_MS``, all enabled
        #: and disabled together (they share the one precondition: a drawn curve).
        self._frame_onset_btns: list[QtWidgets.QPushButton] = []
        for span_ms in _ONSET_ZOOM_MS:
            btn = QtWidgets.QPushButton(f"+{span_ms:g} ms")
            btn.setToolTip(
                f"Snap the X range to {_ONSET_ZOOM_START_MS:g}"
                f"–{_ONSET_ZOOM_START_MS + span_ms:g} ms\n"
                "(the onset transient, in detail)."
            )
            btn.setEnabled(False)
            # Default-arg binding, not a bare closure over the loop variable,
            # which would leave every button framing the last span. ``*_`` eats
            # the ``checked`` flag ``clicked`` sends.
            btn.clicked.connect(lambda *_, ms=span_ms: on_onset(ms))
            toolbar.addWidget(btn)
            self._frame_onset_btns.append(btn)

    def set_curve_enabled(self, enabled: bool) -> None:
        """Arm (or disarm) the buttons whose only precondition is a drawn curve.

        Auto Frame needs the curve's extent; the onset close-ups frame fixed
        times, so a drawn curve is all they need either -- they want no
        calculation window at all.
        """
        self._auto_frame_btn.setEnabled(enabled)
        for btn in self._frame_onset_btns:
            btn.setEnabled(enabled)

    def set_window_enabled(self, enabled: bool) -> None:
        """Arm (or disarm) Frame Calc Window, which needs both window edges."""
        self._frame_window_btn.setEnabled(enabled)
