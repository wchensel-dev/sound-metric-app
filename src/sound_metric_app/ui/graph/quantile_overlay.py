"""The quantile-curve mode button, and how that curve is drawn.

The estimator is :mod:`sound_metric_app.dsp.quantile`. Modes:

* **Hidden** — nothing fitted or drawn.
* **Overlay** — the fitted curve over its point cloud.
* **Replace** — only the fitted curves, for comparing shape across shots.

The reported level bar and calc-window brackets are drawn in every mode.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtWidgets

from ...dsp.quantile import DEFAULT_QUANTILE

__all__ = [
    "QUANTILE_HIDDEN",
    "QUANTILE_MODES",
    "QUANTILE_OVERLAY",
    "QUANTILE_REPLACE",
    "QuantileModeButton",
    "draw_quantile_curve",
    "quantile_short_label",
]

#: The modes, in cycle order.
QUANTILE_HIDDEN = "hidden"
QUANTILE_OVERLAY = "overlay"
QUANTILE_REPLACE = "replace"
QUANTILE_MODES = (QUANTILE_HIDDEN, QUANTILE_OVERLAY, QUANTILE_REPLACE)

#: Heavier than a trace's 1 px so it reads through a same-colour cloud.
_CURVE_WIDTH = 2


def quantile_short_label(quantile: float = DEFAULT_QUANTILE) -> str:
    """``0.95`` as ``"P95"``."""
    return f"P{round(quantile * 100)}"


class QuantileModeButton:
    """One button cycling Hidden -> Overlay -> Replace; its text shows the mode.

    :meth:`set_available` hides rather than disables it; the mode survives.
    """

    def __init__(self, layout: QtWidgets.QHBoxLayout, on_change, quantile: float = DEFAULT_QUANTILE):
        self._on_change = on_change
        self._quantile = quantile
        self._mode = QUANTILE_HIDDEN
        #: Tracked here: ``isVisible()`` is False until the window is shown.
        self._available = False
        name = quantile_short_label(quantile)
        self._labels = {
            QUANTILE_HIDDEN: f"{name} curve: off",
            QUANTILE_OVERLAY: f"{name} curve: overlay",
            QUANTILE_REPLACE: f"{name} curve: replace",
        }
        self.button = QtWidgets.QPushButton(self._labels[QUANTILE_HIDDEN])
        self.button.setToolTip(
            f"A {name} quantile curve tracing the top edge of the sample cloud.\n"
            "Visual aid only; no reported value is affected.\n"
            "Click to cycle: off / overlay (over the samples) / replace (curve only)."
        )
        # Widest label, so the header does not reflow on a click.
        metrics = self.button.fontMetrics()
        widest = max(metrics.horizontalAdvance(text) for text in self._labels.values())
        self.button.setMinimumWidth(widest + 24)
        self.button.clicked.connect(self._advance)
        layout.addWidget(self.button)

    def mode(self) -> str:
        """The mode currently selected, one of :data:`QUANTILE_MODES`."""
        return self._mode

    def set_available(self, available: bool) -> None:
        """Show or hide the button."""
        self._available = bool(available)
        self.button.setVisible(self._available)

    def is_available(self) -> bool:
        """Whether the button is being offered."""
        return self._available

    def set_mode(self, mode: str) -> None:
        """Select ``mode`` without notifying, for a caller restoring a saved state."""
        if mode not in QUANTILE_MODES:
            raise ValueError(f"Unknown quantile mode: {mode!r}")
        self._mode = mode
        self.button.setText(self._labels[mode])

    def _advance(self) -> None:
        """Cycle to the next mode and report it."""
        index = QUANTILE_MODES.index(self._mode)
        self.set_mode(QUANTILE_MODES[(index + 1) % len(QUANTILE_MODES)])
        self._on_change()


def draw_quantile_curve(plot, curve, color: tuple[int, int, int], name: str | None):
    """Plot a fitted ``curve`` as a solid line, legended as ``name`` if given."""
    return plot.plot(
        curve.t_ms,
        curve.values,
        pen=pg.mkPen(color, width=_CURVE_WIDTH),
        name=name,
    )
