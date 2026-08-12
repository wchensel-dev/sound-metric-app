"""The metric graph's manual axis bounds — typed numbers instead of a snap.

The framing buttons in :mod:`.framing` all snap to spans *derived from the
trace* (its drawn extent, its calculation window, a fixed onset slice). This is
the escape hatch for everything they cannot express: an operator comparing two
shots side by side, or reading a plot against a spec limit, needs to pin the
axes to numbers of their own choosing and have them stay put.

Two pieces, in one module because neither is useful without the other:

* :class:`AxisBoundsButton` — the toolbar button, built into the caller's layout
  the way :class:`~sound_metric_app.ui.graph.framing.FramingBar` builds its own
  (and for the same reason: no extra widget level between the buttons and the
  toolbar they share).
* :class:`AxisBoundsDialog` — the modal form behind it. A pure form, like the
  dialogs in :mod:`sound_metric_app.ui.dialogs`: it validates what was typed and
  exposes it through :meth:`~AxisBoundsDialog.values`, applying nothing itself.

*Applying* the bounds stays with the graph, which owns the plot and the framing
state a manual Y range has to survive.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from ..format import _opt_float

#: What :meth:`AxisBoundsDialog.values` returns per axis: an explicit
#: ``(low, high)`` pair, or None for "let this axis autorange".
AxisRange = tuple[float, float] | None


class AxisBoundsButton:
    """Build the Set Axis Bounds button into ``toolbar``.

    ``on_click`` takes no arguments. Starts disabled: with nothing drawn there
    is no view range to prefill the form from, and no curve for the typed
    numbers to frame.
    """

    def __init__(self, toolbar: QtWidgets.QHBoxLayout, *, on_click):
        self._button = QtWidgets.QPushButton("Set Axis Bounds…")
        self._button.setToolTip(
            "Type exact upper/lower bounds for the X and Y axes\n"
            "(instead of snapping to the trace, as the buttons left of this do)."
        )
        self._button.setEnabled(False)
        self._button.clicked.connect(lambda *_: on_click())
        toolbar.addWidget(self._button)

    def set_curve_enabled(self, enabled: bool) -> None:
        """Arm (or disarm) the button; its precondition is a drawn curve."""
        self._button.setEnabled(enabled)


class AxisBoundsDialog(QtWidgets.QDialog):
    """Type the X and Y bounds to frame the graph at.

    Prefilled with the range currently in view, so opening it and saving is a
    no-op and an operator can nudge one edge without re-deriving the other
    three. Each axis is a pair: fill both boxes to pin that axis, or clear
    *both* to hand it back to autorange — which is how a single axis is
    released without also giving up the one the user meant to keep. Reset to
    Auto empties all four, releasing both at once.

    Purely a form. It validates (numbers, both-or-neither, low < high) and
    exposes the result through :meth:`values`; the caller applies it.
    """

    def __init__(
        self,
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        y_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Set axis bounds")
        self._values: tuple[AxisRange, AxisRange] | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Bounds to frame the graph at. Clear both boxes of an axis to\n"
                "let it scale automatically."
            )
        )
        form = QtWidgets.QFormLayout()
        self.x_min_edit = _bounds_edit(x_range[0])
        self.x_max_edit = _bounds_edit(x_range[1])
        form.addRow("X min (ms):", self.x_min_edit)
        form.addRow("X max (ms):", self.x_max_edit)
        # The Y axis carries whatever unit the drawn metric is in (dB, Pa·ms,
        # …), so its rows are labelled from the plot rather than hard-coded.
        y_unit = f" ({y_label})" if y_label else ""
        self.y_min_edit = _bounds_edit(y_range[0])
        self.y_max_edit = _bounds_edit(y_range[1])
        form.addRow(f"Y min{y_unit}:", self.y_min_edit)
        form.addRow(f"Y max{y_unit}:", self.y_max_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Reset
            | QtWidgets.QDialogButtonBox.Cancel
        )
        apply_btn = buttons.button(QtWidgets.QDialogButtonBox.Apply)
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_accept)
        buttons.button(QtWidgets.QDialogButtonBox.Reset).setText("Reset to Auto")
        buttons.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self._on_reset)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_reset(self) -> None:
        """Empty every box — "both axes automatic", one click.

        Deliberately not an immediate accept: it leaves the user in the form,
        so the common "release Y but keep the X I typed" is one edit away
        rather than a second trip through the dialog.
        """
        for edit in (self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit):
            edit.clear()

    def _axis_range(self, name: str, low_edit, high_edit) -> AxisRange | bool:
        """Validate one axis' pair of boxes, or False when they don't make sense.

        False rather than None because None is a real answer here: it is the
        empty pair, meaning "autorange this axis".
        """
        try:
            low = _opt_float(low_edit.text())
            high = _opt_float(high_edit.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
            return False
        if low is None and high is None:
            return None
        if low is None or high is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete bounds",
                f"Give both a min and a max for {name}, or clear both to let it "
                "scale automatically.",
            )
            return False
        if not low < high:
            QtWidgets.QMessageBox.warning(
                self, "Invalid bounds", f"{name} min must be below {name} max."
            )
            return False
        return (low, high)

    def _on_accept(self) -> None:
        x_range = self._axis_range("X", self.x_min_edit, self.x_max_edit)
        if x_range is False:
            return
        y_range = self._axis_range("Y", self.y_min_edit, self.y_max_edit)
        if y_range is False:
            return
        self._values = (x_range, y_range)
        self.accept()

    def values(self) -> tuple[AxisRange, AxisRange]:
        """The validated ``(x_range, y_range)``. Valid only after Apply."""
        assert self._values is not None, "values() called before an accepted Apply"
        return self._values


def _bounds_edit(value: float) -> QtWidgets.QLineEdit:
    """A bounds box prefilled with ``value``, rounded to something typeable.

    A view range is a float with a long tail (``2.9999999999996``); echoing that
    back as the starting point would make every edit start with a cleanup.
    """
    edit = QtWidgets.QLineEdit(f"{value:.4g}")
    edit.setPlaceholderText("auto")
    return edit
