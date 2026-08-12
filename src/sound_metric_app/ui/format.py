"""Display formatting and input parsing for the workflow window.

The value-level layer under the views: how an optional field renders in a cell or
a pre-filled edit box, how a typed string comes back as a number, and how the two
mic-channel combos are read. Nothing here knows about a view, a tree, or the
controller — the functions take values (or a bare ``QComboBox``) and return
values, so a formatting rule is defined once for every view that shows it.

The two parsers are deliberately a pair with different contracts:
:func:`_opt_int` / :func:`_opt_float` raise ``ValueError`` with a message meant
for a dialog, while :func:`_safe_int` swallows the error — see its docstring.
"""

from __future__ import annotations

from datetime import datetime

from PySide6 import QtWidgets

from ..models import MicPosition

#: Placeholder row in a channel combo meaning "no channel tagged for this mic".
_NONE_LABEL = "(none)"

#: Stand-in shown in a channel combo while the capture's channels are being read.
_LOADING_LABEL = "loading…"

#: Shown where a mic position has no channel tagged / a value is missing.
_EMPTY = "—"


def _str_or_empty(value) -> str:
    """Render an optional field for a pre-filled edit box (``None`` -> "")."""
    return "" if value is None else str(value)


def _safe_int(text: str) -> int | None:
    """Parse an int from a live-edited box, treating anything unparseable as ``None``.

    Unlike :func:`_opt_int` this never raises: it backs the as-you-type role
    preview, where a half-typed value is normal and must not pop a dialog.
    """
    try:
        return int(text.strip())
    except ValueError:
        return None


def _unit_of(y_label: str) -> str:
    """Pull the unit out of a trace's y-axis label for the point readout.

    Trace labels carry the unit in trailing parentheses — ``"SPL (dBA)"`` ->
    ``"dBA"``, ``"Pressure (Pa)"`` -> ``"Pa"``. Falls back to the whole label if
    it has no parenthesised unit, so the readout always shows something sensible.
    """
    start = y_label.rfind("(")
    end = y_label.rfind(")")
    if start != -1 and end > start:
        return y_label[start + 1 : end].strip()
    return y_label.strip()


def _format_metric(value) -> str:
    """Render a metric value for a report cell (``None`` -> "—").

    Metric columns are nullable REAL (and the schema-v1 migration blanks
    ``peak_impulse_db`` on pre-existing rows), so a value can arrive as ``None``.
    Show an em-dash for the one missing cell instead of letting ``f"{None:.2f}"``
    raise and abort the whole report render.
    """
    return "—" if value is None else f"{value:.2f}"


def _format_floor(value) -> str:
    """Render a pre-trigger floor for a report cell (``None`` -> "—").

    Signed and to three decimals, unlike :func:`_format_metric`: the sign says
    which way the baseline is displaced, and these values live in the tenths of
    a Pascal, where two decimals would round the differences between captures
    down to a couple of digits. ``None`` is a row stored before the column
    existed — re-marking the shot re-processes the capture and fills it in.
    """
    return "—" if value is None else f"{value:+.3f}"


def _format_floor_pair(floors) -> str:
    """Both mics' pre-trigger floors for one data-bank shot row.

    A shot row stands for the capture, not for a channel, so it shows ML and SE
    together — the same shape the Detail column uses for the channel tags, which
    keeps the two readable down the same row. A mic with no stored floor (never
    tagged, or a row predating the column) shows an em-dash in its half rather
    than dropping out, so the two positions stay in fixed places and a column of
    rows can be scanned straight down.

    ``None`` (no channel row at all for this shot) collapses to a single dash
    instead of a pair of them: there is nothing measured to line up.
    """
    if not floors:
        return _EMPTY
    return "  ".join(
        f"{position.value}:{_format_floor(floors.get(position))}"
        for position in (MicPosition.ML, MicPosition.SE)
    )


def _format_captured_at(captured_at: str | None) -> str:
    """Render a shot's ISO capture timestamp for display (``None`` -> "—").

    Falls back to the raw stored string if it does not parse as ISO-8601, so an
    unexpected format is still shown rather than hidden.
    """
    if not captured_at:
        return "—"
    try:
        return datetime.fromisoformat(captured_at).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return captured_at


def _select_channel(combo: QtWidgets.QComboBox, name: str | None) -> None:
    """Preselect ``name`` in a channel combo, falling back to ``(none)`` at index 0."""
    if name:
        index = combo.findText(name)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
    combo.setCurrentIndex(0)


def _selected_channel(combo: QtWidgets.QComboBox) -> str | None:
    """The channel a combo names, or ``None`` for the placeholder entries."""
    text = combo.currentText()
    return None if text in (_NONE_LABEL, _LOADING_LABEL, "") else text


def _tagged_channel_map(
    parent: QtWidgets.QWidget,
    ml_combo: QtWidgets.QComboBox,
    se_combo: QtWidgets.QComboBox,
) -> dict[str, MicPosition] | None:
    """The tagged channel map, or ``None`` after warning about a bad tagging.

    Returning ``None`` (rather than an empty map) keeps "the user needs to fix
    something" distinct from "nothing tagged" — the caller aborts either way,
    but the warning has already been shown here.
    """
    ml = _selected_channel(ml_combo)
    se = _selected_channel(se_combo)
    if not ml and not se:
        QtWidgets.QMessageBox.warning(
            parent,
            "No mic tagged",
            "Tag at least one channel as Muzzle Left or Shooter's Ear.",
        )
        return None
    if ml and se and ml == se:
        QtWidgets.QMessageBox.warning(
            parent,
            "Same channel",
            "Muzzle Left and Shooter's Ear cannot be the same channel.",
        )
        return None
    channel_map: dict[str, MicPosition] = {}
    if ml:
        channel_map[ml] = MicPosition.ML
    if se:
        channel_map[se] = MicPosition.SE
    return channel_map


def _opt_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a whole number.") from None


def _opt_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number.") from None
