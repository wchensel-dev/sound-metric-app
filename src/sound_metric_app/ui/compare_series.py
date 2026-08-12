"""One curve pinned to the Compare tab, and the two tabs that can pin it.

The Data bank and the Batch average tabs pin the same kind of thing — one shot's
one mic — to the same Compare tab, so the naming and the button feedback live
here rather than being written twice with room to drift.

Defining :class:`CompareSeries` alongside its builders also fixes an ordering
accident from when all of this shared one module: ``_compare_series`` was written
2500 lines *above* the class it constructs, and only resolved because the call
happens at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from ..models import MicPosition
from .qtwidgets import _flash_button

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from .main_window import MainWindow

#: Text on a shot row's Compare button, and the two confirmations it flashes:
#: pinning happens on another tab, so the button is the only feedback there is.
#: "already" is its own message rather than a silent repeat of "added" — the
#: same shot/mic can only be pinned once, and a click that changed nothing
#: should say so.
_COMPARE_LABEL = "Compare"
_COMPARE_ADDED_LABEL = "Added ✓"
_COMPARE_ALREADY_LABEL = "Pinned ✓"


@dataclass(frozen=True)
class CompareSeries:
    """One curve pinned to the Compare tab: a shot, one of its mics, a name.

    ``(shot_id, position)`` is the identity — the same shot can be pinned once
    per mic, so ML and SE overlay as two series, but neither can be pinned twice.
    ``label`` names the curve in the legend and the pinned list; ``detail`` is
    the longer identification its tooltip carries.
    """

    shot_id: int
    position: MicPosition
    label: str
    detail: str = ""

    @property
    def key(self) -> tuple[int, MicPosition]:
        """What makes this series the same one as another (see the class doc)."""
        return (self.shot_id, self.position)


def _compare_series(
    *,
    shot_id: int,
    position: MicPosition,
    sku: str,
    where: str,
    combo_label: str,
    batch_id: int,
    note: str = "",
) -> CompareSeries:
    """Name one shot/mic as a Compare-tab series.

    Both the Data bank and the Batch average tabs pin the same kind of thing --
    one shot's one mic -- to the same Compare tab, and a curve has to read
    identically no matter which one sent it (``CompareView.add_series`` dedupes
    on ``(shot_id, position)`` across tabs). ``where`` is the caller's
    cluster/shot locator (``"C{cluster}·S{order}"``, or ``"S{order}"`` with no
    cluster); ``note`` is appended to the detail verbatim, letting the Data
    bank tab mark an idle shot without the Batch average tab -- which never
    sees idle shots -- carrying dead code for it.
    """
    return CompareSeries(
        shot_id=shot_id,
        position=position,
        label=f"#{shot_id} · {sku} · {where} · {position.value}",
        detail=f"{combo_label}\nBatch #{batch_id} · {where} · {position.label}{note}",
    )


def _pin_compare_series(
    main: "MainWindow",
    button: QtWidgets.QPushButton,
    series: CompareSeries,
    revert_label: str,
) -> None:
    """Pin one shot/mic to the Compare tab and flash the button to confirm.

    Shared by the Data bank and Batch average tabs' own ``_pin_for_compare``
    methods, which differ only in what the button reverts to afterwards (a mic
    label there, the constant "Compare" here).
    """
    added = main.add_to_compare(series)
    _flash_button(
        button, _COMPARE_ADDED_LABEL if added else _COMPARE_ALREADY_LABEL, revert_label
    )
