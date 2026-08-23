"""SilencerScout paste strings (format ``SSR1``) for the batch-average slots.

The Scout Report editor on the SilencerScout admin renders one row per test with
a small paste box under it: paste a string in and the numbers land in that row's
cells. There is no endpoint, no upload and no auth — the string is a transport
for values that already have columns over there, so all this module has to do is
render one line per averaged slot.

The format is::

    SSR1|<shot type>|<MIC>=<LIAeq100ms dB>,<Peak dB>,<Peak dBA>,<Impulse Pa·ms>

Four values, positional and comma-separated, and a labelled mic position. A
value may be left empty (``132.4,,151.0,3.4``) to mean "not measured"; their
end leaves that cell as it was rather than clearing it.

Three things our hierarchy carries do **not** travel in the string:

* **Which test the numbers belong to.** That is decided by the box the operator
  pastes into, so SKU / platform / ammo and the session stay on our side. Their
  parser therefore cannot catch a string dropped into the wrong row — whatever
  surfaces a copy button has to name the batch and the slot beside it.
* **Both roles at once.** One line is one shot type: our FRP slot maps to
  ``frp`` and our regular slot to ``sub`` (subsequent / steady state). The two
  write to different columns over there and never overwrite each other, so the
  order they are pasted in does not matter.
* **The linear magnitudes.** Only the dB levels (and the Pa·ms impulse) have
  columns on their end; the Pa values that back our averaging are ours alone.

Mic positions are labelled (``SE=`` / ``ML=``) rather than positional, and either
may be omitted — so a single slot's line is a complete, valid string on its own,
which is what lets each averaged row carry its own copy button.

``SSR1`` is a version stamp: **S**ilencer**S**cout **R**eport, version 1. Their
parser rejects any other tag rather than misfiling numbers under it, and it only
moves if a value slot's meaning, unit or order changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..models import MicPosition, ShotRole

#: The format stamp every line starts with. See the module docstring.
SCOUT_FORMAT_TAG = "SSR1"

#: Our derived role -> their shot type. ``sub`` is "subsequent / steady state",
#: which is exactly what a non-FRP shot is.
SCOUT_SHOT_TYPES = {ShotRole.FRP: "frp", ShotRole.REGULAR: "sub"}

#: The four value slots, in the order the string carries them: LIAeq100ms (dB),
#: Peak (dB), Peak (dBA), Impulse (Pa·ms). Positional, so this order *is* the
#: contract — reordering it would need an ``SSR2``.
SCOUT_METRIC_KEYS = ("liaeq_100ms_db", "peak_db", "peak_dba", "impulse_pa_ms")

#: Decimals emitted per value: every value is rounded to the nearest tenth
#: (122.57 -> "122.6"). Their end neither rounds nor requires a fixed number of
#: decimal places, so the rounding is ours to apply here.
_DECIMALS = 1


def format_value(value: float | None) -> str:
    """One value slot: a plain decimal, or ``""`` for "not measured".

    Anything we cannot state as a finite plain decimal — a ``NULL`` metric, or a
    NaN / infinity that slipped through — becomes an empty slot rather than a
    token their parser would reject. That matters because a bad value rejects the
    *whole line* on their end, so degrading one slot is what keeps the other
    three landing. Fixed-point formatting also rules out the scientific notation
    a very small magnitude would otherwise print as, which they likewise reject.

    >>> format_value(132.44)
    '132.4'
    >>> format_value(None)
    ''
    """
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{_DECIMALS}f}"


def slot_line(position: MicPosition, role: ShotRole, average: Mapping) -> str:
    """The ``SSR1`` line for one averaged slot: one mic position, one role.

    ``average`` is a slot entry from
    :meth:`~sound_metric_app.storage.WorkflowRepository.batch_averages` — any
    mapping carrying the :data:`SCOUT_METRIC_KEYS` will do. A missing key reads
    the same as a ``None`` value: an empty slot.

    >>> slot_line(MicPosition.SE, ShotRole.FRP, {
    ...     "liaeq_100ms_db": 137.9, "peak_db": 172.6,
    ...     "peak_dba": 156.4, "impulse_pa_ms": 4.88,
    ... })
    'SSR1|frp|SE=137.9,172.6,156.4,4.9'
    """
    values = ",".join(format_value(average.get(key)) for key in SCOUT_METRIC_KEYS)
    return f"{SCOUT_FORMAT_TAG}|{SCOUT_SHOT_TYPES[role]}|{position.value}={values}"
