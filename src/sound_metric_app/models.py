"""Core data models shared across ingestion, DSP, and storage layers.

Two layers of models live here:

* **Signal models** (:class:`Frame`, :class:`MetricResult`) describe one channel of
  one capture and its computed metrics. These predate the workflow and stay
  single-channel friendly.
* **Hierarchy models** (:class:`Batch`, :class:`Group`, :class:`Shot`) describe the
  Batch -> Group -> Shot -> mic-channel organization from the README. They are
  plain in-memory mirrors of the storage rows; ``id`` fields are ``None`` until a
  row is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np


class MicPosition(str, Enum):
    """Which mic a channel was recorded from within a single capture file.

    ``str`` mixin so the value round-trips to/from SQLite text columns directly.
    """

    SE = "SE"  # Shooter's Ear
    MR = "MR"  # Muzzle Right


# --------------------------------------------------------------------------- #
# Filename convention
# --------------------------------------------------------------------------- #

# Capture files are named ``<suppressor_sku>_<test_platform>_<shot_order>.dxd``
# (or ``.d7d``), e.g. ``SUP-1234_AR15_003.dxd`` -> ("SUP-1234", "AR15", 3).
_CAPTURE_EXTENSIONS = {".dxd", ".d7d"}


class ParsedCaptureName(NamedTuple):
    """Result of :func:`parse_capture_filename`. Unpacks as a 3-tuple."""

    suppressor_sku: str  # batch key
    test_platform: str  # part of the group key (with ammo, tagged later)
    shot_order: int  # seeds Shot Order within the group


def parse_capture_filename(name: str) -> ParsedCaptureName:
    """Parse an app-controlled capture filename into its three keys.

    Accepts a bare filename or a full path, with or without a ``.dxd`` / ``.d7d``
    extension. The stem must be exactly three ``_``-separated, non-empty fields
    and the third must be numeric (the zero-padded shot order).

    >>> parse_capture_filename("SUP-1234_AR15_003.dxd")
    ParsedCaptureName(suppressor_sku='SUP-1234', test_platform='AR15', shot_order=3)

    Raises
    ------
    ValueError
        If the extension is not a capture extension, the field count is wrong,
        any field is empty, or the shot-order field is not numeric.
    """
    p = Path(name)
    suffix = p.suffix.lower()
    if suffix and suffix not in _CAPTURE_EXTENSIONS:
        raise ValueError(
            f"Not a capture file: {name!r} has extension {p.suffix!r}, "
            f"expected one of {sorted(_CAPTURE_EXTENSIONS)}."
        )
    stem = p.stem if suffix else p.name

    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Malformed capture name {name!r}: expected exactly 3 "
            f"'_'-separated fields (<sku>_<platform>_<shot_order>), got {len(parts)}."
        )
    sku, platform, order = parts
    if not sku or not platform or not order:
        raise ValueError(f"Malformed capture name {name!r}: fields must be non-empty.")
    if not (order.isascii() and order.isdigit()):
        raise ValueError(
            f"Malformed capture name {name!r}: shot-order field {order!r} is not numeric."
        )
    return ParsedCaptureName(suppressor_sku=sku, test_platform=platform, shot_order=int(order))


# --------------------------------------------------------------------------- #
# Signal models
# --------------------------------------------------------------------------- #


@dataclass
class Frame:
    """One acquisition frame of calibrated sound-pressure data, in Pascals.

    A DewesoftX ``.dxd`` capture for this application is a single 100 ms frame
    (20,000 samples at 200 kHz), so one file maps to one ``Frame`` per channel.
    """

    samples: np.ndarray  # 1-D float64, Pascals (calibration already applied)
    sample_rate: float  # Hz
    channel: str
    source_file: str
    timestamp: datetime | None = None
    events: list = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass
class MicChannel:
    """A :class:`Frame` tagged with the mic position it was recorded from.

    Produced when the user tags which raw channel is SE and which is MR; a shot
    may carry one or both positions.
    """

    position: MicPosition
    frame: Frame


@dataclass
class MetricResult:
    """Computed acoustic metrics for a single :class:`Frame`.

    Mic position is not carried here: the DSP layer works on bare frames and has
    no concept of SE/MR. The tagged position lives on :class:`MicChannel` and is
    passed explicitly to storage when the metrics are persisted.
    """

    peak_db: float
    peak_dba: float
    peak_impulse_db: float
    liaeq_100ms_db: float
    source_file: str
    channel: str
    sample_rate: float
    n_samples: int
    timestamp: datetime | None = None

    def as_row(self) -> dict:
        """Flat dict suitable for storage / CSV export."""
        return {
            "source_file": self.source_file,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "peak_db": self.peak_db,
            "peak_dba": self.peak_dba,
            "peak_impulse_db": self.peak_impulse_db,
            "liaeq_100ms_db": self.liaeq_100ms_db,
        }


# --------------------------------------------------------------------------- #
# Hierarchy models  (Batch -> Group -> Shot -> mic channels)
# --------------------------------------------------------------------------- #


@dataclass
class Batch:
    """One Suppressor SKU under test. Collects every shot fired against it.

    A batch is *closed* by the user to define it; once closed, further similar
    testing starts a new batch rather than reopening this one.
    """

    sku: str
    closed: bool = False
    id: int | None = None
    created_at: str | None = None
    closed_at: str | None = None


@dataclass
class Group:
    """Shots within a batch that share the same Test Platform + Ammo.

    Groups are the unit of averaging (identical test conditions).
    """

    test_platform: str
    ammo: str
    batch_id: int | None = None
    id: int | None = None
    created_at: str | None = None


@dataclass
class Shot:
    """A single firing event, captured as one file carrying up to two mic streams.

    On ingest a shot is an *Unmarked Data Set*: it knows only the provisional
    batch/group keys parsed from its filename (``suppressor_sku``,
    ``test_platform``) and ``marked`` is ``False``. Marking fills in ``ammo``, the
    per-shot environmental fields, and the SE/MR channel tags, and links it to a
    persisted :class:`Group` via ``group_id``.

    Environmental fields are recorded per shot in imperial units: ``wind_speed``
    in mph, ``temp`` in degrees Fahrenheit, ``relative_humidity`` in percent.
    ``se_channel`` / ``mr_channel`` hold the raw channel *names* the user tagged
    for each mic; either may be ``None`` for a single-mic shot.
    """

    source_file: str
    suppressor_sku: str | None = None  # provisional batch key from filename
    test_platform: str | None = None  # provisional group key from filename
    ammo: str | None = None  # set at marking
    shot_order: int | None = None
    wind_speed: float | None = None  # mph
    temp: float | None = None  # degrees Fahrenheit
    relative_humidity: float | None = None  # percent
    se_channel: str | None = None  # raw channel name tagged as SE
    mr_channel: str | None = None  # raw channel name tagged as MR
    marked: bool = False
    group_id: int | None = None
    id: int | None = None
    created_at: str | None = None
