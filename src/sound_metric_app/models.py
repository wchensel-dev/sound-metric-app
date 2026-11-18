"""Core data models shared across ingestion, DSP, and storage layers.

Two layers of models live here:

* **Signal models** (:class:`Frame`, :class:`MetricResult`) describe one channel of
  one capture and its computed metrics. These predate the workflow and stay
  single-channel friendly.
* **Hierarchy models** (:class:`Combination`, :class:`Batch`, :class:`Cluster`,
  :class:`Shot`) describe the containment tree from the README:

  .. code-block:: text

     SKU -> Platform -> Ammo   (together: one Combination)
       Batch      one test session
         Cluster  one string of fire
           Shot   one gunshot event
             Channel: ML (muzzle left) / SE (shooter's ear)

  They are plain in-memory mirrors of the storage rows; ``id`` fields are
  ``None`` until a row is persisted.

The tree is *pure containment*. What drives the roll-up is separate per-shot
state: :attr:`Shot.shot_order` (from which :class:`ShotRole` is derived),
the channel's :class:`MicPosition`, and :attr:`Shot.included`.
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

    Position lives on the *channel*, not the shot: one shot carries both mics,
    and they diverge only at averaging time. The two map directly to the DAQ
    inputs — AI 1 is the muzzle-left transducer, AI 2 the shooter's-ear one.

    ``str`` mixin so the value round-trips to/from SQLite text columns directly.
    """

    SE = "SE"  # Shooter's Ear   (AI 2)
    ML = "ML"  # Muzzle Left     (AI 1)

    @property
    def label(self) -> str:
        """Human-readable position name for reports and form labels."""
        return _POSITION_LABELS[self]


_POSITION_LABELS = {
    MicPosition.SE: "Shooter's Ear",
    MicPosition.ML: "Muzzle Left",
}


class ShotRole(str, Enum):
    """A shot's role within its cluster, **derived** from its shot order.

    The first shot of order 0 is the FRP (first round pop) — the cold-bore shot
    whose signature differs from the rest of the string — and everything after
    it is regular, including any later shot the ``_0000`` file itself holds.
    Deriving the role rather than storing it means every cluster has exactly one
    FRP by construction, and re-ordering a shot cannot leave a stale role behind.

    ``str`` mixin so the value round-trips to/from SQLite text columns directly.
    """

    FRP = "FRP"
    REGULAR = "REGULAR"

    @property
    def label(self) -> str:
        return "FRP" if self is ShotRole.FRP else "Regular"


#: Shot order that makes a shot its cluster's FRP. Everything above is regular.
#: DewesoftX numbers its exports from zero, so the first round of a string
#: arrives as ``..._0000`` and that trailing zero is what marks the FRP.
FRP_SHOT_ORDER = 0

#: Smallest accepted value for each numeric filename field. Shot orders are
#: 0-based (Dewesoft's own counter); cluster indices stay 1-based, since we
#: number the strings of fire ourselves rather than inheriting them.
MIN_SHOT_ORDER = FRP_SHOT_ORDER
MIN_CLUSTER_INDEX = 1


def role_for_order(shot_order: int | None, file_rank: int = 0) -> ShotRole | None:
    """The :class:`ShotRole` implied by a shot's place in its string, or ``None`` if unordered.

    ``shot_order`` is the file's order (from its name) and ``file_rank`` the
    shot's position among the shots that file holds. Only the first shot of the
    order-0 file is the FRP: a recorder left armed can append more shots to that
    same file, and those are regulars.

    >>> role_for_order(0)
    <ShotRole.FRP: 'FRP'>
    >>> role_for_order(0, file_rank=1)
    <ShotRole.REGULAR: 'REGULAR'>
    >>> role_for_order(4)
    <ShotRole.REGULAR: 'REGULAR'>
    >>> role_for_order(None) is None
    True
    """
    if shot_order is None:
        return None
    if shot_order == FRP_SHOT_ORDER and file_rank == 0:
        return ShotRole.FRP
    return ShotRole.REGULAR


def order_label(shot_order: int | None, file_rank: int = 0) -> str:
    """A shot's place in its string as text: ``"3"``, or ``"3.1"`` for a file's second shot.

    The integer part is the file's order from its name; the fraction counts the
    further shots that one file holds, 0-based like Dewesoft's own counter, so a
    one-shot file reads exactly as it always has. ``""`` when unordered.

    >>> order_label(3), order_label(0, 2), order_label(None)
    ('3', '0.2', '')
    """
    if shot_order is None:
        return ""
    return f"{shot_order}.{file_rank}" if file_rank else str(shot_order)


# --------------------------------------------------------------------------- #
# Filename convention
# --------------------------------------------------------------------------- #

# Capture files are named
# ``<suppressor_sku>_<test_platform>_<cluster>_<shot_order>.dxd`` (or ``.d7d``),
# e.g. ``SUP-1234_AR15_02_0003.dxd`` -> ("SUP-1234", "AR15", 2, 3).
CAPTURE_EXTENSIONS = frozenset({".dxd", ".d7d"})


#: Separates a capture path from the shot number in a multi-shot file's shot
#: key. ``|`` cannot occur in a Windows path, so the split is never ambiguous.
SHOT_SOURCE_SEP = "|"


def shot_source(path: str, segment: int | None) -> str:
    """The stored key of one shot: ``path`` for a whole file, ``path|k`` for shot ``k``.

    A one-shot file keeps its plain path, exactly as every shot was keyed before
    files could hold several, so existing rows need no change. Only the shots
    of a file holding two or more triggers carry a suffix.

    >>> shot_source("C:/x.dxd", None), shot_source("C:/x.dxd", 1)
    ('C:/x.dxd', 'C:/x.dxd|1')
    """
    return path if segment is None else f"{path}{SHOT_SOURCE_SEP}{segment}"


def split_shot_source(source: str) -> tuple[str, int | None]:
    """Inverse of :func:`shot_source`: ``(file path, segment or None)``.

    >>> split_shot_source("C:/x.dxd|2"), split_shot_source("C:/x.dxd")
    (('C:/x.dxd', 2), ('C:/x.dxd', None))
    """
    path, sep, segment = source.rpartition(SHOT_SOURCE_SEP)
    if sep and segment.isdigit():
        return path, int(segment)
    return source, None


class ParsedCaptureName(NamedTuple):
    """Result of :func:`parse_capture_filename`. Unpacks as a 4-tuple."""

    suppressor_sku: str  # part of the combination key
    test_platform: str  # part of the combination key (ammo is tagged later)
    cluster_index: int  # which string of fire within the batch
    shot_order: int  # the file's position within that cluster; 0 holds the FRP


def parse_capture_filename(name: str) -> ParsedCaptureName:
    """Parse an app-controlled capture filename into its four keys.

    Accepts a bare filename or a full path, with or without a ``.dxd`` / ``.d7d``
    extension. The stem must be exactly four ``_``-separated, non-empty fields;
    the last two (cluster index and zero-padded shot order) must be numeric.

    Encoding the cluster in the filename fixes each string of fire at capture
    time, so a shot arrives already knowing which cluster it belongs to and what
    its order within that cluster is — and therefore whether it is the FRP.

    The shot order is DewesoftX's own export counter, which starts at zero, so a
    trailing ``0000`` holds the string's FRP. It orders *files*, not rounds: a
    file the recorder stayed armed into can hold several shots, so ``0003`` is
    the fourth file of the string but not necessarily its fourth round. Cluster
    indices are ours and stay 1-based.

    >>> parse_capture_filename("SUP-1234_AR15_02_0003.dxd")
    ParsedCaptureName(suppressor_sku='SUP-1234', test_platform='AR15', cluster_index=2, shot_order=3)
    >>> parse_capture_filename("SUP-1234_AR15_02_0000.dxd").shot_order
    0

    Raises
    ------
    ValueError
        If the extension is not a capture extension, the field count is wrong,
        any field is empty, either numeric field is not numeric, the cluster is
        below 1, or the shot order is below 0.
    """
    p = Path(name)
    suffix = p.suffix.lower()
    if suffix and suffix not in CAPTURE_EXTENSIONS:
        raise ValueError(
            f"Not a capture file: {name!r} has extension {p.suffix!r}, "
            f"expected one of {sorted(CAPTURE_EXTENSIONS)}."
        )
    stem = p.stem if suffix else p.name

    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Malformed capture name {name!r}: expected exactly 4 '_'-separated "
            f"fields (<sku>_<platform>_<cluster>_<shot_order>), got {len(parts)}."
        )
    sku, platform, cluster, order = parts
    if not all(parts):
        raise ValueError(f"Malformed capture name {name!r}: fields must be non-empty.")
    for label, value, minimum in (
        ("cluster", cluster, MIN_CLUSTER_INDEX),
        ("shot-order", order, MIN_SHOT_ORDER),
    ):
        if not (value.isascii() and value.isdigit()):
            raise ValueError(
                f"Malformed capture name {name!r}: {label} field {value!r} is not numeric."
            )
        if int(value) < minimum:
            raise ValueError(
                f"Malformed capture name {name!r}: {label} field {value!r} "
                f"must be {minimum} or greater."
            )
    return ParsedCaptureName(
        suppressor_sku=sku,
        test_platform=platform,
        cluster_index=int(cluster),
        shot_order=int(order),
    )


# --------------------------------------------------------------------------- #
# Signal models
# --------------------------------------------------------------------------- #


@dataclass
class Frame:
    """One acquisition frame of calibrated sound-pressure data, in Pascals.

    One frame is one shot's stored trigger block on one channel (10 ms
    pre-trigger + 200 ms at 200 kHz). A file the recorder stayed armed into
    holds several blocks, and each becomes its own ``Frame`` per channel.
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

    Produced when the user tags which raw channel is SE and which is ML; a shot
    may carry one or both positions.
    """

    position: MicPosition
    frame: Frame


@dataclass
class MetricResult:
    """Computed acoustic metrics for a single :class:`Frame`.

    Each metric carries both its **linear magnitude** (a pressure in Pa, or the
    positive-phase impulse in Pa·ms) and the **dB level** derived from it. The
    linear value is the source of truth: group aggregation averages the linear
    magnitudes and converts to dB once (MATH.md §9), while a per-shot view shows
    the stored dB directly. ``dB = 20*log10(magnitude / p_ref)`` in every case.

    Mic position is not carried here: the DSP layer works on bare frames and has
    no concept of SE/ML. The tagged position lives on :class:`MicChannel` and is
    passed explicitly to storage when the metrics are persisted.
    """

    peak_pa: float  # raw signed peak pressure, Pa (positive overpressure)
    peak_db: float
    peak_a_pa: float  # A-weighted signed peak pressure, Pa
    peak_dba: float
    impulse_pa_ms: float  # positive-phase acoustic impulse, Pa·ms
    peak_impulse_db: float  # 20*log10(impulse_pa_ms / p_ref), dB·ms
    leq10ms_pa: float  # peak 10 ms-Leq (A-weighted RMS), Pa
    leq10ms_db: float
    liaeq_pa: float  # LIAeq,100ms (A-weighted RMS), Pa
    liaeq_100ms_db: float
    source_file: str
    channel: str
    sample_rate: float
    n_samples: int
    timestamp: datetime | None = None
    #: Pre-trigger baseline diagnostic (signed mean Pa over the capture's first
    #: PRETRIGGER_FLOOR_SAMPLES), not an acoustic metric: it has no dB companion,
    #: is never batch-averaged, and corrects nothing. Defaulted so a MetricResult
    #: built without it (a hand-built test fixture, a caller predating the
    #: column) still constructs, reading as "baseline not displaced".
    pretrigger_floor_pa: float = 0.0

    def as_row(self) -> dict:
        """Flat dict suitable for storage / CSV export."""
        return {
            "source_file": self.source_file,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "peak_pa": self.peak_pa,
            "peak_db": self.peak_db,
            "peak_a_pa": self.peak_a_pa,
            "peak_dba": self.peak_dba,
            "impulse_pa_ms": self.impulse_pa_ms,
            "peak_impulse_db": self.peak_impulse_db,
            "leq10ms_pa": self.leq10ms_pa,
            "leq10ms_db": self.leq10ms_db,
            "liaeq_pa": self.liaeq_pa,
            "liaeq_100ms_db": self.liaeq_100ms_db,
            "pretrigger_floor_pa": self.pretrigger_floor_pa,
        }


# --------------------------------------------------------------------------- #
# Hierarchy models  (Combination -> Batch -> Cluster -> Shot -> mic channels)
# --------------------------------------------------------------------------- #


@dataclass
class Combination:
    """One test combination: a SKU + Platform + Ammo path.

    The three test conditions collapse into a single row rather than three
    nested tables, because they are only ever meaningful together — every SKU
    holds many platforms, every platform many ammo types, and it is the specific
    triple that batches hang from. The tree is still presented SKU -> Platform ->
    Ammo in the UI; this is the leaf those three levels address.
    """

    sku: str
    platform: str
    ammo: str
    id: int | None = None
    created_at: str | None = None

    @property
    def label(self) -> str:
        """``SKU / Platform / Ammo``, the combination's display name."""
        return f"{self.sku} / {self.platform} / {self.ammo}"


@dataclass
class Batch:
    """One test session under a :class:`Combination`.

    A batch is a *session*, not a SKU: it carries the day's context (date, the
    typical weather for the session, free-form notes) and holds the clusters
    fired that day. Per-shot conditions can drift within a session, so each
    :class:`Shot` also carries its own specific weather; the batch fields are the
    session-level typical values.

    A batch is *closed* by the user to define it; once closed, further testing on
    the same combination starts a new batch rather than reopening this one.
    """

    combination_id: int | None = None
    label: str | None = None  # user's name for the session
    session_date: str | None = None  # ISO-8601 date the session was fired
    wind_speed: float | None = None  # typical, mph
    temp: float | None = None  # typical, degrees Fahrenheit
    relative_humidity: float | None = None  # typical, percent
    notes: str | None = None
    closed: bool = False
    id: int | None = None
    created_at: str | None = None
    closed_at: str | None = None

    @property
    def title(self) -> str:
        """The session's name + date, or a placeholder when neither is set yet.

        Named ``title`` rather than ``label`` — unlike the other hierarchy types,
        a batch already carries a user-supplied ``label`` field.
        """
        parts = [p for p in (self.label, self.session_date) if p]
        return " ".join(parts) if parts else "(unnamed session)"

    @property
    def weather_summary(self) -> str:
        """The typical session weather as one line, ``""`` when none is recorded.

        Callers that need a placeholder for "nothing recorded" supply their own;
        it differs by surface (a tree column wants blank, a printed field does not).
        """
        bits = []
        if self.wind_speed is not None:
            bits.append(f"wind {self.wind_speed:g} mph")
        if self.temp is not None:
            bits.append(f"{self.temp:g} °F")
        if self.relative_humidity is not None:
            bits.append(f"RH {self.relative_humidity:g}%")
        return ", ".join(bits)


@dataclass
class Cluster:
    """One string of fire within a batch, holding its shots in order.

    Clusters are containment only — they do not themselves average. A cluster of
    3 contributes one FRP and two regulars; a cluster of 4 contributes one FRP
    and three regulars. That is exactly why inclusion is tracked per *shot*
    rather than per cluster: whole clusters cannot cleanly land on a target of 5
    regulars.
    """

    batch_id: int | None = None
    cluster_index: int | None = None  # 1-based, from the filename
    id: int | None = None
    created_at: str | None = None

    @property
    def label(self) -> str:
        return f"Cluster {self.cluster_index}" if self.cluster_index else "Cluster"


@dataclass
class Shot:
    """A single gunshot event: one stored trigger block, carrying up to two mic streams.

    A capture file holds one or more shots. ``source_file`` is the shot's key:
    the plain file path for a one-shot file, or ``<path>|<k>`` for shot ``k`` of
    a file the recorder stayed armed into (see :func:`shot_source`).

    On ingest a shot is an *Unmarked Data Set*: it knows only the provisional
    keys parsed from its filename (``suppressor_sku``, ``test_platform``,
    ``cluster_index``, ``shot_order``) and ``marked`` is ``False``. Marking fills
    in ``ammo``, the per-shot environmental fields, and the ML/SE channel tags,
    and links it to a persisted :class:`Cluster` via ``cluster_id``.

    Three pieces of per-shot state drive the roll-up, separately from where the
    shot sits in the tree:

    * ``shot_order`` — its file's position within the cluster, with
      ``file_rank`` its position within that file. Together they give firing
      order; the first shot of order 0 is the FRP. See :attr:`role`, which is
      derived, never stored.
    * ``included`` — whether the shot feeds its batch's average. Idle
      (``False``) by default; flipping it on is what brings a shot forward out of
      the data bank. ``exclusion_reason`` records *why* a shot was left behind
      (high winds, ambient noise, ...) and is only meaningful while idle.
    * mic position, which lives on the channel rather than here.

    Environmental fields are this shot's *specific* weather, in imperial units:
    ``wind_speed`` in mph, ``temp`` in degrees Fahrenheit, ``relative_humidity``
    in percent. ``se_channel`` / ``ml_channel`` hold the raw channel *names*
    tagged for each mic; either may be ``None`` for a single-mic shot.
    """

    source_file: str
    suppressor_sku: str | None = None  # provisional combination key from filename
    test_platform: str | None = None  # provisional combination key from filename
    ammo: str | None = None  # set at marking; completes the combination key
    cluster_index: int | None = None  # provisional cluster key from filename
    shot_order: int | None = None  # its file's position within the cluster; 0 holds the FRP
    wind_speed: float | None = None  # mph
    temp: float | None = None  # degrees Fahrenheit
    relative_humidity: float | None = None  # percent
    #: Onset trigger threshold used to analyse this shot (Pa). Set at marking
    #: (default ``config.DEFAULT_TRIGGER_PA``); ``None`` means legacy/unknown, so
    #: the shot is analysed at the ``config.ONSET_THRESHOLD_PA`` (1 Pa) fallback.
    trigger_pa: float | None = None
    se_channel: str | None = None  # raw channel name tagged as SE
    ml_channel: str | None = None  # raw channel name tagged as ML
    marked: bool = False
    #: Whether this shot feeds the batch average. Idle by default.
    included: bool = False
    #: Why the shot was left out of the average; only meaningful while idle.
    exclusion_reason: str | None = None
    cluster_id: int | None = None
    id: int | None = None
    created_at: str | None = None
    #: When the shot was fired, pulled from the capture file's start-store time
    #: (Dewesoft ``start_store_time``). ISO-8601 string, or ``None`` if the file
    #: carried no timestamp. Set at marking, when the capture is read.
    captured_at: str | None = None
    #: Position among the shots *still recorded* for the same file (0-based).
    #: Derived by the store on read, never written: discarding a false trigger
    #: ahead of a shot moves that shot up, so the FRP is the first real shot of
    #: the ``_0000`` file even when the recorder fired on something else first.
    file_rank: int = 0

    @property
    def file_path(self) -> str:
        """The capture file on disk, without any shot suffix."""
        return split_shot_source(self.source_file)[0]

    @property
    def segment_index(self) -> int | None:
        """Which stored trigger block of :attr:`file_path` this is; ``None`` = whole file."""
        return split_shot_source(self.source_file)[1]

    @property
    def role(self) -> ShotRole | None:
        """FRP or Regular, derived from its order and file rank (``None`` if unordered)."""
        return role_for_order(self.shot_order, self.file_rank)

    @property
    def order_label(self) -> str:
        """Firing position as text: ``"2"``, or ``"0.1"`` for a file's second shot."""
        return order_label(self.shot_order, self.file_rank)


@dataclass
class DiscardedFile:
    """A capture path, or one shot inside it, that the operator has told the app to ignore.

    Recorded so a permanently-bad file (corrupt, unfixable filename) does not
    resurface as malformed/unreadable on every ingest scan. Restoring just
    deletes the row; the file itself is untouched and will resurface as
    malformed/unreadable on the next scan if it's still bad.

    ``source_file`` is a shot key (:func:`shot_source`), so a single shot of a
    multi-shot file, such as a false trigger, is discarded on its own while its
    sibling shots stay in the app.
    """

    source_file: str
    reason: str | None = None
    discarded_at: str | None = None

    @property
    def file_path(self) -> str:
        return split_shot_source(self.source_file)[0]

    @property
    def segment_index(self) -> int | None:
        """The discarded shot within a multi-shot file; ``None`` for a whole file."""
        return split_shot_source(self.source_file)[1]
