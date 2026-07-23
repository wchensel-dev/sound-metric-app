"""Read DewesoftX ``.dxd`` / ``.d7d`` files via the ``dwdatareader`` package.

The package bundles the native ``DWDataReaderLib`` binary, so no separate SDK
install is required. Channel values returned by ``.series()`` are already scaled
to engineering units (Pascals for the sound-pressure channels).

Two entry points:

* :func:`read_frame` — one auto-detected channel into a :class:`Frame`
  (CLI / back-compat).
* :func:`read_capture` — *every* synchronous Pa channel into a list of
  :class:`Frame`, so a two-mic (SE + ML) capture yields both streams. A single-mic
  capture simply yields one frame.

Raw channels map to mic positions by the DAQ input convention — AI 1 is the
muzzle-left transducer, AI 2 the shooter's-ear one — which :func:`autotag_map`
applies. The mapping stays overridable so a capture that breaks the convention
can still be tagged by hand; :func:`tag_channels` attaches the final ML/SE
labels either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import dwdatareader as dw
import numpy as np

from ..models import Frame, MicChannel, MicPosition


@dataclass
class ChannelInfo:
    name: str
    unit: str
    sample_rate: float
    n_samples: int

    @property
    def is_synchronous(self) -> bool:
        return self.sample_rate > 0


def _channels_of(f: "dw.DWFile") -> list[ChannelInfo]:
    """Enumerate channels of an already-open file with their unit and rate."""
    out: list[ChannelInfo] = []
    for name in f.keys():
        ch = f[name]
        out.append(
            ChannelInfo(
                name=name,
                unit=getattr(ch, "unit", ""),
                sample_rate=float(getattr(ch, "sample_rate", 0.0) or 0.0),
                n_samples=int(getattr(ch, "number_of_samples", 0) or 0),
            )
        )
    return out


def list_channels(path: str) -> list[ChannelInfo]:
    """Enumerate channels in a Dewesoft file with their unit and rate."""
    with dw.DWFile(path) as f:
        return _channels_of(f)


def _pressure_channels(channels: list[ChannelInfo]) -> list[ChannelInfo]:
    """Every capture (mic) channel: synchronous 'Pa' channels, in file order.

    Falls back to all synchronous channels if none are explicitly tagged 'Pa',
    which keeps files whose unit metadata is missing usable.
    """
    pa = [c for c in channels if c.is_synchronous and c.unit.strip().lower() == "pa"]
    if pa:
        return pa
    sync = [c for c in channels if c.is_synchronous]
    if sync:
        return sync
    raise ValueError("No synchronous channel found in file.")


def _pick_pressure_channel(channels: list[ChannelInfo]) -> str:
    """Choose a single sound-pressure channel (first Pa channel, else first sync)."""
    return _pressure_channels(channels)[0].name


def _frame_from_channel(
    f: "dw.DWFile", channel: str, path: str, timestamp: datetime | None
) -> Frame:
    """Build a :class:`Frame` from an already-open file and a channel name."""
    ch = f[channel]
    samples = ch.series().to_numpy().astype(np.float64)
    sample_rate = float(ch.sample_rate)
    return Frame(
        samples=samples,
        sample_rate=sample_rate,
        channel=channel,
        source_file=path,
        timestamp=timestamp,
    )


def _start_store_time(f: "dw.DWFile") -> datetime | None:
    """The capture's start-store time, or ``None`` if the file carried none.

    ``dwdatareader`` exposes this on the file's :class:`DWMeasurementInfo`
    (``f.info.start_store_time``), *not* on the ``DWFile`` object itself, so it
    must be read through ``info``.
    """
    info = getattr(f, "info", None)
    return getattr(info, "start_store_time", None) if info is not None else None


def read_frame(path: str, channel: str | None = None) -> Frame:
    """Read one channel of a Dewesoft file into a :class:`Frame` (Pascals)."""
    with dw.DWFile(path) as f:
        if channel is None:
            channel = _pick_pressure_channel(_channels_of(f))
        start = _start_store_time(f)
        return _frame_from_channel(f, channel, path, start)


def read_capture(path: str) -> list[Frame]:
    """Read *all* synchronous Pa (mic) channels of a capture into frames.

    Returns one :class:`Frame` per mic channel, in file order, each with a
    distinct :attr:`Frame.channel` name. A two-mic file yields two frames; a
    single-mic test file yields one. The channel->SE/ML mapping is applied
    separately via :func:`tag_channels`.
    """
    with dw.DWFile(path) as f:
        selected = _pressure_channels(_channels_of(f))
        start = _start_store_time(f)
        return [_frame_from_channel(f, c.name, path, start) for c in selected]


#: DAQ input -> mic position, the standing rig convention. AI 1 carries the
#: muzzle-left transducer and AI 2 the shooter's-ear one, so a conforming capture
#: needs no manual tagging. Names are matched case-insensitively with internal
#: whitespace collapsed (see :func:`_normalize_channel`), because Dewesoft
#: renders the same input as "AI 1", "ai1", or "AI  1" depending on setup.
DAQ_CHANNEL_POSITIONS: dict[str, MicPosition] = {
    "ai1": MicPosition.ML,  # muzzle left
    "ai2": MicPosition.SE,  # shooter's ear
}


def _normalize_channel(name: str) -> str:
    """Channel name reduced to its comparison key: lowercase, no whitespace."""
    return "".join(name.split()).lower()


def _channel_name(item: ChannelInfo | Frame) -> str:
    """The raw channel name of either a :class:`ChannelInfo` or a :class:`Frame`.

    The two carry it under different attributes (``name`` vs ``channel``), so
    :func:`autotag_map` can accept whichever list its caller already has.
    """
    return item.name if isinstance(item, ChannelInfo) else item.channel


def autotag_map(channels: list[ChannelInfo] | list[Frame]) -> dict[str, MicPosition]:
    """Best-guess channel -> position mapping from the AI 1 / AI 2 convention.

    Accepts either the :class:`ChannelInfo` list from :func:`list_channels` or
    the :class:`Frame` list from :func:`read_capture`, and returns the mapping
    :func:`tag_channels` consumes. Channels that do not match a known DAQ input
    are simply left out, so a non-conforming capture yields a partial (or empty)
    map rather than a wrong one — the caller then falls back to asking the user.

    A position is only ever claimed once: if two channels normalize to the same
    input, the first in file order wins and the duplicate is dropped, so the
    result is always safe to hand to :func:`tag_channels`.
    """
    mapping: dict[str, MicPosition] = {}
    claimed: set[MicPosition] = set()
    for channel in channels:
        name = _channel_name(channel)
        position = DAQ_CHANNEL_POSITIONS.get(_normalize_channel(name))
        if position is None or position in claimed:
            continue
        claimed.add(position)
        mapping[name] = position
    return mapping


def tag_channels(frames: list[Frame], mapping: dict[str, MicPosition]) -> list[MicChannel]:
    """Attach SE/ML labels to captured frames.

    ``mapping`` maps a raw channel name (as it appears in :attr:`Frame.channel`)
    to a :class:`MicPosition`. Tag one channel (single-mic shot) or two. Each mic
    position may be assigned at most once.

    Raises
    ------
    ValueError
        If ``mapping`` is empty, a named channel is not among ``frames``, or a
        position is reused.
    """
    if not mapping:
        raise ValueError("Tag at least one channel (SE and/or ML); the channel map is empty.")
    by_name = {fr.channel: fr for fr in frames}
    tagged: list[MicChannel] = []
    used: set[MicPosition] = set()
    for name, position in mapping.items():
        if name not in by_name:
            raise ValueError(
                f"Channel {name!r} is not in this capture; available: {sorted(by_name)}."
            )
        if position in used:
            raise ValueError(f"Mic position {position.value} assigned to more than one channel.")
        used.add(position)
        tagged.append(MicChannel(position=position, frame=by_name[name]))
    return tagged
