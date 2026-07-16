"""Read DewesoftX ``.dxd`` / ``.d7d`` files via the ``dwdatareader`` package.

The package bundles the native ``DWDataReaderLib`` binary, so no separate SDK
install is required. Channel values returned by ``.series()`` are already scaled
to engineering units (Pascals for the sound-pressure channels).

Two entry points:

* :func:`read_frame` — one auto-detected channel into a :class:`Frame`
  (CLI / back-compat).
* :func:`read_capture` — *every* synchronous Pa channel into a list of
  :class:`Frame`, so a two-mic (SE + MR) capture yields both streams. A single-mic
  capture simply yields one frame.

Mapping raw channels to mic positions is user-defined for now (README): use
:func:`tag_channels` to attach SE/MR labels once the user has chosen.
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


def list_channels(path: str) -> list[ChannelInfo]:
    """Enumerate channels in a Dewesoft file with their unit and rate."""
    out: list[ChannelInfo] = []
    with dw.DWFile(path) as f:
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


def read_frame(path: str, channel: str | None = None) -> Frame:
    """Read one channel of a Dewesoft file into a :class:`Frame` (Pascals)."""
    channels = list_channels(path)
    if channel is None:
        channel = _pick_pressure_channel(channels)

    with dw.DWFile(path) as f:
        start = getattr(f, "start_store_time", None)
        return _frame_from_channel(f, channel, path, start)


def read_capture(path: str) -> list[Frame]:
    """Read *all* synchronous Pa (mic) channels of a capture into frames.

    Returns one :class:`Frame` per mic channel, in file order, each with a
    distinct :attr:`Frame.channel` name. A two-mic file yields two frames; a
    single-mic test file yields one. The channel->SE/MR mapping is applied
    separately via :func:`tag_channels`.
    """
    channels = list_channels(path)
    selected = _pressure_channels(channels)
    with dw.DWFile(path) as f:
        start = getattr(f, "start_store_time", None)
        return [_frame_from_channel(f, c.name, path, start) for c in selected]


def tag_channels(frames: list[Frame], mapping: dict[str, MicPosition]) -> list[MicChannel]:
    """Attach user-chosen SE/MR labels to captured frames.

    ``mapping`` maps a raw channel name (as it appears in :attr:`Frame.channel`)
    to a :class:`MicPosition`. Tag one channel (single-mic shot) or two. Each mic
    position may be assigned at most once.

    Raises
    ------
    ValueError
        If a named channel is not among ``frames`` or a position is reused.
    """
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
