"""Read DewesoftX ``.dxd`` / ``.d7d`` files via the ``dwdatareader`` package.

The package bundles the native ``DWDataReaderLib`` binary, so no separate SDK
install is required. Channel values returned by ``.series()`` are already scaled
to engineering units (Pascals for the sound-pressure channel).
"""

from __future__ import annotations

from dataclasses import dataclass

import dwdatareader as dw
import numpy as np

from ..models import Frame


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


def _pick_pressure_channel(channels: list[ChannelInfo]) -> str:
    """Choose the sound-pressure channel: prefer a synchronous 'Pa' channel."""
    pa = [c for c in channels if c.is_synchronous and c.unit.strip().lower() == "pa"]
    if pa:
        return pa[0].name
    sync = [c for c in channels if c.is_synchronous]
    if sync:
        return sync[0].name
    raise ValueError("No synchronous channel found in file.")


def read_frame(path: str, channel: str | None = None) -> Frame:
    """Read one channel of a Dewesoft file into a :class:`Frame` (Pascals)."""
    channels = list_channels(path)
    if channel is None:
        channel = _pick_pressure_channel(channels)

    with dw.DWFile(path) as f:
        ch = f[channel]
        samples = ch.series().to_numpy().astype(np.float64)
        sample_rate = float(ch.sample_rate)
        start = getattr(f, "start_store_time", None)

    return Frame(
        samples=samples,
        sample_rate=sample_rate,
        channel=channel,
        source_file=path,
        timestamp=start,
    )
