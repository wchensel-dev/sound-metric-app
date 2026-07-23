"""Ingestion layer: read DewesoftX files into normalized Frame objects."""

from .dewesoft_reader import (
    DAQ_CHANNEL_POSITIONS,
    ChannelInfo,
    autotag_map,
    list_channels,
    read_capture,
    read_frame,
    tag_channels,
)

__all__ = [
    "ChannelInfo",
    "DAQ_CHANNEL_POSITIONS",
    "autotag_map",
    "list_channels",
    "read_capture",
    "read_frame",
    "tag_channels",
]
