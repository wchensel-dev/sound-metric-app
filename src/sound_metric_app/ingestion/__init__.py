"""Ingestion layer: read DewesoftX files into normalized Frame objects."""

from .dewesoft_reader import (
    DAQ_CHANNEL_POSITIONS,
    ChannelInfo,
    autotag_map,
    count_shots,
    list_channels,
    read_capture,
    read_frame,
    split_segments,
    tag_channels,
)

__all__ = [
    "ChannelInfo",
    "DAQ_CHANNEL_POSITIONS",
    "autotag_map",
    "count_shots",
    "list_channels",
    "read_capture",
    "read_frame",
    "split_segments",
    "tag_channels",
]
