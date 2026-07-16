"""Ingestion layer: read DewesoftX files into normalized Frame objects."""

from .dewesoft_reader import ChannelInfo, list_channels, read_frame

__all__ = ["ChannelInfo", "list_channels", "read_frame"]
