"""Partition a corpus into cohorts that can honestly be compared."""

from .split import (
    VERSION,
    Part,
    Split,
    group_of,
    hold_out,
    match_frames,
    moving_fraction,
)

__all__ = [
    "VERSION",
    "Part",
    "Split",
    "group_of",
    "hold_out",
    "match_frames",
    "moving_fraction",
]
