"""Structural retargeters that are defensible without knowing the machines.

Everything harder needs a model of the specific pair and belongs with whoever
owns it. Stopping here is deliberate: a retargeter that is nearly right produces
motion that looks reasonable and is wrong.
"""

from .retargeters import ROTATION_WIDTHS, VERSION, DropDimensions, PoseToPosition

__all__ = ["ROTATION_WIDTHS", "VERSION", "DropDimensions", "PoseToPosition"]
