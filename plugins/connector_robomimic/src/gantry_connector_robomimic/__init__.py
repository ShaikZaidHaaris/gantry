"""Read RoboMimic HDF5 demonstration files as Gantry episodes."""

# Importing this registers the meaning tags the connector reports.
import gantry_semantics_manipulation as _semantics  # noqa: F401

from .connector import CONVENTIONS, NOT_BEHAVIOUR, VERSION, RoboMimicConnector

__all__ = ["CONVENTIONS", "NOT_BEHAVIOUR", "VERSION", "RoboMimicConnector"]
