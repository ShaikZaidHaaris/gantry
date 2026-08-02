"""Gantry: a frame that arbitrary payloads move over.

Five independent planes -- datasets, embodiments, policies, evaluation,
feedback -- each swappable without touching the other four. See ARCHITECTURE.md.
"""

__version__ = "0.1.0.dev0"

from . import errors, spine
from .errors import (
    ComponentError,
    ConfigError,
    FaultError,
    GantryError,
    SafetyAbort,
    disposition,
    must_halt,
)

__all__ = [
    "ComponentError",
    "ConfigError",
    "FaultError",
    "GantryError",
    "SafetyAbort",
    "__version__",
    "disposition",
    "errors",
    "must_halt",
    "spine",
]
