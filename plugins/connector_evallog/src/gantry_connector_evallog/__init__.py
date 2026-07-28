"""Read Inspect Robots evaluation logs as Gantry records.

The seam between the two projects. Inspect Robots runs the evaluation; this
turns its log into records the feedback plane can diagnose.
"""

from .connector import (
    DEFAULT_SUCCESS_KEYS,
    STAGE_KEY,
    SUPPORTED_SCHEMAS,
    VERSION,
    EvalLogConnector,
    read_run,
    stage_metadata,
)

__all__ = [
    "DEFAULT_SUCCESS_KEYS",
    "STAGE_KEY",
    "SUPPORTED_SCHEMAS",
    "VERSION",
    "EvalLogConnector",
    "read_run",
    "stage_metadata",
]
