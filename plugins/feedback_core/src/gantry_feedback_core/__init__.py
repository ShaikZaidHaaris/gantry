"""Feedback modules: screen a dataset, diagnose a run, attribute, and harden.

Consumes records and nothing else. No simulator, no policy, no connector, and
no knowledge of which of those produced the episodes it was handed.
"""

from .attribution import Attribution, eligible_statistics, quarantined_statistics
from .funnel import Funnel, Step, build, end_to_end, stage_order, uplift
from .harden import ATTRIBUTABLE, INCONSISTENT, UNIVERSAL, Harden, classify
from .metrics import (
    Statistic,
    known_statistics,
    prescribable_statistics,
    register_statistic,
    tabulate,
)
from .screen import Screen, Threshold

__all__ = [
    "ATTRIBUTABLE",
    "INCONSISTENT",
    "UNIVERSAL",
    "Attribution",
    "Funnel",
    "Harden",
    "Screen",
    "Statistic",
    "Step",
    "Threshold",
    "build",
    "classify",
    "eligible_statistics",
    "end_to_end",
    "known_statistics",
    "prescribable_statistics",
    "quarantined_statistics",
    "register_statistic",
    "stage_order",
    "tabulate",
    "uplift",
]
