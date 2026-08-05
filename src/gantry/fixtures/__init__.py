"""Generated episodes with known flaws.

Every layer above the spine is acceptance-tested against these: a detector is
judged on whether it finds the planted defects *and* stays quiet on the clean
and decoy suites. Runs in CI on a laptop -- no GPU, no simulator, no checkpoint.

Shipped inside core rather than kept in the test tree so that third-party
plugins can test themselves against exactly the same material the first-party
ones are held to.
"""

from .defects import DECOYS, Defect, get_defect, known_defects, register_defect
from .synthetic import (
    RATE_HZ,
    SEGMENT_STEPS,
    STAGES,
    Trajectory,
    make_clean,
    make_defective,
    make_duration_confound,
    make_suite,
)
from .truth import FixtureSuite, GroundTruth

__all__ = [
    "DECOYS",
    "Defect",
    "FixtureSuite",
    "GroundTruth",
    "RATE_HZ",
    "SEGMENT_STEPS",
    "STAGES",
    "Trajectory",
    "get_defect",
    "known_defects",
    "make_clean",
    "make_defective",
    "make_duration_confound",
    "make_suite",
    "register_defect",
]
