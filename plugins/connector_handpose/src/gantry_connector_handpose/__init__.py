"""Another connector's clips, with hands estimated onto them.

Two wires ship. ``mediapipe`` gives hand shape and image position; ``metric``
adds perspective-n-point and gives a pose in metres, which is what a retargeter
can actually use. Both are Apache-2.0, which is the reason they are here rather
than the better-reconstructing research models — see :mod:`.pnp`.
"""

from .handpose import (
    FAST,
    FOUND,
    HANDS,
    JOINTS,
    MIN_STEPS,
    VERSION,
    Estimator,
    HandPoseConnector,
    Track,
    aperture_from,
    mediapipe,
    metric,
    wrist_from,
)
from .pnp import (
    MAX_REPROJECTION,
    RIGS,
    SOURCES,
    Intrinsics,
    Pose,
    for_rig,
    intrinsics_from,
    plausible,
    rotations_to_quaternions,
    solve,
    solve_sequence,
)

__all__ = [
    "FAST",
    "FOUND",
    "HANDS",
    "JOINTS",
    "MAX_REPROJECTION",
    "MIN_STEPS",
    "RIGS",
    "SOURCES",
    "VERSION",
    "Estimator",
    "HandPoseConnector",
    "Intrinsics",
    "Pose",
    "Track",
    "aperture_from",
    "for_rig",
    "intrinsics_from",
    "mediapipe",
    "metric",
    "plausible",
    "rotations_to_quaternions",
    "solve",
    "solve_sequence",
    "wrist_from",
]
