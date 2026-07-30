"""Channel semantics for egocentric human video.

Install it and ego channels get full checking — including the two failures that
are invisible in the array: a monocular hand pose used as though it were metres,
and MANO joint indices read as MediaPipe ones. Don't install it, and nothing in
Gantry notices or cares.
"""

from .vocabulary import (
    FRAMES,
    HANDS,
    KEYPOINTS,
    MEANINGS,
    SCALES,
    aperture_channel,
    describe,
    ego_camera,
    gaze_channel,
    hand_channel,
    head_pose_channel,
    is_metric,
    register_all,
    scale_of,
    sequence_of,
    wrist_channel,
)

register_all()

__all__ = [
    "FRAMES",
    "HANDS",
    "KEYPOINTS",
    "MEANINGS",
    "SCALES",
    "aperture_channel",
    "describe",
    "ego_camera",
    "gaze_channel",
    "hand_channel",
    "head_pose_channel",
    "is_metric",
    "register_all",
    "scale_of",
    "sequence_of",
    "wrist_channel",
]
