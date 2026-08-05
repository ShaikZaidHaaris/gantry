"""Channel semantics for egocentric human video.

Ego data is the cheapest robot training data in existence -- a person with a
camera on their head produces demonstrations at roughly the cost of doing their
own chores -- and it is the most dangerous to read carelessly, because two of its
properties are invisible in the array and change what every downstream number
means.

The two that bite
-----------------
**Scale.** A hand pose estimated from a single camera has no metric scale. The
estimator returns a plausible hand in plausible proportions and cannot know
whether it is a child's hand thirty centimetres away or an adult's at fifty. Every
number is right up to an unknown multiplier. A calibrated rig -- a device with a
known baseline and a SLAM trajectory -- returns metres.

Both arrive as float32 arrays of identical shape. Retarget the first onto a robot
as though it were the second and the arm reaches for a point that does not exist,
consistently, in a way that looks like a policy that has not learned to reach. So
``scale`` is a discriminator: a channel must say ``metric`` or ``unscaled``, and a
consumer that needs metres is entitled to refuse.

**Keypoint convention.** MANO gives 21 joints per hand. MediaPipe gives 21. They
are not the same 21 -- the orderings differ and MANO's root is the wrist while
MediaPipe's index 0 is the palm base. ARKit gives 26. Feed a MANO-trained
retargeter MediaPipe indices and every finger is attached to the wrong knuckle;
the shapes agree, the dtypes agree, nothing complains, and the resulting robot
trajectories are subtly and unfixably wrong.

So ``keypoints`` is a discriminator too. This is the same class of failure the
manipulation vocabulary catches with ``rotation_repr``: identical width,
different meaning, silent when wrong.

Handedness is a third, smaller one
----------------------------------
Left and right hands are mirror images, and an estimator that reports them in a
fixed array slot rather than by identity will swap them whenever the person
crosses their arms. Which slot is which is carried on the channel rather than
assumed from the index.

What is deliberately not here
-----------------------------
Any notion of a robot. This vocabulary describes what a *person* did. Turning
that into what a *machine* should do is the retargeter plane's job, and keeping
the description honest at this layer is what lets the retargeting be argued with
later.
"""

from __future__ import annotations

from typing import Sequence

from gantry.spine import ChannelSpec, register_semantics, units

#: Metres, as the units module spells it. Attached only to metric channels -- a
#: unit on an unscaled one would be a lie with a dimension on it.
METRE = "m"

#: How the hand was measured, and whether its numbers are in metres.
#:
#: ``metric`` -- a calibrated rig: stereo baseline, depth sensor, or a SLAM
#: trajectory with known scale. Positions are metres and mean it.
#:
#: ``unscaled`` -- a monocular estimate. Internally consistent, correct up to one
#: unknown multiplier, and unusable for anything that has to touch a real object
#: at a real distance.
#:
#: ``normalized`` -- deliberately unit-free, e.g. keypoints in image coordinates
#: or a hand normalised to unit bone length. Distinguished from ``unscaled``
#: because it is a *choice* rather than a limitation, and the recovery is
#: different: a normalised hand needs a bone length, an unscaled one needs a
#: measurement.
SCALES = {
    "metric": "positions are metres, from a calibrated capture",
    "unscaled": "monocular estimate, correct up to one unknown multiplier",
    "normalized": "deliberately unit-free; needs a reference length to become metric",
}

#: Hand keypoint conventions, and how many joints each has. Two of them are the
#: same width and a different ordering, which is the whole reason this is a
#: discriminator rather than a comment.
KEYPOINTS = {
    "mano": 21,
    "mediapipe": 21,
    "arkit": 26,
    "openpose": 21,
    #: Wrist only -- no finger joints at all. Common from rigs that track the
    #: controller or the device rather than the hand.
    "wrist_only": 1,
}

#: Which hand. ``either`` is for a channel that carries whichever hand was
#: visible and says which in a parallel channel; it is honest and awkward, and
#: preferable to a fixed slot that silently swaps when arms cross.
HANDS = ("left", "right", "either")

#: The frame ego measurements live in.
#:
#: ``camera`` -- relative to the head-mounted camera, which is moving. Fine for
#: what the person saw; wrong for anything that has to be consistent across a
#: clip, because the reference itself is in motion.
#:
#: ``head`` -- relative to the device, gravity-aligned where the rig reports it.
#:
#: ``world`` -- a fixed frame recovered by SLAM. The only one in which a
#: trajectory means the same thing at the start and the end of a clip.
FRAMES = ("camera", "head", "world", "image")

#: Per-channel meanings, with their physical dimension where they have one.
MEANINGS: dict[str, tuple[str, object]] = {
    "ego.rgb": ("what the person saw, from the head-mounted camera", None),
    "ego.depth": ("per-pixel distance from the head-mounted camera", units.LENGTH),
    "ego.hand_keypoints": ("hand joint positions, one row per joint", units.LENGTH),
    "ego.wrist_pose": ("wrist position and orientation", None),
    "ego.wrist_position": ("wrist position", units.LENGTH),
    "ego.aperture": ("how open the hand is, thumb-to-finger distance", units.LENGTH),
    "ego.grasp": ("whether the hand is holding something, 0 to 1", None),
    "ego.head_pose": ("device position and orientation, from SLAM", None),
    "ego.gaze": ("where the person was looking, as a direction", None),
    "ego.handedness": ("which hand this row is, 0 left 1 right", None),
    "ego.contact": ("whether the hand is touching an object", None),
}


def register_all() -> None:
    """Register the ego vocabulary. Called on import of the package."""
    for name, (description, dimension) in MEANINGS.items():
        register_semantics(name, description, dimension)


def scale_of(spec: ChannelSpec) -> str | None:
    """What this channel says about its own metric scale, or ``None`` if it does not.

    ``None`` is the answer that should worry a caller. A channel that never
    considered the question is not the same as one that says ``unscaled``, and a
    retargeter treating both as "probably fine" is the failure this module exists
    to make loud.
    """
    value = spec.metadata.get("scale")
    return str(value) if value is not None else None


def is_metric(spec: ChannelSpec) -> bool | None:
    """Whether these numbers are metres. ``None`` when the channel does not say."""
    scale = scale_of(spec)
    return None if scale is None else scale == "metric"


def hand_channel(
    name: str,
    *,
    hand: str = "right",
    keypoints: str = "mano",
    scale: str = "unscaled",
    frame: str = "camera",
    rate_hz: float | None = None,
    dtype: str = "float32",
) -> ChannelSpec:
    """A hand-keypoints channel, with everything load-bearing declared.

    The joint count follows from the convention rather than being passed
    separately: a caller who could get them out of step is a caller who will.
    """
    _check(hand=hand, keypoints=keypoints, scale=scale, frame=frame)
    joints = KEYPOINTS[keypoints]
    return ChannelSpec(
        name=name,
        kind="vector" if joints == 1 else "tensor",
        shape=(3,) if joints == 1 else (joints, 3),
        dtype=dtype,
        units=METRE if scale == "metric" else None,
        frame=frame,
        rate_hz=rate_hz,
        semantics="ego.hand_keypoints",
        # Both invisible in the shape, and both change what every number means.
        # `mano` and `mediapipe` are 21 joints each in different orders; `metric`
        # and `unscaled` are the same floats with and without a meaning.
        discriminators=("keypoints", "scale", "hand"),
        metadata={"keypoints": keypoints, "scale": scale, "hand": hand, "joints": joints},
    )


def wrist_channel(
    name: str,
    *,
    hand: str = "right",
    scale: str = "unscaled",
    rotation_repr: str = "quat_wxyz",
    frame: str = "camera",
    rate_hz: float | None = None,
    dtype: str = "float32",
) -> ChannelSpec:
    """A 6-DoF wrist pose channel -- the thing a retargeter actually consumes.

    ``rotation_repr`` is borrowed from the manipulation vocabulary deliberately,
    rather than given its own ego-flavoured spelling. A wrist pose and a robot
    end-effector pose are the same kind of object, and the whole point of the
    retargeter is to relate them -- which it cannot do if the two planes disagree
    about what a quaternion is called.
    """
    _check(hand=hand, scale=scale, frame=frame)
    widths = {"quat_wxyz": 4, "quat_xyzw": 4, "euler_xyz": 3, "rotmat": 9, "axis_angle": 3}
    if rotation_repr not in widths:
        raise ValueError(
            f"unknown rotation_repr {rotation_repr!r}; expected one of {sorted(widths)}"
        )
    width = 3 + widths[rotation_repr]
    return ChannelSpec(
        name=name,
        kind="vector",
        shape=(width,),
        dtype=dtype,
        frame=frame,
        rate_hz=rate_hz,
        semantics="ego.wrist_pose",
        discriminators=("rotation_repr", "scale", "hand"),
        metadata={"rotation_repr": rotation_repr, "scale": scale, "hand": hand},
    )


def aperture_channel(
    name: str,
    *,
    hand: str = "right",
    scale: str = "unscaled",
    rate_hz: float | None = None,
    dtype: str = "float32",
) -> ChannelSpec:
    """Thumb-to-fingertip distance -- what becomes a gripper command.

    Carries its scale like everything else here, because the retargeting from
    aperture to gripper needs a hand-sized reference to be anything but a guess:
    "eight centimetres open" maps to a gripper, "0.31 of an unknown unit" does not.
    """
    _check(hand=hand, scale=scale)
    return ChannelSpec(
        name=name,
        kind="scalar",
        shape=(),
        dtype=dtype,
        units=METRE if scale == "metric" else None,
        rate_hz=rate_hz,
        semantics="ego.aperture",
        discriminators=("scale", "hand"),
        metadata={"scale": scale, "hand": hand},
    )


def ego_camera(
    name: str = "ego_rgb",
    *,
    height: int,
    width: int,
    rate_hz: float | None = None,
    channels: int = 3,
) -> ChannelSpec:
    """The head-mounted camera.

    Frame is ``camera`` and cannot be anything else, which is worth stating: an
    ego camera moves with the person, so a fixed-camera assumption anywhere
    downstream is wrong in a way that only shows up as poor performance.
    """
    return ChannelSpec(
        name=name,
        kind="image",
        shape=(height, width, channels),
        dtype="uint8",
        frame="camera",
        rate_hz=rate_hz,
        semantics="ego.rgb",
        metadata={"mounted": "head", "moving": True},
    )


def head_pose_channel(
    name: str = "head_pose",
    *,
    scale: str = "metric",
    rotation_repr: str = "quat_wxyz",
    rate_hz: float | None = None,
) -> ChannelSpec:
    """The device's own trajectory, from SLAM.

    Its presence is close to a proxy for the whole Tier-2 question: a capture
    that recovered a device trajectory generally recovered metric scale with it,
    and one that did not, did not.
    """
    _check(scale=scale)
    widths = {"quat_wxyz": 4, "quat_xyzw": 4, "euler_xyz": 3, "rotmat": 9}
    if rotation_repr not in widths:
        raise ValueError(f"unknown rotation_repr {rotation_repr!r}")
    return ChannelSpec(
        name=name,
        kind="vector",
        shape=(3 + widths[rotation_repr],),
        dtype="float32",
        frame="world",
        rate_hz=rate_hz,
        semantics="ego.head_pose",
        discriminators=("rotation_repr", "scale"),
        metadata={"rotation_repr": rotation_repr, "scale": scale},
    )


def gaze_channel(name: str = "gaze", *, rate_hz: float | None = None) -> ChannelSpec:
    """Where the person was looking, as a unit direction in the camera frame.

    Worth carrying even though no current policy consumes it: gaze precedes the
    hand by a few hundred milliseconds in reaching, which makes it the best
    available signal for segmenting a long recording into attempts.
    """
    return ChannelSpec(
        name=name,
        kind="vector",
        shape=(3,),
        dtype="float32",
        frame="camera",
        rate_hz=rate_hz,
        semantics="ego.gaze",
        dim_labels=("x", "y", "z"),
    )


def _check(
    *,
    hand: str | None = None,
    keypoints: str | None = None,
    scale: str | None = None,
    frame: str | None = None,
) -> None:
    if hand is not None and hand not in HANDS:
        raise ValueError(f"unknown hand {hand!r}; expected one of {list(HANDS)}")
    if keypoints is not None and keypoints not in KEYPOINTS:
        raise ValueError(
            f"unknown keypoint convention {keypoints!r}; expected one of {sorted(KEYPOINTS)}. "
            "These are not interchangeable: mano and mediapipe are both 21 joints "
            "in different orders, so getting it wrong attaches every finger to the "
            "wrong knuckle with no error anywhere"
        )
    if scale is not None and scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}")
    if frame is not None and frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; expected one of {list(FRAMES)}")


def describe(spec: ChannelSpec) -> str:
    """A one-line human summary of an ego channel, for a report.

    Exists because "hand_keypoints (21, 3) float32" tells a user nothing and
    "right hand, MANO joints, monocular estimate -- no real-world scale" tells
    them the thing that will limit what their data can be used for.
    """
    parts: list[str] = []
    metadata = spec.metadata
    if metadata.get("hand"):
        parts.append(f"{metadata['hand']} hand")
    if metadata.get("keypoints"):
        convention = metadata["keypoints"]
        parts.append("wrist only" if convention == "wrist_only" else f"{convention.upper()} joints")
    scale = scale_of(spec)
    if scale:
        parts.append(SCALES[scale] if scale != "metric" else "metres, calibrated capture")
    if spec.frame:
        parts.append(f"{spec.frame} frame")
    return ", ".join(parts) if parts else (spec.semantics or spec.name)


def sequence_of(specs: Sequence[ChannelSpec]) -> dict[str, list[str]]:
    """Group a schema's ego channels by what they say about scale.

    The shape a report wants: a dataset whose hands are unscaled and whose head
    pose is metric is a real and common situation -- the rig tracked itself and an
    estimator filled in the hands -- and it is worth showing as two rows rather
    than one verdict.
    """
    out: dict[str, list[str]] = {}
    for spec in specs:
        if not (spec.semantics or "").startswith("ego."):
            continue
        out.setdefault(scale_of(spec) or "undeclared", []).append(spec.name)
    return out
