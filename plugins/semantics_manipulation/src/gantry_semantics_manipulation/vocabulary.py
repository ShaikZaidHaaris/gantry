"""The tags themselves, plus builders that make a correct channel the easy one.

Each control mode becomes one Gantry semantics tag, so the resolver's existing
machinery does the work: a channel declaring ``action.joint_delta`` cannot be
bound to one wanting ``action.joint_pos``, and the refusal comes back as
``semantics.mismatch`` with both names, before anything moves.

Encoding the mode *in the tag* rather than as a side field is deliberate. A
separate attribute would need every consumer to remember to compare it, and the
one that forgets is the one that drives an arm to a displacement.
"""

from __future__ import annotations

from typing import Sequence

from gantry.spine import ChannelSpec, register_semantics, units

#: Control modes, from the Inspect Robots ``ControlMode`` literal.
CONTROL_MODES: dict[str, str] = {
    "joint_pos": "absolute joint positions",
    "joint_vel": "joint velocities",
    "joint_delta": "per-step change in joint positions",
    "eef_abs_pose": "absolute end-effector pose",
    "eef_delta_pose": "per-step change in end-effector pose",
    "eef_delta_pos": "per-step change in end-effector position only",
}

#: Modes that command a change rather than a target. The distinction the width
#: of the vector cannot express.
DELTA_MODES = frozenset({"joint_vel", "joint_delta", "eef_delta_pose", "eef_delta_pos"})

#: Rotation encodings and how many numbers each takes.
ROTATION_REPRS: dict[str, str] = {
    "none": "no rotation component",
    "quat_wxyz": "quaternion, scalar first",
    "quat_xyzw": "quaternion, scalar last",
    "rot6d": "6-D continuous rotation",
    "axis_angle": "axis-angle, 3-D",
    "euler_xyz": "intrinsic XYZ Euler angles",
}

ROTATION_WIDTHS: dict[str, int] = {
    "none": 0,
    "quat_wxyz": 4,
    "quat_xyzw": 4,
    "rot6d": 6,
    "axis_angle": 3,
    "euler_xyz": 3,
}

GRIPPER_KINDS: dict[str, str] = {
    "none": "no gripper dimension",
    "continuous": "gripper opening as a continuous fraction",
    "binary": "gripper open or closed",
}

#: Reference frames. Open, unlike the source vocabulary's closed literal: a
#: sensor-relative or object-relative frame is a real thing that a fixed list
#: would force someone to misdeclare as one of these three.
FRAMES: dict[str, str] = {
    "base": "the robot base",
    "world": "a fixed world origin",
    "camera": "a camera's optical frame",
}

#: Canonical proprioception keys and their units, aligned with the Inspect
#: Robots ``CANONICAL_STATE_UNITS`` table. There they are a convention stated in
#: a comment; here they are checked, because a rig reporting millimetres where
#: another reports metres passes every structural check ever written.
STATE_UNITS: dict[str, str] = {
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "eef_pos": "m",
    "eef_quat": "1",
    "gripper": "fraction",
    "gripper_width": "m",
}

_DIMENSIONS = {
    "joint_pos": units.ANGLE,
    "joint_vel": units.ANGLE / units.TIME,
    "eef_pos": units.LENGTH,
    "gripper_width": units.LENGTH,
}


def register_all() -> None:
    """Register every tag. Idempotent, and called on import."""
    for mode, description in CONTROL_MODES.items():
        register_semantics(f"action.{mode}", f"action: {description}")
    for key, description in STATE_UNITS.items():
        register_semantics(f"state.{key}", f"proprioception: {key}", _DIMENSIONS.get(key))
    register_semantics("state.eef_pose", "proprioception: end-effector position and rotation")


def control_mode_of(spec: ChannelSpec) -> str | None:
    """The control mode a channel declares, or ``None`` if it is not an action."""
    if spec.semantics and spec.semantics.startswith("action."):
        return spec.semantics.removeprefix("action.")
    return None


def is_absolute(spec: ChannelSpec) -> bool | None:
    """Whether this action commands a target rather than a displacement.

    ``None`` when the channel does not say. Guardrails and any consumer that
    integrates commands need this, and a default would be a guess about which
    way an arm moves.
    """
    mode = control_mode_of(spec)
    return None if mode is None else mode not in DELTA_MODES


def action_channel(
    name: str,
    control_mode: str,
    *,
    width: int,
    dim_labels: Sequence[str] | None = None,
    rotation_repr: str = "none",
    gripper: str = "none",
    frame: str = "base",
    rate_hz: float | None = None,
    dtype: str = "float32",
) -> ChannelSpec:
    """Build a fully described action channel.

    Every distinguishing property is a required or explicitly defaulted
    argument, and the width is cross-checked against the rotation encoding, so
    an 8-wide pose channel claiming a 4-wide quaternion is caught here rather
    than by whatever it is eventually handed to.
    """
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"unknown control mode {control_mode!r}; expected one of {sorted(CONTROL_MODES)}"
        )
    if rotation_repr not in ROTATION_REPRS:
        raise ValueError(
            f"unknown rotation_repr {rotation_repr!r}; expected one of {sorted(ROTATION_REPRS)}"
        )
    if gripper not in GRIPPER_KINDS:
        raise ValueError(f"unknown gripper {gripper!r}; expected one of {sorted(GRIPPER_KINDS)}")
    if dim_labels is not None and len(dim_labels) != width:
        raise ValueError(f"{len(dim_labels)} labels for a width of {width}")

    rotation_width = ROTATION_WIDTHS[rotation_repr]
    if control_mode.startswith("eef") and rotation_width:
        minimum = 3 + rotation_width
        if width < minimum:
            raise ValueError(
                f"{control_mode} with {rotation_repr} needs at least {minimum} dimensions "
                f"(3 position + {rotation_width} rotation), got {width}"
            )

    return ChannelSpec(
        name=name,
        kind="vector",
        shape=(width,),
        dtype=dtype,
        frame=frame,
        rate_hz=rate_hz,
        semantics=f"action.{control_mode}",
        dim_labels=tuple(dim_labels) if dim_labels is not None else None,
        # Both are load-bearing and neither is visible in the shape: two poses
        # differing only in rotation encoding are the same width, and two
        # grippers differing only in kind both take one number and disagree
        # about what 0.5 means.
        discriminators=("rotation_repr", "gripper"),
        metadata={"rotation_repr": rotation_repr, "gripper": gripper},
    )


def state_channel(
    name: str,
    key: str,
    *,
    width: int,
    dim_labels: Sequence[str] | None = None,
    frame: str = "base",
    rate_hz: float | None = None,
    units_override: str | None = None,
    dtype: str = "float32",
) -> ChannelSpec:
    """Build a proprioception channel with its canonical units already attached."""
    if key not in STATE_UNITS and key != "eef_pose":
        raise ValueError(f"unknown state key {key!r}; expected one of {sorted(STATE_UNITS)}")
    return ChannelSpec(
        name=name,
        kind="vector" if width > 1 else "scalar",
        shape=(width,) if width > 1 else (),
        dtype=dtype,
        units=units_override if units_override is not None else STATE_UNITS.get(key),
        frame=frame,
        rate_hz=rate_hz,
        semantics=f"state.{key}",
        dim_labels=tuple(dim_labels) if dim_labels is not None else None,
    )
