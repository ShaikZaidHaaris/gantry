"""What an SO-ARM101 pair is, described well enough that a wrong run refuses.

This is plane 2 for the real-hardware evaluator: **data, not code**. Nothing
here talks to a serial port. It exists so that the resolver can decide, before a
single servo moves, whether a policy's action space and this arm's action space
are the same quantity — and say precisely which of units, frame, normalization,
calibration convention, or joint order they disagree on when they are not.

Two machines, not one rig
-------------------------
A leader-follower pair is one operator station with **two descriptors**. The
follower is the machine under evaluation: it is commanded and it is scored. The
leader is a 6-DoF input device that is read and never commanded, so it declares
no action channels at all and the conformance kit says so out loud
(``embodiment.no_action``). That note is the correct description of a leader
arm, and it is why nothing can accidentally plan a rollout onto it.

Their joint channels carry **different frame tags** (``so101_leader_joints`` vs
``so101_follower_joints``) even though the numbers are the same width, the same
units and the same order. That is deliberate. Binding one directly to the other
comes back as ``frame.mismatch``, which forces teleoperation to go through
:class:`LeaderToFollower` — a named transform that states, in the run's
provenance, the one assumption teleop rests on: that both arms were calibrated
under the same convention. LeRobot issue #3758 is what happens when they were
not, and the whole point of making that assumption a declared transform is that
it then appears in the record instead of in the data.

The numbers are normalized, and that is a discriminator
-------------------------------------------------------
LeRobot reports and accepts SO-101 joints as ``float32[6]`` in ``RANGE_0_100``:
percent of each joint's *calibrated travel*, not radians. So these channels are
dimensionless, and this module registers its own meaning tags rather than
reusing ``state.joint_pos`` from ``gantry-semantics-manipulation`` — that tag is
pinned to the angle dimension, and declaring percent against it is a
dimensional lie the spine would (correctly) refuse. Percent-of-travel and
radians are genuinely different quantities; converting between them needs a
calibration, which is exactly what :class:`NormalizedToJointAngles` takes as an
argument and refuses to guess.

Because the mapping runs through a calibration, ``calibration_variant`` and
``normalization`` are declared as :attr:`~gantry.spine.ChannelSpec.discriminators`.
A corpus recorded under ``so101_old_calib`` (zero at horizontal extension)
and one recorded under ``so101_new_calib`` (zero at mid-range) are the same
dtype, the same width, the same units and the same joint order, and every
number in one is offset from the other. Nothing structural catches that. The
discriminator does, and it is the difference between "our SO-101 numbers" and
"anyone's SO-101 numbers".

What this hardware cannot do
----------------------------
Six STS3215 serial-bus servos on one Waveshare driver board, position control
only. There is **no force sensing and no torque sensing anywhere on either
arm**, so no wrench channel is declared, and none may be inferred from motor
current: current is not calibrated to force on this rig and a channel claiming
newtons would be an invention. The published kinematics carry two defects
recorded in :data:`KINEMATICS_DEFECTS`, one of which (no linear gripper joint)
means no jaw opening in metres can be derived from the sixth number by any
means available here.

Three declarations the analysis plane plans from
------------------------------------------------
These belong to the evaluator, not to this descriptor, but they are pinned here
in :data:`EVALUATOR_CAPABILITIES` so the two files cannot drift apart, and
because they are the claims that decide which analyses are legal:

``seedable = False``
    A seedable evaluator returns identical records for identical seeds, and a
    paired comparison rests on exactly that. A physical rig cannot: object
    placement is a human act with its own error, and no seed reaches it.
    Declaring ``True`` would let the resolver plan a paired analysis that is not
    valid. **The paired-seed power calculation in CORPUS-PROTOCOL.md therefore
    applies to the sim tier only** — on hardware the same effect needs the
    unpaired n.

``stage_events = False`` unless a pose source is configured
    The funnel needs object pose at each step. On hardware that is a perception
    rig which does not exist yet, and gripper stall gives a contact proxy for
    GRASP but says nothing about lift or place. An overstated ``True`` produces
    an empty funnel that reads as a finished analysis.

``closed_loop = True``, ``outcomes = True``
    The policy sees the consequence of its last action, and a human can always
    say what happened.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.embodiment import (
    CAP_RESETTABLE,
    EmbodimentDescriptor,
    Retargeter,
)
from gantry.errors import ConfigError
from gantry.serial import spec_from_dict, spec_to_dict
from gantry.spine import ChannelSpec, Measurement, Verdict, register_semantics, units

VERSION = "0.1.0.dev0"

# --------------------------------------------------------------------------
# the facts about the machine
# --------------------------------------------------------------------------

#: The six actuated degrees of freedom, in the exact order LeRobot writes them
#: into ``action`` and ``observation.state``. Spelled with the ``.pos`` suffix
#: the datasets carry, so a LeRobot connector's ``dim_labels`` compare equal to
#: these without a rename step: a rename is where an order swap hides.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: The gripper is the sixth DoF, not a flag riding alongside the arm. Named
#: rather than left as the literal 5, because "the last one" stops being true
#: the moment somebody describes a two-gripper variant.
GRIPPER_INDEX = 5

DOF = len(JOINT_NAMES)

#: Bus servos ship from the factory with ``id=1``. Assigning IDs therefore has
#: to happen one servo at a time on an otherwise empty bus, and a rig that was
#: provisioned any other way has two servos answering to the same address and
#: reports whichever replies first. Recorded because it is the failure that
#: looks like a wiring fault and is not.
PROVISIONING = {
    "servo": "Feetech STS3215",
    "bus": "single daisy-chained TTL serial bus per arm",
    "driver": "Waveshare Serial Bus Servo driver board over USB",
    "factory_id": 1,
    "id_assignment": (
        "one servo at a time on an otherwise empty bus; every servo ships as id=1 "
        "and duplicate IDs on a live bus produce silent, intermittent wrong reads"
    ),
    "force_sensing": False,
    "torque_sensing": False,
}

#: Normalization conventions LeRobot can record an SO-101 in, and the closed
#: interval each maps a joint's calibrated travel onto. Listed rather than
#: assumed: ``RANGE_0_100`` and ``RANGE_M100_100`` are the same dtype and the
#: same width, and a corpus recorded in one and replayed as the other drives
#: every joint to the wrong half of its travel.
NORMALIZATION_RANGES: dict[str, tuple[float, float]] = {
    "RANGE_0_100": (0.0, 100.0),
    "RANGE_M100_100": (-100.0, 100.0),
}

#: What this rig records. Stated as a default rather than a fact about SO-101s
#: in general, because it is a recording choice and the alternative is legal.
NORMALIZATION = "RANGE_0_100"

#: The calibration convention every number here is relative to. The older
#: convention puts zero at horizontal extension where this one puts it at
#: mid-range, which shifts the entire action space by a constant per joint.
#: Two labs quoting SO-101 results under different variants are quoting
#: different experiments, and neither log would show it.
CALIBRATION = "so101_new_calib"

#: The two channel names, as the LeRobot datasets spell them. Exported as
#: strings because the arm and the evaluator key dictionaries by them, and a
#: second spelling appearing in one of those modules is how an observation
#: quietly stops being recorded.
STATE = "observation.state"
ACTION = "action"

#: Where the kinematics live. A reference, never a parsed model.
KINEMATICS_REF = (
    "https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/"
    "so101_new_calib.urdf"
)

#: Known defects in the upstream asset, recorded because a descriptor that
#: points at a model without saying what the model omits invites downstream
#: code to trust it for things it cannot answer.
KINEMATICS_DEFECTS: tuple[Mapping[str, str], ...] = (
    {
        "id": "base_collision_meshes_removed",
        "what": (
            "collision meshes were removed from the base links upstream because they "
            "made simulation unstable"
        ),
        "consequence": (
            "any collision or reachability query near the base silently returns clear; "
            "a self-collision check against this URDF is not evidence of anything"
        ),
    },
    {
        "id": "gripper_linear_joint_absent",
        "what": (
            "the gripper's linear jaw joint is not represented in the current URDF/MJCF; "
            "only the rotary drive servo appears"
        ),
        "consequence": (
            "there is no kinematic path from gripper.pos to a jaw opening in metres, so "
            "no gripper_width channel may be declared and no grasp aperture metric "
            "computed from this model"
        ),
    },
)

#: Frames. Joint values are not expressed *in* a frame the way a position is,
#: but the tag still carries the one thing that matters here: which physical
#: arm's joint space these numbers belong to. Two arms with identical channels
#: and different frames refuse to bind, which is the intended outcome.
FOLLOWER_JOINT_FRAME = "so101_follower_joints"
LEADER_JOINT_FRAME = "so101_leader_joints"

CONTROL_HZ = 30.0

#: A budget, derived from the bus, and explicitly not a measurement of this rig.
#: Six position reads plus six position writes over one 1 Mbaud TTL bus per
#: control tick. Anyone quoting latency in a result should replace this with a
#: measured value; a Measurement with ``n=None`` is how it says it has none.
EXPECTED_LATENCY = Measurement(
    value=0.010,
    n=None,
    units="s",
    method=(
        "declared budget: 6 sync position reads + 6 position writes on one 1 Mbaud "
        "serial bus; never measured on this rig"
    ),
    detail={"control_period_s": 1.0 / CONTROL_HZ, "budget_fraction_of_period": 0.30},
)

#: How a policy's action chunk is spent. Hardware consumes exactly one action
#: per tick, so a chunk of k is k ticks of open loop during which the arm cannot
#: react to anything. At 30 Hz a 50-step chunk is 1.67 s of that, which on a
#: physical arm is long enough to finish pushing an object off the table. The
#: ceiling is declared here so the number is in the record rather than in
#: somebody's default.
CHUNKING = {
    "actions_per_tick": 1,
    "chunk_execution": "open_loop",
    "seconds_per_step": 1.0 / CONTROL_HZ,
    "max_open_loop_steps": 15,
    "max_open_loop_seconds": 15 / CONTROL_HZ,
    "note": (
        "execute is a result-changing variable and belongs in the RunRecord protocol, "
        "never in a default"
    ),
}

#: Per-tick command clamp, in units of percent-of-travel. LeRobot's
#: ``max_relative_target`` defaults to ``None``, i.e. no clamp at all, and an
#: unclamped bad action commands the follower across its whole range in one tick
#: and drives it into the table. 8 %/tick is ~240 %/s: roughly three times brisk
#: human teleop, so it does not shape normal motion, while bounding a full-range
#: slam to ~0.4 s — long enough for an operator to reach the stop. A safety
#: default, not a physical constant, which is why it is declared and not baked in.
DEFAULT_MAX_RELATIVE_TARGET = 8.0

#: Pinned here so the evaluator cannot quietly disagree with the descriptor.
#: See the module docstring for why each one is what it is.
EVALUATOR_CAPABILITIES = {
    "seedable": False,
    "stage_events": False,
    "outcomes": True,
    "closed_loop": True,
    "stage_events_note": (
        "True only when a pose source is explicitly configured; the gripper-stall "
        "contact proxy yields GRASP and nothing above it"
    ),
    "seedable_note": (
        "a physical scene reset is a human act with its own error, so the paired-seed "
        "power calculation in CORPUS-PROTOCOL.md applies to the sim tier only"
    ),
}

# --------------------------------------------------------------------------
# meaning tags
# --------------------------------------------------------------------------

#: Namespaced, because these tags mean "percent of *this* family's calibrated
#: travel" and a generic name would invite a robot with different travel to
#: register the same word for something else. The registry refuses redefinition,
#: so claiming a general name here would also lock everyone else out of it.
SEMANTICS_STATE = "so101.joint_position_normalized"
SEMANTICS_ACTION = "so101.action.joint_position_normalized"


def register_all() -> None:
    """Register this embodiment's meaning tags. Idempotent, called on import.

    Both are pinned dimensionless. That is the check that matters: it makes a
    channel claiming these semantics *and* units of radians a hard refusal
    rather than a plausible mislabel.
    """
    register_semantics(
        SEMANTICS_STATE,
        "measured joint position as percent of that joint's calibrated travel",
        units.DIMENSIONLESS,
    )
    register_semantics(
        SEMANTICS_ACTION,
        (
            "commanded absolute joint position as percent of that joint's calibrated "
            "travel; absolute target, never a displacement"
        ),
        units.DIMENSIONLESS,
    )


register_all()


# --------------------------------------------------------------------------
# channel builders
# --------------------------------------------------------------------------


def _joint_metadata(normalization: str, calibration_variant: str) -> dict[str, Any]:
    low, high = NORMALIZATION_RANGES[normalization]
    return {
        "normalization": normalization,
        "range_min": low,
        "range_max": high,
        "calibration_variant": calibration_variant,
        # Absolute, not delta. Same width, same units, same dimension as a delta
        # command; executing one as the other sends the arm to a displacement.
        "control_mode": "joint_pos",
        # Continuous, not binary. Both take one number and disagree about what
        # 50 means: here it is half-open, not "closed".
        "gripper": "continuous",
        "gripper_index": GRIPPER_INDEX,
        "gripper_dim": JOINT_NAMES[GRIPPER_INDEX],
    }


#: Metadata keys that must agree before two joint channels may bind. Every one
#: of them is invisible in dtype, width and joint order, and wrong in a way that
#: still produces motion.
JOINT_DISCRIMINATORS = ("normalization", "calibration_variant", "control_mode", "gripper")


def joint_channel(
    name: str,
    *,
    frame: str,
    semantics: str,
    normalization: str = NORMALIZATION,
    calibration_variant: str = CALIBRATION,
    rate_hz: float = CONTROL_HZ,
    extra: Mapping[str, Any] | None = None,
) -> ChannelSpec:
    """One fully described 6-vector of normalized joint values.

    Refuses an unknown normalization rather than defaulting, because the default
    would be the recording convention of whoever wrote this file.
    """
    if normalization not in NORMALIZATION_RANGES:
        raise ConfigError(
            f"unknown normalization {normalization!r}; known: {sorted(NORMALIZATION_RANGES)}"
        )
    metadata = _joint_metadata(normalization, calibration_variant)
    metadata.update(extra or {})
    return ChannelSpec(
        name=name,
        kind="vector",
        shape=(DOF,),
        dtype="float32",
        # Percent of calibrated travel. Dimensionless, and deliberately not
        # "fraction": the two differ by a factor of 100 and the spine reports
        # that as units.scale rather than letting it through.
        units="percent",
        frame=frame,
        rate_hz=rate_hz,
        semantics=semantics,
        dim_labels=JOINT_NAMES,
        discriminators=JOINT_DISCRIMINATORS,
        metadata=metadata,
    )


def camera_channel(
    name: str,
    *,
    frame: str,
    mount: str,
    height: int = 480,
    width: int = 640,
    rate_hz: float = CONTROL_HZ,
    policy_visible: bool = True,
    intrinsics_ref: str | None = None,
    color_space: str = "rgb8",
) -> ChannelSpec:
    """One camera stream.

    No ``dim_labels``: labels address one element each, and 921,600 of them
    would be noise rather than description. The description that matters for an
    image is the frame, the mount, and whether it is calibrated.

    ``intrinsics_ref`` defaults to ``None`` and that is the honest state of this
    rig. Without intrinsics, no pixel measurement becomes a metric quantity —
    which is the concrete reason ``stage_events`` is False by default rather
    than a policy decision made elsewhere.
    """
    # The frame size goes straight into ``shape``, which is what every consumer
    # allocates against and what the spine binds on. Unchecked, ``height=nan``
    # produces the shape ``(nan, 640, 3)`` and passes strict conformance: a
    # channel describing an image with no number of rows, declared as though
    # somebody had measured it. Non-integers are refused too rather than
    # truncated, because a stream is a whole number of pixels tall or it is not
    # this stream.
    for label, size in (("height", height), ("width", width)):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ConfigError(
                f"camera {name!r} declares {label}={size!r}; a frame size must be a "
                "positive whole number of pixels, and this one goes into the channel's "
                "shape where every consumer allocates against it"
            )
    return ChannelSpec(
        name=name,
        kind="image",
        shape=(height, width, 3),
        dtype="uint8",
        # 8-bit intensity counts. Dimensionless, and declared rather than left
        # blank so an RGB stream cannot be bound to a depth stream in metres.
        units="count",
        frame=frame,
        rate_hz=rate_hz,
        semantics="observation",
        discriminators=("policy_visible", "color_space"),
        metadata={
            "policy_visible": policy_visible,
            "color_space": color_space,
            "mount": mount,
            "intrinsics_ref": intrinsics_ref,
            "calibrated": intrinsics_ref is not None,
        },
    )


#: The two streams a policy may see. Both are named under ``observation.images.``
#: because that is what LeRobot writes and what training pipelines glob.
FRONT_CAMERA = "observation.images.front"
WRIST_CAMERA = "observation.images.wrist"

#: A referee stream is named *outside* the ``observation.`` namespace on
#: purpose. A pipeline that collects ``observation.images.*`` would otherwise
#: hand the ground-truth view straight to the policy, and a channel the policy
#: can see is a channel it can exploit. The name is the guard.
REFEREE_CAMERA = "referee.images.overhead"


def referee_camera_channel(
    name: str = REFEREE_CAMERA,
    *,
    height: int = 480,
    width: int = 640,
    rate_hz: float = CONTROL_HZ,
    intrinsics_ref: str | None = None,
) -> ChannelSpec:
    """A ground-truth stream, refused if it is named where a policy would find it."""
    if name.startswith("observation."):
        raise ConfigError(
            f"referee camera {name!r} is in the observation namespace; rename it "
            "(e.g. 'referee.images.*') so no training pipeline can glob it into the "
            "policy's inputs"
        )
    return camera_channel(
        name,
        frame="so101_referee_cam_optical",
        mount="fixed, off-rig, viewing the workspace",
        height=height,
        width=width,
        rate_hz=rate_hz,
        policy_visible=False,
        intrinsics_ref=intrinsics_ref,
    )


# --------------------------------------------------------------------------
# the descriptors
# --------------------------------------------------------------------------

_NOTES_FOLLOWER = """\
SO-ARM101 follower: 6x Feetech STS3215 on one Waveshare serial-bus driver board,
position control only. No force or torque sensing exists anywhere on this arm,
so no wrench channel is declared and none may be inferred from motor current.

Joint order is the LeRobot order and the gripper is the sixth DoF, continuous,
commanded as an absolute position in percent of calibrated travel. It is not a
binary open/close flag riding alongside a 5-DoF arm.

Every number is relative to the so101_new_calib convention. Under the older
convention zero sits at horizontal extension instead of mid-range, which offsets
the whole action space by a per-joint constant; results are not comparable
across the two and nothing structural would show it, so the variant is a channel
discriminator.

max_relative_target is declared (8 percent-of-travel per tick). LeRobot's
default is None, meaning no clamp, and one bad action then commands the arm
across its full range in a single tick.

Resetting this arm to a home pose is repeatable. Resetting the *scene* is not:
object placement is a human act, which is why the evaluator declares
seedable=False.

The published URDF has two known defects; see metadata.kinematics_defects.
"""

_NOTES_LEADER = """\
SO-ARM101 leader: the same six STS3215 servos, run passively and read at 30 Hz
as a 6-DoF input device. It declares no action channels because it is never
commanded — the conformance kit reports embodiment.no_action, which is the
correct description of a teleoperator, not a gap to be filled.

There is no force feedback: nothing here can render contact back to the
operator's hand, so an operator has no haptic evidence of a collision and the
recorded corpus contains none either.

Its joint frame differs from the follower's so the two cannot bind directly.
Teleoperation goes through the LeaderToFollower retargeter, which states the
calibration assumption it rests on into the run's provenance.
"""


def so101_embodiment(
    *,
    version: str = "1.0",
    # The identity is "so101_follower", not "so101": the leader is an SO-101
    # too, and a record naming the family rather than the role cannot say which
    # arm produced it.
    name: str = "so101_follower",
    normalization: str = NORMALIZATION,
    calibration_variant: str = CALIBRATION,
    cameras: Sequence[ChannelSpec] | None = None,
    referee_camera: ChannelSpec | None = None,
    max_relative_target: float = DEFAULT_MAX_RELATIVE_TARGET,
) -> EmbodimentDescriptor:
    """The arm under evaluation: commanded, scored, and physically present.

    ``cameras`` defaults to the two policy streams this rig records. A referee
    camera, if there is one, is passed separately and deliberately never lands
    in ``state``: ``state`` is what a policy is offered, and the ground-truth
    view has to be unreachable by construction rather than by convention.
    """
    # `<= 0` alone is blind to NaN and to infinity, and both then land in the
    # action channel's metadata as this machine's advertised per-tick clamp —
    # published, conformant, and describing a limit the follower never ran under.
    # That is the defect the agreement guard exists to catch, reached from the
    # other side: there it is a clamp of 5.0 recorded as 8.0, here it is a clamp
    # of nothing recorded as a number. Same wording as transport.SafetyLimits,
    # which has always refused this at the layer that executes it.
    if not math.isfinite(max_relative_target) or max_relative_target <= 0:
        raise ConfigError(
            f"max_relative_target={max_relative_target} does not bound anything; "
            "the whole point of the clamp is that it is a positive per-tick limit, and "
            "a value that is zero, negative, infinite or NaN is not one. This number is "
            "published as the limit every episode was recorded under"
        )
    policy_cameras = tuple(
        cameras
        if cameras is not None
        else (
            camera_channel(
                FRONT_CAMERA,
                frame="so101_front_cam_optical",
                mount="fixed, facing the workspace from the operator side",
            ),
            camera_channel(
                WRIST_CAMERA,
                frame="so101_follower_wrist_cam_optical",
                mount="rigid to the follower's wrist_roll link, moves with the arm",
            ),
        )
    )
    leaked = [
        spec.name
        for spec in policy_cameras
        if spec.metadata.get("policy_visible") is not True
    ]
    if leaked:
        raise ConfigError(
            f"camera(s) {leaked} are in the policy observation set but declare "
            "policy_visible != True; a stream is either offered to the policy or it "
            "is referee ground truth, and it cannot be described as both"
        )

    state_channel = joint_channel(
        STATE,
        frame=FOLLOWER_JOINT_FRAME,
        semantics=SEMANTICS_STATE,
        normalization=normalization,
        calibration_variant=calibration_variant,
    )
    action_channel = joint_channel(
        ACTION,
        frame=FOLLOWER_JOINT_FRAME,
        semantics=SEMANTICS_ACTION,
        normalization=normalization,
        calibration_variant=calibration_variant,
        extra={"max_relative_target": max_relative_target},
    )

    return EmbodimentDescriptor(
        name=name,
        version=version,
        state=(state_channel, *policy_cameras),
        action=(action_channel,),
        control_hz=CONTROL_HZ,
        kinematics=KINEMATICS_REF,
        notes=_NOTES_FOLLOWER,
        # Resettable: the arm goes to a home pose repeatably. Not seedable and
        # not simulated, so neither tag appears — an absent capability is a
        # claim not made, and the resolver reads it as one.
        capabilities=frozenset({CAP_RESETTABLE}),
        metadata={
            "provisioning": dict(PROVISIONING),
            "calibration": {
                "variant": calibration_variant,
                "normalization": normalization,
                "range": list(NORMALIZATION_RANGES[normalization]),
                "why_it_matters": (
                    "so101_old_calib zeroes at horizontal extension, so101_new_calib at "
                    "mid-range; the same recorded numbers describe different poses under "
                    "the two and no structural check can tell"
                ),
                "known_issue": (
                    "LeRobot #3758: a ~17 deg joint calibration offset produced ~6 cm of "
                    "gripper error, was invisible during training because the policy "
                    "learns the biased mapping, and the loss looked healthy"
                ),
            },
            "kinematics_ref": KINEMATICS_REF,
            "kinematics_defects": [dict(defect) for defect in KINEMATICS_DEFECTS],
            "control": {
                "rate_hz": CONTROL_HZ,
                "expected_latency": EXPECTED_LATENCY.as_dict(),
                "chunking": dict(CHUNKING),
            },
            "safety": {
                "max_relative_target": max_relative_target,
                "units": "percent of calibrated travel per control tick",
                "lerobot_default": None,
                "why": (
                    "LeRobot leaves this None, i.e. unclamped; one bad action then "
                    "commands the follower across its full range in a single tick and "
                    "into the table"
                ),
            },
            "sensors": {
                "policy_visible": [spec.name for spec in policy_cameras],
                # Stored as data, outside `state`, so nothing that iterates the
                # observation channels can reach it. Read it back with
                # referee_channels().
                "referee": (
                    [spec_to_dict(referee_camera)] if referee_camera is not None else []
                ),
                "intrinsics": (
                    "uncalibrated on this rig; without intrinsics no pixel measurement "
                    "becomes a metric quantity, which is why stage_events is False"
                ),
            },
            "evaluator_capabilities": dict(EVALUATOR_CAPABILITIES),
        },
    )


def so101_leader(
    *,
    version: str = "1.0",
    name: str = "so101_leader",
    normalization: str = NORMALIZATION,
    calibration_variant: str = CALIBRATION,
) -> EmbodimentDescriptor:
    """The teleoperator arm: read at 30 Hz, never commanded.

    Declaring no action channels is the point, not an omission. The conformance
    kit notes ``embodiment.no_action`` and passes, which is exactly right: the
    leader is describable and uncommandable, and a run that tried to drive it
    would be refused before anything moved.
    """
    return EmbodimentDescriptor(
        name=name,
        version=version,
        state=(
            joint_channel(
                STATE,
                frame=LEADER_JOINT_FRAME,
                semantics=SEMANTICS_STATE,
                normalization=normalization,
                calibration_variant=calibration_variant,
            ),
        ),
        action=(),
        control_hz=CONTROL_HZ,
        kinematics=KINEMATICS_REF,
        notes=_NOTES_LEADER,
        # Not resettable: a human moves this arm. There is nothing to command.
        capabilities=frozenset(),
        metadata={
            "provisioning": dict(PROVISIONING),
            "role": "teleoperator input device",
            "force_feedback": False,
            "calibration": {
                "variant": calibration_variant,
                "normalization": normalization,
            },
            "kinematics_ref": KINEMATICS_REF,
            "kinematics_defects": [dict(defect) for defect in KINEMATICS_DEFECTS],
        },
    )


def policy_observation_channels(
    embodiment: EmbodimentDescriptor,
) -> tuple[ChannelSpec, ...]:
    """Everything a policy may be shown. By construction, all of ``state``."""
    return tuple(embodiment.state)


def referee_channels(embodiment: EmbodimentDescriptor) -> tuple[ChannelSpec, ...]:
    """Ground-truth streams, which live outside the observation set.

    Kept as serialized data in metadata rather than as channels, so the only way
    to reach them is to ask for them by this name. Anything that walks ``state``
    — a policy adapter, an observation builder, a training pipeline — cannot
    stumble into them.
    """
    sensors = embodiment.metadata.get("sensors") or {}
    return tuple(spec_from_dict(payload) for payload in sensors.get("referee", ()))


# --------------------------------------------------------------------------
# retargeters
# --------------------------------------------------------------------------
#
# Three, and no more. What is deliberately absent and why:
#
#   * Dropping the gripper to feed a 5-DoF consumer is already
#     ``gantry_retargeters_core.DropDimensions(["gripper.pos"])``, which names
#     what it drops. Reimplementing it here would be a second opinion about the
#     same transform.
#   * Joint angles to an end-effector pose needs forward kinematics through a
#     URDF whose gripper linkage is missing (KINEMATICS_DEFECTS), so the tool
#     frame it would produce is not the frame anyone means by "gripper". A
#     nearly-right EE pose is worse than none: every spatial metric downstream
#     would be biased by a constant nobody can see.
#   * Absolute to delta is a temporal transform over any absolute channel, not a
#     fact about this arm, and belongs with the generic adapters.


def _check_joint_source(source: ChannelSpec, expected_frame: str | None = None) -> Verdict:
    """Is this actually a normalized SO-101 joint vector?"""
    if source.dim_labels != JOINT_NAMES:
        return Verdict.no(
            "retarget.not_so101_joints",
            f"{source.name!r} is labelled {list(source.dim_labels or ())}, not the "
            "SO-101 joint order",
            hint=f"expected {list(JOINT_NAMES)}",
        )
    if source.metadata.get("normalization") not in NORMALIZATION_RANGES:
        return Verdict.no(
            "retarget.no_normalization",
            f"{source.name!r} declares no known normalization, so its numbers cannot "
            "be placed on a travel range",
            hint=f"set metadata['normalization'] to one of {sorted(NORMALIZATION_RANGES)}",
        )
    if expected_frame is not None and source.frame != expected_frame:
        return Verdict.no(
            "frame.mismatch",
            f"{source.name!r} is in frame {source.frame!r}, this retargeter reads "
            f"{expected_frame!r}",
        )
    return Verdict.yes()


def _in_range(values: np.ndarray, spec: ChannelSpec, who: str) -> None:
    """Refuse out-of-travel numbers instead of clamping them.

    Clamping is the coercion this framework exists to prevent: it turns a
    calibration bug or a diverged policy into motion that looks deliberate.
    """
    low = float(spec.metadata.get("range_min", 0.0))
    high = float(spec.metadata.get("range_max", 100.0))
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != DOF:
        raise ConfigError(f"{who}: expected (steps, {DOF}), got {array.shape}")
    # NaN is the one bad value a range test cannot see: it compares False against
    # both bounds, so `< low | > high` catches +/-inf and lets NaN straight
    # through. A NaN is not a position at one end of travel, it is a number
    # naming no pose at all. ``SO101Arm`` does refuse non-finite values before
    # they reach the bus, but two things are lost by leaving it to that: the
    # refusal there names the arm rather than the transform that produced the
    # NaN, and a converted trajectory that is never executed — angles compared
    # against a simulated run, a corpus being written — never passes an arm at
    # all and would be refused nowhere. Same check, same wording, as bimanual's
    # ``_in_travel``, so the two modules do not disagree about the same value.
    finite = np.isfinite(array)
    if not finite.all():
        first = np.argwhere(~finite)[0]
        step, column = int(first[0]), int(first[1])
        raise ConfigError(
            f"{who}: {spec.name!r} step {step} has {JOINT_NAMES[column]}="
            f"{array[step, column]:g}, which is not a finite position; refusing rather "
            "than passing it on, because a non-finite command names no pose and a "
            "refusal here can still say which transform produced it"
        )
    bad = np.argwhere((array < low) | (array > high))
    if bad.size:
        step, column = int(bad[0][0]), int(bad[0][1])
        label = JOINT_NAMES[column]
        raise ConfigError(
            f"{who}: {spec.name!r} step {step} has {label}={array[step, column]:g}, "
            f"outside the declared travel [{low:g}, {high:g}]; "
            "refusing rather than clamping, because a clamped bad command still moves "
            "the arm and reads as intended motion"
        )


class LeaderToFollower(Retargeter):
    """Leader joint readings, as an absolute follower command.

    Numerically an identity, which is precisely why it has to be a named
    transform rather than a direct binding. The two arms' channels differ only
    in ``frame``, so the spine refuses to bind them and forces this object into
    the adapter chain — where its loss statement lands in the run's provenance
    and tells a later reader what teleoperation assumed.
    """

    @property
    def name(self) -> str:
        return "so101-leader-to-follower"

    @property
    def version(self) -> str:
        return VERSION

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        checks = [
            _check_joint_source(source, LEADER_JOINT_FRAME),
            _check_joint_source(target, FOLLOWER_JOINT_FRAME),
        ]
        if source.metadata.get("normalization") != target.metadata.get("normalization"):
            checks.append(
                Verdict.no(
                    "retarget.normalization",
                    f"leader is {source.metadata.get('normalization')!r} and follower is "
                    f"{target.metadata.get('normalization')!r}",
                    hint="the same number means a different pose on the two ranges",
                )
            )
        if target.semantics != SEMANTICS_ACTION:
            checks.append(
                Verdict.no(
                    "retarget.not_an_action",
                    f"{target.name!r} is {target.semantics!r}; this produces an absolute "
                    f"follower command ({SEMANTICS_ACTION})",
                )
            )
        return Verdict.all(checks)

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        return (
            "assumes both arms were calibrated under "
            f"{source.metadata.get('calibration_variant')!r} and "
            f"{target.metadata.get('calibration_variant')!r} respectively and that those "
            "calibrations agree; a residual offset between them is invisible here and "
            "lands in the corpus as a constant Cartesian bias (LeRobot #3758: ~17 deg "
            "became ~6 cm of gripper error with a healthy training loss)",
            "the two grippers' mechanical travel is normalized away, so equal percentages "
            "are equal fractions of each jaw's own range and not equal apertures",
            "no force feedback exists on the leader, so contact the follower experiences "
            "is never reflected in the recorded leader trajectory",
        )

    def apply(
        self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != DOF:
            raise ConfigError(
                f"{self.name}: expected (steps, {DOF}), got {array.shape}"
            )
        _in_range(array, source, self.name)
        # A copy, not the caller's buffer: a follower command that aliases the
        # leader's readings turns any later in-place edit into a silent retro-
        # active change to what was commanded.
        return array.astype(target.dtype, copy=True)


class _Calibrated(Retargeter):
    """Shared machinery for the two travel-limit conversions.

    The per-joint travel is an argument, never a default. Every SO-101 has
    different limits because they are read off *that* arm during calibration,
    and a built-in table would be one lab's arm silently standing in for
    everyone's.
    """

    def __init__(self, travel_deg: Mapping[str, tuple[float, float]]):
        missing = [label for label in JOINT_NAMES if label not in travel_deg]
        if missing:
            raise ConfigError(
                f"no calibrated travel given for {missing}; all {DOF} joints are "
                "required, because a partially calibrated conversion produces angles "
                "for some joints and invented numbers for the rest"
            )
        # Before the span test, because the span test cannot see a NaN: `nan ==
        # nan` is False, so a travel of (nan, nan) — the most degenerate entry
        # there is — walks straight through the check written to catch degenerate
        # entries. What follows is worse than a refusal: `_in_range` validates
        # the *input*, and this NaN is manufactured downstream of it, so
        # NormalizedToJointAngles turns clean, in-travel numbers into NaN degrees
        # on that joint and declares them converted. Infinity is refused for the
        # same reason and has no legitimate reading here: a joint whose travel
        # ends nowhere is not a calibration anyone measured.
        unusable = [
            label
            for label in JOINT_NAMES
            if not all(math.isfinite(bound) for bound in travel_deg[label])
        ]
        if unusable:
            raise ConfigError(
                f"the calibrated travel for {unusable} is not finite; a travel limit that "
                "is NaN or infinite is not a measurement of any arm, and every angle "
                "converted through it would be non-finite while the input was in range"
            )
        degenerate = [
            label for label in JOINT_NAMES if travel_deg[label][0] == travel_deg[label][1]
        ]
        if degenerate:
            raise ConfigError(
                f"travel for {degenerate} has zero span; the conversion would divide by "
                "zero or map every command onto one angle"
            )
        self._travel = {label: tuple(map(float, travel_deg[label])) for label in JOINT_NAMES}

    @property
    def version(self) -> str:
        return VERSION

    @property
    def _low(self) -> np.ndarray:
        return np.array([self._travel[label][0] for label in JOINT_NAMES], dtype=float)

    @property
    def _high(self) -> np.ndarray:
        return np.array([self._travel[label][1] for label in JOINT_NAMES], dtype=float)

    def _travel_notes(self) -> tuple[str, ...]:
        return (
            "the result is only valid for the calibration these travel limits were read "
            "from; another SO-101 whose servo horns were indexed differently returns a "
            "different angle for the identical normalized number, and no check "
            "downstream can see it (LeRobot #3758)",
            "the sixth value is the gripper drive servo's shaft angle, not a jaw opening: "
            "the published SO-101 URDF carries no linear gripper joint, so no aperture in "
            "metres can be derived from it",
            "the STS3215 reports position as one of 4096 counts per revolution, so ~0.088 "
            "deg of quantisation is present in the input and is not recovered by widening "
            "the dtype",
        )


class NormalizedToJointAngles(_Calibrated):
    """Percent-of-travel to degrees, for anything that needs the URDF's units.

    The direction used to compare a real trajectory against a simulated one, or
    to run forward kinematics. Same width, so nothing is dropped — the cost is
    entirely in what the numbers are *conditional on*, which is why the losses
    are stated anyway.
    """

    @property
    def name(self) -> str:
        return "so101-normalized-to-joint-angles"

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        checks = [_check_joint_source(source)]
        if target.width != DOF:
            checks.append(
                Verdict.no(
                    "retarget.width",
                    f"{target.name!r} is {target.width} wide; this produces {DOF} angles",
                )
            )
        if target.units is None:
            checks.append(
                Verdict.no(
                    "retarget.target_units",
                    f"{target.name!r} declares no units, so there is nothing to convert to",
                    hint="declare 'deg' or 'rad'",
                )
            )
        elif units.parse(target.units).dimension != units.ANGLE:
            checks.append(
                Verdict.no(
                    "units.dimension",
                    f"{target.name!r} is in {target.units!r}, which is not an angle",
                )
            )
        if target.dim_labels is not None and target.dim_labels != JOINT_NAMES:
            checks.append(
                Verdict.no(
                    "dim_labels.mismatch",
                    f"{target.name!r} labels {list(target.dim_labels)}, this emits "
                    f"{list(JOINT_NAMES)} in that order",
                )
            )
        return Verdict.all(checks)

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        return self._travel_notes()

    def apply(
        self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != DOF:
            raise ConfigError(f"{self.name}: expected (steps, {DOF}), got {array.shape}")
        _in_range(array, source, self.name)
        low = float(source.metadata.get("range_min", 0.0))
        high = float(source.metadata.get("range_max", 100.0))
        fraction = (array - low) / (high - low)
        degrees = self._low + fraction * (self._high - self._low)
        factor = units.conversion_factor("deg", target.units)
        return (degrees * factor).astype(target.dtype, copy=False)


class JointAnglesToNormalized(_Calibrated):
    """Degrees to percent-of-travel, for commanding the arm from a joint-space policy.

    The direction that moves hardware, so it refuses rather than clamps. An
    angle outside the calibrated travel is a command the arm cannot reach; a
    clamped version of it is a different command that still executes, and the
    record would show the clamped one as though it were what the policy asked
    for.
    """

    @property
    def name(self) -> str:
        return "so101-joint-angles-to-normalized"

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        checks = [_check_joint_source(target)]
        if source.width != DOF:
            checks.append(
                Verdict.no(
                    "retarget.width",
                    f"{source.name!r} is {source.width} wide; this reads {DOF} angles",
                )
            )
        if source.units is None:
            checks.append(
                Verdict.no(
                    "retarget.source_units",
                    f"{source.name!r} declares no units, so its numbers cannot be read "
                    "as angles",
                    hint="declare 'deg' or 'rad'",
                )
            )
        elif units.parse(source.units).dimension != units.ANGLE:
            checks.append(
                Verdict.no(
                    "units.dimension",
                    f"{source.name!r} is in {source.units!r}, which is not an angle",
                )
            )
        if source.dim_labels is not None and source.dim_labels != JOINT_NAMES:
            checks.append(
                Verdict.no(
                    "dim_labels.mismatch",
                    f"{source.name!r} labels {list(source.dim_labels)}, this reads "
                    f"{list(JOINT_NAMES)} in that order",
                )
            )
        return Verdict.all(checks)

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        return self._travel_notes() + (
            "angles outside the calibrated travel have no normalized representation and "
            "are refused, so a policy trained in a wider joint space loses exactly those "
            "commands rather than having them silently pulled to the nearest limit",
        )

    def apply(
        self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != DOF:
            raise ConfigError(f"{self.name}: expected (steps, {DOF}), got {array.shape}")
        degrees = array * units.conversion_factor(source.units, "deg")
        span = self._high - self._low
        fraction = (degrees - self._low) / span
        low = float(target.metadata.get("range_min", 0.0))
        high = float(target.metadata.get("range_max", 100.0))
        normalized = low + fraction * (high - low)
        # Checked against the *target*'s declared range, which is the range the
        # arm will actually be driven over.
        _in_range(normalized, target, self.name)
        return normalized.astype(target.dtype, copy=False)


__all__ = [
    "ACTION",
    "CALIBRATION",
    "CHUNKING",
    "CONTROL_HZ",
    "DEFAULT_MAX_RELATIVE_TARGET",
    "DOF",
    "EVALUATOR_CAPABILITIES",
    "EXPECTED_LATENCY",
    "FOLLOWER_JOINT_FRAME",
    "FRONT_CAMERA",
    "GRIPPER_INDEX",
    "JOINT_DISCRIMINATORS",
    "JOINT_NAMES",
    "KINEMATICS_DEFECTS",
    "KINEMATICS_REF",
    "LEADER_JOINT_FRAME",
    "NORMALIZATION",
    "NORMALIZATION_RANGES",
    "PROVISIONING",
    "REFEREE_CAMERA",
    "SEMANTICS_ACTION",
    "SEMANTICS_STATE",
    "STATE",
    "VERSION",
    "WRIST_CAMERA",
    "JointAnglesToNormalized",
    "LeaderToFollower",
    "NormalizedToJointAngles",
    "camera_channel",
    "joint_channel",
    "policy_observation_channels",
    "referee_camera_channel",
    "referee_channels",
    "register_all",
    "so101_embodiment",
    "so101_leader",
]
