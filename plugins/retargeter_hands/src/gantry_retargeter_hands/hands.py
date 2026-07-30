"""A person's hand as an arm's command, and an honest account of what that costs.

This is the load-bearing step of the whole ego pipeline. Everything before it is
description; everything after it is a normal Gantry run. If this transform is
wrong, every number downstream is wrong in a way that looks exactly like the
answer the product is trying to measure — a policy that did not learn much.

Where it stops
--------------
It produces an **end-effector pose and a gripper command**, and it refuses to
produce joint positions. Going from a pose to joint angles is inverse kinematics,
which needs that specific arm's link lengths, joint limits, and a choice of
elbow configuration when several solutions exist. This plugin has none of those
and could not check them if it did.

That refusal has a real consequence and it should be stated rather than buried: a
π₀.₅ server under the ALOHA config expects *joint positions*, so ego data cannot
reach it through this retargeter alone. Either the policy is trained on an
end-effector action space, or somebody who owns the arm's kinematics puts an IK
step between. Both are fine. Silently emitting something joint-shaped would not
be, and would produce an arm that moves confidently to the wrong configuration.

What is genuinely discarded
---------------------------
A human hand has more than twenty degrees of freedom. A parallel-jaw gripper has
one. So every retargeting through here throws away the entire manner of the grasp
— which fingers, what opposition, how the object is cradled — and keeps only how
far apart the jaws should be. Two demonstrations that a person would describe
completely differently can come out identical.

That is not a defect to be apologised for; it is the actual information content
of the transform, and it is declared in ``losses`` so that a result produced
through it carries the fact. What would be a defect is implying the mapping is
faithful, because then a policy's failure to reproduce a delicate grasp reads as
the policy's fault.

The number worth reporting
--------------------------
:func:`reach_report` is the most useful thing in this file for a user. It says
what fraction of their reaching happens somewhere the robot cannot go — which is
simultaneously a hard limit on their data, a thing they can fix by standing
closer to what they are doing, and a number nobody else will tell them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from gantry_adapters_rotation import from_matrix, to_matrix

from gantry.contracts.embodiment import Retargeter
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec, Verdict

from .rig import Hand, Mount, Reach

VERSION = "0.1.0.dev0"

#: Width of the rotation block for each encoding this accepts.
ROTATIONS = {"quat_wxyz": 4, "quat_xyzw": 4, "euler_xyz": 3, "axis_angle": 3, "rot6d": 6}

#: Control modes this can produce. Pose commands only — see the module docstring
#: on why joint targets are refused rather than approximated.
POSE_MODES = ("eef_abs_pose", "eef_delta_pose")

#: Control modes that need inverse kinematics, named so the refusal can say which.
JOINT_MODES = ("joint_pos", "joint_delta", "joint_vel")

#: Frames whose origin is attached to the person and moves with them. A
#: trajectory measured in one of these is not a trajectory through space: it is a
#: trajectory relative to something that was itself travelling, and the two are
#: indistinguishable once the numbers are out of context.
MOVING_FRAMES = ("camera", "head", "image")

#: A typical adult hand span, used only to sanity-check a supplied one.
KEY_SCALE = "scale"
KEY_HAND = "hand"
KEY_ROTATION = "rotation_repr"


def hand_command(
    name: str,
    *,
    hand: str = "right",
    scale: str = "unscaled",
    rotation_repr: str = "quat_wxyz",
    frame: str = "world",
    rate_hz: float | None = None,
) -> ChannelSpec:
    """The retargeter's input: a wrist pose with an aperture on the end.

    Defined here rather than in the ego vocabulary because it is not a
    measurement — it is the *derived* per-step command that a hand-tracking
    pipeline produces once it has decided which numbers matter. Keeping it out of
    the vocabulary keeps the distinction between what was seen and what was
    computed from it.
    """
    if rotation_repr not in ROTATIONS:
        raise ConfigError(
            f"unknown rotation encoding {rotation_repr!r}; known: {sorted(ROTATIONS)}"
        )
    width = 3 + ROTATIONS[rotation_repr] + 1
    return ChannelSpec(
        name,
        "vector",
        (width,),
        "float32",
        frame=frame,
        rate_hz=rate_hz,
        semantics="ego.wrist_pose",
        discriminators=(KEY_ROTATION, KEY_SCALE, KEY_HAND),
        metadata={
            KEY_ROTATION: rotation_repr,
            KEY_SCALE: scale,
            KEY_HAND: hand,
            "layout": "position, rotation, aperture",
        },
    )


def arm_command(
    name: str,
    *,
    arm: str = "right",
    rotation_repr: str = "euler_xyz",
    control_mode: str = "eef_abs_pose",
    rate_hz: float | None = None,
) -> ChannelSpec:
    """The retargeter's output: one arm's pose command with a gripper on the end."""
    if rotation_repr not in ROTATIONS:
        raise ConfigError(f"unknown rotation encoding {rotation_repr!r}")
    width = 3 + ROTATIONS[rotation_repr] + 1
    return ChannelSpec(
        name,
        "vector",
        (width,),
        "float32",
        frame="base",
        rate_hz=rate_hz,
        semantics=f"action.{control_mode}",
        discriminators=(KEY_ROTATION, "arm"),
        metadata={KEY_ROTATION: rotation_repr, "arm": arm, "gripper": "continuous"},
    )


class HandToArm(Retargeter):
    """One human hand's trajectory as one robot arm's pose-and-gripper command.

    One hand, one arm, one direction — the same restraint the rest of the
    retargeter plane keeps. Two of these compose into a bimanual command through
    :func:`assemble`, which reads the target's dimension labels rather than
    assuming an order.
    """

    def __init__(
        self,
        *,
        mount: Mount,
        hand: Hand,
        reach: Reach | None = None,
        clip_to_reach: bool = False,
        name: str = "hands",
    ):
        """``reach`` is optional for the transform and load-bearing for the report.

        Without it the retargeting still works and simply cannot say whether any
        of the resulting poses are somewhere the arm can go — which for the
        purposes of this product is most of the value.
        """
        self._mount = mount
        self._hand = hand
        self._reach = reach
        self._clip = bool(clip_to_reach)
        self._name = name
        if clip_to_reach and reach is None:
            raise ConfigError(f"{name}: asked to clip to the workspace with no workspace measured")

    # -- contract ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return VERSION

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        checks = [self._accepts_source(source), self._accepts_target(target)]
        return Verdict.all([check for check in checks if check is not None])

    def _accepts_source(self, source: ChannelSpec) -> Verdict:
        encoding = source.metadata.get(KEY_ROTATION)
        if encoding not in ROTATIONS:
            return Verdict.no(
                "hands.encoding_undeclared",
                f"{source.name} does not say how its rotation is encoded "
                f"({encoding!r}); known: {sorted(ROTATIONS)}",
                hint="a quaternion in wxyz and one in xyzw are four floats each and "
                "differ by a permutation, so nothing else notices",
            )
        expected = 3 + ROTATIONS[encoding] + 1
        if source.width != expected:
            return Verdict.no(
                "hands.source_width",
                f"{source.name} is {source.width} wide; a {encoding} hand command is "
                f"{expected} (3 position, {ROTATIONS[encoding]} rotation, 1 aperture)",
            )
        if source.frame in MOVING_FRAMES:
            return Verdict.no(
                "hands.moving_frame",
                f"{source.name} is in the {source.frame!r} frame, which is bolted to "
                "the person's head",
                hint="a hand position measured relative to a moving camera changes "
                "when the head turns and the hand does not. Retargeted as though it "
                "were fixed, the arm's target swings every time the person looks "
                "around — smooth, plausible motion toward the wrong place. Compose "
                "the head pose in first to get a world-frame trajectory",
            )
        scale = source.metadata.get(KEY_SCALE)
        if scale == "metric":
            return Verdict.yes()
        if scale == "normalized":
            return Verdict.no(
                "hands.image_coordinates",
                f"{source.name} is in normalized coordinates — pixel fractions, not "
                "a metric hand at an unknown scale",
                hint="a hand span rescues an unscaled *metric* hand; it cannot "
                "rescue an image coordinate, because x and y are fractions of the "
                "frame and z is a relative offset near zero. Multiplying them by a "
                "span produces a smooth trajectory inside a box the size of a hand, "
                "which reads as an arm that never reaches anything. Monocular video "
                "gives hand shape and orientation; position needs depth, SLAM, or a "
                "calibrated capture",
            )
        if scale is None:
            return Verdict.no(
                "hands.scale_undeclared",
                f"{source.name} says nothing about whether its positions are metres",
                hint="a channel that never considered scale is not the same as one "
                "that says unscaled; this one cannot be retargeted either way "
                "until it says which",
            )
        if self._hand.span is None:
            return Verdict.no(
                "hands.unscaled_without_reference",
                f"{source.name} is {scale} and no hand span was measured, so there is "
                "nothing to recover the missing factor from",
                hint="a monocular hand pose is correct up to one unknown multiplier. "
                "Retargeting it as though it were metres sends the arm to a point "
                "that does not exist, consistently — which looks exactly like a "
                "policy that never learned to reach. Measure the person's hand span, "
                "or capture with a calibrated rig",
            )
        return Verdict.yes()

    def _accepts_target(self, target: ChannelSpec) -> Verdict:
        mode = (target.semantics or "").removeprefix("action.")
        if mode in JOINT_MODES:
            return Verdict.no(
                "hands.needs_inverse_kinematics",
                f"{target.name} wants {mode}, and this produces an end-effector pose",
                hint="going from a pose to joint angles needs that arm's link "
                "lengths, its joint limits, and a choice of elbow configuration "
                "where several solutions exist. None of those are here. Put an IK "
                "step owned by whoever owns the arm in between, or train the policy "
                "on a pose action space",
            )
        if mode not in POSE_MODES:
            return Verdict.no(
                "hands.target_mode",
                f"{target.name} declares {mode or 'no control mode'}; this produces "
                f"one of {list(POSE_MODES)}",
            )
        encoding = target.metadata.get(KEY_ROTATION)
        if encoding not in ROTATIONS:
            return Verdict.no(
                "hands.target_encoding_undeclared",
                f"{target.name} does not say how its rotation is encoded ({encoding!r})",
            )
        expected = 3 + ROTATIONS[encoding] + 1
        if target.width != expected:
            return Verdict.no(
                "hands.target_width",
                f"{target.name} is {target.width} wide; a {encoding} pose command with "
                f"a gripper is {expected}",
            )
        return Verdict.yes()

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        """What this discards, written for whoever reads the result months later."""
        out = [
            "the whole manner of the grasp: a hand has more than twenty degrees of "
            "freedom and a gripper has one, so which fingers were used, how the "
            "object was opposed, and how it was cradled are all replaced by a "
            "single jaw distance",
            "contact and force: nothing in a video says how hard anything was held",
        ]
        if source.metadata.get(KEY_SCALE) != "metric":
            out.append(
                f"absolute scale, recovered from a measured hand span of "
                f"{self._hand.span} m rather than observed — every position carries "
                "that measurement's error, multiplied"
            )
        if self._clip:
            out.append(
                "reaches outside the arm's workspace, pulled onto its boundary — the "
                "demonstration becomes a different demonstration wherever that happened"
            )
        if self._mount.workspace_ratio != 1.0:
            out.append(
                f"true distances: human motion is scaled by "
                f"{self._mount.workspace_ratio:g} to fit the arm, so speeds and "
                "clearances are no longer the person's"
            )
        return tuple(out)

    def apply(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        self.check(source, target).raise_if_refused(f"{self._name} cannot retarget")
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != source.width:
            raise ConfigError(f"{self._name}: expected (steps, {source.width}), got {array.shape}")
        source_encoding = str(source.metadata[KEY_ROTATION])
        target_encoding = str(target.metadata[KEY_ROTATION])
        rotation_width = ROTATIONS[source_encoding]

        positions = array[:, :3]
        rotations = array[:, 3 : 3 + rotation_width]
        aperture = array[:, 3 + rotation_width]

        if source.metadata.get(KEY_SCALE) != "metric":
            positions = positions * self._metric_factor(positions)

        placed = self._mount.place(positions)
        if self._clip and self._reach is not None:
            placed = self._reach.clip(placed)
        turned = self._mount.turn(to_matrix(rotations, source_encoding))
        grip = self._hand.fraction(aperture)

        return np.concatenate(
            [placed, from_matrix(turned, target_encoding), grip[:, None]], axis=1
        ).astype("float32")

    # -- the number worth reporting ----------------------------------------

    def reach_report(self, values: np.ndarray, source: ChannelSpec) -> dict[str, Any]:
        """What fraction of this reaching is somewhere the arm can actually go.

        Simultaneously a hard limit on the data, a thing the person can fix by
        standing closer to what they are doing, and a number nobody else is going
        to tell them. Returns the fraction plus *why* the out-of-reach points
        were out, because "too far away" and "below the table" are different
        pieces of advice.
        """
        if self._reach is None:
            return {"measured": False, "why": "no workspace was measured for this arm"}
        array = np.asarray(values, dtype=float)
        positions = array[:, :3]
        if source.metadata.get(KEY_SCALE) != "metric":
            positions = positions * self._metric_factor(positions)
        placed = self._mount.place(positions)

        distance = np.linalg.norm(placed, axis=1)
        too_far = distance > self._reach.radius
        too_near = distance < self._reach.inner
        too_low = placed[:, 2] < self._reach.floor
        too_high = placed[:, 2] > self._reach.ceiling
        holds = self._reach.holds(placed)
        steps = max(1, len(placed))
        return {
            "measured": True,
            "arm": self._reach.name or "unnamed",
            "in_reach": round(float(holds.mean()), 4),
            "steps": int(len(placed)),
            "why": {
                "too_far": round(float(too_far.sum()) / steps, 4),
                "too_close_to_the_base": round(float(too_near.sum()) / steps, 4),
                "below_the_workspace": round(float(too_low.sum()) / steps, 4),
                "above_the_workspace": round(float(too_high.sum()) / steps, 4),
            },
            "furthest": round(float(distance.max()) if len(distance) else 0.0, 4),
            "radius": float(self._reach.radius),
        }

    def provenance(self) -> dict[str, Any]:
        """Everything that was measured rather than observed, for the record."""
        return {
            "retargeter": f"{self._name}@{VERSION}",
            "clipped": self._clip,
            "identity_mount": self._mount.is_identity,
            **self._mount.as_dict(),
            **{f"hand_{key}": value for key, value in self._hand.as_dict().items()},
            **(self._reach.as_dict() if self._reach else {}),
        }

    def _metric_factor(self, positions: np.ndarray) -> float:
        """The multiplier that turns unscaled positions into metres.

        Recovered from the ratio between the measured hand span and the span the
        estimator reports, which for a unit-normalised hand is one — so the
        factor is the span itself. Crude, and the honest description of what
        every monocular ego pipeline does.
        """
        if self._hand.span is None:  # pragma: no cover - guarded by accepts()
            raise ConfigError(f"{self._name}: no hand span to recover scale from")
        return float(self._hand.span)


def assemble(per_arm: Mapping[str, np.ndarray], target: ChannelSpec) -> np.ndarray:
    """Two arms' commands into one vector, in the order the target declares.

    Reads ``dim_labels`` rather than assuming left-then-right. That is the whole
    point: a fourteen-wide float32 vector is the same object whichever arm comes
    first, and a swap produces perfectly valid actions sent to the wrong arm.
    The labels are the only written record of the order, so this consults them
    and refuses when they are absent.
    """
    labels = target.dim_labels
    if not labels:
        raise ConfigError(
            f"{target.name} carries no dimension labels, so there is no way to know "
            "which half of it is which arm. A swap here produces valid, smooth "
            "actions sent to the wrong arm and nothing downstream can detect it"
        )
    blocks = {name: np.asarray(value, dtype=float) for name, value in per_arm.items()}
    if not blocks:
        raise ConfigError("nothing to assemble")
    lengths = {name: len(value) for name, value in blocks.items()}
    if len(set(lengths.values())) > 1:
        raise ConfigError(f"arms disagree on length: {lengths}")

    widths = {name: value.shape[1] for name, value in blocks.items()}
    if sum(widths.values()) != target.width:
        raise ConfigError(
            f"{sum(widths.values())} values from {sorted(blocks)} do not fill "
            f"{target.name}'s {target.width}"
        )

    order = _arm_order(labels, sorted(blocks))
    return np.concatenate([blocks[name] for name in order], axis=1).astype("float32")


def _arm_order(labels: Sequence[str], arms: Sequence[str]) -> list[str]:
    """Which arm's block comes first, from where its name appears in the labels."""
    firsts: dict[str, int] = {}
    for arm in arms:
        positions = [index for index, label in enumerate(labels) if arm in str(label)]
        if not positions:
            raise ConfigError(
                f"no dimension label mentions {arm!r}; the target's labels are "
                f"{list(labels)}, so which block belongs where cannot be established"
            )
        firsts[arm] = min(positions)
    return sorted(arms, key=lambda arm: firsts[arm])


def bimanual(
    *,
    mount: Mount,
    left: Hand,
    right: Hand,
    reach: Reach | None = None,
    clip_to_reach: bool = False,
    name: str = "hands",
) -> dict[str, HandToArm]:
    """One retargeter per hand, keyed by which.

    Two objects rather than one two-armed object, because each hand is calibrated
    separately — people are not symmetric, and a single calibration applied to
    both puts the difference straight into the gripper signal of whichever hand
    was not measured.
    """
    return {
        "left": HandToArm(
            mount=mount,
            hand=left,
            reach=reach,
            clip_to_reach=clip_to_reach,
            name=f"{name}.left",
        ),
        "right": HandToArm(
            mount=mount,
            hand=right,
            reach=reach,
            clip_to_reach=clip_to_reach,
            name=f"{name}.right",
        ),
    }
