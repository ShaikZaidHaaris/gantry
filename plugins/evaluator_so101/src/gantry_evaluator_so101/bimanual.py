"""Two SO-ARM101 followers as one machine, described — and not claimed to run.

This is plane 2 only: **data, not code**. Nothing here opens a serial port, and
nothing here is an evaluator. The single-arm module next door describes one
follower; this one describes a rig of two followers whose bases are bolted to a
shared table at a known separation, and whose twelve joints are commanded as a
single vector.

Read this before you read anything else in the file
---------------------------------------------------
**This embodiment is policy-evaluable and NOT teleop-recordable on two arms.**

Two SO-101 arms is a *leader-follower pair*: one operator station driving one
controlled arm. A bimanual rig consumes both arms as **followers**, which leaves
nothing to teleoperate them with. Bimanual teleoperation needs one leader per
follower, so recording a bimanual demonstration takes **four** arms, not two.

That has a consequence larger than the wiring: a bimanual policy needs bimanual
demonstrations, and the corpus factory in ``CORPUS-PROTOCOL.md`` records through
a leader-follower pair. On a two-arm bench it therefore **cannot produce a corpus
for** ``embodiment:so101_bimanual@1.0``. :class:`SingleToBimanual` does not
manufacture one either — it pads a single-arm trajectory with a constant, and its
loss statement says exactly that, in the record, forever.

None of this makes the descriptor useless. Describing a machine costs one
descriptor and is checkable by anyone; *claiming to run* it is a claim nobody can
check, which is why this module ships no evaluator (see "No evaluator" below).
The constraint is declared in :data:`TELEOPERATION` and lands in the descriptor's
metadata, so it is discoverable from the artifact rather than by buying two more
arms.

The order is the contract
-------------------------
Twelve values, **left arm's six first (indices 0-5), then the right arm's six
(indices 6-11)**, each half in the LeRobot SO-101 order that
:data:`~.embodiment.JOINT_NAMES` fixes. Every dimension is labelled
(``left_shoulder_pan.pos`` … ``right_gripper.pos``) and the split is exported as
data (:data:`SIDE_BLOCKS`, :func:`side_slice`, :func:`halves`) rather than left
to a reader's arithmetic.

This is not decoration. A twelve-vector whose halves can be swapped silently is a
robot that does the task mirrored: same width, same dtype, same units, same
normalization, every structural check green, and the arms reaching to each
other's side of the table. Labels and a declared block layout are what turn that
into ``dim_labels.order`` — a refusal — instead of motion.

One source of truth for what the numbers mean
---------------------------------------------
The channels here are built by calling :func:`~.embodiment.joint_channel` and
widening the result, so units (``percent``), dtype, meaning tags, rate and the
load-bearing discriminator *keys* come from the single-arm module and cannot
drift from it. This package has already had that defect once — two modules
spelling a discriminator differently, which does not weaken a check, it deletes
it — and the fix was to have one place define it. Reusing the builder is that fix
applied to the second embodiment instead of re-litigated in it.

Two things are deliberately *not* inherited, because on twelve values they would
be false:

``gripper_index``
    The single-arm metadata says the gripper is element 5. Here there are two
    grippers, at 5 and 11. The singular key is removed rather than left pointing
    at the left arm's jaw, and :data:`GRIPPER_INDICES` replaces it.

``mounting_id``
    Added as an extra discriminator, because it is a fact about *this* machine
    that the single-arm one has no opinion about. See below.

The base separation is part of the machine
------------------------------------------
Two arms whose bases are 40 cm apart and two arms 20 cm apart are different
machines. They have different shared workspace, different collision geometry and
different handover kinematics, and a policy trained on one will not transfer to
the other — while producing numerically identical twelve-vectors that pass every
check in the framework. So :class:`MountingGeometry` is an explicit declared
field, and its :attr:`~MountingGeometry.mounting_id` — base separation, both
yaws, and measured-or-not — rides on the joint channels as the **discriminator**
(:data:`MOUNTING_DISCRIMINATOR`), so two corpora recorded on differently mounted
rigs refuse to bind instead of averaging.

The rest of the declaration rides beside it in the same channel's metadata,
together with a :attr:`~MountingGeometry.digest` of all of it, as provenance
rather than as a refusal. Which fields sit on which side of that line, and why,
is stated on :attr:`~MountingGeometry.mounting_id`: a discriminator that covers
less than the whole declaration is a defensible choice, a discriminator whose
docstring claims to cover the whole declaration is a check that reads as
complete and is not.

The default is :meth:`MountingGeometry.nominal`, which declares ``measured =
False``. That is the honest state of a rig nobody has put a tape measure on, and
it is part of the ``mounting_id`` itself: a nominal 0.400 m and a measured
0.400 m are not the same claim, and the framework should not let one pass for
the other silently.

Cameras
-------
Four policy observations: ``top`` and ``front`` are scene views of the shared
table, ``left_wrist`` and ``right_wrist`` ride their own arms. A referee camera,
if configured, is **not** a policy observation: it is carried outside the
``observation.`` namespace and outside ``state``, exactly as the single-arm
descriptor does it, so nothing that globs ``observation.images.*`` into a policy
can reach the ground-truth view. That separation already exists; this module
honours it rather than reinventing it.

What this hardware still cannot do
----------------------------------
Everything the single arm cannot do, twice. Twelve Feetech STS3215 servos,
position control only, **no force and no torque sensing anywhere**, so no wrench
channel is declared and none may be inferred from motor current. The published
URDF's two defects (:data:`~.embodiment.KINEMATICS_DEFECTS`) apply to both arms,
which is why the workspace overlap on :class:`MountingGeometry` is a declared
measurement and not something derived from the model.

Two arms on separate USB buses adds one failure mode the single arm does not
have: the two halves of a twelve-vector are read and written over **two
independent serial buses**, so they are not simultaneous. That skew is declared
in :data:`BUS_SKEW` rather than assumed to be zero — an unstated skew between the
halves is a bimanual dataset in which the arms appear to react to each other
slightly earlier or later than they did.

No evaluator, on purpose
------------------------
``SO101Evaluator`` drives a leader-follower pair. A bimanual rig is a different
wiring — two followers, no leader — and there is no hardware here to test an
implementation against. Shipping a "thin bimanual evaluator" would be a plugin
that passes its conformance kit against a mock and has never seen a bus, while
its mere presence in ``gantry list`` reads as "this rig runs". Describing the
machine is data and costs one descriptor; claiming to run it is a claim nobody
can check. So this module registers on ``gantry.embodiments`` and on nothing
else.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.embodiment import (
    CAP_RESETTABLE,
    EmbodimentDescriptor,
    Retargeter,
)
from gantry.errors import ConfigError
from gantry.serial import spec_to_dict
from gantry.spine import ChannelSpec, Verdict

from .embodiment import (
    ACTION,
    CALIBRATION,
    CHUNKING,
    CONTROL_HZ,
    DEFAULT_MAX_RELATIVE_TARGET,
    DOF,
    EXPECTED_LATENCY,
    FOLLOWER_JOINT_FRAME,
    FRONT_CAMERA,
    GRIPPER_INDEX,
    JOINT_NAMES,
    KINEMATICS_DEFECTS,
    KINEMATICS_REF,
    NORMALIZATION,
    NORMALIZATION_RANGES,
    PROVISIONING,
    SEMANTICS_ACTION,
    SEMANTICS_STATE,
    STATE,
    VERSION,
    camera_channel,
    joint_channel,
)

# --------------------------------------------------------------------------
# the order, which is the contract
# --------------------------------------------------------------------------

#: The two arms, in the order their joints appear in the vector. This tuple is
#: the definition of "left half" and "right half"; everything else below is
#: derived from it so there is exactly one place the order can be changed and
#: exactly one place a reader has to look to learn it.
SIDES: tuple[str, ...] = ("left", "right")

#: Twelve labels: every dimension named, side first. Spelled the way LeRobot's
#: bimanual follower spells them (``left_shoulder_pan.pos``), so a connector's
#: ``dim_labels`` compare equal to these without a rename step — and a rename is
#: where a side swap hides.
BIMANUAL_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{side}_{joint}" for side in SIDES for joint in JOINT_NAMES
)

BIMANUAL_DOF = len(BIMANUAL_JOINT_NAMES)

#: Half-open index ranges, as data. Exported so nothing downstream has to
#: rediscover that the right arm starts at 6 — an off-by-six is a mirrored robot,
#: and it is not visible in any shape, dtype or unit.
SIDE_BLOCKS: Mapping[str, tuple[int, int]] = {
    side: (index * DOF, (index + 1) * DOF) for index, side in enumerate(SIDES)
}

#: Both grippers, by side. The single-arm descriptor's ``gripper_index`` is 5 and
#: stays 5 there; here the singular key would name the left jaw and quietly imply
#: the right one does not exist.
GRIPPER_INDICES: Mapping[str, int] = {
    side: SIDE_BLOCKS[side][0] + GRIPPER_INDEX for side in SIDES
}

#: The sentence a later reader needs, carried as data rather than as prose in a
#: docstring nobody serialises.
ORDER_CONTRACT = (
    "twelve values: the left arm's six joints at indices 0-5, then the right arm's "
    "six at indices 6-11, each half in the LeRobot SO-101 order "
    f"{list(JOINT_NAMES)}. The halves are not interchangeable and a swapped pair "
    "is a robot performing the task mirrored, with every structural check passing."
)

#: Which physical rig's joint space these numbers belong to. Distinct from the
#: single arm's frame so that taking half of a bimanual vector and calling it a
#: follower command has to go through :class:`BimanualToSingle`, where the loss
#: is stated, rather than through a direct binding, where it is not.
BIMANUAL_JOINT_FRAME = "so101_bimanual_joints"

#: The metadata key that says which physical mounting these numbers were produced
#: on. Declared as a channel discriminator: see :class:`MountingGeometry`.
MOUNTING_DISCRIMINATOR = "mounting_id"

# --------------------------------------------------------------------------
# the honest constraint
# --------------------------------------------------------------------------

#: Why this descriptor exists and what it cannot be used for, machine-readable.
#: Stamped into the descriptor's metadata so the limit travels with the artifact.
#: The alternative is that somebody discovers it by buying two more arms.
TELEOPERATION: Mapping[str, Any] = {
    "policy_evaluable": True,
    "teleop_recordable": False,
    "followers_required": len(SIDES),
    "leaders_required": len(SIDES),
    "arms_required_total": 2 * len(SIDES),
    "why": (
        "two SO-101 arms is a leader-follower pair: one operator station driving one "
        "controlled arm. A bimanual rig consumes both arms as followers, so there is "
        "nothing left to teleoperate them with. Bimanual teleop recording needs one "
        "leader per follower, i.e. four arms."
    ),
    "corpus_consequence": (
        "a bimanual policy needs bimanual demonstrations. The corpus factory in "
        "CORPUS-PROTOCOL.md records through a leader-follower pair, so on a two-arm "
        "rig it cannot feed embodiment:so101_bimanual@1.0. SingleToBimanual does not "
        "manufacture a bimanual corpus either: it pads one arm's trajectory with a "
        "declared constant and says so in its losses."
    ),
    "what_would_lift_it": (
        "two further SO-101 arms wired as leaders, or a demonstration source that is "
        "not teleoperation at all: scripted trajectories, kinesthetic replay with "
        "torque disabled, or transfer from a simulated twin"
    ),
}

#: No evaluator is registered against this ref, and that is a decision rather than
#: an omission. Carried in metadata so a resolver refusal can quote a reason.
NO_EVALUATOR = (
    "no evaluator is registered for embodiment:so101_bimanual@1.0. SO101Evaluator "
    "drives a leader-follower pair (one leader read, one follower commanded), which "
    "is a different wiring from two followers on two buses, and there is no bimanual "
    "hardware here to test an implementation against. A descriptor is checkable data; "
    "an untested runner that appears in `gantry list` reads as a rig that runs."
)

#: The failure mode two USB buses add that one bus does not have. Declared as a
#: budget with no measurement behind it, the same way the single arm declares its
#: latency: a skew silently assumed to be zero is a bimanual corpus in which each
#: arm appears to react to the other slightly early or slightly late, and no frame
#: in the dataset carries evidence of it.
BUS_SKEW: Mapping[str, Any] = {
    "buses": len(SIDES),
    "simultaneous": False,
    "measured": False,
    "budget_s": 0.010,
    "note": (
        "the two halves of a twelve-vector are read and written over two independent "
        "serial buses and are therefore not simultaneous. The budget is one arm's "
        "read+write round trip, not a measurement of any rig; anyone quoting a "
        "bimanual timing result should measure it and replace this number."
    ),
}


# --------------------------------------------------------------------------
# mounting geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MountingGeometry:
    """Where the two bases are, as a declared field rather than an assumption.

    Base separation is not context, it is part of the machine. Two arms 40 cm
    apart share a different workspace, collide in different places, and can hand
    an object over where two arms 20 cm apart cannot — and the twelve-vectors the
    two rigs produce are identical in width, dtype, units, normalization and
    joint order. Nothing structural distinguishes them, so :attr:`mounting_id`
    is put on the joint channels as a discriminator and the distinction becomes a
    refusal.

    ``measured`` is part of the identity on purpose. A separation somebody typed
    in and a separation somebody measured are different claims about the same
    number, and letting them bind silently is how a nominal figure becomes a
    citation.
    """

    #: Centre-to-centre distance between the two base mounting holes, in metres.
    base_separation_m: float
    #: Yaw of each base about the table normal, in degrees, in :data:`SIDES`
    #: order. Zero means both bases face the same way (parallel mounting);
    #: negative/positive pairs mean the arms are toed in towards each other.
    base_yaw_deg: tuple[float, ...]
    #: Free-form name of the arrangement, for a human reading a record.
    layout: str
    #: Whether these numbers were taken off the physical rig.
    measured: bool
    #: How they were obtained. Required, and recorded: "nominal, from the CAD" and
    #: "callipers across the base plates" are different provenance.
    method: str
    #: Overlap of the two arms' reachable volumes, in metres of shared width, or
    #: ``None`` when nobody has established it. Deliberately not derived from the
    #: published URDF: its base collision meshes were removed upstream, so a
    #: reachability query against it returns clear near the base and would report
    #: an overlap that does not exist. See KINEMATICS_DEFECTS.
    reach_overlap_m: float | None = None
    #: Height of the shared work surface, if it is known.
    table_height_m: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_separation_m) or self.base_separation_m <= 0:
            raise ConfigError(
                f"base_separation_m={self.base_separation_m!r} is not a separation; two "
                "arms at zero or negative separation are not a bimanual rig, and the "
                "number is load-bearing enough to be a channel discriminator"
            )
        if len(self.base_yaw_deg) != len(SIDES):
            raise ConfigError(
                f"base_yaw_deg has {len(self.base_yaw_deg)} entries for {len(SIDES)} arms "
                f"{list(SIDES)}; a partial mounting description is one arm placed and one "
                "arm assumed"
            )
        if not all(math.isfinite(yaw) for yaw in self.base_yaw_deg):
            raise ConfigError(f"base_yaw_deg={list(self.base_yaw_deg)} contains a non-finite value")
        if not self.method.strip():
            raise ConfigError(
                "mounting geometry requires a written method; it is carried into every "
                "record taken on this rig, and 'where did this number come from' is the "
                "question that decides whether two campaigns are comparable"
            )
        if self.reach_overlap_m is not None and self.reach_overlap_m < 0:
            raise ConfigError(
                f"reach_overlap_m={self.reach_overlap_m!r} is negative; use None for "
                "'nobody has established it', which is a different statement from 'zero'"
            )

    @classmethod
    def nominal(
        cls,
        base_separation_m: float = 0.400,
        *,
        layout: str = "two bases in a row on a shared table, both facing the operator",
    ) -> "MountingGeometry":
        """A declared reference layout that nobody has measured.

        Exists so the entry-point factory is callable with no arguments, and
        marked ``measured=False`` so that fact is in the digest rather than in
        somebody's memory. A run that matters should pass
        :meth:`from_measurement` instead.
        """
        return cls(
            base_separation_m=base_separation_m,
            base_yaw_deg=(0.0,) * len(SIDES),
            layout=layout,
            measured=False,
            method=(
                "nominal reference layout declared in gantry_evaluator_so101.bimanual; "
                "never measured on any rig, and not comparable to a measured rig of the "
                "same nominal separation"
            ),
        )

    @classmethod
    def from_measurement(
        cls,
        *,
        base_separation_m: float,
        method: str,
        base_yaw_deg: Sequence[float] | None = None,
        layout: str = "measured on the rig",
        reach_overlap_m: float | None = None,
        table_height_m: float | None = None,
    ) -> "MountingGeometry":
        """A mounting somebody actually put a tape measure on.

        ``method`` is mandatory and is not a formality: it is the difference
        between two campaigns being comparable and two campaigns sharing a
        number.
        """
        if not method.strip():
            raise ConfigError("a measured mounting geometry requires a written method")
        return cls(
            base_separation_m=float(base_separation_m),
            base_yaw_deg=tuple(float(y) for y in (base_yaw_deg or (0.0,) * len(SIDES))),
            layout=layout,
            measured=True,
            method=method,
            reach_overlap_m=reach_overlap_m,
            table_height_m=table_height_m,
        )

    @property
    def mounting_id(self) -> str:
        """The discriminator: base separation, both yaws, and measured-or-not.

        Two rigs bind only if all three agree. Separation is rounded to a
        millimetre because that is the resolution at which two bench setups are
        honestly the same rig, stated here so the tolerance is a declared choice
        rather than whatever float formatting happened to do.

        What it deliberately does **not** cover, because a discriminator is a
        refusal and these four do not warrant one: ``layout`` and ``method`` are
        free text, and prose in a discriminator means rewording a note stops two
        corpora from the same bench binding; ``table_height_m`` is the height of
        a surface both bases are bolted to, so it changes the operator's
        ergonomics and not the arm-to-arm geometry; ``reach_overlap_m`` is a
        consequence of the separation rather than an independent fact about the
        rig. All four still ride in the record through :meth:`as_dict` and are
        covered by :attr:`digest`, so a reader can see a disagreement the
        discriminator declines to refuse on.
        """
        yaw = ",".join(f"{value:+.1f}" for value in self.base_yaw_deg)
        state = "measured" if self.measured else "nominal"
        return f"so101_bimanual/sep{self.base_separation_m:.3f}m/yaw{yaw}deg/{state}"

    @property
    def digest(self) -> str:
        """SHA-256 of the full declaration, first 16 hex chars. For provenance.

        Covers everything, including the fields :attr:`mounting_id` leaves out,
        which is the point: the id decides what refuses and the digest records
        what was actually declared.
        """
        payload = repr(sorted(self._declaration().items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _declaration(self) -> dict[str, Any]:
        """Everything declared, minus the digest. Kept separate from
        :meth:`as_dict` so the digest is never an input to itself."""
        return {
            "base_separation_m": self.base_separation_m,
            "base_yaw_deg": list(self.base_yaw_deg),
            "sides": list(SIDES),
            "layout": self.layout,
            "measured": self.measured,
            "method": self.method,
            "reach_overlap_m": self.reach_overlap_m,
            "table_height_m": self.table_height_m,
            "mounting_id": self.mounting_id,
            "why_it_is_a_discriminator": (
                "two rigs whose bases are at different separations share different "
                "workspace and different collision geometry, and produce twelve-vectors "
                "identical in width, dtype, units, normalization and joint order; a "
                "policy trained on one does not transfer to the other and nothing "
                "structural can see the difference"
            ),
            "what_the_id_does_not_cover": (
                "layout, method, table_height_m and reach_overlap_m are recorded here and "
                "in the digest but are not in mounting_id, so two rigs differing only in "
                "those bind: free text in a discriminator makes rewording a note a "
                "refusal, the table height is shared by both bases and does not change "
                "the arm-to-arm geometry, and the reach overlap follows from the "
                "separation rather than standing alongside it"
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        """The declaration as it rides in the record, digest included.

        The digest travels with the data rather than being recomputable only by
        importing this module: a reader comparing two records has to be able to
        see that two mountings differ without running any of this code.
        """
        return {**self._declaration(), "digest": self.digest}


#: What :func:`so101_bimanual` uses when the caller says nothing. Declared, not
#: assumed, and flagged unmeasured in its own digest.
DEFAULT_MOUNTING = MountingGeometry.nominal()


# --------------------------------------------------------------------------
# index helpers — the order, usable
# --------------------------------------------------------------------------


def _require_side(side: str) -> str:
    if side not in SIDES:
        raise ConfigError(f"unknown side {side!r}; this rig has {list(SIDES)}")
    return side


def side_slice(side: str) -> slice:
    """The half of a twelve-vector belonging to one arm."""
    start, stop = SIDE_BLOCKS[_require_side(side)]
    return slice(start, stop)


def other_side(side: str) -> str:
    """The arm that is *not* this one. Two-arm-specific and named as such."""
    _require_side(side)
    return SIDES[1 - SIDES.index(side)]


def side_labels(side: str) -> tuple[str, ...]:
    """The six labels of one arm, in order."""
    return BIMANUAL_JOINT_NAMES[side_slice(side)]


def dim_index(label: str) -> int:
    """Index of one labelled dimension. Raises rather than returning -1."""
    try:
        return BIMANUAL_JOINT_NAMES.index(label)
    except ValueError:
        raise ConfigError(
            f"{label!r} is not a dimension of this machine; it has {list(BIMANUAL_JOINT_NAMES)}"
        ) from None


def halves(values: Any) -> dict[str, np.ndarray]:
    """Split ``(steps, 12)`` into ``{'left': (steps, 6), 'right': (steps, 6)}``.

    Views, not copies, so this is free; and a refusal rather than a reshape if
    the width is wrong, because the whole point of the split is that the two
    halves are not interchangeable.
    """
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != BIMANUAL_DOF:
        raise ConfigError(f"expected (steps, {BIMANUAL_DOF}), got {array.shape}")
    return {side: array[:, side_slice(side)] for side in SIDES}


def join(left: Any, right: Any) -> np.ndarray:
    """Two ``(steps, 6)`` halves into one ``(steps, 12)``, left first.

    The inverse of :func:`halves`, and the only assembly path exported, so the
    order is applied in one place instead of at every call site that happens to
    know it.
    """
    arrays = []
    for side, values in zip(SIDES, (left, right)):
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != DOF:
            raise ConfigError(f"{side}: expected (steps, {DOF}), got {array.shape}")
        arrays.append(array)
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ConfigError(
            f"the two halves have {arrays[0].shape[0]} and {arrays[1].shape[0]} steps; "
            "a bimanual vector is one instant of both arms, so unequal lengths mean one "
            "arm's timeline has been shifted against the other's"
        )
    return np.concatenate(arrays, axis=1)


# --------------------------------------------------------------------------
# channels — built from the single-arm builder, widened here
# --------------------------------------------------------------------------


def bimanual_joint_channel(
    name: str,
    *,
    semantics: str,
    mounting: MountingGeometry = DEFAULT_MOUNTING,
    normalization: str = NORMALIZATION,
    calibration_variant: str = CALIBRATION,
    rate_hz: float = CONTROL_HZ,
    extra: Mapping[str, Any] | None = None,
) -> ChannelSpec:
    """One fully described twelve-vector of normalized joint values.

    Built by calling :func:`~.embodiment.joint_channel` and widening it, on
    purpose. Units, dtype, kind, the meaning tags and the load-bearing
    discriminator *keys* therefore have exactly one definition in this package.
    The alternative — writing them out again here — is how the single-arm and
    bimanual descriptors end up disagreeing about what ``normalization`` means,
    which is the defect this package already had once, and which does not weaken
    a check so much as delete it.
    """
    single = joint_channel(
        name,
        frame=BIMANUAL_JOINT_FRAME,
        semantics=semantics,
        normalization=normalization,
        calibration_variant=calibration_variant,
        rate_hz=rate_hz,
        extra=extra,
    )

    metadata = dict(single.metadata)
    # There are two grippers. Leaving the singular key would name the left jaw
    # and imply the right one is not in the vector, which is exactly the kind of
    # positional half-truth that reads as description.
    metadata.pop("gripper_index", None)
    metadata.pop("gripper_dim", None)
    metadata.update(
        {
            "arms": len(SIDES),
            "sides": list(SIDES),
            "dof_per_arm": DOF,
            "side_blocks": {side: list(SIDE_BLOCKS[side]) for side in SIDES},
            "gripper_indices": dict(GRIPPER_INDICES),
            "gripper_dims": {side: BIMANUAL_JOINT_NAMES[GRIPPER_INDICES[side]] for side in SIDES},
            "order": ORDER_CONTRACT,
            MOUNTING_DISCRIMINATOR: mounting.mounting_id,
            "mounting": mounting.as_dict(),
            "bus_skew": dict(BUS_SKEW),
        }
    )

    return replace(
        single,
        shape=(BIMANUAL_DOF,),
        dim_labels=BIMANUAL_JOINT_NAMES,
        # The single arm's keys, plus the one fact about this machine the single
        # arm has no opinion about.
        discriminators=(*single.discriminators, MOUNTING_DISCRIMINATOR),
        metadata=metadata,
    )


#: The four streams a policy may see. Named under ``observation.images.`` because
#: that is what LeRobot writes and what training pipelines glob.
TOP_CAMERA = "observation.images.top"
LEFT_WRIST_CAMERA = "observation.images.left_wrist"
RIGHT_WRIST_CAMERA = "observation.images.right_wrist"

#: Per-side wrist camera names, keyed the way everything else here is keyed.
WRIST_CAMERAS: Mapping[str, str] = {
    "left": LEFT_WRIST_CAMERA,
    "right": RIGHT_WRIST_CAMERA,
}


def default_cameras(
    *, height: int = 480, width: int = 640, rate_hz: float = CONTROL_HZ
) -> tuple[ChannelSpec, ...]:
    """Two scene views of the shared table, plus one wrist view per arm.

    Two scene views rather than one because a bimanual workspace is wider than a
    single arm's and the arms occlude each other: a front view alone loses the
    far arm behind the near one for most of a handover. Neither is calibrated
    (``intrinsics_ref=None``), which is the honest state of this rig and the
    reason no pixel measurement here becomes a metric quantity.
    """
    scene = (
        camera_channel(
            TOP_CAMERA,
            frame="so101_bimanual_top_cam_optical",
            mount="fixed overhead, viewing the whole shared table and both arms",
            height=height,
            width=width,
            rate_hz=rate_hz,
        ),
        camera_channel(
            FRONT_CAMERA,
            frame="so101_bimanual_front_cam_optical",
            mount="fixed, facing the shared table from the operator side",
            height=height,
            width=width,
            rate_hz=rate_hz,
        ),
    )
    wrists = tuple(
        camera_channel(
            WRIST_CAMERAS[side],
            frame=f"so101_bimanual_{side}_wrist_cam_optical",
            mount=f"rigid to the {side} arm's wrist_roll link, moves with that arm",
            height=height,
            width=width,
            rate_hz=rate_hz,
        )
        for side in SIDES
    )
    return scene + wrists


# --------------------------------------------------------------------------
# the descriptor
# --------------------------------------------------------------------------

_NOTES = """\
SO-ARM101 bimanual rig: two followers, 12x Feetech STS3215 across two independent
USB serial buses, position control only. No force or torque sensing exists
anywhere on either arm, so no wrench channel is declared and none may be inferred
from motor current.

ORDER IS THE CONTRACT. Twelve values: the left arm's six joints at indices 0-5,
then the right arm's six at indices 6-11, each half in the LeRobot SO-101 order.
Every dimension is labelled and the split is exported as data. A swapped pair of
halves is a robot performing the task mirrored, and it passes every structural
check.

POLICY-EVALUABLE, NOT TELEOP-RECORDABLE ON TWO ARMS. Two SO-101s is a
leader-follower pair: one operator station driving one arm. A bimanual rig
consumes both arms as followers, so nothing is left to teleoperate them with.
Bimanual teleop recording needs one leader per follower, i.e. four arms. A
bimanual policy needs bimanual demonstrations, so the corpus factory in
CORPUS-PROTOCOL.md cannot feed this embodiment on a two-arm rig.

NO EVALUATOR IS REGISTERED FOR THIS REF. SO101Evaluator drives a leader-follower
pair, which is a different wiring, and there is no bimanual hardware to test an
implementation against.

BASE SEPARATION IS PART OF THE MACHINE. Two arms 40 cm apart and two arms 20 cm
apart share different workspace and different collision geometry while producing
identical twelve-vectors. The mounting is declared, and its id is a channel
discriminator so differently mounted corpora refuse to bind.

THE TWO HALVES ARE NOT SIMULTANEOUS. Two buses, read and written in sequence.
The skew is declared as a budget and has never been measured.

Every number is relative to the so101_new_calib convention, in LeRobot
RANGE_0_100 percent of each joint's calibrated travel. The published URDF carries
two known defects; see metadata.kinematics_defects.
"""


def so101_bimanual(
    *,
    version: str = "1.0",
    name: str = "so101_bimanual",
    mounting: MountingGeometry | None = None,
    normalization: str = NORMALIZATION,
    calibration_variant: str = CALIBRATION,
    cameras: Sequence[ChannelSpec] | None = None,
    referee_camera: ChannelSpec | None = None,
    max_relative_target: float = DEFAULT_MAX_RELATIVE_TARGET,
) -> EmbodimentDescriptor:
    """Two SO-101 followers as one twelve-DoF machine.

    Callable with no arguments, so the entry point works, and every default it
    then uses is a *declared* one: :data:`DEFAULT_MOUNTING` says in its own digest
    that nobody measured it.

    A referee camera, if there is one, is passed separately and never lands in
    ``state``: ``state`` is what a policy is offered, and the ground-truth view
    has to be unreachable by construction rather than by convention. Read it back
    with :func:`~.embodiment.referee_channels`.
    """
    mounting = mounting if mounting is not None else DEFAULT_MOUNTING
    # Finite first: `<= 0` is blind to NaN and to infinity, and both then ride
    # into the action channel's metadata as this rig's advertised per-tick clamp.
    # Same check and same wording as the single arm's factory, because a
    # bimanual rig that published a clamp of nothing would be describing twelve
    # servos' motion limits rather than six.
    if not math.isfinite(max_relative_target) or max_relative_target <= 0:
        raise ConfigError(
            f"max_relative_target={max_relative_target} does not bound anything; "
            "the whole point of the clamp is that it is a positive per-tick limit, and "
            "a value that is zero, negative, infinite or NaN is not one. This number is "
            "published as the limit every episode was recorded under"
        )

    policy_cameras = tuple(cameras if cameras is not None else default_cameras())
    leaked = [
        spec.name for spec in policy_cameras if spec.metadata.get("policy_visible") is not True
    ]
    if leaked:
        raise ConfigError(
            f"camera(s) {leaked} are in the policy observation set but declare "
            "policy_visible != True; a stream is either offered to the policy or it is "
            "referee ground truth, and it cannot be described as both"
        )
    if referee_camera is not None:
        # The single-arm builder already refuses a referee stream named inside
        # the observation namespace. A caller can hand us any spec, so the same
        # two refusals are made here rather than assumed to have happened.
        if referee_camera.name.startswith("observation."):
            raise ConfigError(
                f"referee camera {referee_camera.name!r} is in the observation namespace; "
                "rename it (e.g. 'referee.images.*') so no training pipeline can glob it "
                "into the policy's inputs"
            )
        if referee_camera.metadata.get("policy_visible") is not False:
            raise ConfigError(
                f"referee camera {referee_camera.name!r} does not declare "
                "policy_visible=False; a ground-truth stream the policy may see is a "
                "stream the policy can exploit"
            )

    state_channel = bimanual_joint_channel(
        STATE,
        semantics=SEMANTICS_STATE,
        mounting=mounting,
        normalization=normalization,
        calibration_variant=calibration_variant,
    )
    action_channel = bimanual_joint_channel(
        ACTION,
        semantics=SEMANTICS_ACTION,
        mounting=mounting,
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
        notes=_NOTES,
        # Each arm returns to a home pose repeatably. Not seedable and not
        # simulated, so neither tag appears: an absent capability is a claim not
        # made, and the resolver reads it as one.
        capabilities=frozenset({CAP_RESETTABLE}),
        metadata={
            "arms": len(SIDES),
            "sides": list(SIDES),
            "dof_per_arm": DOF,
            "dof_total": BIMANUAL_DOF,
            "order": {
                "contract": ORDER_CONTRACT,
                "dim_labels": list(BIMANUAL_JOINT_NAMES),
                "side_blocks": {side: list(SIDE_BLOCKS[side]) for side in SIDES},
                "gripper_indices": dict(GRIPPER_INDICES),
                "per_arm_joint_names": list(JOINT_NAMES),
            },
            "mounting": mounting.as_dict(),
            "teleoperation": dict(TELEOPERATION),
            "evaluator": {"registered": False, "why": NO_EVALUATOR},
            "bus_skew": dict(BUS_SKEW),
            "provisioning": {
                **dict(PROVISIONING),
                "arms": len(SIDES),
                "bus": (
                    "one daisy-chained TTL serial bus per arm, usually on two separate "
                    "USB devices; the two halves of a command are therefore written in "
                    "sequence and not simultaneously"
                ),
            },
            "calibration": {
                "variant": calibration_variant,
                "normalization": normalization,
                "range": list(NORMALIZATION_RANGES[normalization]),
                "per_arm": (
                    "each arm carries its own calibration file and its own hash; two arms "
                    "under different variants are two different mappings from these "
                    "numbers to poses, and the vector cannot show it"
                ),
                "why_it_matters": (
                    "so101_old_calib zeroes at horizontal extension, so101_new_calib at "
                    "mid-range; the same recorded numbers describe different poses under "
                    "the two and no structural check can tell"
                ),
                "known_issue": (
                    "LeRobot #3758: a ~17 deg joint calibration offset produced ~6 cm of "
                    "gripper error, was invisible during training because the policy "
                    "learns the biased mapping, and the loss looked healthy. On a "
                    "bimanual rig it is worse: an offset on one arm alone breaks the "
                    "spatial relationship between the two, which is the only thing a "
                    "bimanual task is about"
                ),
            },
            "kinematics_ref": KINEMATICS_REF,
            "kinematics_defects": [dict(defect) for defect in KINEMATICS_DEFECTS],
            "control": {
                "rate_hz": CONTROL_HZ,
                "expected_latency": EXPECTED_LATENCY.as_dict(),
                "chunking": dict(CHUNKING),
                "note": (
                    "the latency budget is one arm's; two buses are driven in sequence, "
                    "so a bimanual tick costs more than a single-arm tick by an amount "
                    "nobody here has measured (see bus_skew)"
                ),
            },
            "safety": {
                "max_relative_target": max_relative_target,
                "units": "percent of calibrated travel per control tick",
                "applies_to": "all twelve joints, per arm, independently",
                "lerobot_default": None,
                "why": (
                    "LeRobot leaves this None, i.e. unclamped; one bad action then "
                    "commands an arm across its full range in a single tick. On this rig "
                    "the two arms can also reach each other, so an unclamped command is a "
                    "collision between two machines rather than one machine and a table"
                ),
            },
            "sensors": {
                "policy_visible": [spec.name for spec in policy_cameras],
                # Stored as data, outside `state`, so nothing that iterates the
                # observation channels can reach it. Read it back with
                # embodiment.referee_channels().
                "referee": (
                    [spec_to_dict(referee_camera)] if referee_camera is not None else []
                ),
                "intrinsics": (
                    "uncalibrated on this rig; without intrinsics no pixel measurement "
                    "becomes a metric quantity"
                ),
                "force_sensing": False,
                "torque_sensing": False,
            },
        },
    )


# --------------------------------------------------------------------------
# retargeters
# --------------------------------------------------------------------------
#
# Two, and no more. What is deliberately absent:
#
#   * A left<->right mirror. Mirroring a trajectory is not a relabelling: the two
#     arms are mounted at different yaws in a workspace that is not symmetric
#     about the midline, and every joint's calibrated travel is that arm's own.
#     A "mirror" retargeter would look correct on a symmetric fixture and be
#     wrong by whatever the two calibrations differ by, which is #3758 again.
#   * Anything that composes with the single arm's angle conversions. Chaining
#     two lossy judgements produces a third judgement nobody made, and the
#     resolver deliberately does not compose retargeters.


def _range_of(spec: ChannelSpec) -> tuple[float, float]:
    return (
        float(spec.metadata.get("range_min", 0.0)),
        float(spec.metadata.get("range_max", 100.0)),
    )


def _in_travel(values: np.ndarray, spec: ChannelSpec, labels: Sequence[str], who: str) -> None:
    """Refuse out-of-travel numbers instead of clamping them.

    A near-copy of the single arm's check, kept local for one reason: this one
    reports ``left_gripper.pos`` where that one would report ``gripper.pos``. On a
    two-arm rig "the gripper is out of range" names neither arm, and an operator
    reading it walks to the wrong side of the bench.

    Clamping is the coercion this framework exists to prevent: it turns a
    calibration bug or a diverged policy into motion that looks deliberate.
    """
    low, high = _range_of(spec)
    array = np.asarray(values, dtype=float)
    width = len(labels)
    if array.ndim != 2 or array.shape[1] != width:
        raise ConfigError(f"{who}: expected (steps, {width}), got {array.shape}")
    # NaN is the one bad value a range test cannot see: it compares False against
    # both bounds, so `< low | > high` catches +/-inf and lets NaN straight
    # through. A NaN in a commanded half is not a position at one end of travel,
    # it is a number naming no pose at all, and it is checked here for the same
    # reason SingleToBimanual already refuses a non-finite rest pose. There is no
    # bimanual arm layer to catch it later: on this ref these two transforms are
    # the last thing between a diverged policy and a recorded action.
    finite = np.isfinite(array)
    if not finite.all():
        first = np.argwhere(~finite)[0]
        step, column = int(first[0]), int(first[1])
        raise ConfigError(
            f"{who}: {spec.name!r} step {step} has {labels[column]}="
            f"{array[step, column]:g}, which is not a finite position; refusing rather "
            "than passing it on, because a non-finite command names no pose and a "
            "refusal here can still say which transform produced it"
        )
    bad = np.argwhere((array < low) | (array > high))
    if bad.size:
        step, column = int(bad[0][0]), int(bad[0][1])
        raise ConfigError(
            f"{who}: {spec.name!r} step {step} has {labels[column]}="
            f"{array[step, column]:g}, outside the declared travel [{low:g}, {high:g}]; "
            "refusing rather than clamping, because a clamped bad command still moves "
            "the arm and reads as intended motion"
        )


_SO101_SEMANTICS = (SEMANTICS_STATE, SEMANTICS_ACTION)


def _check_bimanual(spec: ChannelSpec, what: str) -> Verdict:
    """Is this actually a normalized twelve-value SO-101 bimanual vector?"""
    if spec.dim_labels != BIMANUAL_JOINT_NAMES:
        return Verdict.no(
            "retarget.not_so101_bimanual",
            f"{what} {spec.name!r} is labelled {list(spec.dim_labels or ())}, not the "
            "SO-101 bimanual joint order",
            hint=f"expected {list(BIMANUAL_JOINT_NAMES)}",
        )
    if spec.frame != BIMANUAL_JOINT_FRAME:
        return Verdict.no(
            "frame.mismatch",
            f"{what} {spec.name!r} is in frame {spec.frame!r}, this retargeter reads "
            f"{BIMANUAL_JOINT_FRAME!r}",
        )
    if spec.metadata.get("normalization") not in NORMALIZATION_RANGES:
        return Verdict.no(
            "retarget.no_normalization",
            f"{what} {spec.name!r} declares no known normalization, so its numbers cannot "
            "be placed on a travel range",
            hint=f"set metadata['normalization'] to one of {sorted(NORMALIZATION_RANGES)}",
        )
    return Verdict.yes()


def _check_single(spec: ChannelSpec, what: str) -> Verdict:
    """Is this the single-arm follower's six-value channel?"""
    if spec.dim_labels != JOINT_NAMES:
        return Verdict.no(
            "retarget.not_so101_joints",
            f"{what} {spec.name!r} is labelled {list(spec.dim_labels or ())}, not the "
            "SO-101 joint order",
            hint=f"expected {list(JOINT_NAMES)}",
        )
    if spec.frame != FOLLOWER_JOINT_FRAME:
        return Verdict.no(
            "frame.mismatch",
            f"{what} {spec.name!r} is in frame {spec.frame!r}, this retargeter reads "
            f"{FOLLOWER_JOINT_FRAME!r}",
        )
    if spec.metadata.get("normalization") not in NORMALIZATION_RANGES:
        return Verdict.no(
            "retarget.no_normalization",
            f"{what} {spec.name!r} declares no known normalization, so its numbers cannot "
            "be placed on a travel range",
            hint=f"set metadata['normalization'] to one of {sorted(NORMALIZATION_RANGES)}",
        )
    return Verdict.yes()


def _agree_on(source: ChannelSpec, target: ChannelSpec, key: str) -> Verdict:
    """Both sides must declare the same value for a load-bearing key."""
    mine, theirs = source.metadata.get(key), target.metadata.get(key)
    if mine is None or theirs is None:
        return Verdict.no(
            "retarget.undeclared",
            f"{key!r} is declared on only one side ({source.name!r}={mine!r}, "
            f"{target.name!r}={theirs!r}); an undeclared discriminator is not a match, "
            "it is an unanswered question",
        )
    if mine != theirs:
        return Verdict.no(
            f"retarget.{key}",
            f"{source.name!r} is {mine!r} and {target.name!r} is {theirs!r}",
            hint="the same number means a different pose under the two",
        )
    return Verdict.yes()


def _same_semantics(source: ChannelSpec, target: ChannelSpec) -> Verdict:
    """A measurement is not a command, whatever its width.

    Both SO-101 meaning tags are dimensionless percent-of-travel, so nothing
    dimensional separates them; the tag is the only thing that does. Letting a
    state channel feed an action channel here would turn "where the arm is" into
    "where the arm is told to go", which on a two-arm rig means the idle arm gets
    commanded to wherever it was last seen — motion, from a transform that was
    only asked to reshape.
    """
    if source.semantics not in _SO101_SEMANTICS or target.semantics not in _SO101_SEMANTICS:
        return Verdict.no(
            "semantics.mismatch",
            f"{source.name!r}={source.semantics!r} and {target.name!r}={target.semantics!r}; "
            f"this retargeter reads and writes {list(_SO101_SEMANTICS)}",
        )
    if source.semantics != target.semantics:
        return Verdict.no(
            "semantics.mismatch",
            f"{source.name!r} is {source.semantics!r} and {target.name!r} is "
            f"{target.semantics!r}; a measured position is not a commanded one",
        )
    return Verdict.yes()


class BimanualToSingle(Retargeter):
    """One arm of a bimanual vector, as a single-arm SO-101 channel.

    Structurally a slice. Semantically the deletion of half a machine, which is
    why it is a named transform whose loss lands in the run's provenance instead
    of a plain binding that would tell a later reader nothing was given up.
    """

    def __init__(self, side: str):
        self.side = _require_side(side)
        self.dropped = other_side(self.side)

    @property
    def name(self) -> str:
        return f"so101-bimanual-to-single-{self.side}"

    @property
    def version(self) -> str:
        return VERSION

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        return Verdict.all(
            [
                _check_bimanual(source, "source"),
                _check_single(target, "target"),
                _same_semantics(source, target),
                _agree_on(source, target, "normalization"),
                _agree_on(source, target, "calibration_variant"),
                _agree_on(source, target, "control_mode"),
            ]
        )

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        mounting = source.metadata.get(MOUNTING_DISCRIMINATOR)
        return (
            f"drops the {self.dropped} arm's six degrees of freedom entirely "
            f"({list(side_labels(self.dropped))}): half the action space is gone, and the "
            "six-wide result carries no trace that it was ever there",
            "any task requiring both arms is unrepresentable in the result (a handover, "
            "a two-handed lift, one arm stabilising while the other works), so a success "
            f"rate computed from the {self.side} half alone is a success rate on a "
            "different task, not a worse measurement of the same one",
            f"the {self.dropped} arm is still physically on the table at whatever pose it "
            "was last commanded to. This transform says nothing about where that is, and "
            "it remains an obstacle, an occluder and a possible collision partner that "
            "the six remaining values cannot describe",
            f"the mounting geometry ({MOUNTING_DISCRIMINATOR}={mounting!r}) leaves with the "
            "second arm: the single-arm channel declares no mounting discriminator, so a "
            "corpus produced through this transform no longer records which bimanual rig "
            "it came from and can be pooled with corpora from a different one",
        )

    def apply(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != BIMANUAL_DOF:
            raise ConfigError(f"{self.name}: expected (steps, {BIMANUAL_DOF}), got {array.shape}")
        _in_travel(array, source, BIMANUAL_JOINT_NAMES, self.name)
        # A copy, not a view onto the caller's buffer: a single-arm command that
        # aliases half of a bimanual array turns any later in-place edit into a
        # silent retroactive change to what was commanded.
        return array[:, side_slice(self.side)].astype(target.dtype, copy=True)


class SingleToBimanual(Retargeter):
    """A single-arm trajectory placed in one half; the other half held, by declaration.

    The other six values are **not** in the source. They are invented here, and
    the only honest way to invent them is to say out loud which pose is being
    commanded and why. Hence ``rest_pose`` and ``rest_pose_source``: both
    required, no defaults.

    An implicit zero would be the dangerous version. In ``RANGE_0_100`` zero is
    not "do nothing" and not "off" — it is a commanded absolute position at one
    end of every joint's calibrated travel. A padded half of zeros drives six
    servos into their stops, at speed, on the first tick, and every episode of it
    looks like data.
    """

    def __init__(
        self,
        side: str,
        *,
        rest_pose: Sequence[float],
        rest_pose_source: str,
        rest_pose_normalization: str = NORMALIZATION,
    ):
        self.side = _require_side(side)
        self.idle = other_side(self.side)
        pose = np.asarray(rest_pose, dtype=float)
        if pose.ndim != 1 or pose.shape[0] != DOF:
            raise ConfigError(
                f"{self.name}: rest_pose must be {DOF} values in the order "
                f"{list(JOINT_NAMES)}, got shape {pose.shape}"
            )
        if not np.all(np.isfinite(pose)):
            raise ConfigError(f"{self.name}: rest_pose contains a non-finite value")
        if not rest_pose_source.strip():
            raise ConfigError(
                f"{self.name}: rest_pose_source is required. This transform commands six "
                "servos to a pose that no data asked for; where that pose came from is "
                "carried into every record it produces, and 'someone chose it once' is "
                "not recoverable afterwards"
            )
        if rest_pose_normalization not in NORMALIZATION_RANGES:
            raise ConfigError(
                f"{self.name}: unknown rest_pose_normalization "
                f"{rest_pose_normalization!r}; known: {sorted(NORMALIZATION_RANGES)}"
            )
        low, high = NORMALIZATION_RANGES[rest_pose_normalization]
        outside = [
            f"{JOINT_NAMES[i]}={pose[i]:g}"
            for i in range(DOF)
            if not low <= pose[i] <= high
        ]
        if outside:
            raise ConfigError(
                f"{self.name}: rest_pose {outside} is outside {rest_pose_normalization} "
                f"[{low:g}, {high:g}]; a rest pose the arm cannot reach is a command that "
                "will be refused on the first tick of every episode"
            )
        self.rest_pose = pose
        self.rest_pose_source = rest_pose_source
        self.rest_pose_normalization = rest_pose_normalization

    @classmethod
    def at_mid_travel(cls, side: str) -> "SingleToBimanual":
        """The idle arm parked at 50 % of every joint's calibrated travel.

        A convenience with its justification attached, not a default. Mid-travel
        is the one pose guaranteed to be inside every joint's envelope whatever
        that arm's calibration says, which makes it safe to command and says
        nothing at all about whether it is out of the working arm's way. On a rig
        where it is not, pass a measured pose instead.
        """
        low, high = NORMALIZATION_RANGES[NORMALIZATION]
        middle = (low + high) / 2.0
        return cls(
            side,
            rest_pose=(middle,) * DOF,
            rest_pose_normalization=NORMALIZATION,
            rest_pose_source=(
                f"mid-travel ({middle:g} in {NORMALIZATION}) on every joint: the one pose "
                "inside every joint's calibrated envelope regardless of that arm's "
                "calibration. Chosen for reachability only: it is not a measured "
                "stow pose and it is not known to be clear of the working arm"
            ),
        )

    @property
    def name(self) -> str:
        return f"so101-single-to-bimanual-{getattr(self, 'side', '?')}"

    @property
    def version(self) -> str:
        return VERSION

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        checks = [
            _check_single(source, "source"),
            _check_bimanual(target, "target"),
            _same_semantics(source, target),
            _agree_on(source, target, "normalization"),
            _agree_on(source, target, "calibration_variant"),
            _agree_on(source, target, "control_mode"),
        ]
        # The rest pose is six numbers on a scale. Written under one convention
        # and executed under another, 50 means mid-travel in RANGE_0_100 and
        # three quarters of the way to one end in RANGE_M100_100 — same number,
        # different pose, no structural difference.
        if target.metadata.get("normalization") != self.rest_pose_normalization:
            checks.append(
                Verdict.no(
                    "retarget.rest_pose_normalization",
                    f"the rest pose is written in {self.rest_pose_normalization!r} and "
                    f"{target.name!r} is {target.metadata.get('normalization')!r}",
                    hint="the same number names a different pose on the two ranges",
                )
            )
        else:
            low, high = _range_of(target)
            outside = [
                f"{JOINT_NAMES[i]}={self.rest_pose[i]:g}"
                for i in range(DOF)
                if not low <= self.rest_pose[i] <= high
            ]
            if outside:
                checks.append(
                    Verdict.no(
                        "retarget.rest_pose_range",
                        f"rest pose {outside} is outside {target.name!r}'s declared travel "
                        f"[{low:g}, {high:g}]",
                    )
                )
        return Verdict.all(checks)

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        pose = ", ".join(
            f"{label}={value:g}" for label, value in zip(JOINT_NAMES, self.rest_pose)
        )
        low, high = _range_of(target)
        # Where zero actually sits on THIS channel's range, rather than printing
        # the bottom of the range and calling it zero. The two coincide in
        # RANGE_0_100 and do not in RANGE_M100_100, where zero is mid-travel and
        # "a zero-filled half drives six servos into their stops" is simply
        # false. A loss statement is the one thing a later reader cannot check
        # against anything, so one that is wrong under a legal normalization is
        # worse than a vague one: it lands in the provenance reading as a fact.
        if low < 0.0 < high:
            zero_note = (
                f"0 in {self.rest_pose_normalization} is mid-travel on this range rather "
                "than an end of it, so a zero-filled half is a commanded absolute position "
                "at that pose, which is still not an idle arm"
            )
        else:
            zero_note = (
                f"0 in {self.rest_pose_normalization} is a commanded absolute position at "
                f"one end of every joint's calibrated travel (the declared range is "
                f"[{low:g}, {high:g}]), so an unset or zero-filled half would drive six "
                "servos into their stops rather than leaving them alone"
            )
        return (
            f"the {self.idle} arm's six values are not in the source: they are invented "
            f"here and held at the declared rest pose [{pose}] in "
            f"{self.rest_pose_normalization}. Provenance: {self.rest_pose_source}",
            "holding is an explicit choice, not the absence of one: every number in the "
            "idle half is an absolute commanded position and no value means 'leave that "
            "arm where it is'. " + zero_note,
            f"the result is not a bimanual demonstration. It is a single-arm trajectory "
            f"with a constant attached, and a policy or metric consuming it sees a "
            f"{self.idle} arm that never moves: a distribution no bimanual rig produces "
            "while performing a bimanual task",
            f"the {self.idle} arm is somewhere already, and this transform does not know "
            "where. The first commanded tick can therefore be a full-range move from its "
            "actual pose to the rest pose; the per-tick relative-target clamp spreads that "
            "move over many ticks but does not prevent it",
            "no information about the two arms' coordination is added, because none "
            "exists in the source: inter-arm timing, contact and handover are exactly what "
            "a bimanual corpus is for and exactly what padding cannot supply",
        )

    def apply(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != DOF:
            raise ConfigError(f"{self.name}: expected (steps, {DOF}), got {array.shape}")
        _in_travel(array, source, JOINT_NAMES, self.name)
        out = np.empty((array.shape[0], BIMANUAL_DOF), dtype=float)
        out[:, side_slice(self.side)] = array
        out[:, side_slice(self.idle)] = self.rest_pose
        # Checked against the target's declared travel, which is the range the
        # rig will actually be driven over — the constructor only saw the
        # module's convention, not this channel's.
        _in_travel(out, target, BIMANUAL_JOINT_NAMES, self.name)
        return out.astype(target.dtype, copy=False)


__all__ = [
    "ACTION",
    "BIMANUAL_DOF",
    "BIMANUAL_JOINT_FRAME",
    "BIMANUAL_JOINT_NAMES",
    "BUS_SKEW",
    "DEFAULT_MOUNTING",
    "FRONT_CAMERA",
    "GRIPPER_INDICES",
    "LEFT_WRIST_CAMERA",
    "MOUNTING_DISCRIMINATOR",
    "NO_EVALUATOR",
    "ORDER_CONTRACT",
    "RIGHT_WRIST_CAMERA",
    "SIDES",
    "SIDE_BLOCKS",
    "STATE",
    "TELEOPERATION",
    "TOP_CAMERA",
    "VERSION",
    "WRIST_CAMERAS",
    "BimanualToSingle",
    "MountingGeometry",
    "SingleToBimanual",
    "bimanual_joint_channel",
    "default_cameras",
    "dim_index",
    "halves",
    "join",
    "other_side",
    "side_labels",
    "side_slice",
    "so101_bimanual",
]
