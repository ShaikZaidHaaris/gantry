"""One SO-ARM101, and a leader-follower pair of them.

This is the layer between :mod:`.transport` (bytes on a serial bus) and the
evaluator (RunRecords). It owns three things nothing above or below it can own:
calibration, wall-clock rate, and the health signals that tell you a corpus is
being recorded through a rig that has quietly moved.

What the hardware is
--------------------
Six Feetech STS3215 serial-bus servos per arm, daisy-chained to a Waveshare
driver board over USB. **Position control only. No force sensing, no torque
sensing, anywhere.** Every contact signal in this file is therefore a *proxy*
inferred from position error, and is labelled as one.

Two arms is one leader-follower pair — one operator station — not two rigs.

The vector
----------
LeRobot's ``action`` and ``observation.state`` for this arm are ``float32[6]``
in exactly this order::

    shoulder_pan.pos  shoulder_lift.pos  elbow_flex.pos
    wrist_flex.pos    wrist_roll.pos     gripper.pos

The gripper is the sixth DoF, not a separate binary channel. The order is
carried as ``dim_labels`` on every channel this module describes, because a
six-wide float32 vector matches another six-wide float32 vector by coincidence
otherwise — and two arms whose joints are in different orders are compatible on
every other axis and produce garbage.

The values are LeRobot's ``RANGE_0_100`` normalisation, which is a *percent of
the calibrated travel of that joint*, not an angle. Percent is what the bus
speaks; degrees and millimetres are what a person can act on, so this module
converts to both — using the calibrated tick range from the calibration file,
never a nominal constant.

Why the calibration hash is in here
-----------------------------------
``CORPUS-PROTOCOL.md`` requires the SHA-256 of both arms' calibration JSONs in
every episode's metadata. That is not bookkeeping: it is the only thing that
makes a mid-campaign recalibration detectable *retrospectively*, after four
weeks of recording, when the corpus is already on disk and the operator does
not remember which Tuesday they re-homed the wrist.

Why the drift detector is the most important thing here
-------------------------------------------------------
LeRobot issue #3758 is open with no maintainer response. A joint calibration
offset of roughly 17 degrees produced roughly 6 cm of gripper error, and it was
**invisible during training** — a constant offset is a bijection, so the policy
learns the biased mapping and the loss looks healthy the whole way down. It
only appears at evaluation, on an un-offset robot, as a policy that reaches
confidently for the wrong place.

Nothing in a training curve will find this. A stable-frame check will: command
the arm to hold still, and ``commanded - measured`` should centre on zero for
every joint. A persistent non-zero centre is an offset. This module reports
that centre per joint in degrees *and* in approximate Cartesian millimetres at
the gripper, because millimetres are the unit in which a person decides whether
to stop and recalibrate.

The honest limits of that check are stated on :class:`DriftReport` and are not
decoration: it cannot see an error common to both arms, it cannot see a
kinematic error, and gravity sag masquerades as drift unless you compare
against a baseline captured at the same pose.

What this module refuses to do
------------------------------
Pad an action vector. Guess a width. Clamp silently. Coerce a normalisation.
Report a bare float. If the data does not fit, it refuses, and the refusal
names the thing that did not fit.

Required transport surface
--------------------------
:mod:`.transport` must expose, on both ``Transport`` and ``MockTransport``:

============================================  ====================================
``open()``                                    claim the port
``close()``                                   release it
``ping(motor_id) -> bool``                    is that id on the bus
``read_present_position(ids) -> Reading``     raw encoder counts plus the
                                              instant they were taken
``write_goal_position({id: ticks})``          raw encoder counts
``torque_enable(ids)`` / ``torque_disable``   holding torque on or off
============================================  ====================================

Raw ticks, not normalised units: normalisation needs the calibration, and the
calibration lives here. A transport that normalised on its own would be a
second, uncheckable copy of the mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.embodiment import (
    CAP_RESETTABLE,
    EmbodimentDescriptor,
)
from gantry.errors import ComponentError, ConfigError, FaultError, SafetyAbort
from gantry.spine import ChannelSpec, Measurement, Verdict

# Both modules define a Calibration and a SafetyLimits, and they are genuinely
# different types describing different things — a verified file here, a register
# map and a tick budget there. Aliased rather than re-exported so that every use
# site below says which one it means; a bare `Calibration` in this file is always
# the file.
#
# _round_half_away is imported rather than reimplemented, private name and all,
# because rounding halves away from zero is a *decision* the bus documents —
# `round` alone is banker's rounding, which makes a conversion depend on the
# parity of the tick it lands between. A second copy of that rule here would be a
# second thing to fall out of step with the first, and the symptom would be a
# one-tick disagreement confined to the commanded values that land exactly
# halfway between two counts: too small to notice, too systematic to be noise.
from .transport import Calibration as BusCalibration
from .transport import MockTransport, Transport, _round_half_away
from .transport import SafetyLimits as BusLimits

VERSION = "0.1.0.dev0"

# --------------------------------------------------------------------------
# the vector, and the hardware constants behind it
# --------------------------------------------------------------------------

#: The LeRobot action / observation.state key order for SO-ARM101. Load-bearing
#: as an *order*, not just as a set: see the module docstring.
JOINT_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: The same joints as the calibration file names them — no ``.pos`` suffix.
MOTOR_NAMES: tuple[str, ...] = tuple(key.split(".")[0] for key in JOINT_KEYS)

DOF = len(JOINT_KEYS)

#: Index of the gripper within the vector. Named rather than written as ``5``
#: at four call sites, because "element 5" is exactly the kind of positional
#: reference this framework exists to delete.
GRIPPER = DOF - 1

#: Wall-clock control rate of the pair. Not a target — a *contract*. A corpus
#: recorded at a variable rate is confounded in a way that is invisible
#: afterwards, because the frames carry no evidence of how far apart they were.
CONTROL_HZ = 30.0

#: STS3215 encoder resolution. One mechanical revolution of the output horn.
TICKS_PER_REV = 4096

#: The normalisation LeRobot applies to this arm, carried as a channel
#: discriminator so a corpus recorded under a different convention cannot be
#: matched to this one by width alone.
NORMALIZATION = "RANGE_0_100"

#: The calibration flavour these files are written by. Also a discriminator:
#: the old and new SO-101 calibration procedures produce different homing
#: offsets for the same physical arm.
CALIBRATION_VARIANT = "so101_new_calib"

#: How far outside 0..100 a commanded value may sit before it is refused.
#: A leader arm at the very end of its travel legitimately reads a little past
#: 100 — refusing that would abort a session on a shrug. Anything past this is
#: not slop, it is a wrong number, and it is refused rather than clamped.
NORM_TOLERANCE = 2.0

#: Approximate distance from each joint axis to the gripper tip, in millimetres,
#: with the arm at a nominal mid-workspace extension.
#:
#: These are **declared approximations for reporting only** — a first-order,
#: small-angle lever arm used to turn an angular offset into the unit a person
#: can act on. They are not kinematics and nothing actuates through them. They
#: are configurable per rig (:class:`Geometry`) because an arm with a different
#: end effector has different arms; the defaults are the stock SO-101.
#:
#: Sanity check against the issue this exists for: 17 degrees on ``elbow_flex``
#: is 0.297 rad x 190 mm = 56 mm, which is the ~6 cm reported in LeRobot #3758.
LEVER_ARM_MM: Mapping[str, float] = {
    "shoulder_pan.pos": 300.0,
    "shoulder_lift.pos": 250.0,
    "elbow_flex.pos": 190.0,
    "wrist_flex.pos": 100.0,
    "wrist_roll.pos": 25.0,
}

#: Full jaw travel of the stock two-finger gripper, in millimetres of aperture.
#: The gripper's error is an aperture, not a displacement of the tip, so it is
#: reported in a different frame and never mixed into a max over the others.
GRIPPER_APERTURE_MM = 35.0

#: Which end of the gripper's normalised range is the closed jaw. A *claim*
#: about the ``so101_new_calib`` convention, overridable per rig, and load-
#: bearing for exactly one thing: whether a stall is labelled ``closing``.
#: Stall *detection* itself is polarity-free, so getting this wrong mislabels a
#: field rather than losing a grasp signal.
GRIPPER_CLOSED_AT_ZERO = True

#: Frames the millimetre figures are expressed in. Two different physical
#: quantities; they are never summed or maxed together.
BASE_FRAME = "base"
JAW_FRAME = "gripper_jaw"

#: Everything :mod:`.transport` must expose. Checked once, by name, at connect,
#: so a mismatch is a refusal that names the missing method rather than an
#: AttributeError from inside a 30 Hz loop with an arm in the air.
REQUIRED_TRANSPORT = (
    "open",
    "close",
    "ping",
    "read_present_position",
    "write_goal_position",
    # Enable and disable are separate calls on the bus rather than one boolean,
    # and they are not symmetric: enabling first writes the present position
    # into the goal register so the arm does not snap to a stale target the
    # instant it is energised. Naming both here is what makes that asymmetry a
    # requirement of the transport rather than a detail one implementation
    # happens to honour.
    "torque_enable",
    "torque_disable",
)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def sha256_of_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, exactly as it sits on disk.

    Of the bytes and not of the parsed contents: a re-serialisation with
    different key order or whitespace is a different file, and the question
    this hash answers is "is this the same file the corpus was recorded
    against", not "does it mean the same thing".
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class JointCalibration:
    """One servo's calibration, as LeRobot writes it."""

    name: str
    id: int
    drive_mode: int
    #: Written into the servo's EEPROM by the calibration procedure, and
    #: therefore already reflected in every raw count the chain reports. It is
    #: kept here, and deliberately absent from the arithmetic below — see
    #: :meth:`to_ticks`.
    homing_offset: int
    range_min: int
    range_max: int

    @property
    def span_ticks(self) -> int:
        return self.range_max - self.range_min

    @property
    def deg_per_unit(self) -> float:
        """Degrees of joint travel per normalised unit, from *this* joint's range.

        Derived from the calibrated span rather than a nominal constant. Two
        joints with different calibrated ranges have different degrees per
        unit, and a drift figure computed from a shared nominal would be wrong
        by whatever the ranges differ by — which is the same class of error the
        drift detector exists to find.
        """
        return self.span_ticks * (360.0 / TICKS_PER_REV) / 100.0

    def to_ticks(self, value: float) -> int:
        """Normalised units -> raw encoder ticks.

        LeRobot's ``RANGE_0_100`` denormalisation and nothing else::

            raw = value / 100 * (range_max - range_min) + range_min

        **There is no ``homing_offset`` term, and its absence is the point.**
        The calibration procedure writes that offset into the servo's own EEPROM
        (``write('Homing_Offset', motor, offset)``), so the chain applies it in
        hardware and every raw count it reports already carries it. Adding it
        again in software applies it twice, and a goal computed that way lands
        ``homing_offset`` ticks away from the pose the number names.

        That is LeRobot #3758 reproduced inside the package written to detect
        it: a constant per-joint offset between the commanded number and the
        pose it produces. It passes every structural check — the vector is the
        right width, the dtype is right, the units are declared, the round trip
        is self-consistent — because it is a bijection, so training absorbs it
        and the loss never mentions it. It surfaces at evaluation, on the
        un-offset robot, as confident reaching to the wrong place.

        It also survived 124 passing checks in this package, because every mock
        here homes at zero and at ``homing_offset=0`` a doubled term is
        invisible. Section [3] of the selftest is the guard: it holds this
        against :class:`transport.JointCalibration` over the full range of every
        joint of a calibration whose offsets are non-zero and of mixed sign.
        """
        # Mirrored statement for statement from the bus's own conversion,
        # including the order of the drive_mode reflection and the rounding
        # rule, so the two agree bit for bit rather than to within a tick.
        if self.drive_mode:
            value = 100.0 - value
        ratio = value / 100.0
        return _round_half_away(ratio * self.span_ticks + self.range_min)

    def to_units(self, ticks: int) -> float:
        """Raw encoder ticks -> normalised units. The inverse of :meth:`to_ticks`.

        ``((ticks - range_min) / (range_max - range_min)) * 100``, which is
        LeRobot's ``_normalize`` for ``RANGE_0_100``. No ``homing_offset`` term
        here either, for the reason given on :meth:`to_ticks`: the count that
        arrives from the servo has already had it applied.

        Values outside 0..100 are returned as they are. A joint sitting outside
        its calibrated envelope is a fact about the arm, and a reader that
        clamped it would be hiding the single clearest symptom of a servo horn
        that has slipped.
        """
        value = (ticks - self.range_min) / self.span_ticks * 100.0
        return 100.0 - value if self.drive_mode else value


@dataclass(frozen=True)
class Calibration:
    """One arm's calibration file, verified, with its hash."""

    path: Path
    sha256: str
    joints: tuple[JointCalibration, ...]
    variant: str = CALIBRATION_VARIANT

    def __post_init__(self) -> None:
        if len(self.joints) != DOF:
            raise ConfigError(
                f"{self.path}: {len(self.joints)} joint(s), this arm has {DOF}"
            )

    @property
    def short(self) -> str:
        """First 16 hex characters. For logs; never for identity comparisons."""
        return self.sha256[:16]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(joint.id for joint in self.joints)

    def joint(self, key: str) -> JointCalibration:
        index = JOINT_KEYS.index(key)
        return self.joints[index]

    @property
    def deg_per_unit(self) -> np.ndarray:
        return np.array([joint.deg_per_unit for joint in self.joints], dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        """The provenance stamp. Path *and* hash: one says where to look, the
        other says whether what is there now is what was used."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "variant": self.variant,
            "ids": list(self.ids),
        }


def load_calibration(
    path: str | Path, *, variant: str = CALIBRATION_VARIANT
) -> Calibration:
    """Read and *verify* a LeRobot SO-101 calibration file.

    Verification is the point. A calibration that parses is not a calibration
    that is usable: a missing joint, a duplicated motor id, or an inverted
    range each produce an arm that moves and is wrong, and each of them is
    cheap to catch here and expensive to catch in a corpus.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path}: no calibration file, so this arm cannot be trusted")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: not readable as JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path}: expected an object keyed by joint name")

    missing = [name for name in MOTOR_NAMES if name not in payload]
    if missing:
        raise ConfigError(
            f"{path}: missing joint(s) {missing}; this arm has exactly {list(MOTOR_NAMES)}"
        )
    extra = [name for name in payload if name not in MOTOR_NAMES]
    if extra:
        # Not padded over and not ignored: an extra joint means this file
        # belongs to a different arm, and using it would put six of seven
        # calibrations on the right motors and be silently wrong about which.
        raise ConfigError(
            f"{path}: describes joint(s) {sorted(extra)} this arm does not have"
        )

    joints: list[JointCalibration] = []
    for name in MOTOR_NAMES:
        entry = payload[name]
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{path}: joint {name!r} is not an object")
        try:
            joint = JointCalibration(
                name=name,
                id=int(entry["id"]),
                drive_mode=int(entry.get("drive_mode", 0)),
                homing_offset=int(entry["homing_offset"]),
                range_min=int(entry["range_min"]),
                range_max=int(entry["range_max"]),
            )
        except KeyError as error:
            raise ConfigError(f"{path}: joint {name!r} has no {error.args[0]!r}") from error
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{path}: joint {name!r} has a non-integer field: {error}") from error
        if joint.span_ticks <= 0:
            raise ConfigError(
                f"{path}: joint {name!r} has range_min={joint.range_min} >= "
                f"range_max={joint.range_max}, which is not a travel"
            )
        if not 0 < joint.span_ticks <= TICKS_PER_REV:
            raise ConfigError(
                f"{path}: joint {name!r} spans {joint.span_ticks} ticks; "
                f"an STS3215 has {TICKS_PER_REV} per revolution"
            )
        joints.append(joint)

    ids = [joint.id for joint in joints]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        # Every STS3215 ships with id=1, so a bus assigned in a hurry really
        # does end up with two motors answering to the same address. They then
        # both accept every command meant for either.
        raise ConfigError(
            f"{path}: motor id(s) {duplicates} used by more than one joint; "
            "assign ids one servo at a time on an otherwise empty bus"
        )
    return Calibration(
        path=path,
        sha256=sha256_of_file(path),
        joints=tuple(joints),
        variant=variant,
    )


# --------------------------------------------------------------------------
# geometry, safety, clock
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """The declared approximations used to report angles as millimetres.

    Kept as data, kept separate from the calibration, and stamped into
    provenance, so a number reported in millimetres can always be traced back
    to the lever arms that produced it. Nothing here is used to command
    anything.
    """

    lever_arm_mm: Mapping[str, float] = field(
        default_factory=lambda: dict(LEVER_ARM_MM)
    )
    gripper_aperture_mm: float = GRIPPER_APERTURE_MM
    note: str = (
        "first-order small-angle lever arms at a nominal mid-workspace pose; "
        "for reporting only, never for control"
    )

    def mm_per_deg(self, key: str) -> float:
        if key == JOINT_KEYS[GRIPPER]:
            raise KeyError("the gripper's error is an aperture, not a tip displacement")
        return math.radians(1.0) * self.lever_arm_mm[key]

    def frame_of(self, key: str) -> str:
        return JAW_FRAME if key == JOINT_KEYS[GRIPPER] else BASE_FRAME

    def as_dict(self) -> dict[str, Any]:
        return {
            "lever_arm_mm": dict(self.lever_arm_mm),
            "gripper_aperture_mm": self.gripper_aperture_mm,
            "note": self.note,
        }


REFUSE = "refuse"
CLAMP = "clamp"


@dataclass(frozen=True)
class SafetyLimits:
    """How far one joint may be commanded to move in one control tick.

    LeRobot's ``max_relative_target`` defaults to ``None``, meaning no limit at
    all, and an unclamped bad action drives the follower into the table at full
    speed. That default is a hardware failure waiting for a bug, so this class
    has no default: constructing an arm forces the decision to be made and
    written down, and the decision travels into provenance.

    ``on_exceed`` is ``refuse`` by default. ``clamp`` exists because aborting a
    twenty-second episode on one leader jerk destroys the take, but a clamp is
    a transform that discards what the operator actually did, so every clamped
    tick is counted and the count is reported with the episode. A corpus with a
    high clamp count is not the corpus the operator thinks they recorded.
    """

    max_relative_target: tuple[float, ...]
    on_exceed: str = REFUSE
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.max_relative_target) != DOF:
            raise ConfigError(
                f"max_relative_target has {len(self.max_relative_target)} entries, "
                f"this arm has {DOF} joints"
            )
        if self.on_exceed not in (REFUSE, CLAMP):
            raise ConfigError(f"on_exceed must be {REFUSE!r} or {CLAMP!r}, got {self.on_exceed!r}")
        # NaN is refused separately and before the sign test, because it passes
        # the sign test — it compares False against everything — and then passes
        # the guard itself: `abs(delta) > nan` is False on every joint, so
        # `_apply_relative_limit` never fires, `clamped_ticks` stays 0, and the
        # episode records a clean run under a limit that bounded nothing.
        #
        # That is precisely what `unlimited()` does, and the difference is the
        # whole point of this class: `unlimited()` demands a written reason and
        # carries it into provenance, so "we turned the guard off" is a recorded
        # fact. A NaN gets the same behaviour, asks for no reason, and serialises
        # as `nan` where `as_dict` uses None to mean "no limit" — so it is not
        # even legible as a disabled guard afterwards.
        #
        # Infinity is deliberately *not* refused here. It is how `unlimited()`
        # encodes itself, and the reason requirement lives on that constructor.
        undefined = [
            JOINT_KEYS[i] for i, limit in enumerate(self.max_relative_target) if math.isnan(limit)
        ]
        if undefined:
            raise ConfigError(
                f"max_relative_target is NaN on {undefined}; a NaN limit bounds nothing "
                "(every comparison against it is False) and so silently disables the "
                "relative-target guard. To run without it, say so: "
                "SafetyLimits.unlimited(reason), which records why with every episode"
            )
        if any(limit <= 0 for limit in self.max_relative_target):
            raise ConfigError("a max_relative_target of zero or less permits no motion at all")

    @classmethod
    def teleop(cls, on_exceed: str = CLAMP) -> "SafetyLimits":
        """The preset for leader-follower recording.

        8 percent of travel per tick at 30 Hz is a full-range joint move in
        0.42 s, which is faster than a person teleoperates and slower than a
        dropped serial frame. The gripper gets 15: the jaws are light, they
        move fast on purpose, and a limited gripper misses grasps. Both numbers
        are rig-specific and both are recorded.
        """
        return cls(
            max_relative_target=(8.0, 8.0, 8.0, 8.0, 8.0, 15.0),
            on_exceed=on_exceed,
            note="teleop preset; tune per rig",
        )

    @classmethod
    def unlimited(cls, reason: str) -> "SafetyLimits":
        """No limit, which requires saying in writing why.

        Available because there are legitimate reasons (a bench with the arm
        off the table, a replay of a known-good trajectory). The reason is
        mandatory and is carried into provenance, so "we turned the guard off"
        is a recorded fact rather than a thing someone remembers.
        """
        if not reason.strip():
            raise ConfigError(
                "disabling the relative-target guard requires a written reason; "
                "it is recorded with every episode taken under it"
            )
        return cls(max_relative_target=(math.inf,) * DOF, on_exceed=REFUSE, note=reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_relative_target": [
                None if math.isinf(limit) else limit for limit in self.max_relative_target
            ],
            "on_exceed": self.on_exceed,
            "note": self.note,
        }


class Clock:
    """Wall clock and sleep, injectable so the loop can be tested at speed.

    A real control loop and a test of a real control loop differ in exactly one
    thing — whether time passes — and pulling that one thing out is what lets
    the rate discipline below be checked with no hardware and no twenty-second
    wait.
    """

    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        # Sleep to within a millisecond, then spin. time.sleep's granularity on
        # Windows is coarse enough relative to a 33 ms period to be visible in
        # the period histogram, and a control rate that wobbles is the confound
        # this whole class exists to prevent.
        coarse = seconds - 0.001
        if coarse > 0:
            time.sleep(coarse)
        deadline = self.now() + max(0.0, seconds - max(0.0, coarse))
        while self.now() < deadline:
            pass


class VirtualClock(Clock):
    """A clock that only moves when told to. For tests and for the selftest."""

    def __init__(self, start: float = 0.0):
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += seconds

    def advance(self, seconds: float) -> None:
        """Simulate work taking time — a slow bus read, a policy forward pass."""
        self._now += seconds


# --------------------------------------------------------------------------
# rate discipline
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Overrun:
    """One tick that did not fit in its period."""

    tick: int
    late_by_s: float


@dataclass(frozen=True)
class RateReport:
    """What the loop actually ran at, as opposed to what it was asked to.

    Reported rather than corrected. There is no way to command in the past, so
    an overrun cannot be made up; the only useful response is to record it, and
    the only harmful response is to let the schedule quietly slide. A corpus
    recorded at a rate that drifted is confounded on every timing-derived
    quantity — velocity, smoothness, episode duration — and the frames carry no
    evidence of it afterwards.
    """

    nominal_hz: float
    ticks: int
    elapsed_s: float
    overruns: tuple[Overrun, ...]
    periods_s: tuple[float, ...]

    @property
    def realised_hz(self) -> float:
        return (self.ticks / self.elapsed_s) if self.elapsed_s > 0 else 0.0

    @property
    def worst_late_s(self) -> float:
        return max((o.late_by_s for o in self.overruns), default=0.0)

    @property
    def late_fraction(self) -> float:
        return (len(self.overruns) / self.ticks) if self.ticks else 0.0

    def measurements(self) -> dict[str, Measurement]:
        periods = np.asarray(self.periods_s, dtype=np.float64)
        out = {
            "rate.realised_hz": Measurement(
                self.realised_hz,
                n=self.ticks,
                units="Hz",
                method=f"ticks / elapsed wall clock, nominal {self.nominal_hz} Hz",
            ),
            "rate.late_fraction": Measurement(
                self.late_fraction,
                n=self.ticks,
                units="fraction",
                method="ticks whose deadline had already passed",
            ),
            "rate.worst_late_ms": Measurement(
                self.worst_late_s * 1e3,
                n=len(self.overruns),
                units="ms",
                method="longest single overrun",
            ),
        }
        if periods.size:
            out["rate.period_jitter_ms"] = Measurement(
                float(periods.std(ddof=1) * 1e3) if periods.size > 1 else 0.0,
                n=int(periods.size),
                units="ms",
                method="standard deviation of realised tick period",
            )
        return out

    def verdict(self, *, tolerate: float = 0.01) -> Verdict:
        """Refuse a run whose rate did not hold.

        ``tolerate`` is the fraction of ticks allowed to be late. The default
        of one percent is not a physics constant; it is the level at which a
        30 Hz corpus is still honestly a 30 Hz corpus. Callers recording a
        campaign should set it and record what they set.
        """
        if not self.ticks:
            return Verdict.no("rate.no_ticks", "the loop ran no ticks")
        if self.late_fraction > tolerate:
            return Verdict.no(
                "rate.overrun",
                f"{len(self.overruns)}/{self.ticks} tick(s) missed the "
                f"{1e3 / self.nominal_hz:.1f} ms deadline "
                f"(worst {self.worst_late_s * 1e3:.1f} ms); realised "
                f"{self.realised_hz:.2f} Hz against a nominal {self.nominal_hz:.2f} Hz",
                hint="a variable control rate confounds every timing-derived metric in "
                "the corpus and leaves no trace in the frames; fix the loop or "
                "record the corpus at the rate it actually ran",
                late_fraction=self.late_fraction,
            )
        if self.overruns:
            return Verdict.note(
                "rate.overrun_within_tolerance",
                f"{len(self.overruns)}/{self.ticks} tick(s) were late, worst "
                f"{self.worst_late_s * 1e3:.1f} ms",
            )
        return Verdict.yes()


class RateKeeper:
    """Holds a fixed wall-clock period, and says so when it cannot.

    Deadlines advance from a fixed anchor so that ordinary jitter does not
    accumulate. When a deadline has already passed the anchor is moved forward
    to *now* rather than the loop being asked to catch up: firing three ticks
    back to back to recover lost time commands a physical arm faster than the
    operator moved, which is worse than the gap. The slip is recorded instead.
    """

    def __init__(self, hz: float = CONTROL_HZ, clock: Clock | None = None):
        if hz <= 0:
            raise ConfigError(f"a control rate must be positive, got {hz}")
        self.hz = float(hz)
        self.period = 1.0 / self.hz
        self._clock = clock or Clock()
        self._start = self._clock.now()
        self._next = self._start + self.period
        self._tick = 0
        self._last = self._start
        self._overruns: list[Overrun] = []
        self._periods: list[float] = []

    @property
    def clock(self) -> Clock:
        return self._clock

    def wait(self) -> float:
        """Block until the next tick boundary. Returns the realised period."""
        now = self._clock.now()
        remaining = self._next - now
        if remaining < 0:
            self._overruns.append(Overrun(self._tick, -remaining))
            self._next = now + self.period
        else:
            self._clock.sleep(remaining)
            self._next += self.period
        now = self._clock.now()
        realised = now - self._last
        self._periods.append(realised)
        self._last = now
        self._tick += 1
        return realised

    def report(self) -> RateReport:
        return RateReport(
            nominal_hz=self.hz,
            ticks=self._tick,
            elapsed_s=self._clock.now() - self._start,
            overruns=tuple(self._overruns),
            periods_s=tuple(self._periods),
        )


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------


def _as_vector(action: Any, what: str) -> np.ndarray:
    """A float32[6], or a refusal naming what arrived instead.

    Deliberately the only entry point for an externally supplied vector, and
    deliberately unforgiving. A policy that emits five values and an arm that
    accepts six by padding the sixth with a zero produces a gripper command of
    "fully closed" on every step, at speed, forever, and every episode of it
    looks like data.
    """
    array = np.asarray(action)
    if array.ndim != 1:
        raise ConfigError(
            f"{what}: expected a 1-D vector of {DOF} values, got shape {array.shape}"
        )
    if array.shape[0] != DOF:
        raise ConfigError(
            f"{what}: expected {DOF} values in the order {list(JOINT_KEYS)}, "
            f"got {array.shape[0]}",
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ConfigError(f"{what}: values are {array.dtype}, which is not numeric")
    values = array.astype(np.float32, copy=True)
    if not np.all(np.isfinite(values)):
        bad = [JOINT_KEYS[i] for i in np.flatnonzero(~np.isfinite(values))]
        raise ConfigError(f"{what}: non-finite value(s) on {bad}")
    return values


class SO101Arm:
    """One SO-ARM101: six STS3215 servos, one calibration, no force sensing."""

    def __init__(
        self,
        transport: Transport,
        calibration: Calibration,
        limits: SafetyLimits,
        *,
        role: str,
        geometry: Geometry | None = None,
    ):
        """``limits`` has no default on purpose — see :class:`SafetyLimits`.

        ``role`` is ``"leader"`` or ``"follower"`` or whatever names this arm
        in the rig; it appears in every refusal, which is the difference
        between "joint out of range" and "the follower's wrist is out of range"
        when there are two arms on the bench and one of them is in the air.
        """
        self.transport = transport
        self.calibration = calibration
        self.limits = limits
        self.role = role
        self.geometry = geometry or Geometry()
        self._connected = False
        self._last_command: np.ndarray | None = None
        self.clamped_ticks = 0
        self.envelope_excursions = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<SO101Arm {self.role} calib={self.calibration.short}>"

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Open the bus and confirm the arm on it is the arm the file describes.

        The ping sweep is not a formality. Every STS3215 ships with ``id=1``, so
        a bus that was assigned in a hurry has two motors on the same address,
        both accepting every command meant for either. Catching that here costs
        a second; catching it in a corpus costs the corpus.
        """
        missing = [
            name
            for name in REQUIRED_TRANSPORT
            if not callable(getattr(self.transport, name, None))
        ]
        if missing:
            raise ConfigError(
                f"{self.role}: transport {type(self.transport).__name__} has no "
                f"callable {missing}; this arm needs {list(REQUIRED_TRANSPORT)}"
            )
        self.transport.open()
        absent = [
            f"{name}(id={joint.id})"
            for name, joint in zip(MOTOR_NAMES, self.calibration.joints)
            if not self.transport.ping(joint.id)
        ]
        if absent:
            self.transport.close()
            raise FaultError(
                f"{self.role}: no reply from {absent} on the bus described by "
                f"{self.calibration.path} (sha256 {self.calibration.short})",
                partial={"role": self.role, "absent": absent},
            )
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self.transport.close()
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise ComponentError(f"{self.role}: not connected; call connect() first")

    def set_torque(self, enable: bool) -> None:
        """Holding torque on all six. Off is how a leader arm is backdriven.

        Two calls rather than one flag with a boolean, because the bus does not
        treat them as opposites: enabling writes the present position into every
        goal register first, so an arm that was moved by hand while de-energised
        holds where it is instead of snapping back to whatever it was last told.
        """
        self._require_connected()
        ids = list(self.calibration.ids)
        if enable:
            self.transport.torque_enable(ids)
        else:
            self.transport.torque_disable(ids)

    # -- state -------------------------------------------------------------

    def read_state(self) -> np.ndarray:
        """``float32[6]`` in :data:`JOINT_KEYS` order, normalised units.

        A partial reply is refused rather than filled in. A dropped serial
        frame on one motor is common; a state vector with a stale value in it
        is indistinguishable from a real one, and it is the input to every
        tracking and drift number below.
        """
        self._require_connected()
        ids = list(self.calibration.ids)
        reading = self.transport.read_present_position(ids)
        # The bus returns a Reading — ticks plus the instant they were taken —
        # because the clamp below it needs the timestamp. Unwrapped here rather
        # than by the transport, so the freshness rule stays owned by the layer
        # that enforces it and nothing above can hand it a bare dict.
        raw = getattr(reading, "ticks", reading)
        if not isinstance(raw, Mapping):
            raise ComponentError(
                f"{self.role}: read_present_position returned "
                f"{type(reading).__name__}, expected a Reading or a mapping of id -> ticks"
            )
        absent = [
            f"{name}(id={joint.id})"
            for name, joint in zip(MOTOR_NAMES, self.calibration.joints)
            if joint.id not in raw
        ]
        if absent:
            raise ComponentError(
                f"{self.role}: no present position for {absent}; a state vector with "
                "a stale joint in it reads exactly like a real one"
            )
        values = np.empty(DOF, dtype=np.float32)
        for index, joint in enumerate(self.calibration.joints):
            values[index] = joint.to_units(int(raw[joint.id]))
        return values

    # -- action ------------------------------------------------------------

    def write_action(self, action: Any) -> np.ndarray:
        """Command the arm. Returns what was actually sent, ``float32[6]``.

        Three separate refusals, none of which is a clamp:

        * a vector of the wrong width, rank or dtype (:func:`_as_vector`)
        * a value further than :data:`NORM_TOLERANCE` outside 0..100
        * a tick that lands outside the servo's physical 0..4095

        And one *declared* transform, off by default: the relative-target
        guard, which clamps only when the caller asked for clamping and counts
        every tick it touched.
        """
        self._require_connected()
        values = _as_vector(action, f"{self.role}: action")

        low, high = -NORM_TOLERANCE, 100.0 + NORM_TOLERANCE
        out_of_range = [
            (JOINT_KEYS[i], float(values[i]))
            for i in range(DOF)
            if not low <= values[i] <= high
        ]
        if out_of_range:
            raise SafetyAbort(
                f"{self.role}: {out_of_range} outside the calibrated 0..100 travel "
                f"(tolerance {NORM_TOLERANCE}); this is a wrong number, not slop, "
                "and it is refused rather than clamped"
            )
        self.envelope_excursions += int(
            sum(1 for i in range(DOF) if not 0.0 <= values[i] <= 100.0)
        )

        values = self._apply_relative_limit(values)

        goals: dict[int, int] = {}
        for index, joint in enumerate(self.calibration.joints):
            ticks = joint.to_ticks(float(values[index]))
            if not 0 <= ticks < TICKS_PER_REV:
                raise SafetyAbort(
                    f"{self.role}: {JOINT_KEYS[index]} at {values[index]:.2f} maps to "
                    f"tick {ticks}, outside the encoder's 0..{TICKS_PER_REV - 1}; "
                    "the command is not representable on this servo"
                )
            goals[joint.id] = ticks
        commanded = self._commanded(self.transport.write_goal_position(goals), values)
        self._last_command = commanded
        return commanded

    def _commanded(self, result: Any, asked: np.ndarray) -> np.ndarray:
        """What the bus reports it actually put in the goal registers.

        The bus has two limits of its own beneath this one — a per-write tick
        budget and each joint's measured travel — and it reports what it did
        rather than swallowing it. Reading that back is what makes the value this
        method returns the value the servos received, which is the value the
        record downstream calls ``action``.

        It matters most exactly where it is easiest to miss. Both layers now map
        0..100 onto the same measured travel, so the declared range is reachable
        end to end — but the bus still has a per-write tick budget above that,
        and a command that outruns it is cut back silently as far as this class
        can see. Returning ``asked`` there would put a number in the corpus that
        no servo was ever given, and near a travel limit it would be wrong by a
        constant, at one end of the range, in one direction — which is the shape
        that survives training.
        """
        writes = getattr(result, "writes", None)
        if writes is None:
            raise ComponentError(
                f"{self.role}: write_goal_position returned {type(result).__name__} with "
                "no per-joint report, so what the servos were actually told is unknown"
            )
        by_id = {write.id: write.commanded for write in writes}
        out = np.empty(DOF, dtype=np.float32)
        for index, joint in enumerate(self.calibration.joints):
            if joint.id not in by_id:
                raise ComponentError(
                    f"{self.role}: the bus reported no write for {JOINT_KEYS[index]} "
                    f"(id={joint.id}); a partial report cannot be recorded as a command"
                )
            out[index] = joint.to_units(int(by_id[joint.id]))
        # Whether the bus altered anything is asked in ticks, where the question
        # has an exact answer. Asking it in normalised units would compare a
        # value against its own round trip through an integer encoder and call
        # every step clamped.
        if any(write.was_clamped for write in writes):
            self.clamped_ticks += 1
        return out

    def _apply_relative_limit(self, values: np.ndarray) -> np.ndarray:
        if self._last_command is None:
            # Nothing to be relative to. The first command of a session is
            # bounded by the envelope check above and by whoever set the pose.
            return values
        limits = np.asarray(self.limits.max_relative_target, dtype=np.float64)
        delta = values.astype(np.float64) - self._last_command.astype(np.float64)
        exceeded = np.flatnonzero(np.abs(delta) > limits)
        if not exceeded.size:
            return values
        offenders = [
            f"{JOINT_KEYS[i]} {delta[i]:+.2f} > {limits[i]:.2f}" for i in exceeded
        ]
        if self.limits.on_exceed == REFUSE:
            raise SafetyAbort(
                f"{self.role}: action moves {offenders} in one {1e3 / CONTROL_HZ:.1f} ms "
                "tick; refused. An unclamped bad action drives this arm into the table."
            )
        bounded = values.astype(np.float64).copy()
        bounded[exceeded] = self._last_command.astype(np.float64)[exceeded] + np.clip(
            delta[exceeded], -limits[exceeded], limits[exceeded]
        )
        self.clamped_ticks += 1
        return bounded.astype(np.float32)

    def reset_session_counters(self) -> None:
        """Zero the per-episode counters. The last command is *not* cleared:
        the relative-target guard must still be relative to where the arm is."""
        self.clamped_ticks = 0
        self.envelope_excursions = 0

    # -- holding still, which is how drift is found ------------------------

    def hold(
        self,
        pose: Any,
        *,
        ticks: int = 90,
        settle_ticks: int = 15,
        keeper: RateKeeper | None = None,
    ) -> np.ndarray:
        """Command one pose repeatedly and return ``commanded - measured``.

        Shape ``(ticks - settle_ticks, 6)``, normalised units. The settle
        window is discarded because the first frames measure the approach, not
        the steady state, and an approach transient reads as an offset.
        """
        self._require_connected()
        target = _as_vector(pose, f"{self.role}: hold pose")
        if ticks <= settle_ticks:
            raise ConfigError(
                f"{self.role}: hold({ticks=}, {settle_ticks=}) keeps no samples"
            )
        keeper = keeper or RateKeeper(CONTROL_HZ)
        residuals: list[np.ndarray] = []
        for tick in range(ticks):
            self.write_action(target)
            keeper.wait()
            measured = self.read_state()
            if tick >= settle_ticks:
                residuals.append(target.astype(np.float64) - measured.astype(np.float64))
        return np.asarray(residuals, dtype=np.float64)

    def check_drift(
        self,
        pose: Any,
        *,
        ticks: int = 90,
        settle_ticks: int = 15,
        keeper: RateKeeper | None = None,
        baseline: "DriftBaseline | None" = None,
        actionable_mm: float = 2.0,
    ) -> "DriftReport":
        """The stable-frame calibration check. See :class:`DriftReport`."""
        residuals = self.hold(pose, ticks=ticks, settle_ticks=settle_ticks, keeper=keeper)
        return analyse_drift(
            residuals,
            calibration=self.calibration,
            geometry=self.geometry,
            role=self.role,
            pose=_as_vector(pose, f"{self.role}: hold pose"),
            baseline=baseline,
            actionable_mm=actionable_mm,
        )

    # -- description -------------------------------------------------------

    def channels(self) -> tuple[ChannelSpec, ...]:
        return so101_channels(rate_hz=CONTROL_HZ)

    def embodiment(self, version: str = VERSION) -> EmbodimentDescriptor:
        return so101_embodiment(self.calibration, version=version)


# --------------------------------------------------------------------------
# how this arm is described to the rest of the framework
# --------------------------------------------------------------------------


def _joint_channel(name: str, *, rate_hz: float | None) -> ChannelSpec:
    return ChannelSpec(
        name=name,
        kind="vector",
        shape=(DOF,),
        dtype="float32",
        # Percent of calibrated travel. Dimensionless on purpose: these are not
        # angles, and declaring them in degrees would be a lie that every
        # dimensional check downstream would then believe.
        units="percent",
        frame="joint",
        rate_hz=rate_hz,
        semantics="joint_position",
        dim_labels=JOINT_KEYS,
        discriminators=("normalization", "calibration_variant"),
        metadata={
            "normalization": NORMALIZATION,
            "calibration_variant": CALIBRATION_VARIANT,
            "gripper_index": GRIPPER,
        },
    )


def so101_channels(*, rate_hz: float | None = CONTROL_HZ) -> tuple[ChannelSpec, ...]:
    """The channels a teleop recording carries, named as LeRobot names them."""
    return (
        _joint_channel("observation.state", rate_hz=rate_hz),
        _joint_channel("action", rate_hz=rate_hz),
        _joint_channel("leader.state", rate_hz=rate_hz),
        replace(
            _joint_channel("tracking_error", rate_hz=rate_hz),
            semantics="joint_position",
            metadata={
                "normalization": NORMALIZATION,
                "calibration_variant": CALIBRATION_VARIANT,
                "gripper_index": GRIPPER,
                "definition": "previous tick's command minus this tick's measured position",
            },
        ),
        ChannelSpec(
            name="timestamp",
            kind="scalar",
            shape=(),
            dtype="float32",
            units="s",
            rate_hz=rate_hz,
            semantics="time",
            metadata={"origin": "first tick of the session, wall clock"},
        ),
    )


def so101_embodiment(
    calibration: Calibration, *, version: str = VERSION
) -> EmbodimentDescriptor:
    """Plane 2's view of this arm.

    ``seedable`` is absent, and that absence is the whole point — see
    :data:`SEEDABLE_IS_FALSE`. ``resettable`` is present and means only that
    the *arm* can be driven back to a repeatable joint configuration. It says
    nothing about the scene: putting the cube back is a human act with its own
    error, and no capability flag here covers it.
    """
    state = _joint_channel("observation.state", rate_hz=CONTROL_HZ)
    return EmbodimentDescriptor(
        name="so101",
        version=version,
        state=(state,),
        action=(replace(state, name="action"),),
        control_hz=CONTROL_HZ,
        kinematics=None,
        notes=(
            "SO-ARM101, 6x Feetech STS3215 on a Waveshare serial bus driver. "
            "Position control only: no force or torque sensing anywhere on the "
            "arm, so every contact signal available here is a position-error "
            "proxy. Gripper is the sixth DoF, continuous, "
            f"{'0 = closed' if GRIPPER_CLOSED_AT_ZERO else '100 = closed'}. "
            f"Values are LeRobot {NORMALIZATION} percent of calibrated travel, "
            f"calibration variant {calibration.variant}, "
            f"sha256 {calibration.sha256}."
        ),
        capabilities=frozenset({CAP_RESETTABLE}),
        metadata={
            "calibration": calibration.as_dict(),
            "force_sensing": False,
            "torque_sensing": False,
        },
    )


#: Stated here so it can be cited from the evaluator's descriptor rather than
#: re-argued: a physical rig **cannot** be seedable. Identical seeds do not give
#: identical records, because the initial state of a real scene is set by a
#: person putting an object down, and that placement carries its own error.
#:
#: The contract says a seedable evaluator "returns identical records for
#: identical seeds, which is what a paired comparison rests on". Declaring True
#: here would let the resolver plan a paired analysis on data that cannot
#: support one, and the output of that analysis looks exactly like a real one.
#:
#: Consequence, and it is a large one: the paired-seed power calculation in
#: CORPUS-PROTOCOL.md (n=127 unpaired -> n=76 paired) applies to the **sim tier
#: only**. Any real-hardware arm of the campaign is sized unpaired.
SEEDABLE_IS_FALSE = (
    "a physical rig cannot return identical records for identical seeds: object "
    "placement is a human act with its own error, so paired-seed analyses are "
    "invalid here and the paired power calculation applies to the sim tier only"
)

#: The other half of the honesty budget. The funnel needs object pose; on real
#: hardware that needs a perception rig, which does not exist. The gripper-stall
#: proxy below yields a GRASP candidate and nothing else — no lift, no place.
#: An evaluator that declared stage events on the strength of GRASP alone would
#: produce a funnel with one populated rung and several empty ones, which reads
#: as a finished analysis of a policy that never lifts anything.
STAGE_EVENTS_NEED_A_POSE_SOURCE = (
    "stage events require object pose; the gripper-stall proxy supplies a GRASP "
    "candidate only, so stage_events stays False until a pose source is "
    "explicitly configured"
)


# --------------------------------------------------------------------------
# calibration drift — the stable-frame check
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JointDrift:
    """One joint's steady-state ``commanded - measured`` centre."""

    joint: str
    n: int
    mean_units: float
    median_units: float
    std_units: float
    ci_units: tuple[float, float]
    deg_per_unit: float
    frame: str
    mm_per_unit: float
    drifted: bool

    @property
    def mean_deg(self) -> float:
        return self.mean_units * self.deg_per_unit

    @property
    def mean_mm(self) -> float:
        return self.mean_units * self.mm_per_unit

    @property
    def ci_mm(self) -> tuple[float, float]:
        low, high = (bound * self.mm_per_unit for bound in self.ci_units)
        return (low, high) if low <= high else (high, low)

    @property
    def excludes_zero(self) -> bool:
        return self.ci_units[0] > 0.0 or self.ci_units[1] < 0.0

    def measurement(self) -> Measurement:
        return Measurement(
            self.mean_mm,
            n=self.n,
            ci=self.ci_mm,
            units="mm",
            method="stable-frame commanded-minus-measured centre, "
            "converted through the calibrated joint span and a declared lever arm",
            detail={
                "joint": self.joint,
                "frame": self.frame,
                "mean_deg": self.mean_deg,
                "mean_units": self.mean_units,
                "median_units": self.median_units,
                "std_units": self.std_units,
                "drifted": self.drifted,
            },
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.joint:<18} {self.mean_units:+7.2f} pct  {self.mean_deg:+7.2f} deg  "
            f"{self.mean_mm:+8.2f} mm [{self.ci_mm[0]:+.2f}, {self.ci_mm[1]:+.2f}] "
            f"{self.frame:<12} {'DRIFTED' if self.drifted else 'ok'}"
        )


@dataclass(frozen=True)
class DriftBaseline:
    """A drift signature captured when the arm was known good.

    Exists because of one confound that would otherwise sink the whole check:
    under position control a joint holding against gravity settles with a
    *real* steady-state error, and that error is indistinguishable from a
    calibration offset in a single measurement. Subtracting a baseline taken at
    the same pose cancels the gravity term.

    The calibration hash is stored with it and is checked on use. Comparing
    today's drift against a baseline captured under a different calibration is
    meaningless, and it is the exact mistake this check exists to catch, so it
    is refused rather than warned about.
    """

    pose: tuple[float, ...]
    mean_units: tuple[float, ...]
    calibration_sha256: str
    role: str
    n: int

    @classmethod
    def from_report(cls, report: "DriftReport") -> "DriftBaseline":
        if report.baseline is not None:
            raise ConfigError(
                "a baseline must be captured from an unbaselined measurement, "
                "otherwise it stores a difference of differences"
            )
        return cls(
            pose=tuple(float(value) for value in report.pose),
            mean_units=tuple(joint.mean_units for joint in report.joints),
            calibration_sha256=report.calibration_sha256,
            role=report.role,
            n=report.joints[0].n if report.joints else 0,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose": list(self.pose),
            "mean_units": list(self.mean_units),
            "calibration_sha256": self.calibration_sha256,
            "role": self.role,
            "n": self.n,
        }


@dataclass(frozen=True)
class DriftReport:
    """Per-joint calibration drift, in the units a person can act on.

    **What this finds.** A persistent, non-zero centre of ``commanded -
    measured`` while the arm is commanded to hold still. That is what a servo
    horn that has slipped, a re-homing that landed somewhere else, or a
    corrupted ``homing_offset`` looks like. It is LeRobot #3758's failure mode,
    and it is invisible in a training loss because a constant offset is a
    bijection the policy simply learns.

    **What this does not find**, stated because a check whose limits are not
    written down gets trusted past them:

    * An error common to leader *and* follower. Both arms wrong by the same
      amount looks like perfect tracking.
    * A kinematic error. Wrong link lengths, a bent horn, a mis-seated
      gripper — the joints all report exactly where they were told to be.
    * Gravity sag, unless ``baseline`` is supplied. A loaded joint has a real
      steady-state error under position control. Without a baseline at the
      same pose this check will call that drift, which is a false positive on
      an arm that is fine.
    * Anything at a pose other than the one held. Offsets interact with the
      configuration; one pose is one sample of the workspace.

    The millimetre figures come from :class:`Geometry`'s declared lever arms
    and are approximate by construction. The degrees are exact given the
    calibration file. The percent is the raw measurement.
    """

    role: str
    pose: tuple[float, ...]
    joints: tuple[JointDrift, ...]
    calibration_sha256: str
    geometry: Mapping[str, Any]
    actionable_mm: float
    baseline: DriftBaseline | None = None
    underpowered: bool = False

    @property
    def drifted(self) -> tuple[JointDrift, ...]:
        return tuple(joint for joint in self.joints if joint.drifted)

    @property
    def worst_base_frame_mm(self) -> float:
        """Largest tip displacement among the base-frame joints.

        The gripper is excluded on purpose: its error is a jaw aperture in a
        different frame, and a maximum taken across two frames is a number with
        no physical meaning.
        """
        return max(
            (abs(joint.mean_mm) for joint in self.joints if joint.frame == BASE_FRAME),
            default=0.0,
        )

    def measurements(self) -> dict[str, Measurement]:
        out = {f"drift.{joint.joint}": joint.measurement() for joint in self.joints}
        out["drift.worst_base_frame_mm"] = Measurement(
            self.worst_base_frame_mm,
            n=len(self.joints) - 1,
            units="mm",
            method="largest absolute base-frame tip displacement across joints",
            detail={"excludes": JOINT_KEYS[GRIPPER], "reason": "different frame"},
        )
        return out

    def verdict(self) -> Verdict:
        if self.underpowered:
            return Verdict.no(
                "drift.underpowered",
                f"{self.role}: too few stable frames to say anything about drift",
                hint="hold the pose longer; a short sample cannot distinguish an "
                "offset from noise, and 'no drift found' from a short sample is "
                "the most dangerous output this check has",
            )
        drifted = self.drifted
        if not drifted:
            return Verdict.note(
                "drift.clear",
                f"{self.role}: no joint centres away from zero by more than "
                f"{self.actionable_mm:.1f} mm at the gripper "
                f"(calibration {self.calibration_sha256[:16]})",
            )
        detail = "; ".join(
            f"{joint.joint} {joint.mean_deg:+.1f} deg = {joint.mean_mm:+.1f} mm"
            for joint in drifted
        )
        return Verdict.no(
            "drift.offset",
            f"{self.role}: {len(drifted)} joint(s) hold with a persistent offset: {detail}",
            hint="recalibrate before recording. A constant joint offset is a "
            "bijection, so a policy trained on a corpus recorded through it "
            "learns the biased mapping and the training loss stays healthy "
            "(LeRobot #3758); it surfaces only at evaluation, on the un-offset "
            "robot, as confident reaching to the wrong place",
            joints=[joint.joint for joint in drifted],
            worst_mm=self.worst_base_frame_mm,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pose": list(self.pose),
            "calibration_sha256": self.calibration_sha256,
            "geometry": dict(self.geometry),
            "actionable_mm": self.actionable_mm,
            "baseline": self.baseline.as_dict() if self.baseline else None,
            "underpowered": self.underpowered,
            "joints": {
                joint.joint: {
                    "n": joint.n,
                    "mean_units": joint.mean_units,
                    "median_units": joint.median_units,
                    "std_units": joint.std_units,
                    "mean_deg": joint.mean_deg,
                    "mean_mm": joint.mean_mm,
                    "ci_mm": list(joint.ci_mm),
                    "frame": joint.frame,
                    "drifted": joint.drifted,
                }
                for joint in self.joints
            },
        }

    def table(self) -> str:
        lines = [f"drift {self.role} (calibration {self.calibration_sha256[:16]})"]
        lines += [f"  {joint}" for joint in self.joints]
        return "\n".join(lines)


#: Minimum stable frames before this check will say anything at all. At 30 Hz
#: that is one second of holding after the settle window. Below it the standard
#: error is wide enough that "no drift" means "did not look", and reporting a
#: clean result from a sample that could not have found one is worse than
#: reporting nothing.
MIN_DRIFT_SAMPLES = 30

#: Two-sided 95% normal quantile. n is at least 30 by the check above, where
#: the t correction is under 4% and smaller than the lever-arm approximation
#: it is multiplied by.
_Z95 = 1.959963984540054


def analyse_drift(
    residuals: np.ndarray,
    *,
    calibration: Calibration,
    geometry: Geometry,
    role: str,
    pose: np.ndarray,
    baseline: DriftBaseline | None = None,
    actionable_mm: float = 2.0,
) -> DriftReport:
    """Turn a stable-frame residual sample into a per-joint verdict.

    A joint is called drifted when **both** hold: its 95% interval excludes
    zero, and its centre exceeds ``actionable_mm`` at the gripper. Two
    conditions rather than one because they fail differently — the interval
    alone calls a 0.2 mm bias on a thousand samples a problem, and the
    magnitude alone calls a noisy joint drifted on a lucky window.
    """
    samples = np.asarray(residuals, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != DOF:
        raise ConfigError(
            f"{role}: drift residuals must be (frames, {DOF}), got {samples.shape}"
        )

    if baseline is not None:
        if baseline.calibration_sha256 != calibration.sha256:
            raise ConfigError(
                f"{role}: the drift baseline was captured under calibration "
                f"{baseline.calibration_sha256[:16]} and this arm is running "
                f"{calibration.short}. Subtracting it would cancel exactly the "
                "change this check exists to find."
            )
        offset = np.asarray(baseline.mean_units, dtype=np.float64)
        if offset.shape != (DOF,):
            raise ConfigError(f"{role}: baseline has {offset.shape[0]} joints, expected {DOF}")
        if not np.allclose(np.asarray(baseline.pose, dtype=np.float64), pose, atol=0.5):
            raise ConfigError(
                f"{role}: the baseline was captured at a different pose. A gravity "
                "steady-state error is pose-dependent, so a baseline from another "
                "pose subtracts the wrong constant."
            )
        samples = samples - offset

    n = samples.shape[0]
    underpowered = n < MIN_DRIFT_SAMPLES

    joints: list[JointDrift] = []
    for index, key in enumerate(JOINT_KEYS):
        column = samples[:, index]
        mean = float(column.mean()) if n else 0.0
        median = float(np.median(column)) if n else 0.0
        std = float(column.std(ddof=1)) if n > 1 else 0.0
        half = _Z95 * std / math.sqrt(n) if n else math.inf
        ci = (mean - half, mean + half)
        cal = calibration.joints[index]
        frame = geometry.frame_of(key)
        if frame == JAW_FRAME:
            mm_per_unit = geometry.gripper_aperture_mm / 100.0
        else:
            mm_per_unit = cal.deg_per_unit * geometry.mm_per_deg(key)
        drifted = (
            not underpowered
            and (ci[0] > 0.0 or ci[1] < 0.0)
            and abs(mean * mm_per_unit) >= actionable_mm
        )
        joints.append(
            JointDrift(
                joint=key,
                n=n,
                mean_units=mean,
                median_units=median,
                std_units=std,
                ci_units=ci,
                deg_per_unit=cal.deg_per_unit,
                frame=frame,
                mm_per_unit=mm_per_unit,
                drifted=drifted,
            )
        )
    return DriftReport(
        role=role,
        pose=tuple(float(value) for value in np.asarray(pose, dtype=np.float64)),
        joints=tuple(joints),
        calibration_sha256=calibration.sha256,
        geometry=geometry.as_dict(),
        actionable_mm=actionable_mm,
        baseline=baseline,
        underpowered=underpowered,
    )


# --------------------------------------------------------------------------
# gripper stall — the only contact signal this hardware has
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StallEvent:
    """A sustained gripper position error: the jaw is on something.

    A *proxy*, and the word is load-bearing. There is no force sensor on this
    arm. What is measured is that the jaw was commanded somewhere and did not
    get there for several consecutive frames, which happens when it closes on
    an object — and identically when it closes on the table, on itself, on a
    cable, or on a servo that has stopped answering.

    This is what makes a GRASP stage event possible at all on real hardware.
    It is not what makes a *funnel* possible: lift and place need object pose.
    See :data:`STAGE_EVENTS_NEED_A_POSE_SOURCE`.
    """

    step: int
    steps: int
    mean_residual_units: float
    mean_residual_mm: float
    closing: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "steps": self.steps,
            "mean_residual_units": self.mean_residual_units,
            "mean_residual_mm": self.mean_residual_mm,
            "closing": self.closing,
            "proxy": "position error; this arm has no force sensor",
        }


class GripperStallDetector:
    """Confirms a stall after it has persisted, not on the frame it starts.

    Every commanded step produces a transient position error simply because
    the jaw takes time to arrive. Requiring persistence is what separates "the
    gripper is moving" from "the gripper has stopped moving against something",
    and it is the whole difference between a grasp signal and a noise
    generator.
    """

    def __init__(
        self,
        *,
        threshold_units: float = 4.0,
        min_steps: int = 4,
        aperture_mm: float = GRIPPER_APERTURE_MM,
        closed_at_zero: bool = GRIPPER_CLOSED_AT_ZERO,
    ):
        if threshold_units <= 0:
            raise ConfigError("a stall threshold of zero fires on every frame")
        if min_steps < 2:
            raise ConfigError(
                "a stall confirmed on one frame is the jaw's own settling transient"
            )
        self.threshold_units = float(threshold_units)
        self.min_steps = int(min_steps)
        self.aperture_mm = float(aperture_mm)
        self.closed_at_zero = bool(closed_at_zero)
        self._run_start: int | None = None
        self._run: list[float] = []
        self._emitted = False
        self.events: list[StallEvent] = []

    @property
    def stalled(self) -> bool:
        """Currently in a confirmed stall."""
        return self._emitted

    def update(self, step: int, commanded: float, measured: float) -> StallEvent | None:
        """Feed one tick. Returns the event on the frame a stall is confirmed."""
        residual = float(commanded) - float(measured)
        if abs(residual) < self.threshold_units:
            self._run_start = None
            self._run = []
            self._emitted = False
            return None
        if self._run_start is None:
            self._run_start = step
        self._run.append(residual)
        if self._emitted or len(self._run) < self.min_steps:
            return None
        self._emitted = True
        mean = float(np.mean(self._run))
        # commanded past measured towards the closed end == the jaw is being
        # driven shut and is not getting there.
        closing = (mean < 0.0) if self.closed_at_zero else (mean > 0.0)
        event = StallEvent(
            step=self._run_start,
            steps=len(self._run),
            mean_residual_units=mean,
            mean_residual_mm=mean * self.aperture_mm / 100.0,
            closing=closing,
        )
        self.events.append(event)
        return event


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingReport:
    """Per-joint leader-follower tracking error, mean and max.

    Required per episode by CORPUS-PROTOCOL.md, and it earns its place twice:
    as a free covariate for the analysis (a demonstration recorded through a
    badly tracking follower is a different demonstration) and as the operator's
    real-time health signal.

    The error is ``previous tick's command minus this tick's measurement``, not
    the current command minus the current measurement. The follower cannot have
    responded to a command issued microseconds earlier in the same tick, and an
    error computed against it measures the loop's own latency rather than the
    follower's tracking.
    """

    n: int
    mean_units: tuple[float, ...]
    max_units: tuple[float, ...]
    deg_per_unit: tuple[float, ...]

    @classmethod
    def from_errors(cls, errors: np.ndarray, calibration: Calibration) -> "TrackingReport":
        samples = np.asarray(errors, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != DOF:
            raise ConfigError(f"tracking errors must be (frames, {DOF}), got {samples.shape}")
        magnitude = np.abs(samples)
        empty = samples.shape[0] == 0
        return cls(
            n=int(samples.shape[0]),
            mean_units=tuple(
                float(v) for v in (magnitude.mean(axis=0) if not empty else np.zeros(DOF))
            ),
            max_units=tuple(
                float(v) for v in (magnitude.max(axis=0) if not empty else np.zeros(DOF))
            ),
            deg_per_unit=tuple(float(v) for v in calibration.deg_per_unit),
        )

    def mean_deg(self) -> tuple[float, ...]:
        return tuple(m * d for m, d in zip(self.mean_units, self.deg_per_unit))

    def max_deg(self) -> tuple[float, ...]:
        return tuple(m * d for m, d in zip(self.max_units, self.deg_per_unit))

    def as_dict(self) -> dict[str, Any]:
        """The shape CORPUS-PROTOCOL.md asks for: mean and max, per joint."""
        return {
            "n": self.n,
            "definition": "previous command minus current measurement, absolute",
            "per_joint": {
                key: {
                    "mean_units": mean,
                    "max_units": peak,
                    "mean_deg": mean * scale,
                    "max_deg": peak * scale,
                }
                for key, mean, peak, scale in zip(
                    JOINT_KEYS, self.mean_units, self.max_units, self.deg_per_unit
                )
            },
        }

    def measurements(self) -> dict[str, Measurement]:
        out: dict[str, Measurement] = {}
        for key, mean, peak, scale in zip(
            JOINT_KEYS, self.mean_units, self.max_units, self.deg_per_unit
        ):
            out[f"tracking.{key}.mean_deg"] = Measurement(
                mean * scale,
                n=self.n,
                units="deg",
                method="mean absolute leader-follower tracking error",
            )
            out[f"tracking.{key}.max_deg"] = Measurement(
                peak * scale,
                n=self.n,
                units="deg",
                method="worst single-tick leader-follower tracking error",
            )
        return out


# --------------------------------------------------------------------------
# the pair
# --------------------------------------------------------------------------


@dataclass
class TeleopFrames:
    """The trajectory a teleop session produced, ready to become an episode.

    Channel names are LeRobot's, so the connector that reads the corpus back
    needs no rename and no positional guess.
    """

    timestamps: list[float] = field(default_factory=list)
    leader_state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    follower_state: list[np.ndarray] = field(default_factory=list)
    tracking_error: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timestamps)

    def arrays(self) -> dict[str, np.ndarray]:
        """``{channel: array}`` with a leading time axis, all lengths equal.

        The first tick has no tracking error — there is no previous command to
        compare against — so it carries NaN rather than a zero. A zero there
        would read as perfect tracking on the first frame of every episode in
        the corpus, which is a bias of exactly one frame per episode in the
        favourable direction.
        """
        n = len(self)
        errors = np.full((n, DOF), np.nan, dtype=np.float32)
        if n > 1:
            errors[1:] = np.asarray(self.tracking_error, dtype=np.float32)
        return {
            "timestamp": np.asarray(self.timestamps, dtype=np.float32),
            "leader.state": np.asarray(self.leader_state, dtype=np.float32).reshape(n, DOF),
            "action": np.asarray(self.action, dtype=np.float32).reshape(n, DOF),
            "observation.state": np.asarray(self.follower_state, dtype=np.float32).reshape(n, DOF),
            "tracking_error": errors,
        }

    def schema(self) -> tuple[ChannelSpec, ...]:
        return so101_channels(rate_hz=CONTROL_HZ)


@dataclass(frozen=True)
class TeleopReport:
    """Everything a teleop session knows about itself except the trajectory."""

    ticks: int
    rate: RateReport
    tracking: TrackingReport
    stalls: tuple[StallEvent, ...]
    clamped_ticks: int
    envelope_excursions: int
    leader_calibration_sha256: str
    follower_calibration_sha256: str

    def verdict(self, *, tolerate_late: float = 0.01) -> Verdict:
        """Is this session's recording fit to enter a corpus?"""
        checks = [self.rate.verdict(tolerate=tolerate_late)]
        if self.clamped_ticks:
            checks.append(
                Verdict.note(
                    "teleop.clamped",
                    f"{self.clamped_ticks}/{self.ticks} tick(s) hit the relative-target "
                    "guard, so the recorded action is not what the operator did on those ticks",
                    hint="a corpus with a high clamp count is not the corpus the "
                    "operator thinks they recorded",
                )
            )
        if self.envelope_excursions:
            checks.append(
                Verdict.note(
                    "teleop.envelope",
                    f"{self.envelope_excursions} commanded value(s) sat outside 0..100 "
                    "but within tolerance",
                )
            )
        return Verdict.all(checks)

    def as_metadata(self) -> dict[str, Any]:
        """The per-episode metadata block CORPUS-PROTOCOL.md requires from this layer.

        The two calibration hashes are the point of the exercise: they are what
        makes a mid-campaign recalibration detectable retrospectively.
        """
        return {
            "ticks": self.ticks,
            "control_hz_nominal": self.rate.nominal_hz,
            "control_hz_realised": self.rate.realised_hz,
            "rate_overruns": len(self.rate.overruns),
            "rate_worst_late_ms": self.rate.worst_late_s * 1e3,
            "tracking_error": self.tracking.as_dict(),
            "gripper_stalls": [event.as_dict() for event in self.stalls],
            "clamped_ticks": self.clamped_ticks,
            "envelope_excursions": self.envelope_excursions,
            "leader_calibration_sha256": self.leader_calibration_sha256,
            "follower_calibration_sha256": self.follower_calibration_sha256,
        }

    def measurements(self) -> dict[str, Measurement]:
        out: dict[str, Measurement] = {}
        out.update(self.rate.measurements())
        out.update(self.tracking.measurements())
        out["teleop.clamped_fraction"] = Measurement(
            (self.clamped_ticks / self.ticks) if self.ticks else 0.0,
            n=self.ticks,
            units="fraction",
            method="ticks where the relative-target guard altered the commanded action",
        )
        out["teleop.gripper_stalls"] = Measurement(
            float(len(self.stalls)),
            n=self.ticks,
            units="count",
            method="sustained gripper position error; a contact proxy, not a force reading",
        )
        return out


class SO101Pair:
    """A leader arm and a follower arm: one operator station.

    Two jobs. Drive the follower from the leader to *record* demonstration
    corpora (the corpus factory), and expose the health signals that say
    whether what was recorded is worth keeping. Running a policy on the
    follower is the evaluator's job and uses :class:`SO101Arm` directly.
    """

    def __init__(
        self,
        leader: SO101Arm,
        follower: SO101Arm,
        *,
        hz: float = CONTROL_HZ,
        clock: Clock | None = None,
        stall_detector: GripperStallDetector | None = None,
    ):
        if leader is follower:
            raise ConfigError("a pair needs two arms; this is one arm twice")
        if leader.calibration.path == follower.calibration.path:
            # Not pedantry: two arms sharing one calibration file means one of
            # them is running someone else's homing offsets, which is a
            # constant joint offset — the #3758 failure, installed on purpose.
            raise ConfigError(
                f"leader and follower both calibrated from {leader.calibration.path}; "
                "each arm needs its own calibration or one of them is running the "
                "other's homing offsets"
            )
        self.leader = leader
        self.follower = follower
        self.hz = float(hz)
        self.clock = clock or Clock()
        self.stall = stall_detector or GripperStallDetector()

    # -- what the evaluator drives this through ----------------------------
    #
    # The evaluator holds a pair and nothing else, and it names exactly these
    # six things. They are spelled out here rather than left for the evaluator
    # to reach through `.follower.write_action`, because a caller reaching past
    # this facade would be one refactor away from reaching past the safety
    # limits attached to the arm it reached through.

    @property
    def joints(self) -> tuple[str, ...]:
        """The action-vector order. Not a set: a permutation is not a rename."""
        return JOINT_KEYS

    @property
    def control_hz(self) -> float:
        return self.hz

    @property
    def calibration(self) -> dict[str, str]:
        """Both arms' calibration hashes, which every episode is stamped with.

        Both, not the follower's alone. A leader whose calibration moved records
        a demonstration in a shifted joint space, and the follower's hash would
        say nothing about it — that is #3758 arriving through the teleoperator
        instead of through the arm under test.
        """
        return {
            "leader": self.leader.calibration.sha256,
            "follower": self.follower.calibration.sha256,
        }

    def read_leader(self) -> np.ndarray:
        """``float32[6]`` measured on the leader, in RANGE_0_100."""
        return self.leader.read_state()

    def read_follower(self) -> np.ndarray:
        """``float32[6]`` measured on the follower, in RANGE_0_100."""
        return self.follower.read_state()

    def write_follower(self, target: Any) -> np.ndarray:
        """Command the follower and return what was actually sent.

        This is not a bypass of :meth:`SO101Arm.write_action`, and deliberately
        not a thinner path than teleoperation uses. A caller above may well have
        clamped already — the evaluator does — but the arm's own envelope check,
        tick-representability check and relative guard still run, and they are
        the last thing between a wrong number and the table. Two clamps in
        series is not redundancy to be optimised away; the outer one is a
        measurement decision and the inner one is a safety device, and they are
        allowed to be set differently.
        """
        return self.follower.write_action(target)

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Leader torque off so it can be backdriven, follower torque on."""
        self.leader.connect()
        self.follower.connect()
        self.leader.set_torque(False)
        self.follower.set_torque(True)

    def disconnect(self) -> None:
        for arm in (self.follower, self.leader):
            try:
                arm.disconnect()
            except Exception:  # noqa: BLE001 - releasing the other port matters more
                pass

    def __enter__(self) -> "SO101Pair":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    # -- the loop ----------------------------------------------------------

    def teleoperate(
        self,
        *,
        ticks: int,
        record: bool = True,
        on_tick: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    ) -> tuple[TeleopFrames, TeleopReport]:
        """Drive the follower from the leader for ``ticks`` ticks at ``hz``.

        The ordering within a tick is fixed and matters:

        1. read the leader
        2. command the follower with it
        3. read the follower — which reflects the *previous* command
        4. score the previous command against that measurement

        Step 4 is why tracking error is computed with a one-tick lag. Comparing
        a measurement against the command issued microseconds earlier in the
        same tick measures the serial round trip, not how well the follower is
        tracking, and it would report a well-tracking arm as badly tracking and
        a stuck arm as fine.
        """
        if ticks < 1:
            raise ConfigError(f"teleoperate(ticks={ticks}) records nothing")
        self.leader.reset_session_counters()
        self.follower.reset_session_counters()

        keeper = RateKeeper(self.hz, clock=self.clock)
        frames = TeleopFrames()
        errors: list[np.ndarray] = []
        previous_command: np.ndarray | None = None
        started = self.clock.now()

        for tick in range(ticks):
            leader_state = self.leader.read_state()
            command = self.follower.write_action(leader_state)
            keeper.wait()
            follower_state = self.follower.read_state()

            if previous_command is not None:
                error = previous_command.astype(np.float64) - follower_state.astype(np.float64)
                errors.append(error)
                self.stall.update(
                    tick,
                    float(previous_command[GRIPPER]),
                    float(follower_state[GRIPPER]),
                )
            if record:
                frames.timestamps.append(self.clock.now() - started)
                frames.leader_state.append(leader_state)
                frames.action.append(command)
                frames.follower_state.append(follower_state)
                if previous_command is not None:
                    frames.tracking_error.append(errors[-1].astype(np.float32))
            if on_tick is not None:
                on_tick(tick, command, follower_state)
            previous_command = command

        report = TeleopReport(
            ticks=ticks,
            rate=keeper.report(),
            tracking=TrackingReport.from_errors(
                np.asarray(errors, dtype=np.float64).reshape(len(errors), DOF),
                self.follower.calibration,
            ),
            stalls=tuple(self.stall.events),
            clamped_ticks=self.follower.clamped_ticks,
            envelope_excursions=self.follower.envelope_excursions,
            leader_calibration_sha256=self.leader.calibration.sha256,
            follower_calibration_sha256=self.follower.calibration.sha256,
        )
        return frames, report

    # -- health ------------------------------------------------------------

    def check_calibration_drift(
        self,
        pose: Any,
        *,
        ticks: int = 90,
        settle_ticks: int = 15,
        baseline: DriftBaseline | None = None,
        actionable_mm: float = 2.0,
    ) -> DriftReport:
        """Run the stable-frame check on the follower.

        The follower only: the leader is backdriven and has no holding torque,
        so "commanded minus measured" is not defined for it. A leader offset
        shows up instead as a systematic tracking error at rest, which is what
        :class:`TrackingReport` is for.
        """
        return self.follower.check_drift(
            pose,
            ticks=ticks,
            settle_ticks=settle_ticks,
            keeper=RateKeeper(self.hz, clock=self.clock),
            baseline=baseline,
            actionable_mm=actionable_mm,
        )

    def capture_drift_baseline(self, pose: Any, *, ticks: int = 150) -> DriftBaseline:
        """Record the follower's known-good signature at one pose.

        Run this once at commissioning, with the arm in the pose the campaign
        will check at, and store it beside the calibration file. It is what
        turns the drift check from "is there a steady-state error" (there
        always is, gravity supplies one) into "has the steady-state error
        changed since the arm was known good", which is the question.
        """
        report = self.follower.check_drift(
            pose, ticks=ticks, keeper=RateKeeper(self.hz, clock=self.clock)
        )
        if report.underpowered:
            raise ConfigError(
                f"a baseline from {ticks} tick(s) is too short to be a baseline; "
                f"at least {MIN_DRIFT_SAMPLES} stable frames are needed"
            )
        return DriftBaseline.from_report(report)

    def embodiment(self) -> EmbodimentDescriptor:
        return so101_embodiment(self.follower.calibration)


# --------------------------------------------------------------------------
# selftest — runs on MockTransport, no hardware anywhere
# --------------------------------------------------------------------------

#: Fault-injection hooks the selftest needs from ``MockTransport``. Looked up by
#: name from a short candidate list, and refused loudly if none is present.
#: This lookup exists only in the selftest harness — nothing on the control path
#: guesses at an API — and its whole purpose is that a mismatch with
#: :mod:`.transport` is a one-line fix with the name printed rather than an
#: AttributeError from inside a loop.
_OFFSET_HOOKS = ("inject_offset", "set_offset", "inject_position_offset")
_STALL_HOOKS = ("inject_stall", "set_stall", "inject_position_limit")


def _hook(transport: Any, candidates: Sequence[str]) -> Callable[..., Any]:
    for name in candidates:
        found = getattr(transport, name, None)
        if callable(found):
            return found
    raise ConfigError(
        f"{type(transport).__name__} has none of {list(candidates)}; the selftest "
        "needs that fault-injection hook. Add it to MockTransport or rename the "
        "candidate here — the production path does not use this lookup."
    )


#: Nominal measured travel per joint, in raw encoder counts. Different per
#: joint on purpose, so the degrees-per-unit conversion is exercised rather than
#: collapsing to one constant that would hide a per-joint error.
_FIXTURE_SPANS: Mapping[str, tuple[int, int]] = {
    "shoulder_pan": (900, 3200),
    "shoulder_lift": (1000, 3000),
    "elbow_flex": (950, 3100),
    "wrist_flex": (1100, 2900),
    "wrist_roll": (500, 3600),
    "gripper": (1800, 2600),
}


#: Every path :func:`_write_calibration` has written this process. Membership,
#: not location or content: a fixture file and a real calibration are the same
#: shape by construction — that is why ``measured`` exists — so the only
#: trustworthy way to tell them apart is to remember which ones we invented.
_FIXTURE_FILES: set[Path] = set()


def _write_calibration(
    path: Path,
    *,
    homing: Mapping[str, int] | None = None,
    widen: int = 0,
) -> Path:
    """A plausible so101_new_calib file.

    ``widen`` shifts every joint's measured travel by a few counts. It is how
    two mock arms get genuinely different calibrations: two SO-101s off the same
    print do not share travel limits, and giving a leader and a follower the same
    file would install the #3758 failure on purpose while also making their two
    hashes in the episode metadata indistinguishable.
    """
    homing = homing or {}
    payload = {
        name: {
            "id": index + 1,
            "drive_mode": 0,
            "homing_offset": int(homing.get(name, 0)),
            "range_min": low - widen,
            "range_max": high + widen,
        }
        for index, (name, (low, high)) in enumerate(_FIXTURE_SPANS.items())
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _FIXTURE_FILES.add(path.resolve())
    return path


def bus_calibration(calibration: Calibration) -> BusCalibration:
    """The same arm's calibration, as the bus layer needs it.

    Two types for one physical fact, and they are not redundant: this module's
    :class:`Calibration` is a *file* — a path and the hash of its bytes, which is
    what makes a corpus traceable to the numbers it was recorded under — while
    the bus's is the tick arithmetic and a ``measured`` flag that decides whether
    it may drive a motor at all. Derived here, from the one file, so the two can
    never be built from different sources and quietly disagree about what a tick
    means.

    ``measured`` is carried across honestly: a file this package wrote for a mock
    is not a measurement, and the bus refuses to drive real hardware from one.
    """
    payload = {
        joint.name: {
            "id": joint.id,
            "drive_mode": joint.drive_mode,
            "homing_offset": joint.homing_offset,
            "range_min": joint.range_min,
            "range_max": joint.range_max,
        }
        for joint in calibration.joints
    }
    bus = BusCalibration.from_lerobot(payload, source=str(calibration.path))
    return replace(bus, measured=not _is_fixture(calibration.path))


def _is_fixture(path: Path) -> bool:
    """Whether this file was written by :func:`_write_calibration`."""
    return path.resolve() in _FIXTURE_FILES


#: Where mock calibration files live for the life of the process. One directory,
#: created on first use, so a test suite that builds a hundred mock pairs writes
#: two files rather than two hundred.
_MOCK_ROOM: Path | None = None


def _mock_room() -> Path:
    global _MOCK_ROOM
    if _MOCK_ROOM is None:
        import tempfile

        _MOCK_ROOM = Path(tempfile.mkdtemp(prefix="so101-mock-"))
    return _MOCK_ROOM


def mock_arm(
    role: str,
    *,
    path: Path | None = None,
    homing: Mapping[str, int] | None = None,
    widen: int = 0,
    on_exceed: str = CLAMP,
    park: float = 50.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    **chain: Any,
) -> SO101Arm:
    """One arm over a :class:`MockTransport`, with no hardware anywhere.

    The calibration is written to a real file and read back through
    :func:`load_calibration`, so the mock exercises the verification path and the
    hash stamped into provenance is a real SHA-256 of real bytes rather than a
    constant somebody typed. The file lives under a temp directory that
    :func:`bus_calibration` recognises, which is what keeps ``measured=False``
    travelling with it.

    The bus clamp is not optional here either. It is set wider than the arm's own
    relative guard on purpose: the arm's is the behavioural limit a campaign
    chooses, the bus's is the floor nobody may configure away.

    ``clock`` is the one argument worth reading twice. The simulated servos
    integrate their motion against it, so leaving it at wall time while the loop
    above runs on a virtual clock gives a mock whose arm moves by however long
    the interpreter took — two identical runs then produce different records, and
    every number measured off the mock is host scheduling noise. Pass the loop's
    own clock and the mock becomes reproducible.

    ``park`` is where the arm sits before anything commands it, in percent of
    each joint's *own* calibrated travel. Not one tick for the whole chain:
    2048 is the middle of the encoder, and the gripper's measured range is not
    centred on the encoder, so a chain parked at 2048 starts the gripper a third
    of the way through its travel and every run opens with the clamp binding
    while it walks to where it was told. That is a property of the fixture, and
    it would be read off the record as a property of the policy.
    """
    if path is None:
        path = _write_calibration(_mock_room() / f"{role}.json", homing=homing, widen=widen)
    calibration = load_calibration(path)
    transport = MockTransport(
        bus_calibration(calibration),
        max_relative_target=BusLimits.normalized(_BUS_CLAMP_NORMALIZED),
        start={joint.id: joint.to_ticks(park) for joint in calibration.joints},
        clock=clock,
        sleep=sleep,
        **chain,
    )
    return SO101Arm(
        transport,
        calibration,
        SafetyLimits.teleop(on_exceed=on_exceed),
        role=role,
    )


#: The bus-level per-write budget for a mock, in percent of each joint's travel.
#: Above the teleop preset's widest joint (the gripper's 15) so that the arm's
#: guard is what shapes motion and this one is what catches a caller who got past
#: it. A bus clamp below the arm's would silently re-clamp every command and the
#: episode would attribute the difference to the wrong layer.
_BUS_CLAMP_NORMALIZED = 20.0

# There is deliberately no second way to build a mock arm. An earlier version of
# this file had one that constructed the bus from `Calibration.fixture(...)`
# while the arm on top of it was loaded from a calibration file — two different
# travel ranges for one physical joint. Everything still ran: ticks converted,
# the clamp clamped, the stall detector produced events, and every number was
# nonsense, because the two layers disagreed about what a tick meant. That is
# the failure this whole package is written against, so the constructor that
# could express it is gone rather than documented.


def _selftest() -> int:  # noqa: C901 - a linear script; splitting it hides the order
    import tempfile

    passed: list[str] = []
    failed: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        # ASCII separator on purpose: this prints to whatever console the rig is
        # plugged into, and a cp1252 terminal renders an em dash as a mojibake
        # box in the middle of the one line someone is reading at the bench.
        (passed if condition else failed).append(name)
        print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  ..  ' + detail) if detail else ''}")

    room = Path(tempfile.mkdtemp(prefix="so101-selftest-"))
    home = np.full(DOF, 50.0, dtype=np.float32)

    def build(role: str, path: Path, *, on_exceed: str = CLAMP) -> tuple[SO101Arm, Any]:
        # One calibration file, both layers. The bus's tick arithmetic and the
        # arm's have to come from the same measured travel or every conversion
        # between them is off by the difference, silently.
        arm = mock_arm(role, path=path, on_exceed=on_exceed)
        arm.connect()
        arm.set_torque(True)
        arm.write_action(home)
        return arm, arm.transport

    print("gantry-evaluator-so101 arm.py selftest (MockTransport, no hardware)")

    # -- 1. a wrong-width action is refused, never padded ------------------
    print("\n[1] action width")
    # Two physical arms have different homing offsets. Giving them the same file
    # would be the #3758 failure installed on purpose, and would also make the
    # two calibration hashes in the episode metadata indistinguishable.
    leader_file = _write_calibration(
        room / "leader.json", homing={"shoulder_pan": 3, "wrist_roll": -5}
    )
    follower_file = _write_calibration(room / "follower.json")
    arm, _ = build("follower", follower_file)
    for bad, label in (
        (np.zeros(5, dtype=np.float32), "5 values"),
        (np.zeros(7, dtype=np.float32), "7 values"),
        (np.zeros((2, DOF), dtype=np.float32), "a (2,6) chunk"),
    ):
        try:
            arm.write_action(bad)
        except ConfigError as error:
            check(f"refuses {label}", True, str(error).split(";")[0][:78])
        else:
            check(f"refuses {label}", False, "it was accepted")
    try:
        arm.write_action(np.array([50, 50, 50, 50, 50, np.nan], dtype=np.float32))
    except ConfigError as error:
        check("refuses a NaN gripper", True, str(error))
    else:
        check("refuses a NaN gripper", False, "it was accepted")
    check("accepts a correct 6-vector", arm.write_action(home).shape == (DOF,))
    arm.disconnect()

    # -- 2. the calibration hash is stable, and moves when the file does ---
    print("\n[2] calibration hash")
    first = load_calibration(follower_file)
    again = load_calibration(follower_file)
    check("stable across reloads", first.sha256 == again.sha256, first.short)
    check("64 hex characters of SHA-256", len(first.sha256) == 64)
    _write_calibration(follower_file, homing={"wrist_flex": 12})
    moved = load_calibration(follower_file)
    check(
        "changes when the file changes",
        moved.sha256 != first.sha256,
        f"{first.short} -> {moved.short}",
    )
    _write_calibration(follower_file)
    check(
        "returns to the original hash when the file does",
        load_calibration(follower_file).sha256 == first.sha256,
    )
    duplicate = room / "duplicate-ids.json"
    _write_calibration(duplicate)
    payload = json.loads(duplicate.read_text())
    payload["elbow_flex"]["id"] = 1
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_calibration(duplicate)
    except ConfigError as error:
        check("refuses two motors on one id", True, str(error).split(": ", 1)[1])
    else:
        check("refuses two motors on one id", False, "accepted")

    # -- 3. the tick mapping is the bus's, not a second one ----------------
    print("\n[3] tick conversion agrees with the bus")
    # Every other calibration this package builds for a mock homes at zero, and
    # with homing_offset=0 the two mappings above and below this line agree by
    # accident. That is how a disagreement between them survived every check in
    # this file: #3758's own failure mode, a constant per-joint offset between
    # the commanded number and the pose it produces, reproduced inside the
    # package written to detect it. So this calibration gives *every* joint a
    # non-zero offset and mixes the signs, where a term applied on one side and
    # not the other cannot cancel and cannot be mistaken for rounding.
    offset_file = _write_calibration(
        room / "offsets.json",
        homing={
            "shoulder_pan": 11,
            "shoulder_lift": -7,
            "elbow_flex": 17,
            "wrist_flex": -23,
            "wrist_roll": -9,
            "gripper": 5,
        },
    )
    offset_calib = load_calibration(offset_file)
    offset_bus = bus_calibration(offset_calib)
    tick_gap: list[str] = []
    unit_gap: list[str] = []
    unreachable: list[str] = []
    for key in JOINT_KEYS:
        mine = offset_calib.joint(key)
        theirs = offset_bus.by_name(mine.name)
        # Quarter-unit steps because the narrowest fixture joint spans 1800
        # ticks: on a coarser grid no commanded value ever lands exactly halfway
        # between two ticks, and halfway is where two rounding rules diverge.
        for step in range(401):
            value = step / 4.0
            if mine.to_ticks(value) != theirs.to_ticks(value):
                tick_gap.append(
                    f"{mine.name} at {value:g}: {mine.to_ticks(value)} vs {theirs.to_ticks(value)}"
                )
        for ticks in range(mine.range_min, mine.range_max + 1):
            if mine.to_units(ticks) != theirs.to_normalized(ticks):
                unit_gap.append(
                    f"{mine.name} at {ticks}: {mine.to_units(ticks):.4f} vs "
                    f"{theirs.to_normalized(ticks):.4f}"
                )
        if mine.to_ticks(0.0) < theirs.range_min or mine.to_ticks(100.0) > theirs.range_max:
            unreachable.append(
                f"{mine.name} declares 0..100 as ticks "
                f"{mine.to_ticks(0.0)}..{mine.to_ticks(100.0)}, bus travel "
                f"{theirs.range_min}..{theirs.range_max}"
            )
    check(
        "units -> ticks matches the bus on every joint, 0..100 inclusive",
        not tick_gap,
        f"{len(tick_gap)} disagree, first {tick_gap[0]}" if tick_gap
        else f"{DOF} joints x 401 values",
    )
    check(
        "ticks -> units matches the bus on every joint, end to end",
        not unit_gap,
        f"{len(unit_gap)} disagree, first {unit_gap[0]}" if unit_gap
        else f"{DOF} joints, every tick of measured travel",
    )
    check(
        "the declared 0..100 is reachable within the travel the bus will accept",
        not unreachable,
        unreachable[0] if unreachable else "no joint maps its own range outside the bus clamp",
    )
    check(
        "homing_offset is still carried, it is only no longer applied twice",
        all(joint.homing_offset for joint in offset_calib.joints)
        and all(
            offset_bus.by_name(joint.name).homing_offset == joint.homing_offset
            for joint in offset_calib.joints
        ),
        "out of the arithmetic, still in the digest and in what the live chain is compared to",
    )

    # -- 4. drift: an injected constant offset, reported in mm ------------
    print("\n[4] calibration drift detector")
    clock = VirtualClock()
    arm, transport = build("follower", follower_file)
    calibration = arm.calibration
    # 17 degrees on elbow_flex is the offset from LeRobot #3758.
    target_deg = 17.0
    elbow = calibration.joint("elbow_flex.pos")
    offset_ticks = int(round(target_deg / (360.0 / TICKS_PER_REV)))
    _hook(transport, _OFFSET_HOOKS)(elbow.id, offset_ticks)
    report = arm.check_drift(
        home, ticks=120, settle_ticks=15, keeper=RateKeeper(CONTROL_HZ, clock=clock)
    )
    print(report.table())
    found = {joint.joint for joint in report.drifted}
    check("finds the offset joint", found == {"elbow_flex.pos"}, f"drifted={sorted(found)}")
    hit = report.joints[JOINT_KEYS.index("elbow_flex.pos")]
    check(
        "reports it in degrees, close to the injected 17",
        abs(abs(hit.mean_deg) - target_deg) < 1.0,
        f"{hit.mean_deg:+.2f} deg",
    )
    expected_mm = math.radians(target_deg) * LEVER_ARM_MM["elbow_flex.pos"]
    check(
        "reports it in mm at the gripper, near the ~6 cm of #3758",
        abs(abs(hit.mean_mm) - expected_mm) < 3.0,
        f"{hit.mean_mm:+.2f} mm vs {expected_mm:.2f} mm expected",
    )
    check(
        "its interval excludes zero",
        hit.excludes_zero,
        f"ci {hit.ci_mm[0]:+.2f}..{hit.ci_mm[1]:+.2f} mm",
    )
    verdict = report.verdict()
    check("the verdict refuses", not verdict.ok, verdict.codes()[0])
    print("      " + verdict.explain(indent="        ").replace("\n", "\n      "))
    measured = report.measurements()["drift.elbow_flex.pos"]
    check(
        "reported as a Measurement, not a bare float",
        measured.n > 0 and measured.units == "mm",
        str(measured),
    )
    check(
        "a clean joint is not called drifted",
        not report.joints[JOINT_KEYS.index("wrist_roll.pos")].drifted,
    )

    clean_arm, _ = build("follower-clean", follower_file)
    clean = clean_arm.check_drift(
        home, ticks=120, settle_ticks=15, keeper=RateKeeper(CONTROL_HZ, clock=clock)
    )
    check("a clean arm produces no drifted joints", not clean.drifted)
    check("and its verdict passes", clean.verdict().ok, clean.verdict().codes()[0])
    short = clean_arm.check_drift(
        home, ticks=25, settle_ticks=15, keeper=RateKeeper(CONTROL_HZ, clock=clock)
    )
    check(
        "refuses to clear an arm from too few frames",
        short.underpowered and not short.verdict().ok,
        short.verdict().codes()[0],
    )
    baseline = DriftBaseline.from_report(clean)
    stale = replace(baseline, calibration_sha256="0" * 64)
    try:
        clean_arm.check_drift(home, ticks=120, settle_ticks=15,
                              keeper=RateKeeper(CONTROL_HZ, clock=clock), baseline=stale)
    except ConfigError as error:
        check("refuses a baseline from another calibration", True, str(error)[-58:])
    else:
        check("refuses a baseline from another calibration", False, "accepted")
    arm.disconnect()
    clean_arm.disconnect()

    # -- 5. gripper stall as a contact proxy -------------------------------
    print("\n[5] gripper stall")
    clock = VirtualClock()
    leader, _ = build("leader", leader_file)
    follower, follower_bus = build("follower", follower_file)
    gripper = follower.calibration.joint("gripper.pos")
    # The jaw closes onto an object at 40 percent of travel and goes no further.
    stall_at = gripper.to_ticks(40.0)
    _hook(follower_bus, _STALL_HOOKS)(gripper.id, stall_at)
    pair = SO101Pair(
        leader,
        follower,
        clock=clock,
        stall_detector=GripperStallDetector(threshold_units=4.0, min_steps=4),
    )
    # The operator walks the leader's jaw shut. A real leader is backdriven;
    # here it is driven, which is the same thing from the follower's side.
    leader.set_torque(True)
    closing = np.tile(home, (40, 1)).astype(np.float32)
    closing[:, GRIPPER] = np.linspace(50.0, 10.0, 40)
    poses = iter(closing)

    def move_leader(tick: int, command: np.ndarray, state: np.ndarray) -> None:
        leader.write_action(next(poses))

    move_leader(0, home, home)
    pair.teleoperate(ticks=len(closing) - 1, on_tick=move_leader)
    events = pair.stall.events
    check("a stall is detected", bool(events), f"{len(events)} event(s)")
    if events:
        event = events[0]
        print(f"      {event.as_dict()}")
        check("it is labelled as closing", event.closing)
        check("it persisted before being confirmed", event.steps >= 4, f"{event.steps} ticks")
        check(
            "its residual is reported in mm of jaw aperture",
            abs(event.mean_residual_mm) > 0.5,
            f"{event.mean_residual_mm:+.2f} mm",
        )
    free = GripperStallDetector(threshold_units=4.0, min_steps=4)
    for index in range(20):
        free.update(index, 50.0, 50.0 - 0.1 * (index % 2))
    check("a free-moving jaw produces no stall", not free.events)
    leader.disconnect()
    follower.disconnect()

    # -- 6. an overrun is reported, not swallowed -------------------------
    print("\n[6] rate discipline")
    leader, _ = build("leader", leader_file)
    follower, _ = build("follower", follower_file)
    pair = SO101Pair(leader, follower)  # real clock: this is a timing claim
    period_ms = 1e3 / CONTROL_HZ

    _, healthy = pair.teleoperate(ticks=20)
    # One stray late tick on a shared laptop is the OS scheduler, not the loop.
    # The claim being tested here is that a loop that keeps up is not accused of
    # overrunning; the claim about reporting is tested below.
    check(
        "a loop that keeps up is not accused of overrunning",
        len(healthy.rate.overruns) <= 1,
        f"realised {healthy.rate.realised_hz:.2f} Hz, "
        f"{len(healthy.rate.overruns)} late, jitter "
        f"{healthy.rate.measurements()['rate.period_jitter_ms'].value:.2f} ms",
    )
    check("and its verdict passes", healthy.rate.verdict(tolerate=0.1).ok)

    leader.disconnect()
    follower.disconnect()
    leader, _ = build("leader", leader_file)
    follower, _ = build("follower", follower_file)
    pair = SO101Pair(leader, follower)

    def slow(tick: int, command: np.ndarray, state: np.ndarray) -> None:
        # Work inside the tick that does not fit in the tick.
        time.sleep(0.050)

    _, slowed = pair.teleoperate(ticks=15, on_tick=slow)
    check(
        "a loop that cannot keep up reports every late tick",
        len(slowed.rate.overruns) >= 14,
        f"{len(slowed.rate.overruns)}/{slowed.rate.ticks} late, worst "
        f"{slowed.rate.worst_late_s * 1e3:.1f} ms over a {period_ms:.1f} ms period",
    )
    late_verdict = slowed.rate.verdict()
    check(
        "the verdict refuses rather than passing quietly",
        not late_verdict.ok,
        late_verdict.codes()[0],
    )
    print("      " + late_verdict.explain(indent="        ").replace("\n", "\n      "))
    check(
        "the realised rate is reported, not assumed",
        slowed.rate.realised_hz < CONTROL_HZ * 0.8,
        f"{slowed.rate.realised_hz:.2f} Hz against a nominal {CONTROL_HZ:.0f} Hz",
    )
    check(
        "the session verdict carries it too",
        not slowed.verdict().ok,
        ",".join(slowed.verdict().codes()),
    )
    metadata = slowed.as_metadata()
    check(
        "both calibration hashes are in the episode metadata",
        len(metadata["leader_calibration_sha256"]) == 64
        and len(metadata["follower_calibration_sha256"]) == 64
        and metadata["leader_calibration_sha256"] != metadata["follower_calibration_sha256"],
    )
    check(
        "tracking error is reported per joint, mean and max",
        set(metadata["tracking_error"]["per_joint"]) == set(JOINT_KEYS),
    )
    leader.disconnect()
    follower.disconnect()

    # -- 7. the declarations, restated where they can be read -------------
    print("\n[7] declarations")
    print(f"      seedable       = False   {SEEDABLE_IS_FALSE}")
    print(f"      stage_events   = False   {STAGE_EVENTS_NEED_A_POSE_SOURCE}")
    print("      closed_loop    = True    the world responds to what the follower did")
    print("      outcomes       = True    a human can always say what happened")
    embodiment = so101_embodiment(load_calibration(follower_file))
    check("the embodiment descriptor validates", embodiment.validate().ok)
    check("it does not claim seedable", "seedable" not in embodiment.capabilities)
    action_spec = embodiment.action[0]
    check("its action is float32[6] in LeRobot key order", action_spec.dim_labels == JOINT_KEYS)
    check(
        "and declares its normalisation as a discriminator",
        "normalization" in action_spec.discriminators
        and action_spec.metadata["normalization"] == NORMALIZATION,
    )
    check("every channel spec validates", all(spec.validate().ok for spec in so101_channels()))

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="exercise the arm and pair layer against MockTransport; no hardware needed",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selftest:
        parser.print_help()
        return 2
    return _selftest()


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
