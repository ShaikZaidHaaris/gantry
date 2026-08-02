"""The wire to six Feetech STS3215 servos, and the mock that makes it provable.

This is the lowest layer of the SO-101 plugin and the only one that touches a
serial port. Everything above it — the arm, the clamp's callers, the operator
console, the evaluator — is written against :class:`Transport` and is therefore
runnable, in full, with no hardware plugged in. That is not a testing
convenience. A rig that can only be exercised while it is powered gets exercised
rarely, and the paths that matter most are the ones that only occur when
something has already gone wrong.

What this file is actually for
------------------------------
Position control of a six-servo chain is about forty lines. The rest of this
module exists because of three specific ways a real SO-101 hurts itself, and
each one is answered by a mechanism that cannot be switched off:

**No clamp is the LeRobot default.** ``max_relative_target`` defaults to
``None`` upstream, which means an unclamped goal. A policy that emits one bad
action at 30 Hz then drives the follower into the table at full torque in a
single control step. So a clamp is a *required constructor argument* here:
:class:`Transport` cannot be instantiated without one, there is no default
value, and the public write path is concrete on the base class so a subclass
cannot route around it.

**A stale position read is how an arm runs away.** The clamp is relative to
where the arm *is*. Substitute a last-known value for a read that timed out and
the delta is computed against a fiction; the arm keeps being told to move a
"small" amount away from a position it left some time ago. So a timeout
*invalidates* the cached reference rather than falling back to it, and a write
whose reference is older than :attr:`SafetyLimits.max_reference_age` is refused
with :class:`~gantry.errors.SafetyAbort`. There is no code path in this file
that answers a read with a value the bus did not just produce.

**A calibration error is invisible.** LeRobot issue #3758: a ~17 degree joint
offset produced ~6 cm of gripper error, and the policy trained on it looked
healthy the whole way — it learned the biased mapping and the loss never said
anything. The tick-to-normalized conversion here is therefore stated
explicitly, round-tripped exactly over the full tick range in the selftest, and
refuses to run against a :class:`Calibration` that did not come from a
calibration procedure. A digest of the calibration travels with it so a
mid-campaign recalibration is detectable retrospectively, which is what
CORPUS-PROTOCOL.md asks for.

What this file will not tell you
--------------------------------
There is no force or torque sensing on this arm. ``Present_Load`` is a PWM duty
proxy and ``Present_Current`` is a raw count whose scale this file will not
guess; both are reported as :class:`~gantry.spine.Measurement` values whose
``method`` says what they are, so that nothing downstream can quote them as
newtons. The stall signal derived from them is a *contact* proxy and nothing
more: it can say a gripper closed on something solid, and it cannot say what.

Layering
--------
``Transport``          the narrow interface, and every safety rule, concrete.
``SerialTransport``    the real one: the Feetech STS/SMS packet protocol over
                       anything with ``read``/``write``. No hardware library is
                       imported at module scope.
``ServoChain``         a simulated six-servo bus with fault injection.
``MockTransport``      ``ServoChain`` behind the ``Transport`` interface.
``LoopbackSerial``     ``ServoChain`` behind a *serial port*, so the packet
                       codec in ``SerialTransport`` is exercised too. Without
                       it the mock would prove the layers above and leave the
                       wire format itself untested.

This module imports nothing from its siblings, deliberately: it is the bottom
of the stack and everything else may import it. In particular it does not
restate the canonical joint order, which ``arm.JOINTS`` owns — a second copy of
an ordering is a second thing to fall out of step, and an ordering error is
silent all the way to the loss curve.
"""

from __future__ import annotations

import argparse
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from gantry.errors import ComponentError, ConfigError, FaultError, SafetyAbort
from gantry.spine import Measurement, digest_of

VERSION = "0.1.0.dev0"


# ==========================================================================
# the STS3215 control table
# ==========================================================================
#
# Source: the STS/SMS series control table LeRobot ships in
# ``lerobot/motors/feetech/tables.py`` (STS_SMS_SERIES_CONTROL_TABLE), which
# agrees with the FEETECH "Serial Bus Smart Control servo Communication
# Protocol" manual. Every address below was read off that table rather than
# recalled. Addresses this file does not use are omitted rather than copied,
# so that nothing here is an untested claim.
#
# Two entries carry real uncertainty and are marked UNCERTAIN where they
# appear. Nothing in this module branches on either of them.


@dataclass(frozen=True)
class Register:
    """One control-table entry.

    ``sign_bit`` is set for the registers Feetech stores as sign-magnitude
    rather than two's complement — an easy and very quiet bug, because a
    small negative reads as a large positive and a load of -3 becomes 1027.
    """

    name: str
    address: int
    length: int
    writable: bool = False
    #: Lives in EEPROM: writing needs the lock open and wears the part out.
    eeprom: bool = False
    sign_bit: int | None = None

    def __post_init__(self) -> None:
        if self.length not in (1, 2, 4):
            raise ValueError(f"register {self.name!r} has unusable length {self.length}")


def _register(*args: Any, **kwargs: Any) -> Register:
    return Register(*args, **kwargs)


#: Only the registers this transport actually reads or writes.
CONTROL_TABLE: dict[str, Register] = {
    r.name: r
    for r in (
        _register("Model_Number", 3, 2),
        _register("ID", 5, 1, writable=True, eeprom=True),
        _register("Baud_Rate", 6, 1, writable=True, eeprom=True),
        _register("Min_Position_Limit", 9, 2, writable=True, eeprom=True),
        _register("Max_Position_Limit", 11, 2, writable=True, eeprom=True),
        _register("Homing_Offset", 31, 2, writable=True, eeprom=True, sign_bit=11),
        _register("Operating_Mode", 33, 1, writable=True, eeprom=True),
        _register("Torque_Enable", 40, 1, writable=True),
        _register("Acceleration", 41, 1, writable=True),
        _register("Goal_Position", 42, 2, writable=True),
        _register("Goal_Velocity", 46, 2, writable=True, sign_bit=15),
        _register("Lock", 55, 1, writable=True),
        _register("Present_Position", 56, 2),
        _register("Present_Velocity", 58, 2, sign_bit=15),
        _register("Present_Load", 60, 2, sign_bit=10),
        _register("Present_Voltage", 62, 1),
        _register("Present_Temperature", 63, 1),
        _register("Status", 65, 1),
        _register("Moving", 66, 1),
        # UNCERTAIN. Address 69 is in the STS/SMS table, but I could not confirm
        # from a primary FEETECH source that the STS3215 populates it, nor its
        # scale (6.5 mA/LSB is widely quoted and widely unattributed). This
        # transport therefore reads it as a raw count and refuses to convert it
        # to amps — see read_present_current.
        _register("Present_Current", 69, 2, sign_bit=15),
    )
}

#: Encoder counts over one full turn. 12-bit absolute encoder.
RESOLUTION = 4096
DEGREES_PER_TICK = 360.0 / RESOLUTION

#: UNCERTAIN. The model number reported by an STS3215 is widely given as 777,
#: but this transport only ever reports it back from a ping and never compares
#: against it — a chain that answers is a chain that answers.
STS3215_MODEL_NUMBER = 777

#: Torque_Enable values. Some firmware treats 2 as a damped/brake state; this
#: transport writes only 0 and 1 because those two are the documented pair.
TORQUE_OFF = 0
TORQUE_ON = 1

#: Lock: 0 opens EEPROM for writing, 1 closes it. Backwards from how it reads.
EEPROM_UNLOCKED = 0
EEPROM_LOCKED = 1

#: Present_Load magnitude is documented as 0..1000 = 0..100 % PWM duty, with
#: bit 10 as the direction. Some firmware runs to 1023, which is why the
#: fraction below is reported alongside its raw count rather than instead of it.
LOAD_FULL_SCALE = 1000.0

# -- packet protocol -------------------------------------------------------
#
# Dynamixel-1.0-shaped, which is what the SCS/STS family speaks:
#
#   instruction  FF FF  ID  LEN  INST  PARAM...  CHK
#   status       FF FF  ID  LEN  ERR   PARAM...  CHK
#
# LEN counts the bytes after it, inclusive of the checksum. CHK is the bitwise
# complement of the sum from ID onward. Multi-byte registers are little-endian
# on the STS/SMS series (the older SCS series is big-endian, which is why this
# module is named for STS and refuses to pretend to cover both).

HEADER = b"\xff\xff"

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82  # STS/SMS only; absent on the older SCS series.
INST_SYNC_WRITE = 0x83

BROADCAST_ID = 0xFE
MAX_ID = 0xFC

#: Bits of the status error byte. UNCERTAIN in the naming: vendor documents
#: disagree about the middle bits. Nothing here branches on an individual bit —
#: any non-zero error byte is a fault — so the labels are for the human reading
#: the message and the behaviour does not depend on them being right.
ERROR_BITS: tuple[tuple[int, str], ...] = (
    (0x01, "voltage"),
    (0x02, "angle sensor"),
    (0x04, "overheat"),
    (0x08, "current"),
    (0x10, "angle limit"),
    (0x20, "overload"),
)


def describe_error(bits: int) -> str:
    named = [name for mask, name in ERROR_BITS if bits & mask]
    unknown = bits & ~sum(mask for mask, _ in ERROR_BITS)
    if unknown:
        named.append(f"unrecognised bits 0x{unknown:02x}")
    return ", ".join(named) or f"0x{bits:02x}"


# ==========================================================================
# failures
# ==========================================================================


class BusFailure(ComponentError):
    """A transaction that did not produce a usable answer.

    ``motor_id`` is set when the failure is attributable to one servo and left
    ``None`` when it is genuinely chain-wide. The distinction is for the person
    holding the arm at three in the morning: "servo 4 stopped answering" sends
    them to one cable, and "the chain stopped answering" sends them to the
    power supply. Blaming all six for one dead servo sends them to the wrong
    one, which is worse than saying nothing.
    """

    def __init__(self, message: str, motor_id: int | None = None, partial: Any = None):
        super().__init__(message, partial)
        self.motor_id = motor_id


class BusTimeout(BusFailure):
    """One transaction got no answer, or a short one.

    Recoverable by the taxonomy in :mod:`gantry.errors`: a single dropped frame
    on a shared bus is ordinary, and the layer above may retry or fail the
    trial. What it may *not* do is carry on with the previous value, which is
    why this is raised instead of returning one.
    """


class BusCorruption(BusFailure):
    """A frame arrived and did not survive its own checksum."""


class ServoFault(FaultError):
    """A servo reported a hardware error, or stopped answering entirely.

    Halting, always. A chain with a servo in an unknown state cannot be
    commanded safely, and the episodes it would go on to produce would look
    like every other episode.
    """


class UnsupportedRegister(ConfigError):
    """A register this transport declines to interpret on this hardware."""


# ==========================================================================
# units: ticks <-> the conventions a policy speaks
# ==========================================================================
#
# The conversion is stated here in full because getting it silently wrong is
# the #3758 failure mode at this layer. These three spellings are LeRobot's
# vocabulary, not ours; the plugin's embodiment descriptor names the same
# strings, and they mean the same thing in both places.

RANGE_0_100 = "RANGE_0_100"
RANGE_M100_100 = "RANGE_M100_100"
DEGREES = "DEGREES"
NORM_MODES = (RANGE_0_100, RANGE_M100_100, DEGREES)

#: Units string for each convention, in the vocabulary gantry.spine.units
#: parses. RANGE_* are affine remaps of a joint's own recorded travel: they are
#: dimensionless by construction and are *not* percent of anything physical.
NORM_UNITS = {RANGE_0_100: "fraction", RANGE_M100_100: "fraction", DEGREES: "deg"}


@dataclass(frozen=True)
class JointCalibration:
    """One servo's measured travel, which is what makes a tick mean anything.

    ``range_min`` and ``range_max`` are raw encoder counts observed at the ends
    of that joint's usable travel *on this physical arm*. They are measurements.
    Two SO-101s off the same print will not share them, and an arm that has been
    reassembled does not share them with its former self.
    """

    name: str
    id: int
    range_min: int
    range_max: int
    norm_mode: str = RANGE_0_100
    #: LeRobot's flag for a joint whose positive direction is reversed.
    drive_mode: int = 0
    #: Written into the servo's EEPROM by the calibration procedure. Carried
    #: here for the digest and for comparison against the live chain; this
    #: module never applies it in software, because the servo already has.
    homing_offset: int = 0

    def __post_init__(self) -> None:
        if self.norm_mode not in NORM_MODES:
            raise ConfigError(
                f"joint {self.name!r} declares norm_mode {self.norm_mode!r}; "
                f"known: {NORM_MODES}"
            )
        if not 0 <= self.id <= MAX_ID:
            raise ConfigError(f"joint {self.name!r} has motor id {self.id}, outside 0..{MAX_ID}")
        if self.range_max <= self.range_min:
            raise ConfigError(
                f"joint {self.name!r} has range_min={self.range_min} >= "
                f"range_max={self.range_max}; a joint with no travel cannot be normalised"
            )
        for bound in (self.range_min, self.range_max):
            if not 0 <= bound < RESOLUTION:
                raise ConfigError(
                    f"joint {self.name!r} has a travel bound of {bound} ticks, outside "
                    f"0..{RESOLUTION - 1}"
                )

    @property
    def span(self) -> int:
        return self.range_max - self.range_min

    @property
    def units(self) -> str:
        return NORM_UNITS[self.norm_mode]

    # -- the conversion, both ways ----------------------------------------

    def to_normalized(self, ticks: int) -> float:
        """Raw encoder count -> the convention this joint is declared in.

        ``RANGE_0_100``   ``(t - min) / (max - min) * 100``, mirrored about 50
                          when ``drive_mode`` is set.
        ``RANGE_M100_100`` the same ratio mapped onto [-100, 100], negated when
                          ``drive_mode`` is set.
        ``DEGREES``       ``(t - midpoint) * 360 / 4096``, so zero is the middle
                          of the recorded travel rather than the encoder's zero.

        Deliberately identical to LeRobot's ``_normalize``: a corpus recorded by
        ``lerobot-record`` and one recorded through this transport have to be the
        same numbers, and "close enough" here is a systematic bias that training
        will absorb without complaint.
        """
        if self.norm_mode is DEGREES or self.norm_mode == DEGREES:
            midpoint = (self.range_min + self.range_max) / 2
            return (ticks - midpoint) * 360.0 / RESOLUTION
        ratio = (ticks - self.range_min) / self.span
        if self.norm_mode == RANGE_0_100:
            value = ratio * 100.0
            return 100.0 - value if self.drive_mode else value
        value = ratio * 200.0 - 100.0
        return -value if self.drive_mode else value

    def to_ticks(self, value: float, *, tolerance: float = 0.0) -> int:
        """The convention -> a raw encoder count, refusing anything off-scale.

        Out of range is a refusal rather than a clamp. A policy emitting 137 in
        a 0..100 convention is telling you something — that it is off its
        training distribution, or that the two ends disagree about the
        convention — and silently pulling it back to 100 turns that into a
        slightly wrong motion that nobody ever hears about. The relative clamp
        is a safety device applied to a command that made sense; it is not a
        licence to accept one that did not.

        Rounds where LeRobot truncates. Truncation biases every conversion
        toward zero by up to a tick; over a corpus that is a small constant
        offset in exactly the direction #3758 warns is invisible. One tick is
        0.088 degrees, so the disagreement with upstream is bounded and stated
        rather than hidden.
        """
        low, high = self.normalized_bounds
        if not (low - tolerance) <= value <= (high + tolerance):
            raise SafetyAbort(
                f"joint {self.name!r} was commanded {value:g} {self.units}, outside its "
                f"declared range [{low:g}, {high:g}]; refusing rather than clamping, "
                "because an off-scale command is evidence and a clamped one is not"
            )
        if self.norm_mode == DEGREES:
            midpoint = (self.range_min + self.range_max) / 2
            return _round_half_away(value * RESOLUTION / 360.0 + midpoint)
        if self.norm_mode == RANGE_0_100:
            value = 100.0 - value if self.drive_mode else value
            ratio = value / 100.0
        else:
            value = -value if self.drive_mode else value
            ratio = (value + 100.0) / 200.0
        return _round_half_away(ratio * self.span + self.range_min)

    @property
    def normalized_bounds(self) -> tuple[float, float]:
        """The travel ends, expressed in this joint's own convention."""
        low = self.to_normalized(self.range_min)
        high = self.to_normalized(self.range_max)
        return (low, high) if low <= high else (high, low)

    def clamp_ticks(self, ticks: int) -> int:
        return max(self.range_min, min(self.range_max, ticks))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "id": self.id,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "norm_mode": self.norm_mode,
            "drive_mode": self.drive_mode,
            "homing_offset": self.homing_offset,
        }


def _round_half_away(value: float) -> int:
    """Round halves away from zero. ``round`` alone is banker's rounding, which
    would make a conversion depend on the parity of the tick it lands between."""
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


@dataclass(frozen=True)
class Calibration:
    """Every joint of one arm, and where the numbers came from.

    ``measured`` is the load-bearing field. A calibration invented for a test
    fixture and one produced by moving the arm to both ends of every joint are
    the same shape, and only one of them may drive a motor. :class:`Transport`
    refuses an unmeasured calibration unless the caller says out loud that
    nothing physical is attached.
    """

    joints: tuple[JointCalibration, ...]
    #: True only when these numbers came off a real calibration run.
    measured: bool = False
    #: Where they came from, for the record. A path, a serial number, a commit.
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.joints:
            raise ConfigError("a calibration with no joints describes nothing")
        ids = [joint.id for joint in self.joints]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ConfigError(
                f"motor id(s) {duplicates} appear twice in one chain; every servo on a "
                "bus needs a distinct id, and two answering at once is unreadable"
            )
        names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ConfigError(f"joint name(s) {duplicates} appear twice in one calibration")

    @property
    def ids(self) -> tuple[int, ...]:
        """Motor ids in the order the joints were declared. That order is the
        action-vector order, and it belongs to the caller, not to this file."""
        return tuple(joint.id for joint in self.joints)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    def by_id(self, motor_id: int) -> JointCalibration:
        for joint in self.joints:
            if joint.id == motor_id:
                return joint
        raise ConfigError(f"no joint on this chain has motor id {motor_id}; ids are {self.ids}")

    def by_name(self, name: str) -> JointCalibration:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise ConfigError(f"no joint named {name!r}; joints are {self.names}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "joints": [joint.as_dict() for joint in self.joints],
            "measured": self.measured,
            "source": self.source,
        }

    @property
    def digest(self) -> str:
        """A stable hash of the numbers, for the episode metadata.

        CORPUS-PROTOCOL.md requires the SHA-256 of both arms' calibration with
        every episode, so that a recalibration halfway through a campaign is
        detectable after the fact rather than never. ``source`` and ``measured``
        are inside the digest on purpose: swapping a fixture for a real
        calibration changes what the run means even if every number matched.
        """
        return digest_of(self.as_dict())

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_lerobot(
        cls,
        payload: Mapping[str, Mapping[str, Any]],
        *,
        source: str | None = None,
        norm_modes: Mapping[str, str] | None = None,
    ) -> "Calibration":
        """Read a LeRobot ``so101_new_calib`` JSON body, in its declared order.

        The file is ``{joint_name: {id, drive_mode, homing_offset, range_min,
        range_max}}``. Nothing is renamed and nothing is reordered: a mapping
        read out of order is an action vector permuted, which every check
        downstream passes.

        ``norm_modes`` overrides the convention per joint. The default is
        RANGE_0_100 throughout, which is what this rig's corpora are declared
        in. Upstream ``SO101Follower`` instead uses RANGE_M100_100 for the five
        body joints and RANGE_0_100 only for the gripper — so a corpus recorded
        under stock LeRobot defaults and one recorded here are *not*
        interchangeable, and this argument is where you say which you have. The
        convention is part of the calibration digest for exactly that reason.
        """
        modes = dict(norm_modes or {})
        joints = []
        for name, entry in payload.items():
            missing = [k for k in ("id", "range_min", "range_max") if k not in entry]
            if missing:
                raise ConfigError(
                    f"calibration entry {name!r} is missing {missing}; refusing to fill "
                    "in a travel limit that was never measured"
                )
            joints.append(
                JointCalibration(
                    name=name,
                    id=int(entry["id"]),
                    range_min=int(entry["range_min"]),
                    range_max=int(entry["range_max"]),
                    norm_mode=modes.get(name, RANGE_0_100),
                    drive_mode=int(entry.get("drive_mode", 0)),
                    homing_offset=int(entry.get("homing_offset", 0)),
                )
            )
        return cls(tuple(joints), measured=True, source=source)

    @classmethod
    def fixture(
        cls,
        names: Sequence[str],
        *,
        ids: Sequence[int] | None = None,
        range_min: int = 512,
        range_max: int = 3584,
        norm_mode: str = RANGE_0_100,
    ) -> "Calibration":
        """A calibration for a mock, and unmistakably not a measurement.

        ``measured`` stays False, so a :class:`Transport` built on this refuses
        to drive anything real. Named ``fixture`` rather than ``default``
        because there is no such thing as a default travel limit.
        """
        ids = tuple(ids) if ids is not None else tuple(range(1, len(names) + 1))
        if len(ids) != len(names):
            raise ConfigError(f"{len(names)} joint name(s) but {len(ids)} motor id(s)")
        return cls(
            tuple(
                JointCalibration(name, motor_id, range_min, range_max, norm_mode)
                for name, motor_id in zip(names, ids)
            ),
            measured=False,
            source="fixture (not a measurement)",
        )


# ==========================================================================
# the clamp
# ==========================================================================


@dataclass(frozen=True)
class SafetyLimits:
    """How far a single write may move the arm, and how stale its reference may be.

    Constructed only through :meth:`ticks`, :meth:`degrees` or
    :meth:`normalized`, never from a bare number. Upstream's
    ``max_relative_target`` is a bare int whose unit is implied by whichever
    layer happens to be holding it; on this rig the same integer means 0.088
    degrees, 1 degree, or a joint-dependent slice of travel depending on how you
    read it. A wrong reading of it is a factor-of-eleven error in a safety
    limit, so this type makes the unit part of the value and the constructor
    refuses to guess.
    """

    #: Per-step travel budget and the unit it is quoted in.
    amount: float
    unit: str
    #: A clamp reference older than this is not a reference. 100 ms is three
    #: control periods at 30 Hz — long enough to tolerate one retried read,
    #: short enough that the arm cannot have gone far.
    max_reference_age: float = 0.100
    #: Consecutive unanswered reads on one servo before the chain is faulted
    #: rather than merely retried.
    max_consecutive_timeouts: int = 3
    #: Movement below this is not movement; it is encoder noise.
    stall_epsilon_ticks: int = 3
    #: Consecutive commanded-but-motionless steps that constitute a stall.
    stall_steps: int = 5

    _UNITS = ("ticks", "deg", "normalized")

    def __post_init__(self) -> None:
        if self.unit not in self._UNITS:
            raise ConfigError(f"unknown clamp unit {self.unit!r}; expected one of {self._UNITS}")
        if not self.amount > 0 or self.amount != self.amount or self.amount == float("inf"):
            raise ConfigError(
                f"max_relative_target is {self.amount!r}; a clamp that is zero, negative, "
                "infinite or NaN is not a clamp"
            )
        if not self.max_reference_age > 0:
            raise ConfigError(
                f"max_reference_age is {self.max_reference_age!r}; a write with no freshness "
                "requirement can be clamped against an arbitrarily old position"
            )
        if self.max_consecutive_timeouts < 1:
            raise ConfigError("max_consecutive_timeouts must be at least 1")
        # Written as negated comparisons, not as `< 0` and `< 1`, so a NaN is
        # refused rather than accepted. The positive form lets NaN through, and a
        # NaN epsilon disables the stall proxy exactly the way a NaN clamp
        # disables the clamp above: every `movement <= nan` is False, so no step
        # is ever counted as motionless and the contact annotation reports no
        # stall on a gripper that never moved.
        if not self.stall_epsilon_ticks >= 0 or not self.stall_steps >= 1:
            raise ConfigError(
                f"stall_epsilon_ticks={self.stall_epsilon_ticks!r} and "
                f"stall_steps={self.stall_steps!r}; the first must be >= 0 and the second "
                ">= 1, and neither may be NaN — a NaN threshold is never crossed, so the "
                "stall proxy would report no contact rather than no reading"
            )

    @classmethod
    def ticks(cls, amount: float, **kwargs: Any) -> "SafetyLimits":
        """Raw encoder counts per write. 1 tick is 0.088 degrees."""
        return cls(float(amount), "ticks", **kwargs)

    @classmethod
    def degrees(cls, amount: float, **kwargs: Any) -> "SafetyLimits":
        """Degrees of joint travel per write, which is how a human thinks about it."""
        return cls(float(amount), "deg", **kwargs)

    @classmethod
    def normalized(cls, amount: float, **kwargs: Any) -> "SafetyLimits":
        """A slice of each joint's own travel, in that joint's convention.

        Joint-dependent in ticks, because each joint's recorded travel differs.
        This is the reading that matches upstream ``max_relative_target``, which
        is applied to normalised values.
        """
        return cls(float(amount), "normalized", **kwargs)

    def resolve(self, calibration: Calibration) -> dict[int, int]:
        """Per-motor budget in ticks, which is the only unit the bus has.

        Rounded up, never down, and never to zero: a limit that rounds to zero
        would freeze the arm while looking like a working configuration.
        """
        budgets: dict[int, int] = {}
        for joint in calibration.joints:
            if self.unit == "ticks":
                raw = self.amount
            elif self.unit == "deg":
                raw = self.amount / DEGREES_PER_TICK
            else:
                low, high = joint.normalized_bounds
                raw = self.amount * joint.span / (high - low)
            budgets[joint.id] = max(1, int(raw + 0.999))
        return budgets

    def describe(self) -> str:
        return f"{self.amount:g} {self.unit}/step"


@dataclass(frozen=True)
class JointWrite:
    """What one servo was asked for, what it got, and the difference."""

    id: int
    name: str
    reference: int
    requested: int
    commanded: int

    @property
    def clamped_by(self) -> int:
        return self.requested - self.commanded

    @property
    def was_clamped(self) -> bool:
        return self.commanded != self.requested


@dataclass(frozen=True)
class WriteResult:
    """The outcome of one ``write_goal_position``, with nothing hidden.

    The relative clamp is the single place in this file that deliberately
    alters a command. It is not allowed to be invisible, so every alteration
    comes back here, gets counted on the transport, and can be handed to
    whatever writes the run's provenance.
    """

    writes: tuple[JointWrite, ...]
    reference_age: float

    @property
    def clamped(self) -> tuple[JointWrite, ...]:
        return tuple(write for write in self.writes if write.was_clamped)

    @property
    def any_clamped(self) -> bool:
        return bool(self.clamped)

    def explain(self) -> str:
        if not self.clamped:
            return "no joint was clamped"
        return "; ".join(
            f"{w.name} asked {w.requested} from {w.reference}, sent {w.commanded} "
            f"({w.clamped_by:+d} ticks removed)"
            for w in self.clamped
        )


@dataclass
class StallState:
    """Per-servo evidence that a commanded joint is not moving.

    This is the contact proxy, and it is the *only* one this hardware supports.
    An STS3215 has no load cell; ``Present_Load`` is a duty-cycle number and
    the current register's scale is unverified. What can be said honestly is
    kinematic: the joint was told to move, it did not, repeatedly. On the
    gripper that is very likely something between the fingers. On the shoulder
    it is very likely the table.

    It says nothing about *what* was contacted, which is why the funnel above
    still cannot emit lift or place without a pose source.
    """

    id: int
    name: str
    steps: int = 0
    tracking_error: int = 0
    stalled: bool = False

    def measurement(self, span: int) -> Measurement:
        return Measurement(
            float(self.tracking_error) * DEGREES_PER_TICK,
            n=self.steps,
            units="deg",
            method=(
                "commanded minus measured joint position, in degrees of joint travel. "
                "A kinematic contact proxy only: this arm has no force or torque sensing, "
                "so a stall says a joint stopped, never what stopped it"
            ),
            detail={"ticks": self.tracking_error, "span_ticks": span, "stalled": self.stalled},
        )


# ==========================================================================
# the interface, and every rule that is not negotiable
# ==========================================================================


@dataclass(frozen=True)
class Reading:
    """Positions, and when they were actually obtained."""

    ticks: dict[int, int]
    at: float

    def normalized(self, calibration: Calibration) -> dict[str, float]:
        return {
            calibration.by_id(motor_id).name: calibration.by_id(motor_id).to_normalized(value)
            for motor_id, value in self.ticks.items()
        }

    def vector(self, calibration: Calibration) -> tuple[float, ...]:
        """The reading in declared joint order, which is action-vector order."""
        return tuple(
            calibration.by_id(motor_id).to_normalized(self.ticks[motor_id])
            for motor_id in calibration.ids
        )


@dataclass(frozen=True)
class PingResult:
    id: int
    model_number: int
    error: int = 0


class Transport(ABC):
    """A serial chain of position-controlled servos, with the safety on.

    Subclasses implement four raw primitives — ping, read positions, read a
    register, write goals — and inherit everything that keeps the arm intact.
    The public methods are concrete and final by convention: the clamp, the
    freshness rule and the timeout escalation are not hooks a backend can
    choose to skip, because a backend that could skip them would eventually be
    written and would look exactly like one that could not.
    """

    def __init__(
        self,
        calibration: Calibration,
        *,
        max_relative_target: SafetyLimits,
        allow_unmeasured_calibration: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        """``max_relative_target`` is required and has no default.

        Upstream defaults it to ``None``, meaning no clamp at all. That default
        is fine for a simulator and is how a physical follower gets driven into
        the table, so there is no way to reach that state from here: omitting
        the argument is a ``TypeError`` at the call site and passing a bare
        number is a refusal that says which unit to use.
        """
        if not isinstance(max_relative_target, SafetyLimits):
            raise ConfigError(
                "max_relative_target must be a SafetyLimits, not "
                f"{type(max_relative_target).__name__}. A bare number does not say what it "
                "measures, and on this rig one tick, one degree and one unit of normalised "
                "travel differ by more than a factor of ten. Write "
                "SafetyLimits.degrees(3), SafetyLimits.ticks(34) or "
                "SafetyLimits.normalized(2) and the unit travels with the value."
            )
        if not calibration.measured and not allow_unmeasured_calibration:
            raise ConfigError(
                "this calibration is not marked as measured, and an unmeasured travel "
                "range makes every normalised value a fiction — LeRobot #3758 is a 17 "
                "degree offset that cost 6 cm at the gripper and never showed up in a "
                "loss curve. Load one with Calibration.from_lerobot(...), or pass "
                "allow_unmeasured_calibration=True if nothing physical is attached."
            )
        self._calibration = calibration
        self._limits = max_relative_target
        self._budget = max_relative_target.resolve(calibration)
        self._clock = clock
        self._open = False
        #: The clamp reference. Set only by a successful read, cleared by any
        #: failed one. There is no third way to populate it.
        self._reference: Reading | None = None
        self._timeouts: dict[int, int] = {motor_id: 0 for motor_id in calibration.ids}
        self._stalls: dict[int, StallState] = {
            joint.id: StallState(joint.id, joint.name) for joint in calibration.joints
        }
        self._clamp_events = 0
        self._writes = 0
        self._torque: dict[int, bool] = {motor_id: False for motor_id in calibration.ids}
        #: Called with every WriteResult that clamped something. A hook rather
        #: than a log line, so the evaluator above can put it in the record.
        self.on_clamp: Callable[[WriteResult], None] | None = None

    # -- what a subclass must provide --------------------------------------

    @abstractmethod
    def _open_bus(self) -> None:
        """Acquire the port. Raise ConfigError if it cannot be had."""

    @abstractmethod
    def _close_bus(self) -> None:
        """Release the port. Must be safe to call twice."""

    @abstractmethod
    def _ping(self, motor_id: int) -> PingResult:
        """One ping. Raise BusTimeout if nothing answers."""

    @abstractmethod
    def _read_register(self, motor_id: int, register: Register) -> int:
        """One register, already decoded from sign-magnitude if it is stored so."""

    @abstractmethod
    def _read_positions(self, ids: Sequence[int]) -> dict[int, int]:
        """Present_Position for each id. Raise BusTimeout if any did not answer.

        All or nothing on purpose. A partial position vector is not a smaller
        reading, it is a reading with a hole where the clamp reference for one
        joint should be, and the only safe thing to do with it is refuse it.
        """

    @abstractmethod
    def _write_goals(self, ticks: Mapping[int, int]) -> None:
        """Goal_Position for each id, already clamped. Raw, and the only writer."""

    @abstractmethod
    def _write_register(self, motor_id: int, register: Register, value: int) -> None:
        """One register write."""

    # -- lifecycle ---------------------------------------------------------

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @property
    def limits(self) -> SafetyLimits:
        return self._limits

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> "Transport":
        if self._open:
            return self
        self._open_bus()
        self._open = True
        self._reference = None
        return self

    def close(self) -> None:
        """Release the bus, torque off first.

        Torque is dropped before the port goes away because a chain left
        energised with a stale goal is a chain nobody is holding. If the
        disable itself fails the port still closes — a transport that refuses
        to let go of a port it cannot talk to is how a rig needs a power cycle.
        """
        if not self._open:
            return
        try:
            self.torque_disable()
        except (BusTimeout, BusCorruption, ServoFault):
            pass
        finally:
            self._close_bus()
            self._open = False
            self._reference = None

    def __enter__(self) -> "Transport":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _require_open(self, what: str) -> None:
        if not self._open:
            raise ConfigError(f"{what} needs an open bus; call open() first")

    # -- reading -----------------------------------------------------------

    def ping(self, motor_id: int) -> PingResult:
        self._require_open("ping")
        result = self._ping(motor_id)
        if result.error:
            raise ServoFault(
                f"servo {motor_id} answered a ping with error byte 0x{result.error:02x} "
                f"({describe_error(result.error)}); the chain is not in a state anyone "
                "has confirmed is safe"
            )
        return result

    def scan(self, ids: Sequence[int] | None = None) -> dict[int, PingResult | None]:
        """Ping each id, reporting silence rather than raising on it.

        The one read in this file that tolerates a missing answer, because
        finding out who is on the bus is the question rather than an obstacle
        to it.
        """
        self._require_open("scan")
        found: dict[int, PingResult | None] = {}
        for motor_id in ids if ids is not None else self._calibration.ids:
            try:
                found[motor_id] = self._ping(motor_id)
            except (BusTimeout, BusCorruption):
                found[motor_id] = None
        return found

    def read_present_position(self, ids: Sequence[int] | None = None) -> Reading:
        """Positions in raw ticks, and the only thing that refreshes the clamp reference.

        On timeout the cached reference is *destroyed*, not returned. That one
        line is the difference between a dropped frame and an arm that keeps
        being told to move a little further from a place it is no longer at.
        Consecutive silence from the same servo escalates to a halting
        :class:`ServoFault`, because a servo that has stopped answering three
        times running has not dropped a frame, it has left.
        """
        self._require_open("read_present_position")
        wanted = tuple(ids) if ids is not None else self._calibration.ids
        try:
            ticks = self._read_positions(wanted)
        except BusFailure as error:
            self._reference = None
            # Charge the strike to the servo that earned it when the bus could
            # say which; a chain-wide failure is charged to everyone, because
            # it genuinely was everyone.
            blamed = (error.motor_id,) if error.motor_id in wanted else wanted
            for motor_id in blamed:
                self._timeouts[motor_id] = self._timeouts.get(motor_id, 0) + 1
            stuck = [
                motor_id
                for motor_id in blamed
                if self._timeouts[motor_id] >= self._limits.max_consecutive_timeouts
            ]
            if stuck:
                raise ServoFault(
                    f"servo(s) {stuck} did not answer "
                    f"{self._limits.max_consecutive_timeouts} reads in a row: {error}. "
                    "Halting rather than retrying — the position of this arm is unknown, "
                    "and every clamp downstream is computed from it"
                ) from error
            raise
        missing = [motor_id for motor_id in wanted if motor_id not in ticks]
        if missing:
            self._reference = None
            raise BusTimeout(
                f"the chain answered without servo(s) {missing}; a position vector with a "
                "hole in it is not a partial reading, it is an unusable one"
            )
        for motor_id in wanted:
            self._timeouts[motor_id] = 0
        reading = Reading(dict(ticks), self._clock())
        if tuple(sorted(ticks)) == tuple(sorted(self._calibration.ids)):
            self._reference = reading
        return reading

    def read_present_load(self, ids: Sequence[int] | None = None) -> dict[int, Measurement]:
        """``Present_Load``, reported as what it is rather than as a force.

        Register 60 is a signed PWM duty figure: how hard the servo's controller
        is pushing, not how hard the world is pushing back. There is no load
        cell on an SO-101 and no torque estimate that has been validated on one,
        so this comes back as a dimensionless fraction whose ``method`` says so
        and whose raw count is attached. Anything downstream that wants newtons
        has to add a sensor, not a coefficient.
        """
        self._require_open("read_present_load")
        register = CONTROL_TABLE["Present_Load"]
        out: dict[int, Measurement] = {}
        for motor_id in ids if ids is not None else self._calibration.ids:
            raw = self._read_register(motor_id, register)
            out[motor_id] = Measurement(
                raw / LOAD_FULL_SCALE,
                n=1,
                units="fraction",
                method=(
                    "STS3215 Present_Load (register 60), sign-magnitude on bit 10, scaled "
                    f"by the documented {LOAD_FULL_SCALE:g}-count full-scale PWM duty. This is "
                    "commanded effort, not measured force: the arm has no force or torque "
                    "sensing of any kind"
                ),
                detail={"raw": raw, "register": 60, "joint": self._calibration.by_id(motor_id).name},
            )
        return out

    def read_present_current(self, ids: Sequence[int] | None = None) -> dict[int, Measurement]:
        """``Present_Current`` as a raw count, deliberately not converted to amps.

        Register 69 exists in the STS/SMS control table. Whether an STS3215
        populates it, and at what scale, is not something I could confirm from a
        primary source — 6.5 mA/LSB is quoted widely and attributed nowhere. A
        current reported in milliamps at an unverified scale is exactly the
        number somebody turns into a grasp threshold, so this returns counts and
        says why.
        """
        self._require_open("read_present_current")
        register = CONTROL_TABLE["Present_Current"]
        out: dict[int, Measurement] = {}
        for motor_id in ids if ids is not None else self._calibration.ids:
            raw = self._read_register(motor_id, register)
            out[motor_id] = Measurement(
                float(raw),
                n=1,
                units="count",
                method=(
                    "STS3215 Present_Current (register 69), raw counts. Not converted to "
                    "amps: the per-count scale is unverified for this part, and an "
                    "unverified scale becomes a threshold somebody trusts"
                ),
                detail={"raw": raw, "register": 69, "scale": "unverified"},
            )
        return out

    def read_temperature(self, ids: Sequence[int] | None = None) -> dict[int, int]:
        """Degrees Celsius per servo, straight from register 63."""
        self._require_open("read_temperature")
        register = CONTROL_TABLE["Present_Temperature"]
        return {
            motor_id: self._read_register(motor_id, register)
            for motor_id in (ids if ids is not None else self._calibration.ids)
        }

    # -- writing -----------------------------------------------------------

    def write_goal_position(self, ticks: Mapping[int, int]) -> WriteResult:
        """Command goals in raw ticks, clamped, against a reference that is fresh.

        Three rules, in order, none of them optional:

        1. The reference must be recent. Older than
           ``limits.max_reference_age`` and this refuses, because a delta
           against an old position is a delta against nothing.
        2. Each joint moves at most its per-step budget from where it *is*.
        3. Each goal stays inside that joint's measured travel.

        Rule 1 refuses. Rules 2 and 3 alter the command and report every
        alteration in the result, which is the only altering this file does.
        """
        self._require_open("write_goal_position")
        if not ticks:
            raise ConfigError("write_goal_position was given no joints to command")
        unknown = [motor_id for motor_id in ticks if motor_id not in set(self._calibration.ids)]
        if unknown:
            raise ConfigError(
                f"asked to command motor id(s) {unknown}, which this calibration does not "
                f"describe; the chain is {self._calibration.ids}"
            )

        reference = self._fresh_reference()
        age = self._clock() - reference.at
        writes = []
        commanded: dict[int, int] = {}
        for motor_id, requested in ticks.items():
            joint = self._calibration.by_id(motor_id)
            here = reference.ticks[motor_id]
            budget = self._budget[motor_id]
            step = max(-budget, min(budget, int(requested) - here))
            value = joint.clamp_ticks(here + step)
            commanded[motor_id] = value
            writes.append(JointWrite(motor_id, joint.name, here, int(requested), value))

        result = WriteResult(tuple(writes), age)
        self._update_stalls(reference, commanded)
        self._write_goals(commanded)
        self._writes += 1
        if result.any_clamped:
            self._clamp_events += 1
            if self.on_clamp is not None:
                self.on_clamp(result)
        return result

    def write_goal_normalized(self, values: Mapping[str, float]) -> WriteResult:
        """Command by joint name in each joint's declared convention.

        The conversion refuses off-scale values rather than clamping them; see
        :meth:`JointCalibration.to_ticks` for why an out-of-range command is
        evidence worth keeping.
        """
        return self.write_goal_position(
            {
                self._calibration.by_name(name).id: self._calibration.by_name(name).to_ticks(value)
                for name, value in values.items()
            }
        )

    def write_goal_vector(self, values: Sequence[float]) -> WriteResult:
        """Command from an action vector in declared joint order.

        Width is checked and never padded or truncated. A six-wide action
        arriving at a five-joint chain is a mismatch somebody needs to hear
        about, and quietly dropping the gripper column is how a corpus ends up
        with a constant sixth dimension nobody notices.
        """
        expected = len(self._calibration.joints)
        if len(values) != expected:
            raise ConfigError(
                f"this chain has {expected} joint(s) ({', '.join(self._calibration.names)}) "
                f"and the action vector is {len(values)} wide; refusing to pad or truncate"
            )
        return self.write_goal_normalized(
            {joint.name: float(value) for joint, value in zip(self._calibration.joints, values)}
        )

    def _fresh_reference(self) -> Reading:
        reference = self._reference
        if reference is None:
            raise SafetyAbort(
                "no position reference: the last read failed, or none has been taken. "
                "Refusing to write a goal, because the relative clamp is computed from "
                "where the arm is and there is currently no answer to that. Read "
                "present position again before commanding."
            )
        age = self._clock() - reference.at
        if age > self._limits.max_reference_age:
            self._reference = None
            raise SafetyAbort(
                f"the position reference is {age * 1000:.1f} ms old and the limit is "
                f"{self._limits.max_reference_age * 1000:.1f} ms. A clamp against a stale "
                "position lets the arm walk away one 'small' step at a time; re-read "
                "before commanding"
            )
        return reference

    def _update_stalls(self, reference: Reading, commanded: Mapping[int, int]) -> None:
        """Commanded to move, did not move, repeatedly. The only contact proxy here."""
        epsilon = self._limits.stall_epsilon_ticks
        for motor_id, goal in commanded.items():
            state = self._stalls[motor_id]
            asked = abs(goal - reference.ticks[motor_id])
            previous = state.tracking_error
            state.tracking_error = goal - reference.ticks[motor_id]
            moving = abs(state.tracking_error - previous) > epsilon
            if asked > epsilon and not moving:
                state.steps += 1
            else:
                state.steps = 0
            state.stalled = state.steps >= self._limits.stall_steps

    def stalls(self) -> dict[int, StallState]:
        """A snapshot of the stall monitor, per motor id."""
        return {motor_id: replace(state) for motor_id, state in self._stalls.items()}

    def stall_measurements(self) -> dict[str, Measurement]:
        """Tracking error per joint name, as a measurement rather than a float."""
        return {
            state.name: state.measurement(self._calibration.by_id(motor_id).span)
            for motor_id, state in self._stalls.items()
        }

    # -- torque ------------------------------------------------------------

    def torque_enable(self, ids: Sequence[int] | None = None) -> None:
        """Energise, after pointing each servo at where it already is.

        The goal register holds whatever was last written to it, which after a
        power cycle or a teleop session is somewhere else entirely. Enabling
        torque without writing the present position first makes the arm snap to
        that old goal at full speed the instant it is energised. So the goal is
        set from a fresh read first, every time, and if that read fails nothing
        is energised.
        """
        self._require_open("torque_enable")
        wanted = tuple(ids) if ids is not None else self._calibration.ids
        here = self._read_positions(wanted)
        missing = [motor_id for motor_id in wanted if motor_id not in here]
        if missing:
            raise SafetyAbort(
                f"refusing to enable torque: servo(s) {missing} did not report a position, "
                "and energising a servo whose goal register still holds an old target "
                "makes the arm jump to it"
            )
        self._write_goals({motor_id: here[motor_id] for motor_id in wanted})
        register = CONTROL_TABLE["Torque_Enable"]
        for motor_id in wanted:
            self._write_register(motor_id, register, TORQUE_ON)
            self._torque[motor_id] = True
        self._reference = Reading(dict(here), self._clock())

    def torque_disable(self, ids: Sequence[int] | None = None) -> None:
        """De-energise. The arm becomes backdrivable and will fall under gravity."""
        self._require_open("torque_disable")
        register = CONTROL_TABLE["Torque_Enable"]
        for motor_id in ids if ids is not None else self._calibration.ids:
            self._write_register(motor_id, register, TORQUE_OFF)
            self._torque[motor_id] = False

    def torque_state(self) -> dict[int, bool]:
        """What this transport last commanded, which is not the same as asking."""
        return dict(self._torque)

    # -- what the run should record ----------------------------------------

    def health(self) -> dict[str, Any]:
        """A block for the episode metadata: clamp activity, stalls, provenance."""
        return {
            "transport": type(self).__name__,
            "version": VERSION,
            "calibration_digest": self._calibration.digest,
            "calibration_measured": self._calibration.measured,
            "calibration_source": self._calibration.source,
            "max_relative_target": self._limits.describe(),
            "budget_ticks": dict(self._budget),
            "writes": self._writes,
            "clamped_writes": self._clamp_events,
            "clamp_rate": Measurement(
                self._clamp_events / self._writes if self._writes else 0.0,
                n=self._writes,
                units="fraction",
                method="writes in which the relative clamp altered at least one joint",
            ),
            "stalled_joints": [state.name for state in self._stalls.values() if state.stalled],
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "open" if self._open else "closed"
        return (
            f"<{type(self).__name__} {state} ids={self._calibration.ids} "
            f"clamp={self._limits.describe()}>"
        )


# ==========================================================================
# the packet codec
# ==========================================================================


@dataclass(frozen=True)
class Status:
    """A decoded status packet."""

    id: int
    error: int
    params: tuple[int, ...]


def checksum(payload: Sequence[int]) -> int:
    """Bitwise complement of the sum from the id byte onward."""
    return (~sum(payload)) & 0xFF


def instruction_packet(motor_id: int, instruction: int, params: Sequence[int] = ()) -> bytes:
    if not 0 <= motor_id <= BROADCAST_ID:
        raise ConfigError(f"motor id {motor_id} is outside 0..{BROADCAST_ID}")
    for value in params:
        if not 0 <= value <= 0xFF:
            raise ConfigError(f"packet parameter {value} does not fit in a byte")
    body = [motor_id, len(params) + 2, instruction, *params]
    return bytes([*HEADER, *body, checksum(body)])


def decode_status(frame: bytes) -> Status:
    """Decode one status frame, checksum included. Raises rather than guessing."""
    if len(frame) < 6 or frame[:2] != HEADER:
        raise BusCorruption(f"not a status frame: {frame.hex(' ') or '<empty>'}")
    motor_id, length = frame[2], frame[3]
    if len(frame) != length + 4:
        raise BusCorruption(
            f"status frame from id {motor_id} declares length {length} "
            f"but carries {len(frame) - 4}"
        )
    body = frame[2:-1]
    if checksum(body) != frame[-1]:
        raise BusCorruption(
            f"checksum mismatch from id {motor_id}: computed 0x{checksum(body):02x}, "
            f"frame says 0x{frame[-1]:02x}"
        )
    return Status(motor_id, frame[4], tuple(frame[5:-1]))


def to_bytes(value: int, length: int) -> list[int]:
    """Little-endian, which is the STS/SMS convention. The older SCS series is
    big-endian, and this module does not claim to speak it."""
    if length == 1:
        return [value & 0xFF]
    if length == 2:
        return [value & 0xFF, (value >> 8) & 0xFF]
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


def from_bytes(data: Sequence[int]) -> int:
    return sum(byte << (8 * index) for index, byte in enumerate(data))


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    """Feetech stores signed registers as magnitude plus a sign bit, not two's
    complement. Read as two's complement, a load of -3 becomes 1027."""
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    limit = (1 << sign_bit) - 1
    magnitude = abs(value)
    if magnitude > limit:
        raise ConfigError(f"{value} does not fit in {sign_bit} magnitude bits")
    return magnitude | ((1 << sign_bit) if value < 0 else 0)


class SerialLike(Protocol):
    """The smallest slice of pyserial this module needs.

    A Protocol rather than an import, so the codec above is exercised against
    :class:`LoopbackSerial` in CI with no serial library installed and no port
    on the machine.
    """

    def read(self, size: int) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def reset_input_buffer(self) -> None: ...

    def close(self) -> None: ...


def open_serial(port: str, baudrate: int = 1_000_000, timeout: float = 0.02) -> SerialLike:
    """Open a real port, and say how to get the dependency if it is missing.

    1 Mbps is the SO-101 default. The 20 ms read timeout is two thirds of a
    control period at 30 Hz — long enough for a servo's return delay, short
    enough that a dead servo is discovered inside the step it went silent in
    rather than after it.
    """
    try:
        import serial
    except ImportError as error:  # pragma: no cover - exercised by the extra-less env
        raise ConfigError(
            "talking to a real SO-101 needs pyserial: pip install "
            "'gantry-evaluator-so101[serial]'. Everything else in this plugin, "
            "including the whole safety layer, runs on MockTransport without it"
        ) from error
    try:
        return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    except Exception as error:  # noqa: BLE001 - pyserial raises its own family
        raise ConfigError(
            f"could not open {port!r} at {baudrate} baud: {type(error).__name__}: {error}. "
            "find_port() will tell you which port the arm is on"
        ) from error


# ==========================================================================
# the real one
# ==========================================================================


class SerialTransport(Transport):
    """The Feetech STS/SMS protocol over a serial port.

    Sync-write is used for goals — one frame for the whole chain, which is what
    keeps six servos moving as one gesture rather than as six staggered ones at
    30 Hz. Sync-read is optional and off by default: it is an STS/SMS-only
    instruction, its reply is one status packet per servo, and a chain that
    answers it partially is harder to reason about than six sequential reads
    that each either worked or did not. Turn it on when the loop is tight and
    the chain is known good.
    """

    def __init__(
        self,
        calibration: Calibration,
        *,
        max_relative_target: SafetyLimits,
        port: str | None = None,
        baudrate: int = 1_000_000,
        timeout: float = 0.02,
        serial_factory: Callable[[], SerialLike] | None = None,
        sync_read: bool = False,
        allow_unmeasured_calibration: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(
            calibration,
            max_relative_target=max_relative_target,
            allow_unmeasured_calibration=allow_unmeasured_calibration,
            clock=clock,
        )
        if serial_factory is None and port is None:
            raise ConfigError(
                "SerialTransport needs either a port name or a serial_factory; it will "
                "not go looking for a port on its own, because the wrong port is another "
                "device and this one writes to it"
            )
        self._factory = serial_factory or (lambda: open_serial(port, baudrate, timeout))
        self._port_name = port
        self._timeout = timeout
        self._sync_read = sync_read
        self._serial: SerialLike | None = None

    # -- port --------------------------------------------------------------

    def _open_bus(self) -> None:
        self._serial = self._factory()
        for method in ("read", "write"):
            if not callable(getattr(self._serial, method, None)):
                raise ConfigError(
                    f"the serial object has no callable {method!r}; this transport needs "
                    "read() and write()"
                )

    def _close_bus(self) -> None:
        if self._serial is not None and callable(getattr(self._serial, "close", None)):
            self._serial.close()
        self._serial = None

    @property
    def _bus(self) -> SerialLike:
        if self._serial is None:
            raise ConfigError("the serial port is not open")
        return self._serial

    # -- framing -----------------------------------------------------------

    def _send(self, packet: bytes) -> None:
        if callable(getattr(self._bus, "reset_input_buffer", None)):
            # Anything still in the buffer belongs to a transaction that already
            # failed. Reading it as this one's answer is how a timeout turns
            # into a wrong value, which is worse than a timeout.
            self._bus.reset_input_buffer()
        self._bus.write(packet)

    def _receive(self, expected_params: int, *, fault_on_error: bool = True) -> Status:
        """Read exactly one status frame, or say what arrived instead.

        ``fault_on_error=False`` is for ping alone, where "are you faulted?" is
        the question being asked rather than an obstacle to asking it.
        """
        head = self._read_exact(4)
        if head[:2] != HEADER:
            raise BusCorruption(f"expected 0xFFFF, got {head.hex(' ')}")
        length = head[3]
        if length < 2:
            raise BusCorruption(f"status frame declares length {length}")
        rest = self._read_exact(length)
        status = decode_status(head + rest)
        if len(status.params) != expected_params:
            raise BusCorruption(
                f"servo {status.id} returned {len(status.params)} parameter byte(s), "
                f"expected {expected_params}"
            )
        if status.error and fault_on_error:
            raise ServoFault(
                f"servo {status.id} reports error 0x{status.error:02x} "
                f"({describe_error(status.error)})"
            )
        return status

    def _read_exact(self, count: int) -> bytes:
        """Read ``count`` bytes or raise. Never returns a short buffer.

        A short read is the shape a timeout arrives in, and the only correct
        response is to stop. Returning what came back would hand a truncated
        position to the clamp.
        """
        buffer = bytearray()
        while len(buffer) < count:
            chunk = self._bus.read(count - len(buffer))
            if not chunk:
                raise BusTimeout(
                    f"bus timeout after {self._timeout * 1000:.0f} ms: wanted {count} byte(s), "
                    f"got {len(buffer)}"
                )
            buffer.extend(chunk)
        return bytes(buffer)

    # -- primitives --------------------------------------------------------

    def _ping(self, motor_id: int) -> PingResult:
        self._send(instruction_packet(motor_id, INST_PING))
        # A ping is how you ask whether a servo is faulted, so its error byte is
        # the answer and is carried in the result. Transport.ping is what turns
        # a non-zero one into a ServoFault; every other read treats it as fatal
        # on the spot.
        status = self._receive(0, fault_on_error=False)
        if status.error:
            return PingResult(status.id, 0, status.error)
        model = self._read_register(motor_id, CONTROL_TABLE["Model_Number"])
        return PingResult(status.id, model, status.error)

    def _read_register(self, motor_id: int, register: Register) -> int:
        self._send(
            instruction_packet(motor_id, INST_READ, [register.address, register.length])
        )
        try:
            status = self._receive(register.length)
        except BusFailure as error:
            # The framing layer has no idea who it was talking to. This is the
            # innermost frame that does, so it is where the blame is attached.
            error.motor_id = motor_id
            raise
        raw = from_bytes(status.params)
        if register.sign_bit is not None:
            return decode_sign_magnitude(raw, register.sign_bit)
        return raw

    def _read_positions(self, ids: Sequence[int]) -> dict[int, int]:
        register = CONTROL_TABLE["Present_Position"]
        if not self._sync_read:
            return {motor_id: self._read_register(motor_id, register) for motor_id in ids}
        self._send(
            instruction_packet(
                BROADCAST_ID, INST_SYNC_READ, [register.address, register.length, *ids]
            )
        )
        found: dict[int, int] = {}
        for _ in ids:
            status = self._receive(register.length)
            found[status.id] = from_bytes(status.params)
        return found

    def _write_goals(self, ticks: Mapping[int, int]) -> None:
        register = CONTROL_TABLE["Goal_Position"]
        if len(ticks) == 1:
            (motor_id, value), = ticks.items()
            self._write_register(motor_id, register, value)
            return
        params: list[int] = [register.address, register.length]
        for motor_id, value in ticks.items():
            params.extend([motor_id, *to_bytes(value, register.length)])
        # Sync-write is unacknowledged by design: the reply would be six status
        # frames for one gesture. What confirms the write is the next position
        # read, which the clamp needs anyway.
        self._send(instruction_packet(BROADCAST_ID, INST_SYNC_WRITE, params))

    def _write_register(self, motor_id: int, register: Register, value: int) -> None:
        if not register.writable:
            raise UnsupportedRegister(f"{register.name} is read-only")
        encoded = (
            encode_sign_magnitude(value, register.sign_bit)
            if register.sign_bit is not None
            else value
        )
        self._send(
            instruction_packet(
                motor_id, INST_WRITE, [register.address, *to_bytes(encoded, register.length)]
            )
        )
        self._receive(0)

    # -- EEPROM ------------------------------------------------------------

    def write_eeprom(self, motor_id: int, register_name: str, value: int) -> None:
        """Write a persistent register, opening and closing the lock around it.

        Separate from ``_write_register`` because EEPROM writes wear the part
        and survive a power cycle: a configuration mistake made here is still
        there tomorrow, which a RAM write is not.
        """
        register = CONTROL_TABLE[register_name]
        if not register.eeprom:
            raise UnsupportedRegister(f"{register_name} is not an EEPROM register")
        self._require_open("write_eeprom")
        lock = CONTROL_TABLE["Lock"]
        self._write_register(motor_id, lock, EEPROM_UNLOCKED)
        try:
            self._write_register(motor_id, register, value)
        except BaseException:
            # The write did not land, so the servo is still at its old address
            # and its EEPROM is still open. Close it there.
            self._write_register(motor_id, lock, EEPROM_LOCKED)
            raise
        # Writing ID moves the servo. It stops answering to the id we have been
        # addressing this whole transaction and starts answering to the new one,
        # so the re-lock has to be sent to the new address or the servo is left
        # with its EEPROM open — which is how the next stray write becomes
        # permanent. Spelled out rather than folded into the finally clause,
        # because "the address changed mid-transaction" is not a detail anyone
        # should have to infer from a ternary.
        self._write_register(value if register.name == "ID" else motor_id, lock, EEPROM_LOCKED)


# ==========================================================================
# the simulated chain
# ==========================================================================


@dataclass
class SimulatedServo:
    """One servo's state, plus whatever has been done to it on purpose."""

    id: int
    position: int
    goal: int
    torque: bool = False
    load: int = 0
    temperature: int = 31
    error: int = 0
    #: Answers nothing. A servo whose cable came out or whose board died.
    dead: bool = False
    #: Answers, but with a value frozen at the moment it was frozen. The
    #: nastiest of the three, because everything looks healthy.
    stale_at: int | None = None
    #: Physically stuck: honest reads, no movement. A jam, or a grasp.
    jammed: bool = False
    #: A constant added to every reported position. A slipped servo horn or a
    #: calibration taken against a different zero: the encoder answers, the
    #: number is plausible, and it is wrong by a fixed amount forever. This is
    #: LeRobot #3758 in one integer, and it exists here so the drift detector
    #: above can be made to fire without a bench.
    offset: int = 0
    #: A tick the servo physically cannot travel *through*. The jaws meeting an
    #: object, rather than a jam: the motor keeps pushing and the encoder stops
    #: changing. It is a barrier, not an attractor — a servo already on one side
    #: of it stays on that side, and one placed behind where the servo already
    #: is does nothing at all until the servo comes back to it.
    obstruction: int | None = None
    #: Which side of the obstruction the servo was on when it was installed.
    #: Recorded rather than derived per step, because deriving it would let the
    #: barrier silently change sides as the servo moves through it.
    obstruction_from_above: bool = True

    def reported_position(self) -> int:
        raw = self.position if self.stale_at is None else self.stale_at
        return raw + self.offset


class ServoChain:
    """A six-servo bus with the failures that actually happen, on demand.

    The happy path here matters least. What this exists for is the fault
    injection below: a transport whose timeout handling has never been made to
    fire has timeout handling nobody has read.

    Movement is first-order toward the goal at ``speed_ticks_per_s``, with a
    fixed ``position_error_ticks`` steady-state offset so that a test asserting
    "the arm arrived exactly" fails, as it should — real servos have a dead
    zone and stop short.
    """

    def __init__(
        self,
        ids: Sequence[int],
        *,
        start: int | Mapping[int, int] = 2048,
        speed_ticks_per_s: float = 2000.0,
        position_error_ticks: int = 0,
        latency: float = 0.0005,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        # ``start`` may be one tick for the whole chain or a tick per servo.
        # Per-servo matters more than it looks: 2048 is the middle of the
        # *encoder*, and a joint whose calibrated travel is not centred on the
        # encoder — the gripper's is not — parks nowhere near the middle of its
        # own range. A caller that means "mid-travel" has to say so per joint.
        parked = start if isinstance(start, Mapping) else {motor_id: start for motor_id in ids}
        missing = [motor_id for motor_id in ids if motor_id not in parked]
        if missing:
            raise ConfigError(f"no start position given for servo(s) {missing}")
        self.servos = {
            motor_id: SimulatedServo(motor_id, int(parked[motor_id]), int(parked[motor_id]))
            for motor_id in ids
        }
        self.speed = speed_ticks_per_s
        self.position_error = position_error_ticks
        self.latency = latency
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        #: Transactions to drop wholesale, whoever they were addressed to.
        self._drop = 0
        #: Replies to corrupt, so the checksum path gets exercised.
        self._corrupt = 0
        self.transactions = 0

    # -- fault injection ---------------------------------------------------

    def kill(self, motor_id: int) -> None:
        """This servo stops answering. Reads to it time out from now on."""
        self.servos[motor_id].dead = True

    def revive(self, motor_id: int) -> None:
        self.servos[motor_id].dead = False

    def freeze(self, motor_id: int) -> None:
        """This servo keeps answering with the value it holds right now.

        The failure mode with no symptoms. Position control keeps working, the
        chain keeps replying, and the number feeding the clamp stopped being
        true some time ago.
        """
        self.servos[motor_id].stale_at = self.servos[motor_id].position

    def unfreeze(self, motor_id: int) -> None:
        self.servos[motor_id].stale_at = None

    def jam(self, motor_id: int) -> None:
        """Physically stuck. Honest reads, no movement — a stall, or a grasp."""
        self.servos[motor_id].jammed = True

    def unjam(self, motor_id: int) -> None:
        self.servos[motor_id].jammed = False

    def drop(self, transactions: int = 1) -> None:
        """The next N transactions get no answer. A bus timeout, not a dead servo."""
        self._drop = transactions

    def corrupt(self, replies: int = 1) -> None:
        """The next N replies come back with a broken checksum."""
        self._corrupt = replies

    def set_error(self, motor_id: int, bits: int) -> None:
        """The servo answers, and says it is in trouble."""
        self.servos[motor_id].error = bits

    def offset(self, motor_id: int, ticks: int) -> None:
        """Every reported position is off by ``ticks`` from now on.

        Not a fault the bus can see. The chain answers on time, the checksum is
        good, and the number has a constant bias — which is the whole reason the
        drift check exists rather than a health check.
        """
        self.servos[motor_id].offset = int(ticks)

    def obstruct(self, motor_id: int, at_ticks: int) -> None:
        """This servo cannot travel through ``at_ticks``. The jaws met something.

        The side is fixed at the moment the object is placed, from where the
        servo is standing then — which is what putting an object between two
        jaws does. Deriving the side per step from the goal instead would make
        the barrier flip whenever the command crossed it, and a servo parked on
        the far side of its own obstruction would be teleported onto it.
        """
        servo = self.servos[motor_id]
        servo.obstruction = int(at_ticks)
        servo.obstruction_from_above = servo.position >= int(at_ticks)

    def unobstruct(self, motor_id: int) -> None:
        self.servos[motor_id].obstruction = None

    # -- physics -----------------------------------------------------------

    def advance(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        travel = self.speed * elapsed
        for servo in self.servos.values():
            if not servo.torque or servo.jammed:
                servo.load = 800 if servo.jammed and servo.torque else 0
                continue
            target = servo.goal - self.position_error
            delta = target - servo.position
            if abs(delta) <= travel:
                servo.position = target
                servo.load = 0
            else:
                servo.position += int(travel) if delta > 0 else -int(travel)
                servo.load = 300 if delta > 0 else -300
            servo.position = max(0, min(RESOLUTION - 1, servo.position))
            if servo.obstruction is not None:
                # A barrier the servo cannot cross. It keeps being commanded
                # past it, so position stops changing while load stays high:
                # that conjunction is the whole of what a grasp looks like to
                # an arm whose only sensor is an encoder.
                stopped = (
                    max(servo.position, servo.obstruction)
                    if servo.obstruction_from_above
                    else min(servo.position, servo.obstruction)
                )
                if stopped != servo.position:
                    servo.position = stopped
                    servo.load = 800

    def _transact(self, motor_id: int | None) -> SimulatedServo | None:
        """One bus round trip, applying latency and any injected failure."""
        self.transactions += 1
        if self.latency:
            self._sleep(self.latency)
        self.advance()
        if self._drop > 0:
            self._drop -= 1
            raise BusTimeout("injected bus timeout")
        if motor_id is None:
            return None
        servo = self.servos.get(motor_id)
        if servo is None or servo.dead:
            raise BusTimeout(f"no answer from id {motor_id}", motor_id=motor_id)
        return servo

    # -- the servo-level operations MockTransport speaks --------------------

    def ping(self, motor_id: int) -> PingResult:
        servo = self._transact(motor_id)
        assert servo is not None
        return PingResult(servo.id, STS3215_MODEL_NUMBER, servo.error)

    def read(self, motor_id: int, register: Register) -> int:
        servo = self._transact(motor_id)
        assert servo is not None
        if servo.error:
            raise ServoFault(
                f"servo {motor_id} reports error 0x{servo.error:02x} "
                f"({describe_error(servo.error)})"
            )
        return {
            "Present_Position": servo.reported_position(),
            "Goal_Position": servo.goal,
            "Present_Load": servo.load,
            "Present_Current": abs(servo.load) * 2,
            "Present_Temperature": servo.temperature,
            "Present_Voltage": 120,
            "Torque_Enable": int(servo.torque),
            "Model_Number": STS3215_MODEL_NUMBER,
            "Moving": int(servo.position != servo.goal),
            "Status": servo.error,
        }.get(register.name, 0)

    def write(self, motor_id: int, register: Register, value: int) -> None:
        servo = self._transact(motor_id)
        assert servo is not None
        if register.name == "Goal_Position":
            servo.goal = value
        elif register.name == "Torque_Enable":
            servo.torque = bool(value)
        elif register.name == "ID":
            self.servos.pop(motor_id)
            servo.id = value
            self.servos[value] = servo

    def sync_write_goals(self, ticks: Mapping[int, int]) -> None:
        self._transact(None)
        for motor_id, value in ticks.items():
            servo = self.servos.get(motor_id)
            if servo is None or servo.dead:
                # A sync-write is unacknowledged, so a dead servo simply does
                # not move. The next read is what finds out, which is exactly
                # what happens on real hardware.
                continue
            servo.goal = value

    def take_corrupt(self) -> bool:
        if self._corrupt > 0:
            self._corrupt -= 1
            return True
        return False


class MockTransport(Transport):
    """A :class:`ServoChain` behind the transport interface.

    What everything above this file is tested against. It skips the packet
    codec on purpose — :class:`LoopbackSerial` covers that — so that a test of
    the clamp is a test of the clamp and not of byte order.
    """

    def __init__(
        self,
        calibration: Calibration,
        *,
        max_relative_target: SafetyLimits,
        chain: ServoChain | None = None,
        latency: float = 0.0005,
        position_error_ticks: int = 0,
        speed_ticks_per_s: float = 2000.0,
        start: int | Mapping[int, int] = 2048,
        allow_unmeasured_calibration: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(
            calibration,
            max_relative_target=max_relative_target,
            allow_unmeasured_calibration=allow_unmeasured_calibration,
            clock=clock,
        )
        self.chain = chain or ServoChain(
            calibration.ids,
            start=start,
            speed_ticks_per_s=speed_ticks_per_s,
            position_error_ticks=position_error_ticks,
            latency=latency,
            clock=clock,
            sleep=sleep,
        )

    def _open_bus(self) -> None:
        return None

    def _close_bus(self) -> None:
        return None

    def _ping(self, motor_id: int) -> PingResult:
        return self.chain.ping(motor_id)

    def _read_register(self, motor_id: int, register: Register) -> int:
        if self.chain.take_corrupt():
            raise BusCorruption(
                f"injected checksum failure reading {register.name}", motor_id=motor_id
            )
        return self.chain.read(motor_id, register)

    def _read_positions(self, ids: Sequence[int]) -> dict[int, int]:
        register = CONTROL_TABLE["Present_Position"]
        return {motor_id: self._read_register(motor_id, register) for motor_id in ids}

    def _write_goals(self, ticks: Mapping[int, int]) -> None:
        self.chain.sync_write_goals(ticks)

    def _write_register(self, motor_id: int, register: Register, value: int) -> None:
        self.chain.write(motor_id, register, value)

    # -- fault injection, surfaced at the transport ------------------------
    #
    # The layers above hold a Transport, not a ServoChain, so a fault they need
    # to provoke has to be reachable from here. Named on the mock only: there is
    # no way to reach any of this from SerialTransport, which is the point.

    def inject_offset(self, motor_id: int, ticks: int) -> None:
        """A constant bias on one servo's reported position. See #3758."""
        self.chain.offset(motor_id, ticks)

    def inject_stall(self, motor_id: int, at_ticks: int) -> None:
        """One servo stops travelling at ``at_ticks``. A grasp, or an obstacle."""
        self.chain.obstruct(motor_id, at_ticks)


class LoopbackSerial:
    """A :class:`ServoChain` pretending to be a serial port.

    Exists so that :class:`SerialTransport` — the packet framing, the
    little-endian split, the checksum, the short-read handling — is exercised
    in CI on a machine with no arm and no pyserial. Without it the mock would
    prove every layer above the wire and leave the wire itself as the one part
    that only gets tested by plugging in a robot.
    """

    def __init__(self, chain: ServoChain):
        self.chain = chain
        self._out = bytearray()
        self.is_open = True

    # -- SerialLike --------------------------------------------------------

    def write(self, data: bytes) -> int:
        try:
            self._out.extend(self._respond(bytes(data)))
        except BusTimeout:
            pass  # Silence on the wire is what a timeout looks like from here.
        return len(data)

    def read(self, size: int) -> bytes:
        chunk = bytes(self._out[:size])
        del self._out[: len(chunk)]
        return chunk

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def close(self) -> None:
        self.is_open = False

    # -- the servo side ----------------------------------------------------

    def _respond(self, packet: bytes) -> bytes:
        if packet[:2] != HEADER:
            return b""
        motor_id, _, instruction = packet[2], packet[3], packet[4]
        params = packet[5:-1]
        if instruction == INST_PING:
            result = self.chain.ping(motor_id)
            return self._status(motor_id, result.error, ())
        if instruction == INST_READ:
            address, length = params[0], params[1]
            register = _register_at(address, length)
            value = self.chain.read(motor_id, register)
            if register.sign_bit is not None:
                value = encode_sign_magnitude(value, register.sign_bit)
            return self._status(motor_id, 0, to_bytes(value, length))
        if instruction == INST_WRITE:
            address = params[0]
            data = params[1:]
            register = _register_at(address, len(data))
            value = from_bytes(data)
            if register.sign_bit is not None:
                value = decode_sign_magnitude(value, register.sign_bit)
            self.chain.write(motor_id, register, value)
            return self._status(motor_id, 0, ())
        if instruction == INST_SYNC_WRITE:
            address, width = params[0], params[1]
            register = _register_at(address, width)
            goals = {}
            body = params[2:]
            stride = width + 1
            for offset in range(0, len(body), stride):
                goals[body[offset]] = from_bytes(body[offset + 1 : offset + stride])
            self.chain.sync_write_goals(goals)
            return b""  # Unacknowledged, as on real hardware.
        if instruction == INST_SYNC_READ:
            address, width = params[0], params[1]
            register = _register_at(address, width)
            out = bytearray()
            for target in params[2:]:
                try:
                    value = self.chain.read(target, register)
                except BusTimeout:
                    continue  # A silent servo simply does not appear in the replies.
                out.extend(self._status(target, 0, to_bytes(value, width)))
            return bytes(out)
        return b""

    def _status(self, motor_id: int, error: int, params: Sequence[int]) -> bytes:
        body = [motor_id, len(params) + 2, error, *params]
        check = checksum(body)
        if self.chain.take_corrupt():
            check ^= 0xFF
        return bytes([*HEADER, *body, check])


def _register_at(address: int, length: int) -> Register:
    for register in CONTROL_TABLE.values():
        if register.address == address and register.length == length:
            return register
    raise BusCorruption(f"no known register at address {address} of length {length}")


# ==========================================================================
# finding the arm
# ==========================================================================


@dataclass(frozen=True)
class PortCandidate:
    device: str
    description: str = ""


def list_serial_ports() -> tuple[PortCandidate, ...]:
    """Every serial device the machine currently sees."""
    try:
        from serial.tools import list_ports
    except ImportError as error:  # pragma: no cover
        raise ConfigError(
            "listing serial ports needs pyserial: pip install "
            "'gantry-evaluator-so101[serial]'"
        ) from error
    return tuple(
        PortCandidate(port.device, port.description or "")
        for port in sorted(list_ports.comports(), key=lambda p: p.device)
    )


def port_removed(before: Sequence[PortCandidate], after: Sequence[PortCandidate]) -> str:
    """The one port that disappeared, or a refusal naming what happened instead.

    The whole of ``lerobot-find-port``, as a pure function: list ports, unplug
    the arm, list again, and the difference is the answer. Pure because the
    interesting cases — nothing was unplugged, two things were — are the ones
    worth having a test for, and neither of them needs a USB port.
    """
    gone = sorted({p.device for p in before} - {p.device for p in after})
    if not gone:
        raise ConfigError(
            "no port disappeared. Either the cable was not unplugged, or this arm's "
            "adapter does not enumerate as a serial device on this machine"
        )
    if len(gone) > 1:
        raise ConfigError(
            f"{len(gone)} ports disappeared ({', '.join(gone)}); unplug one arm at a "
            "time, or the leader and the follower end up swapped and every recorded "
            "action is the wrong arm's"
        )
    return gone[0]


def find_port(
    *,
    ports: Callable[[], Sequence[PortCandidate]] = list_serial_ports,
    pause: Callable[[str], Any] = input,
    announce: Callable[[str], Any] = print,
) -> str:
    """Interactive port discovery, equivalent to ``lerobot-find-port``."""
    before = tuple(ports())
    announce(f"{len(before)} serial port(s) present: {', '.join(p.device for p in before) or '-'}")
    pause("Unplug the arm's USB cable, then press Enter. ")
    after = tuple(ports())
    device = port_removed(before, after)
    announce(f"the arm is on {device}. Plug it back in.")
    pause("Press Enter once it is reconnected. ")
    return device


def assign_motor_id(
    transport: Transport,
    new_id: int,
    *,
    candidates: Sequence[int] = tuple(range(0, 21)),
) -> None:
    """Set one servo's id, refusing unless exactly one servo is on the bus.

    Every STS3215 ships as id 1. Put two on a bus before assigning and they
    both answer every packet at once; the replies collide, the reads look
    intermittent rather than wrong, and the usual response is to suspect the
    cable. So this scans first and refuses to write unless the bus holds
    exactly one servo — which is the procedure anyway, one servo at a time on
    an otherwise empty chain.
    """
    if not isinstance(transport, SerialTransport):
        raise ConfigError("assigning a motor id needs a SerialTransport")
    if not 0 <= new_id <= MAX_ID:
        raise ConfigError(f"motor id {new_id} is outside 0..{MAX_ID}")
    present = [motor_id for motor_id, result in transport.scan(candidates).items() if result]
    if len(present) != 1:
        raise ConfigError(
            f"{len(present)} servo(s) answered on this bus ({present or 'none'}). ID "
            "assignment needs exactly one: every servo ships as id 1, so two of them "
            "reply simultaneously and the collision reads as a flaky cable"
        )
    transport.write_eeprom(present[0], "ID", new_id)


# ==========================================================================
# selftest
# ==========================================================================


class _FakeClock:
    """A clock the test drives, so latency and staleness are exact rather than raced."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


_SELFTEST_JOINTS = (
    # A fixture, not a declaration. The canonical joint order for this rig is
    # arm.JOINTS; restating it here would be a second copy of an ordering, and
    # an ordering that has drifted is silent all the way to the loss curve.
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def _fixture(**kwargs: Any) -> tuple[MockTransport, _FakeClock]:
    clock = _FakeClock()
    calibration = Calibration.fixture(_SELFTEST_JOINTS, **kwargs.pop("calibration_args", {}))
    transport = MockTransport(
        calibration,
        max_relative_target=kwargs.pop("limits", SafetyLimits.degrees(3.0)),
        clock=clock,
        sleep=clock.advance,
        **kwargs,
    )
    return transport.open(), clock


def selftest(verbose: bool = True) -> int:
    """Exercise every safety rule and every injected fault. No hardware, no sleeping."""
    checks: list[tuple[str, str]] = []

    def ok(name: str, detail: str = "") -> None:
        checks.append((name, detail))
        if verbose:
            print(f"  pass  {name}" + (f"  -- {detail}" if detail else ""))

    def expect(name: str, exception: type[BaseException], call: Callable[[], Any]) -> None:
        try:
            call()
        except exception as error:
            ok(name, f"{type(error).__name__}: {str(error).splitlines()[0][:88]}")
        except BaseException as error:  # noqa: BLE001
            raise AssertionError(
                f"{name}: expected {exception.__name__}, got {type(error).__name__}: {error}"
            ) from error
        else:
            raise AssertionError(f"{name}: expected {exception.__name__}, nothing was raised")

    print("gantry-evaluator-so101 transport selftest")

    # -- units ------------------------------------------------------------
    print("\nunit conversion (the #3758 failure mode at this layer)")
    for mode in NORM_MODES:
        joint = JointCalibration("probe", 1, 700, 3400, mode)
        bad = [t for t in range(joint.range_min, joint.range_max + 1) if joint.to_ticks(joint.to_normalized(t)) != t]
        assert not bad, f"{mode}: {len(bad)} tick(s) failed to round-trip, first {bad[:3]}"
        low, high = joint.normalized_bounds
        ok(f"round-trip {mode}", f"{joint.span + 1} ticks exact, bounds [{low:g}, {high:g}]")

    mirrored = JointCalibration("mirrored", 1, 700, 3400, RANGE_0_100, drive_mode=1)
    assert abs(mirrored.to_normalized(700) - 100.0) < 1e-9
    assert abs(mirrored.to_normalized(3400) - 0.0) < 1e-9
    bad = [t for t in range(700, 3401) if mirrored.to_ticks(mirrored.to_normalized(t)) != t]
    assert not bad
    ok("drive_mode inverts and still round-trips", "range_min reads 100, range_max reads 0")

    degrees = JointCalibration("deg", 1, 0, RESOLUTION - 1, DEGREES)
    assert abs(degrees.to_normalized(RESOLUTION // 2) - 0.0) < 0.05
    assert abs((degrees.to_normalized(2048) - degrees.to_normalized(2047)) - DEGREES_PER_TICK) < 1e-9
    ok("degree scale", f"1 tick = {DEGREES_PER_TICK:.6f} deg, zero at mid-travel")

    expect(
        "off-scale command refused, not clamped",
        SafetyAbort,
        lambda: JointCalibration("probe", 1, 700, 3400, RANGE_0_100).to_ticks(137.0),
    )

    assert decode_sign_magnitude(encode_sign_magnitude(-3, 10), 10) == -3
    assert decode_sign_magnitude(0x403, 10) == -3
    ok("sign-magnitude round-trip", "load -3 encodes 0x403, not 1027")

    # -- construction refusals --------------------------------------------
    print("\nthe clamp cannot be omitted or misread")
    fixture_cal = Calibration.fixture(_SELFTEST_JOINTS)
    expect(
        "no clamp argument at all",
        TypeError,
        lambda: MockTransport(fixture_cal),  # type: ignore[call-arg]
    )
    expect(
        "a bare number is refused (unit unknown)",
        ConfigError,
        lambda: MockTransport(fixture_cal, max_relative_target=5),  # type: ignore[arg-type]
    )
    expect("a clamp of zero", ConfigError, lambda: SafetyLimits.ticks(0))
    expect("a clamp of infinity", ConfigError, lambda: SafetyLimits.degrees(float("inf")))
    expect(
        "unmeasured calibration on a real transport",
        ConfigError,
        lambda: SerialTransport(
            fixture_cal, max_relative_target=SafetyLimits.degrees(3), port="COM-NOPE"
        ),
    )
    expect(
        "SerialTransport with no port and no factory",
        ConfigError,
        lambda: SerialTransport(
            Calibration.fixture(_SELFTEST_JOINTS[:1]),
            max_relative_target=SafetyLimits.degrees(3),
            allow_unmeasured_calibration=True,
        ),
    )
    ok(
        "clamp budget resolves per joint",
        f"3 deg -> {SafetyLimits.degrees(3).resolve(fixture_cal)[1]} ticks",
    )

    # -- the clamp itself --------------------------------------------------
    print("\nthe relative clamp")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    budget = SafetyLimits.degrees(3.0).resolve(transport.calibration)[1]
    result = transport.write_goal_position({1: 4000})
    assert result.any_clamped
    moved = result.writes[0].commanded - result.writes[0].reference
    assert moved == budget, f"moved {moved}, budget {budget}"
    ok("a 1952-tick command becomes one step", f"{moved} ticks; {result.explain()[:60]}")

    transport.read_present_position()
    small = transport.write_goal_position({1: transport.read_present_position().ticks[1] + 2})
    assert not small.any_clamped
    ok("a command inside the budget passes untouched", "no joint clamped")

    events: list[WriteResult] = []
    transport.on_clamp = events.append
    transport.read_present_position()
    transport.write_goal_position({1: 4000})
    assert len(events) == 1
    ok("every clamp is reported, never silent", f"on_clamp fired, {events[0].explain()[:52]}")

    transport.read_present_position()
    edge = transport.write_goal_position({6: 0})
    assert edge.writes[0].commanded >= transport.calibration.by_id(6).range_min
    ok("absolute travel limit holds", f"gripper floored at {edge.writes[0].commanded} ticks")

    # -- staleness ---------------------------------------------------------
    print("\nfail-closed: a stale reference is not a reference")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    clock.advance(0.5)
    expect(
        "a 500 ms old reference (limit 100 ms)",
        SafetyAbort,
        lambda: transport.write_goal_position({1: 2100}),
    )
    expect(
        "and the reference is dropped, not merely aged",
        SafetyAbort,
        lambda: transport.write_goal_position({1: 2100}),
    )
    transport.read_present_position()
    transport.write_goal_position({1: 2100})
    ok("a fresh read restores the ability to write", "")

    # -- injected fault: bus timeout ---------------------------------------
    print("\ninjected fault: bus timeout")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    transport.chain.drop(1)
    expect("a dropped transaction", BusTimeout, transport.read_present_position)
    expect(
        "and the last known value is NOT substituted",
        SafetyAbort,
        lambda: transport.write_goal_position({1: 2100}),
    )
    ok("this is the runaway-arm rule", "a timeout destroys the clamp reference")

    # -- injected fault: dead servo ----------------------------------------
    print("\ninjected fault: dead servo")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    transport.chain.kill(4)
    for attempt in range(1, 3):
        expect(f"silent servo, read {attempt} of 3", BusTimeout, transport.read_present_position)
    try:
        transport.read_present_position()
    except ServoFault as error:
        assert "[4]" in str(error), f"only servo 4 is silent, message said: {error}"
        ok("third consecutive silence escalates to a halting fault", str(error).split(":")[0])
        ok("and blames only the servo that earned it", "not all six; the operator checks one cable")
    else:
        raise AssertionError("a third silent read should have faulted")
    assert FaultError in type(ServoFault("x")).__mro__
    ok("ServoFault halts by taxonomy", "gantry.errors: halts=True, recoverable=False")

    transport.chain.revive(4)
    transport.read_present_position()
    ok("a revived servo clears its strikes", "")

    # -- injected fault: stale servo ---------------------------------------
    print("\ninjected fault: servo returning a stale value")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    transport.chain.freeze(3)
    frozen = transport.read_present_position().ticks[3]
    for _ in range(SafetyLimits.degrees(3).stall_steps + 2):
        transport.read_present_position()
        transport.write_goal_position({3: 3000})
        clock.advance(0.01)
    assert transport.read_present_position().ticks[3] == frozen
    assert transport.stalls()[3].stalled, "a frozen joint must show as stalled"
    ok("a frozen joint is reported faithfully and flagged", f"held at {frozen} ticks")
    measurement = transport.stall_measurements()["elbow_flex.pos"]
    assert measurement.units == "deg" and measurement.n is not None
    ok("stall arrives as a Measurement, not a float", str(measurement)[:70])
    assert "no force or torque" in (measurement.method or "")
    ok("and says it is not a force reading", "method names the absence of a sensor")

    # -- injected fault: physical jam / contact ----------------------------
    print("\ninjected fault: physical jam (the gripper contact proxy)")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.chain.jam(6)
    for _ in range(SafetyLimits.degrees(3).stall_steps + 2):
        transport.read_present_position()
        transport.write_goal_position({6: 3400})
        clock.advance(0.01)
    assert transport.stalls()[6].stalled
    load = transport.read_present_load([6])[6]
    assert load.units == "fraction" and "not measured force" in (load.method or "")
    ok("a jammed gripper stalls and loads up", f"load {load}")
    current = transport.read_present_current([6])[6]
    assert current.units == "count"
    ok("current stays a raw count", "scale unverified for this part; no amps invented")

    # -- injected fault: servo hardware error ------------------------------
    print("\ninjected fault: servo reports a hardware error")
    transport, clock = _fixture()
    transport.chain.set_error(2, 0x24)
    expect("an error byte on read", ServoFault, lambda: transport.read_present_position())
    ok("described in words", describe_error(0x24))

    # -- injected fault: corrupted reply -----------------------------------
    print("\ninjected fault: corrupted reply")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    transport.chain.corrupt(1)
    expect("a bad checksum", BusCorruption, transport.read_present_position)
    expect(
        "and it invalidates the reference too",
        SafetyAbort,
        lambda: transport.write_goal_position({1: 2100}),
    )

    # -- torque ------------------------------------------------------------
    print("\ntorque")
    transport, clock = _fixture()
    transport.chain.servos[1].goal = 100  # a stale goal from a previous session
    transport.chain.servos[1].position = 2048
    transport.torque_enable([1])
    assert transport.chain.servos[1].goal == 2048
    ok("enabling torque first points the servo at itself", "no snap to the old goal")
    transport.chain.kill(5)
    expect(
        "and refuses to energise a chain it cannot read",
        BusTimeout,
        lambda: transport.torque_enable(),
    )
    transport.chain.revive(5)
    transport.torque_disable()
    assert not any(transport.torque_state().values())
    ok("torque_disable reaches every joint", "")

    # -- the packet codec, over LoopbackSerial ------------------------------
    print("\nthe wire format itself (SerialTransport over LoopbackSerial)")
    packet = instruction_packet(1, INST_READ, [56, 2])
    assert packet == bytes([0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE]), packet.hex(" ")
    ok("instruction packet bytes", packet.hex(" "))
    expect(
        "a corrupted frame is refused, not parsed",
        BusCorruption,
        lambda: decode_status(bytes([0xFF, 0xFF, 0x01, 0x02, 0x00, 0x00])),
    )
    assert to_bytes(0x0123, 2) == [0x23, 0x01]
    ok("two-byte registers are little-endian", "0x0123 -> 23 01 (STS/SMS, not SCS)")

    clock = _FakeClock()
    chain = ServoChain(fixture_cal.ids, clock=clock, sleep=clock.advance)
    wire = SerialTransport(
        fixture_cal,
        max_relative_target=SafetyLimits.degrees(3),
        serial_factory=lambda: LoopbackSerial(chain),
        allow_unmeasured_calibration=True,
        clock=clock,
    )
    with wire:
        assert wire.ping(1).model_number == STS3215_MODEL_NUMBER
        wire.torque_enable()
        wire.read_present_position()
        result = wire.write_goal_position({1: 4000})
        assert result.any_clamped
        ok("real codec path: ping, torque, sync-write, clamp", result.explain()[:64])
        chain.drop(1)
        expect("real codec path: short read is a timeout", BusTimeout, wire.read_present_position)
        wire.read_present_position()
        chain.corrupt(1)
        expect("real codec path: checksum failure", BusCorruption, wire.read_present_position)
        chain.set_error(2, 0x20)
        assert wire._ping(2).error == 0x20, "a ping must carry the error byte, not swallow it"
        expect("real codec path: a faulted servo answers its ping", ServoFault, lambda: wire.ping(2))
        chain.set_error(2, 0)

    clock = _FakeClock()
    chain = ServoChain(fixture_cal.ids, clock=clock, sleep=clock.advance)
    fast = SerialTransport(
        fixture_cal,
        max_relative_target=SafetyLimits.degrees(3),
        serial_factory=lambda: LoopbackSerial(chain),
        sync_read=True,
        allow_unmeasured_calibration=True,
        clock=clock,
    )
    with fast:
        fast.torque_enable()
        assert len(fast.read_present_position().ticks) == 6
        ok("sync-read reads the whole chain in one frame", "6 status packets, matched by id")
        chain.kill(3)
        expect(
            "and a missing servo is a hole, refused",
            BusTimeout,
            fast.read_present_position,
        )

    # -- port discovery ----------------------------------------------------
    print("\nport discovery (lerobot-find-port, as a pure function)")
    before = (PortCandidate("COM3"), PortCandidate("COM7"), PortCandidate("COM9"))
    assert port_removed(before, (PortCandidate("COM3"), PortCandidate("COM9"))) == "COM7"
    ok("one port disappeared", "COM7")
    expect("nothing disappeared", ConfigError, lambda: port_removed(before, before))
    expect(
        "two disappeared (leader and follower would get swapped)",
        ConfigError,
        lambda: port_removed(before, (PortCandidate("COM3"),)),
    )
    seen: list[str] = []
    device = find_port(
        ports=iter([before, (PortCandidate("COM3"), PortCandidate("COM9"))]).__next__,
        pause=lambda _: None,
        announce=seen.append,
    )
    assert device == "COM7" and len(seen) == 2
    ok("the interactive wrapper agrees", device)

    # -- id assignment -----------------------------------------------------
    print("\nmotor id assignment (every servo ships as id 1)")
    clock = _FakeClock()
    single = ServoChain([1], clock=clock, sleep=clock.advance)
    one_servo = Calibration.fixture(("only",), ids=(1,))
    lone = SerialTransport(
        one_servo,
        max_relative_target=SafetyLimits.degrees(3),
        serial_factory=lambda: LoopbackSerial(single),
        allow_unmeasured_calibration=True,
        clock=clock,
    )
    with lone:
        assign_motor_id(lone, 6)
        assert 6 in single.servos and 1 not in single.servos
        ok("one servo on an empty bus is renumbered", "id 1 -> 6")

    clock = _FakeClock()
    crowded = ServoChain([1, 2], clock=clock, sleep=clock.advance)
    two = SerialTransport(
        Calibration.fixture(("a", "b"), ids=(1, 2)),
        max_relative_target=SafetyLimits.degrees(3),
        serial_factory=lambda: LoopbackSerial(crowded),
        allow_unmeasured_calibration=True,
        clock=clock,
    )
    with two:
        expect(
            "two servos on the bus refuses the write",
            ConfigError,
            lambda: assign_motor_id(two, 6),
        )

    # -- calibration provenance --------------------------------------------
    print("\ncalibration provenance")
    payload = {
        name: {"id": index, "range_min": 700 + index, "range_max": 3400, "drive_mode": 0,
               "homing_offset": 11}
        for index, name in enumerate(_SELFTEST_JOINTS, start=1)
    }
    measured = Calibration.from_lerobot(payload, source="so101_new_calib/follower.json")
    assert measured.measured and measured.names == _SELFTEST_JOINTS
    ok("LeRobot calibration reads in declared order", ", ".join(measured.names[:3]) + ", ...")
    other = Calibration.from_lerobot(
        {**payload, "gripper.pos": {**payload["gripper.pos"], "range_max": 3399}},
        source="so101_new_calib/follower.json",
    )
    assert measured.digest != other.digest
    ok("one tick of difference changes the digest", f"{measured.digest[:16]}... vs {other.digest[:16]}...")
    convention = Calibration.from_lerobot(
        payload, source="x", norm_modes={"gripper.pos": RANGE_M100_100}
    )
    assert convention.digest != measured.digest
    ok("so does a change of convention", "a corpus in RANGE_M100_100 is not the same corpus")
    expect(
        "a calibration missing a measured range",
        ConfigError,
        lambda: Calibration.from_lerobot({"j": {"id": 1, "range_min": 0}}),
    )
    expect(
        "two servos sharing an id",
        ConfigError,
        lambda: Calibration.fixture(("a", "b"), ids=(1, 1)),
    )

    # -- vector width ------------------------------------------------------
    print("\naction width")
    transport, clock = _fixture()
    transport.torque_enable()
    transport.read_present_position()
    expect(
        "a five-wide action on a six-joint chain",
        ConfigError,
        lambda: transport.write_goal_vector([50.0] * 5),
    )
    transport.write_goal_vector([50.0] * 6)
    ok("a six-wide action is accepted", "no padding, no truncation, ever")

    health = transport.health()
    assert health["calibration_measured"] is False
    assert isinstance(health["clamp_rate"], Measurement)
    ok("health block is record-ready", f"clamp_rate {health['clamp_rate']}")
    transport.close()

    print(f"\n{len(checks)} checks passed, no hardware present.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="exercise every safety rule and every injected fault on MockTransport",
    )
    parser.add_argument(
        "--find-port",
        action="store_true",
        help="identify which serial port an arm is on, by unplugging it",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the verdict")
    args = parser.parse_args(argv)
    if args.find_port:
        print(find_port())
        return 0
    if args.selftest:
        return selftest(verbose=not args.quiet)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
