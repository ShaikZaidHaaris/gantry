"""One gripper's joint readings expressed in another gripper's terms.

`retargeters_core` stops short of this on purpose — its own docstring says a
gripper conversion "needs a model of the machine and belongs with whoever owns
that machine". This package owns that model, and it is a small one: how far
open each gripper is, as a fraction of its own measured travel.

Why a fraction and not a rescale
--------------------------------
Grippers do not agree on how many numbers they report, what those numbers mean,
or even which direction is closed. Measured in robosuite:

    PandaGripper      2 joints   closed [ 0.0005, -0.0005]  open [ 0.0399, -0.0399]
    RethinkGripper    2 joints   closed [-0.0118,  0.0118]  open [ 0.0114, -0.0114]
    Robotiq85Gripper  6 joints   travel 2.04 rad
    JacoThreeFinger   6 joints   travel 9.38 rad

Panda and Rethink report the same *number* of values, and a policy trained on
one and fed the other reads a closed hand as half-open with the fingers on the
wrong sides. That is the case this exists to prevent: matching widths are what
make the mistake silent, so a width check cannot catch it.

So the conversion goes through the only quantity both grippers actually share —
how far open they are, as a fraction of their own travel — and nothing else
crosses. The reading is projected onto the line from its own measured closed
pose to its own measured open pose, and rebuilt on the target's line.

What this discards, and why that must be said out loud
------------------------------------------------------
Fraction-open is not equivalence. A three-fingered hand at 40% and a parallel
jaw at 40% hold different objects differently, contact at different points, and
fail differently. This retargeter declares that loss rather than implying the
bodies are interchangeable, and a result produced through it carries the
retargeter's name so nobody reads it as a native measurement later.

The calibration is measured, never assumed. Numbers written by hand for a
gripper nobody commanded are the failure mode this whole approach exists to
avoid — see :func:`calibration_from`, which refuses an embodiment that does not
carry one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gantry.contracts.embodiment import Retargeter
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec, Verdict

VERSION = "0.1.0.dev0"

#: How many leading values are the end-effector pose. Position and orientation
#: are frame-and-unit facts that cross bodies unchanged, so they pass through;
#: everything after them is the gripper's own business.
POSE_WIDTH = 6

#: Below this, the two calibration poses are the same pose and the projection
#: has no axis to project onto.
MIN_TRAVEL = 1e-6


@dataclass(frozen=True)
class GripperCalibration:
    """What one gripper reads at each end of its travel, as measured.

    ``closed`` and ``open`` are full joint readings, not summaries, because the
    joints of a multi-finger hand do not move together and the direction
    between the two poses is the only thing that makes "half open" meaningful.
    """

    name: str
    closed: tuple[float, ...]
    open: tuple[float, ...]
    #: Where these numbers came from. Recorded so a reader can tell a measured
    #: calibration from one somebody typed.
    measured_in: str = ""

    @property
    def joints(self) -> int:
        return len(self.open)

    @property
    def travel(self) -> np.ndarray:
        return np.asarray(self.open, dtype=np.float64) - np.asarray(self.closed, dtype=np.float64)

    def validate(self) -> Verdict:
        checks = []
        if len(self.closed) != len(self.open):
            checks.append(
                Verdict.no(
                    "gripper.ragged_calibration",
                    f"{self.name}: closed has {len(self.closed)} values, open has "
                    f"{len(self.open)}",
                )
            )
        elif float(np.linalg.norm(self.travel)) < MIN_TRAVEL:
            checks.append(
                Verdict.no(
                    "gripper.no_travel",
                    f"{self.name}: its open and closed readings are the same, so how "
                    "far open it is cannot be read from them",
                    hint="command it to both stops and record what it reports",
                )
            )
        if not self.measured_in:
            checks.append(
                Verdict.note(
                    "gripper.unattributed_calibration",
                    f"{self.name}: the calibration does not say where it was measured",
                )
            )
        return Verdict.all(checks)

    def fraction(self, reading: np.ndarray) -> np.ndarray:
        """How far open, in [0, 1], for readings of shape ``(steps, joints)``.

        A projection onto the closed-to-open line rather than a per-joint
        rescale: the joints of a coupled hand are not independent, and rescaling
        each on its own produces poses the hand cannot reach.

        Clamped, because a reading outside the measured travel means the
        calibration is narrower than reality, and extrapolating it would put the
        target gripper somewhere it cannot go.
        """
        axis = self.travel
        offset = np.asarray(reading, dtype=np.float64) - np.asarray(self.closed, dtype=np.float64)
        return np.clip(offset @ axis / float(axis @ axis), 0.0, 1.0)

    def reading(self, fraction: np.ndarray) -> np.ndarray:
        """The joint reading this gripper would give at that fraction open."""
        closed = np.asarray(self.closed, dtype=np.float64)
        return closed + np.asarray(fraction, dtype=np.float64)[:, None] * self.travel


def gripper_block(spec: ChannelSpec) -> tuple[str, int] | None:
    """The gripper's name and how many values it occupies, read off the labels.

    A state channel here is a pose followed by whatever the hand reports, and
    the measured embodiment files label those trailing values with the gripper
    class that produced them — ``PandaGripper.0``, ``Robotiq85Gripper.3``. That
    label is what makes two eight-wide states distinguishable, so it is what is
    read, rather than the width.
    """
    labels = spec.dim_labels or ()
    tail = labels[POSE_WIDTH:]
    if not tail:
        return None
    names = {label.split(".")[0] for label in tail}
    if len(names) != 1:
        return None
    return names.pop(), len(tail)


def calibration_from(embodiment: Any) -> GripperCalibration:
    """The calibration a machine description carries, or a refusal.

    Refuses rather than defaulting. A plausible calibration is worse than none:
    it produces a policy that grips at the wrong moment and a result nobody can
    trace back to a number somebody guessed.
    """
    block = getattr(embodiment, "metadata", None) or {}
    if not isinstance(block, Mapping):
        block = {}
    hands = block.get("grippers")
    if isinstance(hands, Mapping) and len(hands) > 1:
        # Said precisely rather than as "no calibration": this body is
        # described, and describing it better will not help. One hand's
        # opening does not stand for two, and which of the two a one-handed
        # policy is reading is not a detail to pick a default for.
        raise ConfigError(
            f"{getattr(embodiment, 'name', embodiment)!r} reports {len(hands)} hands "
            f"({', '.join(sorted(hands))}); this conversion maps one hand onto one "
            "hand and there is no non-arbitrary way to choose between them"
        )
    grip = block.get("gripper")
    if not isinstance(grip, Mapping) or "open" not in grip or "closed" not in grip:
        # A refusal rather than a crash: a body nobody calibrated is a gap in
        # the description, and a sweep across bodies should record it as such
        # next to the ones that ran, not abort on it.
        raise ConfigError(
            f"{getattr(embodiment, 'name', embodiment)!r} carries no gripper calibration, "
            "so how far open its hand is cannot be read; command it to both stops and "
            "record what it reports"
        )
    return GripperCalibration(
        name=str(grip.get("name", "")),
        closed=tuple(float(v) for v in grip["closed"]),
        open=tuple(float(v) for v in grip["open"]),
        measured_in=str(grip.get("measured_in", "")),
    )


class GripperAperture(Retargeter):
    """Re-express a state's gripper block in another gripper's readings.

    The pose passes through untouched; only the trailing gripper values are
    converted, and only through fraction-open.
    """

    def __init__(self, source: GripperCalibration, target: GripperCalibration):
        source.validate().raise_if_refused(f"source gripper {source.name!r}")
        target.validate().raise_if_refused(f"target gripper {target.name!r}")
        self._source = source
        self._target = target

    @property
    def name(self) -> str:
        return f"gripper_aperture:{self._source.name}->{self._target.name}"

    @property
    def version(self) -> str:
        return VERSION

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        pair = []
        for spec, calibration, side in (
            (source, self._source, "source"),
            (target, self._target, "target"),
        ):
            block = gripper_block(spec)
            if block is None:
                pair.append(
                    Verdict.no(
                        "gripper.unlabelled",
                        f"{side} channel {spec.name!r} does not label which gripper its "
                        f"trailing values come from, so they cannot be told apart from "
                        "any other gripper's",
                        hint="dim_labels after the pose should read '<GripperClass>.<n>'",
                    )
                )
                continue
            found, width = block
            if found != calibration.name:
                pair.append(
                    Verdict.no(
                        "gripper.wrong_calibration",
                        f"{side} channel reports a {found}, but this retargeter was "
                        f"built for a {calibration.name}",
                    )
                )
            elif width != calibration.joints:
                pair.append(
                    Verdict.no(
                        "gripper.width_mismatch",
                        f"{side} {found} occupies {width} values here and "
                        f"{calibration.joints} in its calibration",
                    )
                )
        if source.width - POSE_WIDTH < 1 or target.width - POSE_WIDTH < 1:
            pair.append(
                Verdict.no(
                    "gripper.no_block",
                    f"{source.width} -> {target.width}: a state needs {POSE_WIDTH} pose "
                    "values and at least one gripper value for this to mean anything",
                )
            )
        return Verdict.all(pair)

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        return (
            f"reads {self._source.name} as a fraction of its own travel and rebuilds "
            f"that fraction on {self._target.name}",
            "keeps only how far open the hand is; finger count, finger kinematics and "
            "the geometry of where it contacts an object do not cross",
            "grip force and contact state are not represented at either end",
        )

    def apply(
        self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec
    ) -> np.ndarray:
        self.check(source, target).raise_if_refused(f"{self.name} cannot map this pair")
        array = np.atleast_2d(np.asarray(values, dtype=np.float64))
        pose, hand = array[:, :POSE_WIDTH], array[:, POSE_WIDTH:]
        converted = self._target.reading(self._source.fraction(hand))
        return np.concatenate([pose, converted], axis=1).astype(np.float32)


def between(source: Any, target: Any) -> GripperAperture:
    """A retargeter between two machine descriptions that carry calibrations."""
    return GripperAperture(calibration_from(source), calibration_from(target))


def state_spec_for(spec: ChannelSpec, calibration: GripperCalibration) -> ChannelSpec:
    """The same pose, with the gripper block relabelled for the target hand."""
    labels = tuple(spec.dim_labels or ())[:POSE_WIDTH] + tuple(
        f"{calibration.name}.{i}" for i in range(calibration.joints)
    )
    return ChannelSpec(
        spec.name,
        spec.kind,
        (POSE_WIDTH + calibration.joints,),
        spec.dtype,
        semantics=spec.semantics,
        dim_labels=labels,
    )
