"""The three measured things a hand-to-arm retargeting needs, and cannot guess.

Every number in this module is a property of a specific person, a specific
robot, or the specific relationship between them. None of it can be inferred
from a video, and every one of them has a plausible-looking default that is
wrong in a way that produces smooth, confident, incorrect motion.

So they are dataclasses that refuse to be constructed empty, rather than
keyword arguments with defaults. That is the same call ``retargeter_gripper``
made about gripper calibration, for the same reason: numbers written by hand for
a machine nobody measured are the failure this approach exists to avoid.

Why a wrong mount is invisible
------------------------------
A rotation offset that is ninety degrees out produces a trajectory that is
perfectly smooth, stays in the workspace, and reaches for the wrong side of
every object. It does not look like a bug. It looks like a policy that has not
learned the task, which is exactly the conclusion the whole product is trying to
measure -- so the one failure that most resembles the answer is the one that must
be hardest to introduce by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from gantry.errors import ConfigError

#: Below this the two calibration apertures are the same and there is no travel
#: to project onto.
MIN_TRAVEL = 1e-4

#: A human hand span, wrist to fingertip, for an adult. Used only as a sanity
#: bound on a supplied measurement -- never as a substitute for one.
PLAUSIBLE_SPAN = (0.12, 0.28)


@dataclass(frozen=True)
class Hand:
    """One person's hand, measured.

    ``closed`` and ``open`` are that person's thumb-to-index distance with the
    hand shut and with it open around a typical object. They are what turns an
    aperture into a gripper command, and they are per-person: an adult's open
    hand and a child's differ by a factor that would otherwise land directly in
    the gripper signal.

    ``span`` is wrist-to-middle-fingertip, and it is the escape hatch for
    unscaled data -- a monocular estimate is correct up to one multiplier, and a
    known real length is exactly what recovers it.
    """

    closed: float
    open: float
    #: Wrist to middle fingertip, metres. Required only when the source is
    #: unscaled; that is the only thing that can recover the missing factor.
    span: float | None = None
    measured_by: str = ""

    def __post_init__(self) -> None:
        if float(self.open) - float(self.closed) < MIN_TRAVEL:
            raise ConfigError(
                f"a hand that opens to {self.open} and closes to {self.closed} has no "
                "travel to measure against; these are the person's own thumb-to-index "
                "distance shut and open, and they cannot be the same number"
            )
        if self.span is not None:
            low, high = PLAUSIBLE_SPAN
            if not low <= float(self.span) <= high:
                raise ConfigError(
                    f"a hand span of {self.span} m is outside {low}-{high} m. This is "
                    "wrist to middle fingertip in metres, and it is the number that "
                    "recovers scale for monocular data, an error here multiplies "
                    "through every position in the dataset"
                )

    @property
    def travel(self) -> float:
        return float(self.open) - float(self.closed)

    def fraction(self, aperture: np.ndarray) -> np.ndarray:
        """How open the hand is, 0 to 1, on its own measured travel.

        Clipped, because a person will open wider than their calibration at some
        point and a gripper command above one is not a wider gripper.
        """
        values = (np.asarray(aperture, dtype=float) - float(self.closed)) / self.travel
        return np.clip(values, 0.0, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "closed": float(self.closed),
            "open": float(self.open),
            "span": float(self.span) if self.span is not None else None,
            "measured_by": self.measured_by,
        }


@dataclass(frozen=True)
class Reach:
    """The robot's workspace, measured.

    Not to clip trajectories into -- though it can -- but to *report*. What
    fraction of a person's reaching happens somewhere this arm cannot go is one
    of the most useful numbers the whole ego pipeline produces, because it is
    both a real limit on the data and a thing the person can fix by standing
    closer to what they are doing.
    """

    #: Furthest the end-effector reaches from the base, metres.
    radius: float
    #: Lowest and highest the end-effector goes, in base coordinates.
    floor: float = -0.2
    ceiling: float = 0.8
    #: Nearest approach -- most arms cannot fold back onto their own base.
    inner: float = 0.1
    name: str = ""

    def __post_init__(self) -> None:
        if float(self.radius) <= float(self.inner):
            raise ConfigError(
                f"a reach of {self.radius} m with an inner limit of {self.inner} m "
                "leaves no workspace at all"
            )
        if float(self.ceiling) <= float(self.floor):
            raise ConfigError(f"a ceiling of {self.ceiling} m below a floor of {self.floor} m")

    def holds(self, positions: np.ndarray) -> np.ndarray:
        """Boolean per step: is this point somewhere the arm can be?"""
        points = np.asarray(positions, dtype=float).reshape(-1, 3)
        distance = np.linalg.norm(points, axis=1)
        return (
            (distance <= float(self.radius))
            & (distance >= float(self.inner))
            & (points[:, 2] >= float(self.floor))
            & (points[:, 2] <= float(self.ceiling))
        )

    def clip(self, positions: np.ndarray) -> np.ndarray:
        """Pull out-of-reach points onto the workspace boundary.

        A blunt instrument and it is meant to look like one. Clipping turns a
        reach the arm cannot make into a reach it can, which changes the
        demonstration into a different demonstration -- so this is off by default
        and the retargeter declares it as a loss whenever it is on.
        """
        points = np.array(positions, dtype=float).reshape(-1, 3)
        distance = np.linalg.norm(points, axis=1, keepdims=True)
        far = distance[:, 0] > float(self.radius)
        near = (distance[:, 0] < float(self.inner)) & (distance[:, 0] > 1e-9)
        points[far] *= float(self.radius) / distance[far]
        points[near] *= float(self.inner) / distance[near]
        points[:, 2] = np.clip(points[:, 2], float(self.floor), float(self.ceiling))
        return points

    def as_dict(self) -> dict[str, Any]:
        return {
            "reach": self.name or "unnamed",
            "radius": float(self.radius),
            "inner": float(self.inner),
            "floor": float(self.floor),
            "ceiling": float(self.ceiling),
        }


@dataclass(frozen=True)
class Mount:
    """Where the robot stands relative to the person, and how its gripper is turned.

    The hardest thing in this plugin to get right and the easiest to get wrong
    without noticing. ``rotation`` takes a point in the human's frame into the
    robot's base frame; ``origin`` is the point in human coordinates that the
    robot's base sits at; ``palm_to_gripper`` is the fixed turn between the
    orientation of a palm and the orientation of a two-fingered gripper holding
    the same object.

    ``workspace_ratio`` scales human distances into robot distances. Deliberately
    *not* called scale -- the ego vocabulary already uses that word for whether
    numbers are metres at all, and conflating the two would be the same class of
    error this file is about.
    """

    rotation: Any = None
    origin: Any = (0.0, 0.0, 0.0)
    palm_to_gripper: Any = None
    workspace_ratio: float = 1.0
    #: What was actually done to establish this. Free text, required when the
    #: mount is anything other than the explicitly-aligned one.
    established_by: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _rotation(self.rotation))
        object.__setattr__(self, "palm_to_gripper", _rotation(self.palm_to_gripper))
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float).reshape(3))
        if float(self.workspace_ratio) <= 0:
            raise ConfigError("workspace_ratio must be positive")

    @classmethod
    def aligned(cls, **kwargs: Any) -> "Mount":
        """The identity mount: the person's frame *is* the robot's base frame.

        A named constructor rather than a default, because identity is a claim.
        It is the right claim when a rig was set up so that the two frames agree,
        and it is silently wrong the rest of the time -- so it should be something
        somebody typed, and it appears in the record as such.
        """
        return cls(established_by="declared aligned", **kwargs)

    def place(self, positions: np.ndarray) -> np.ndarray:
        """Human-frame points into robot base coordinates."""
        points = np.asarray(positions, dtype=float).reshape(-1, 3) - self.origin
        return (points * float(self.workspace_ratio)) @ np.asarray(self.rotation).T

    def turn(self, matrices: np.ndarray) -> np.ndarray:
        """Human wrist orientations into gripper orientations."""
        rotation = np.asarray(self.rotation)
        offset = np.asarray(self.palm_to_gripper)
        return rotation @ np.asarray(matrices) @ offset

    def as_dict(self) -> dict[str, Any]:
        return {
            "mount_rotation": np.asarray(self.rotation).round(6).tolist(),
            "mount_origin": np.asarray(self.origin).round(6).tolist(),
            "palm_to_gripper": np.asarray(self.palm_to_gripper).round(6).tolist(),
            "workspace_ratio": float(self.workspace_ratio),
            "established_by": self.established_by or "undeclared",
            **dict(self.metadata),
        }

    @property
    def is_identity(self) -> bool:
        return bool(
            np.allclose(np.asarray(self.rotation), np.eye(3))
            and np.allclose(np.asarray(self.palm_to_gripper), np.eye(3))
            and np.allclose(np.asarray(self.origin), 0.0)
            and float(self.workspace_ratio) == 1.0
        )


def _rotation(value: Any) -> np.ndarray:
    """A 3x3 from a matrix, a wxyz quaternion, or nothing.

    ``None`` becomes identity, which is a legitimate value and a dangerous
    default -- hence :meth:`Mount.aligned`, which makes choosing it deliberate.
    """
    if value is None:
        return np.eye(3)
    array = np.asarray(value, dtype=float)
    if array.shape == (3, 3):
        if not np.allclose(array @ array.T, np.eye(3), atol=1e-5):
            raise ConfigError(
                "a mount rotation must be a rotation: this matrix is not orthonormal, "
                "so it scales or shears as well as turning, and every pose through it "
                "would be quietly distorted"
            )
        return array
    if array.shape == (4,):
        w, x, y, z = array / (np.linalg.norm(array) or 1.0)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
    raise ConfigError(f"a rotation is a 3x3 matrix or a wxyz quaternion, got shape {array.shape}")


#: A ViperX 300 6-DoF, the arm an ALOHA station is built from. Published reach,
#: recorded here so a first run has something to report against -- and named, so
#: that a report says which arm it was measured against rather than implying
#: every arm.
VIPERX_300 = Reach(radius=0.75, inner=0.10, floor=-0.05, ceiling=0.70, name="viperx_300")

#: A Franka Emika Panda, for the single-arm configurations.
PANDA = Reach(radius=0.855, inner=0.10, floor=-0.05, ceiling=1.10, name="panda")

REACHES = {reach.name: reach for reach in (VIPERX_300, PANDA)}
