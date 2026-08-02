"""Two structural reductions that are defensible without knowing the machines.

A retargeter maps one machine's channel onto another's, and most such maps need
kinematics, a calibration, or someone's opinion. These two do not:

**Dropping named dimensions.** A seven-jointed arm feeding a six-jointed
consumer has to lose a joint, and *which* joint is a decision -- so it is an
argument, not a guess. Once named, the transform is a selection and the loss is
exactly the named columns.

**Reducing a pose to a position.** A pose is a position followed by a rotation
in a stated encoding. Keeping the leading three numbers is unambiguous, and what
is discarded is the whole orientation.

Everything harder -- joint angles to an end-effector pose, one gripper's opening
onto another's -- needs a model of the machine and belongs with whoever owns that
machine. This package deliberately stops here rather than shipping a plausible
approximation, because a retargeter that is *nearly* right produces motion that
looks reasonable and is wrong.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gantry.contracts.embodiment import Retargeter
from gantry.spine import ChannelSpec, Verdict

VERSION = "0.1.0.dev0"

#: How many numbers each rotation encoding takes, matching the manipulation
#: semantics package. Duplicated rather than imported so this package does not
#: depend on that one; the values are a fact about the encodings, not a choice.
ROTATION_WIDTHS = {
    "none": 0,
    "quat_wxyz": 4,
    "quat_xyzw": 4,
    "rot6d": 6,
    "axis_angle": 3,
    "euler_xyz": 3,
}


class DropDimensions(Retargeter):
    """Feed a narrower consumer by dropping dimensions it does not have.

    The dropped names are given, never inferred. A retargeter that picked which
    joint to discard by position or by heuristic would be making an engineering
    decision on someone's behalf, silently, once per run.
    """

    def __init__(self, drop: Sequence[str]):
        if not drop:
            raise ValueError("name at least one dimension to drop")
        self._drop = tuple(drop)

    @property
    def name(self) -> str:
        return f"drop-dimensions[{','.join(self._drop)}]"

    @property
    def version(self) -> str:
        return VERSION

    def _indices(self, source: ChannelSpec) -> tuple[int, ...] | None:
        if source.dim_labels is None:
            return None
        keep = [
            index for index, label in enumerate(source.dim_labels) if label not in set(self._drop)
        ]
        return tuple(keep)

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        if source.dim_labels is None:
            return Verdict.no(
                "retarget.unlabelled",
                f"{source.name!r} has no dimension labels, so there is no way to say "
                "which dimensions these are",
                hint="label the dimensions on the source channel",
            )
        missing = [label for label in self._drop if label not in source.dim_labels]
        if missing:
            return Verdict.no(
                "retarget.no_such_dimension",
                f"{source.name!r} has no dimension(s) named {missing}",
                hint=f"it has {list(source.dim_labels)}",
            )
        keep = self._indices(source) or ()
        if len(keep) != target.width:
            return Verdict.no(
                "retarget.width",
                f"dropping {list(self._drop)} leaves {len(keep)} dimension(s), but "
                f"{target.name!r} wants {target.width}",
            )
        if target.dim_labels is not None:
            kept = tuple(source.dim_labels[index] for index in keep)
            if kept != target.dim_labels:
                return Verdict.no(
                    "retarget.labels",
                    f"the kept dimensions {list(kept)} are not what {target.name!r} "
                    f"labels: {list(target.dim_labels)}",
                    hint="a differing order needs the permute adapter as well",
                )
        return Verdict.yes()

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        return (
            f"dropped dimension(s) {', '.join(self._drop)} from {source.name!r}; "
            "anything they carried is gone",
        )

    def apply(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        keep = self._indices(source)
        if keep is None:
            raise ValueError(f"{source.name!r} has no dimension labels")
        return np.asarray(values)[:, list(keep)]


class PoseToPosition(Retargeter):
    """Feed a position-only consumer from a full pose, discarding the rotation."""

    @property
    def name(self) -> str:
        return "pose-to-position"

    @property
    def version(self) -> str:
        return VERSION

    def _rotation_width(self, source: ChannelSpec) -> int | None:
        """How many numbers the rotation takes, only if the channel says.

        Deliberately never inferred from the width. Reading "six wide" as
        "position plus axis-angle" would make every six-jointed arm look like a
        pose, and this retargeter would happily hand back its first three joint
        angles as a position in metres. Guessing here is precisely the
        coincidence-matching the framework refuses everywhere else.
        """
        encoding = source.metadata.get("rotation_repr")
        return ROTATION_WIDTHS.get(str(encoding)) if encoding is not None else None

    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        if target.width != 3:
            return Verdict.no(
                "retarget.not_a_position",
                f"{target.name!r} is {target.width} wide; a position is 3",
            )
        rotation = self._rotation_width(source)
        if rotation is None:
            return Verdict.no(
                "retarget.not_a_pose",
                f"{source.name!r} declares no rotation encoding, so it cannot be read as a pose",
                hint="set rotation_repr in the channel's metadata; a width alone "
                "never establishes that something is a pose",
            )
        if source.width != 3 + rotation:
            return Verdict.no(
                "retarget.pose_width",
                f"{source.name!r} is {source.width} wide but a position plus its "
                f"declared rotation would be {3 + rotation}",
            )
        return Verdict.yes()

    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        encoding = source.metadata.get("rotation_repr", "the rotation")
        return (f"discarded the orientation ({encoding}) from {source.name!r}",)

    def apply(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        return np.asarray(values)[:, :3]
