"""Convert the rotation part of a pose between encodings.

Five encodings are in common use and they are mutually unreadable: a quaternion
scalar-first and one scalar-last are both four floats, axis-angle and Euler are
both three, and a channel described only by its width cannot tell you which it
holds. That is why this exists at all — the distinction is invisible to every
structural check, and executing one encoding as another produces motion that
looks plausible and points somewhere else.

A pose channel here is a position followed by a rotation, optionally followed by
a gripper: ``[x, y, z, <rotation>, gripper?]``. Only the rotation block is
touched; the position and anything after it pass through untouched.

Lossless, with one honest exception
-----------------------------------
Every conversion below is exact up to floating point, and the round trip is
tested. The exception is the double cover: ``q`` and ``-q`` are the same
rotation, so a round trip may return the antipodal quaternion. That is the same
rotation and not the same numbers, which is why quaternions are canonicalised to
a non-negative scalar part rather than left to flip. Anything that compared them
elementwise would otherwise see a change that is not one.
"""

from __future__ import annotations

import numpy as np
from gantry.resolve import Adapter
from gantry.spine import ChannelSpec, Verdict

VERSION = "0.1.0.dev0"

#: The metadata key carrying the encoding. Matches the manipulation semantics
#: pack, which is where the vocabulary is defined.
KEY = "rotation_repr"

#: How many numbers each encoding takes.
WIDTHS = {
    "none": 0,
    "quat_wxyz": 4,
    "quat_xyzw": 4,
    "rot6d": 6,
    "axis_angle": 3,
    "euler_xyz": 3,
}

KNOWN = frozenset(WIDTHS) - {"none"}


# --------------------------------------------------------------------------
# everything goes through a rotation matrix
# --------------------------------------------------------------------------


def _canonical(quaternion: np.ndarray) -> np.ndarray:
    """Pick the representative with a non-negative scalar part.

    ``q`` and ``-q`` name the same rotation. Without this, a round trip can
    return the antipode — correct as a rotation, and a sign flip to anything
    comparing numbers.
    """
    flip = quaternion[:, 0] < 0
    return np.where(flip[:, None], -quaternion, quaternion)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Shepperd's method: pick the branch with the largest denominator.

    The naive formula divides by something near zero for rotations close to a
    half turn, so the branch is chosen per sample rather than globally.
    """
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    out = np.empty((len(m), 4))
    for index in range(len(m)):
        r = m[index]
        if trace[index] > 0:
            s = np.sqrt(trace[index] + 1.0) * 2
            out[index] = [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            out[index] = [
                (r[2, 1] - r[1, 2]) / s, 0.25 * s,
                (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s,
            ]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            out[index] = [
                (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                0.25 * s, (r[1, 2] + r[2, 1]) / s,
            ]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            out[index] = [
                (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s, 0.25 * s,
            ]
    return _canonical(out)


def to_matrix(values: np.ndarray, encoding: str) -> np.ndarray:
    """Any encoding to a rotation matrix."""
    values = np.asarray(values, dtype=float)
    if encoding == "quat_wxyz":
        return _quat_to_matrix(_canonical(values))
    if encoding == "quat_xyzw":
        return _quat_to_matrix(_canonical(values[:, [3, 0, 1, 2]]))
    if encoding == "axis_angle":
        angle = np.linalg.norm(values, axis=1, keepdims=True)
        axis = np.divide(values, angle, out=np.zeros_like(values), where=angle > 1e-12)
        half = angle[:, 0] / 2.0
        quaternion = np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)
        return _quat_to_matrix(_canonical(quaternion))
    if encoding == "euler_xyz":
        cos, sin = np.cos(values), np.sin(values)
        matrices = np.empty((len(values), 3, 3))
        for index in range(len(values)):
            cx, cy, cz = cos[index]
            sx, sy, sz = sin[index]
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            matrices[index] = rz @ ry @ rx
        return matrices
    if encoding == "rot6d":
        a, b = values[:, :3], values[:, 3:6]
        e1 = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b - (e1 * b).sum(axis=1, keepdims=True) * e1
        e2 = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        e3 = np.cross(e1, e2)
        return np.stack([e1, e2, e3], axis=2)
    raise ValueError(f"unknown rotation encoding {encoding!r}")


def from_matrix(matrices: np.ndarray, encoding: str) -> np.ndarray:
    """A rotation matrix to any encoding."""
    if encoding == "quat_wxyz":
        return _matrix_to_quat(matrices)
    if encoding == "quat_xyzw":
        q = _matrix_to_quat(matrices)
        return q[:, [1, 2, 3, 0]]
    if encoding == "axis_angle":
        q = _matrix_to_quat(matrices)
        angle = 2.0 * np.arccos(np.clip(q[:, 0], -1.0, 1.0))
        sin_half = np.sqrt(np.clip(1.0 - q[:, 0] ** 2, 0.0, 1.0))
        axis = np.divide(
            q[:, 1:], sin_half[:, None], out=np.zeros_like(q[:, 1:]), where=sin_half[:, None] > 1e-12
        )
        return axis * angle[:, None]
    if encoding == "euler_xyz":
        sy = -matrices[:, 2, 0]
        cy = np.sqrt(np.clip(1.0 - sy**2, 0.0, 1.0))
        y = np.arctan2(sy, cy)
        x = np.arctan2(matrices[:, 2, 1], matrices[:, 2, 2])
        z = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
        return np.stack([x, y, z], axis=1)
    if encoding == "rot6d":
        return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)
    raise ValueError(f"unknown rotation encoding {encoding!r}")


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------


#: Where the rotation block starts, when a channel says. Needed because a
#: channel is ``[prefix][rotation][suffix]`` and the width alone does not say how
#: the non-rotation numbers are split around it.
OFFSET_KEY = "rotation_offset"


def _encoding(spec: ChannelSpec) -> str | None:
    value = spec.metadata.get(KEY)
    return str(value) if value is not None else None


def _offset(spec: ChannelSpec, encoding: str) -> int | None:
    """Where the rotation starts, or ``None`` if it cannot be known.

    Declared wins. Otherwise there is exactly one unambiguous case — the whole
    channel is the rotation — and one conventional one: a pose is a position
    followed by a rotation, so the block starts at three. Anything else is a
    guess about someone's layout, and guessing here reinterprets whichever three
    numbers happen to come first as a position.
    """
    declared = spec.metadata.get(OFFSET_KEY)
    if declared is not None:
        return int(declared)
    width = WIDTHS[encoding]
    if spec.width == width:
        return 0
    if spec.width >= 3 + width and str(spec.semantics or "").endswith("pose"):
        return 3
    return None


def _guard(source: ChannelSpec, target: ChannelSpec) -> Verdict:
    mine, theirs = _encoding(source), _encoding(target)
    if mine is None or theirs is None:
        return Verdict.no(
            "adapter.rotation_undeclared",
            f"cannot convert a rotation that is not declared on both sides "
            f"({source.name!r}={mine!r}, {target.name!r}={theirs!r})",
        )
    unknown = [value for value in (mine, theirs) if value not in KNOWN]
    if unknown:
        return Verdict.no(
            "adapter.rotation_unknown",
            f"unknown rotation encoding(s) {unknown}; known: {sorted(KNOWN)}",
        )
    if source.width - WIDTHS[mine] != target.width - WIDTHS[theirs]:
        return Verdict.no(
            "adapter.rotation_layout",
            f"{source.name!r} has {source.width - WIDTHS[mine]} non-rotation dimension(s) "
            f"and {target.name!r} has {target.width - WIDTHS[theirs]}; "
            "only the rotation block is converted, so the rest must line up",
        )
    here, there = _offset(source, mine), _offset(target, theirs)
    if here is None or there is None:
        unknown = source.name if here is None else target.name
        return Verdict.no(
            "adapter.rotation_offset",
            f"cannot tell where the rotation starts in {unknown!r}",
            hint=f"set {OFFSET_KEY!r} in the channel's metadata; a width alone does "
            "not say how the other numbers sit around the rotation",
        )
    if here != there:
        return Verdict.no(
            "adapter.rotation_layout",
            f"the rotation starts at {here} in {source.name!r} and {there} in "
            f"{target.name!r}; only the block itself is converted",
        )
    return Verdict.yes()


def convert(values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
    """Re-encode the rotation block, leaving position and gripper alone."""
    mine, theirs = _encoding(source), _encoding(target)
    if mine is None or theirs is None:
        raise ValueError("both channels must declare a rotation encoding")
    data = np.asarray(values, dtype=float)
    width = WIDTHS[mine]
    start = _offset(source, mine)
    if start is None:
        raise ValueError(f"cannot tell where the rotation starts in {source.name!r}")
    if data.shape[1] < start + width:
        raise ValueError(
            f"{source.name!r} is {data.shape[1]} wide, too narrow for a {mine} rotation "
            f"starting at {start}"
        )
    before, block, after = (
        data[:, :start],
        data[:, start : start + width],
        data[:, start + width :],
    )
    rotated = from_matrix(to_matrix(block, mine), theirs)
    return np.concatenate([before, rotated, after], axis=1)


ROTATION = Adapter(
    name="rotation",
    version=VERSION,
    closes=("metadata.mismatch", "shape.mismatch"),
    guard=_guard,
    # Exact up to floating point in every direction, and the round trip is
    # tested. An empty loss list here is a claim, and it is checked.
    cost=lambda source, target: (),
    transform=convert,
    metadata={"reversible": True, "key": KEY},
)
