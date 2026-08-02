"""Convert the rotation part of a pose between encodings.

Five encodings are in common use and they are mutually unreadable: a quaternion
scalar-first and one scalar-last are both four floats, axis-angle and Euler are
both three, and a channel described only by its width cannot tell you which it
holds. That is why this exists at all -- the distinction is invisible to every
structural check, and executing one encoding as another produces motion that
looks plausible and points somewhere else.

A pose channel here is a position followed by a rotation, optionally followed by
a gripper: ``[x, y, z, <rotation>, gripper?]``. Only the rotation block is
touched; the position and anything after it pass through untouched.

More than one arm, more than one rotation
-----------------------------------------
A two-armed command is two of those laid end to end, so it holds *two* rotation
blocks and they do not sit at the same offset on both sides of a conversion: a
right-arm Euler block starts at ten where the quaternion one starts at eleven.
Converting only the first would leave the second arm's three numbers in place
and read as the first three of a quaternion -- a well-formed vector, a plausible
trajectory, and the wrong arm pointing somewhere arbitrary.

So a channel may declare several blocks, and what has to agree between the two
sides is not the offsets but the *gaps between them*: the runs of position and
gripper numbers must be the same lengths in the same order. That is the real
invariant, and it holds for one arm as much as for two.

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
    return the antipode -- correct as a rotation, and a sign flip to anything
    comparing numbers.
    """
    flip = quaternion[:, 0] < 0
    return np.where(flip[:, None], -quaternion, quaternion)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
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
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            out[index] = [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            out[index] = [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
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
            q[:, 1:],
            sin_half[:, None],
            out=np.zeros_like(q[:, 1:]),
            where=sin_half[:, None] > 1e-12,
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
#: the non-rotation numbers are split around it. An integer for one block, or a
#: sequence of them for a channel that carries several -- one per arm.
OFFSET_KEY = "rotation_offset"

#: How many arms a command drives, when the channel says. Only ever used to
#: *check* a declared block count, never to infer one: two arms and two rotations
#: is the common case but a channel may hold a wrist rotation as well, and
#: inferring here would silently pick a layout.
ARMS_KEY = "arms"


def _encoding(spec: ChannelSpec) -> str | None:
    value = spec.metadata.get(KEY)
    return str(value) if value is not None else None


def _offsets(spec: ChannelSpec, encoding: str) -> tuple[int, ...] | None:
    """Where each rotation starts, or ``None`` if it cannot be known.

    Declared wins. Otherwise there is exactly one unambiguous case -- the whole
    channel is the rotation -- and one conventional one: a pose is a position
    followed by a rotation, so the block starts at three. Anything else is a
    guess about someone's layout, and guessing here reinterprets whichever three
    numbers happen to come first as a position.

    A two-armed channel is never inferred. Its width is consistent with one
    rotation and a long tail of other numbers just as much as with two, and the
    two readings disagree about the second arm.
    """
    declared = spec.metadata.get(OFFSET_KEY)
    if declared is not None:
        if isinstance(declared, (int, np.integer)) and not isinstance(declared, bool):
            return (int(declared),)
        return tuple(int(value) for value in declared)
    width = WIDTHS[encoding]
    if spec.width == width:
        return (0,)
    if spec.width >= 3 + width and str(spec.semantics or "").endswith("pose"):
        return (3,)
    return None


def _segments(spec: ChannelSpec, offsets: tuple[int, ...], block: int) -> tuple[int, ...] | None:
    """The runs of non-rotation numbers around the blocks, in order.

    This is what has to match between two channels. The offsets themselves
    cannot: an Euler block and the quaternion block replacing it have different
    widths, so everything after the first one shifts.

    ``None`` when the declared offsets do not describe a partition -- overlapping
    blocks, or one running off the end.
    """
    cursor, gaps = 0, []
    for start in offsets:
        if start < cursor or start + block > spec.width:
            return None
        gaps.append(start - cursor)
        cursor = start + block
    gaps.append(spec.width - cursor)
    return tuple(gaps)


def _outside(spec: ChannelSpec, offsets: tuple[int, ...], block: int) -> tuple[str, ...] | None:
    """Dimension names that are not part of any rotation block."""
    if not spec.dim_labels:
        return None
    inside = {index for start in offsets for index in range(start, start + block)}
    return tuple(str(label) for index, label in enumerate(spec.dim_labels) if index not in inside)


def _labels_agree(
    source: ChannelSpec,
    target: ChannelSpec,
    here: tuple[int, ...],
    there: tuple[int, ...],
    mine: int,
    theirs: int,
) -> bool:
    """Whether the non-rotation names match.

    Re-encoding a rotation renames its own block -- ``rx, ry, rz`` becomes
    ``qw, qx, qy, qz`` -- and that is the adapter's own doing. Everything else
    must still line up, because a disagreement out there is a different layout
    rather than a different encoding. Undeclared on either side is not a
    disagreement; it is the absence of a claim, and the spine reports that
    separately.
    """
    ours = _outside(source, here, mine)
    theirs_labels = _outside(target, there, theirs)
    if ours is None or theirs_labels is None:
        return True
    return ours == theirs_labels


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
    here, there = _offsets(source, mine), _offsets(target, theirs)
    if here is None or there is None:
        unknown = source.name if here is None else target.name
        return Verdict.no(
            "adapter.rotation_offset",
            f"cannot tell where the rotation starts in {unknown!r}",
            hint=f"set {OFFSET_KEY!r} in the channel's metadata; a width alone does "
            "not say how the other numbers sit around the rotation, and a two-armed "
            "command holds one rotation per arm",
        )
    if len(here) != len(there):
        return Verdict.no(
            "adapter.rotation_blocks",
            f"{source.name!r} declares {len(here)} rotation block(s) and "
            f"{target.name!r} declares {len(there)}; converting between them would "
            "leave a rotation unconverted and read as part of the next one",
        )
    for spec, offsets, encoding in ((source, here, mine), (target, there, theirs)):
        arms = spec.metadata.get(ARMS_KEY)
        if arms is not None and int(arms) != len(offsets):
            return Verdict.no(
                "adapter.rotation_blocks",
                f"{spec.name!r} drives {int(arms)} arm(s) but declares {len(offsets)} "
                f"rotation block(s); one of the two is wrong about the layout",
            )
    mine_gaps = _segments(source, here, WIDTHS[mine])
    their_gaps = _segments(target, there, WIDTHS[theirs])
    if mine_gaps is None or their_gaps is None:
        broken = source.name if mine_gaps is None else target.name
        return Verdict.no(
            "adapter.rotation_offset",
            f"the rotation blocks declared on {broken!r} overlap or run past its "
            f"{(source if mine_gaps is None else target).width} dimensions",
        )
    if not _labels_agree(source, target, here, there, WIDTHS[mine], WIDTHS[theirs]):
        return Verdict.no(
            "adapter.rotation_labels",
            f"the numbers around the rotations are named differently in "
            f"{source.name!r} and {target.name!r}. Re-encoding a rotation renames "
            "its own block and nothing else, so a disagreement outside the blocks "
            "is a different layout -- most often the arms in the other order, "
            "which produces valid commands sent to the wrong arm",
        )
    if mine_gaps != their_gaps:
        return Verdict.no(
            "adapter.rotation_layout",
            f"the numbers around the rotations run {mine_gaps} in {source.name!r} and "
            f"{their_gaps} in {target.name!r}; only the rotation blocks are converted, "
            "so the positions and grippers between them must line up",
        )
    return Verdict.yes()


def convert(values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
    """Re-encode every rotation block, leaving positions and grippers alone."""
    mine, theirs = _encoding(source), _encoding(target)
    if mine is None or theirs is None:
        raise ValueError("both channels must declare a rotation encoding")
    data = np.asarray(values, dtype=float)
    width = WIDTHS[mine]
    offsets = _offsets(source, mine)
    if offsets is None:
        raise ValueError(f"cannot tell where the rotation starts in {source.name!r}")
    if data.shape[1] != source.width:
        raise ValueError(
            f"{data.shape[1]} values do not fill {source.name!r}'s {source.width} dimensions"
        )
    if _segments(source, offsets, width) is None:
        raise ValueError(
            f"the rotation blocks declared on {source.name!r} overlap or run past its "
            f"{source.width} dimensions"
        )
    pieces, cursor = [], 0
    for start in offsets:
        pieces.append(data[:, cursor:start])
        pieces.append(from_matrix(to_matrix(data[:, start : start + width], mine), theirs))
        cursor = start + width
    pieces.append(data[:, cursor:])
    return np.concatenate(pieces, axis=1)


ROTATION = Adapter(
    name="rotation",
    version=VERSION,
    closes=("metadata.mismatch", "shape.mismatch", "dim_labels.mismatch"),
    guard=_guard,
    # Exact up to floating point in every direction, and the round trip is
    # tested. An empty loss list here is a claim, and it is checked.
    cost=lambda source, target: (),
    transform=convert,
    metadata={"reversible": True, "key": KEY},
)
