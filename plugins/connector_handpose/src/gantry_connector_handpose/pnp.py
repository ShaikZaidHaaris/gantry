"""Metric hand pose from one camera, using the hand as its own ruler.

The problem this solves is the one the first real ego video exposed. A monocular
estimator returns two things that look similar and are not: image landmarks,
which are pixel fractions, and world landmarks, which are real metres but centred
on the hand -- they say what shape the hand is and nothing about where it was.
Neither is a trajectory, and multiplying the first by a hand span produces a
smooth, confident path inside a box the size of a hand.

But together they are enough. A set of 3D points whose true metric size is known,
and their 2D projections in an image, determine the object's pose relative to the
camera. That is perspective-n-point, it is ninety years old, and it is exact up to
the accuracy of the two inputs. The hand's own measured size is the ruler that
recovers the missing scale, so nothing has to be assumed about the person or
supplied by them.

Why this rather than the state of the art
-----------------------------------------
HaWoR and WiLoR reconstruct hands better than this does, and both are
**CC-BY-NC-ND and require MANO**, which is registration-gated and research-only.
A dataset produced through them is encumbered, and a company selling data
evaluation would discover that at legal review rather than at build time.

This path is MediaPipe (Apache-2.0) and OpenCV (Apache-2.0) and nothing else. It
is less accurate and it is unencumbered, and which of those matters more is a
business decision rather than an engineering one -- so the estimator is pluggable,
every estimator declares its licence, and the licence travels onto the derived
dataset. Swapping in HaWoR is one argument; what changes with it is recorded.

Intrinsics are required and never guessed
-----------------------------------------
PnP needs the camera's focal length. Get it wrong by 20% and every distance is
wrong by 20% -- uniformly, smoothly, with no symptom. So :class:`Intrinsics` has
to be constructed deliberately, it records how it was obtained, and a caller who
only knows the field of view says so with :meth:`Intrinsics.from_fov` rather than
having a plausible default chosen for them.

What this still does not give you
---------------------------------
The camera moves. These poses are in the *camera* frame, and for an ego video the
camera is on a head that turns. That is genuinely fine for the ego-to-robot case
where the robot's camera sits where the person's eyes were -- the relationship
being learned is hand-relative-to-viewpoint in both. It is not fine if a
world-frame trajectory is wanted, and for that the camera's own motion has to
come from somewhere else. The frame is declared as ``camera`` so nothing
downstream can mistake one for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gantry.errors import ConfigError

#: Where the intrinsics came from. Recorded because "measured with a checkerboard"
#: and "guessed from the spec sheet" carry different error, and a distance is
#: only as good as the focal length behind it.
SOURCES = ("calibrated", "fov", "exif", "declared")

#: Reprojection error above which a solve is not trusted, as a **fraction of the
#: hand's own extent in the image**.
#:
#: Relative rather than absolute, because an absolute pixel budget is a different
#: standard at every distance: ten pixels is loose on a hand filling the frame
#: and impossible on one across the room. Measured on real footage, two good
#: detectors disagree with each other about where a given knuckle is by roughly
#: 8% of the hand's span -- so demanding a fit tighter than that is demanding
#: better than the inputs support, and it showed: at a fixed 10 px this rejected
#: 109 of 120 frames whose poses were perfectly plausible.
MAX_REPROJECTION = 0.15

#: A floor in pixels, for a hand so small in frame that a fraction of it is
#: sub-pixel and the criterion stops meaning anything.
MIN_REPROJECTION_PX = 6.0

#: How far from the camera a hand can possibly be, in metres.

#:
#: Not a nicety. A degenerate point set -- a hand edge-on, an averaged template
#: that came out planar -- makes the fallback solvers return a pose that
#: reprojects beautifully and sits nine trillion metres away, because at that
#: distance the projection is essentially unchanged by anything. The reprojection
#: check cannot catch it; only a statement about where hands actually are can.
#: This is the same bound the depth path applies, for the same reason.
REACHABLE = (0.05, 3.0)


@dataclass(frozen=True)
class Intrinsics:
    """A camera's focal length and principal point, in pixels.

    Deliberately awkward to construct by accident. Every distance this module
    produces scales linearly with ``fx``, so a focal length that was assumed
    rather than measured makes every number in the dataset wrong by the same
    unknown factor -- which is exactly the class of error the whole ego pipeline
    has already been caught by once.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "declared"
    note: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ConfigError(f"unknown intrinsics source {self.source!r}; {list(SOURCES)}")
        for name in ("fx", "fy"):
            if float(getattr(self, name)) <= 0:
                raise ConfigError(f"{name} must be positive")
        if not (0 < self.cx < self.width and 0 < self.cy < self.height):
            raise ConfigError(
                f"principal point ({self.cx}, {self.cy}) is outside a "
                f"{self.width}x{self.height} frame"
            )

    @classmethod
    def from_fov(
        cls,
        *,
        width: int,
        height: int,
        horizontal_fov_deg: float,
        note: str = "",
    ) -> "Intrinsics":
        """From a stated horizontal field of view, with the assumption recorded.

        The honest fallback when a camera has not been calibrated: a manufacturer's
        FOV figure is usually within a few percent, which is a few percent on every
        distance. Named separately from the calibrated path so a report can say
        which was used.
        """
        if not 10.0 < float(horizontal_fov_deg) < 180.0:
            raise ConfigError(
                f"a horizontal field of view of {horizontal_fov_deg} degrees is not "
                "a camera; expected something between 10 and 180"
            )
        focal = (width / 2.0) / np.tan(np.deg2rad(float(horizontal_fov_deg)) / 2.0)
        return cls(
            fx=float(focal),
            fy=float(focal),
            cx=width / 2.0,
            cy=height / 2.0,
            width=int(width),
            height=int(height),
            source="fov",
            note=note or f"assumed {horizontal_fov_deg:g} deg horizontal FOV, square pixels",
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fx": round(float(self.fx), 3),
            "fy": round(float(self.fy), 3),
            "cx": round(float(self.cx), 3),
            "cy": round(float(self.cy), 3),
            "resolution": [int(self.width), int(self.height)],
            "intrinsics_source": self.source,
            "intrinsics_note": self.note,
        }


@dataclass(frozen=True)
class Pose:
    """One hand's metric pose in the camera frame, and how much to believe it."""

    #: Wrist position in metres, camera frame.
    position: np.ndarray
    #: Hand orientation as a 3x3 rotation, camera frame.
    rotation: np.ndarray
    #: Mean reprojection error in pixels. The honest confidence signal -- a solve
    #: that cannot reproduce the 2D points it was given did not converge on the
    #: right pose, whatever the detector's own score said.
    reprojection: float
    ok: bool


def extent(points: np.ndarray) -> float:
    """How big the hand is in the image, in pixels.

    The scale everything else is judged against. Taken as the bounding-box
    diagonal of the detected keypoints, which is stable under rotation in a way
    that width or height alone is not.
    """
    pixels = np.asarray(points, dtype=float).reshape(-1, 2)
    live = pixels[np.any(pixels != 0.0, axis=1)]
    if len(live) < 2:
        return 0.0
    span = live.max(axis=0) - live.min(axis=0)
    return float(np.hypot(*span))


def solve(
    world: np.ndarray,
    image: np.ndarray,
    intrinsics: Intrinsics,
    *,
    max_reprojection: float = MAX_REPROJECTION,
    robust: bool = True,
) -> Pose:
    """Metric hand pose from metric hand shape plus its projection.

    ``world`` is ``(joints, 3)`` in metres, hand-centred. ``image`` is
    ``(joints, 2)`` in pixels. Returns the wrist's position and the hand's
    orientation in the camera's frame.

    ``robust`` fits through RANSAC, which matters more than it sounds. The 3D
    hand is rigid and a real hand is not, so when the fingers are curled and the
    model is flat, a handful of joints disagree badly while the palm agrees
    perfectly. A least-squares fit is dragged around by those few; RANSAC finds
    the consistent majority and reports the rest as outliers, which is the
    correct reading -- those joints genuinely are not where the model says.
    """
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - needs the extra
        raise ConfigError(
            "recovering metric pose needs OpenCV: pip install 'gantry-connector-handpose[pnp]'"
        ) from error

    obj = np.ascontiguousarray(np.asarray(world, dtype=np.float64).reshape(-1, 3))
    pts = np.ascontiguousarray(np.asarray(image, dtype=np.float64).reshape(-1, 2))
    if len(obj) != len(pts) or len(obj) < 4:
        raise ConfigError(
            f"perspective-n-point needs at least four matched points; got "
            f"{len(obj)} 3D and {len(pts)} 2D"
        )
    if not np.any(obj):
        # An all-zero hand is a frame where nothing was found. Say so rather than
        # letting the solver return an arbitrary pose for a degenerate input.
        return Pose(np.zeros(3), np.eye(3), float("inf"), False)

    # The budget in pixels, from the hand's own size in this frame.
    span = extent(pts)
    budget = (
        max(MIN_REPROJECTION_PX, float(max_reprojection) * span) if span else MIN_REPROJECTION_PX
    )

    # SQPNP is the best of these and it asserts outright on a degenerate point
    # set -- a hand seen edge-on, or an averaged template that came out nearly
    # planar. That is one frame's problem, not the clip's, so it falls back
    # rather than taking the whole episode down with it.
    ok, rvec, tvec = False, None, None
    inliers = None
    if robust and len(obj) >= 6:
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                pts,
                intrinsics.matrix,
                None,
                reprojectionError=budget,
                iterationsCount=200,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            ok = False
    if not ok:
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec = cv2.solvePnP(obj, pts, intrinsics.matrix, None, flags=flag)
            except cv2.error:
                continue
            if ok:
                break
    if not ok or rvec is None:
        return Pose(np.zeros(3), np.eye(3), float("inf"), False)

    projected, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.matrix, None)
    per_point = np.linalg.norm(projected.reshape(-1, 2) - pts, axis=1)
    # Scored on the joints the fit actually claims to explain. A rigid model of a
    # bending hand will always have a few that it does not, and averaging those
    # in judges the fit by the parts it never fitted.
    if inliers is not None and len(inliers) >= 4:
        error = float(per_point[np.asarray(inliers).reshape(-1)].mean())
    else:
        error = float(per_point.mean())
    rotation, _ = cv2.Rodrigues(rvec)
    # The object points are hand-centred, so the wrist sits at the object's own
    # first joint carried through the same transform.
    wrist = (rotation @ obj[0].reshape(3, 1) + tvec).reshape(3)
    distance = float(np.linalg.norm(wrist))
    near, far = REACHABLE
    return Pose(
        position=wrist,
        rotation=np.asarray(rotation, dtype=float),
        reprojection=error,
        # Both conditions, because they catch different failures. Reprojection
        # catches a pose that does not explain the pixels; the distance bound
        # catches one that explains them perfectly from an impossible place.
        ok=error <= budget and near <= distance <= far,
    )


def solve_sequence(
    world: np.ndarray,
    image: np.ndarray,
    intrinsics: Intrinsics,
    *,
    max_reprojection: float = MAX_REPROJECTION,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every step of one hand. Returns ``(positions, rotations, reprojection)``.

    Frames whose solve failed keep their position at the origin and are reported
    with an infinite reprojection error, rather than being interpolated over. A
    gap that is visible is a gap somebody can decide about; one that has been
    filled in silently is a gap nobody can find.
    """
    steps = len(world)
    positions = np.zeros((steps, 3), dtype="float32")
    rotations = np.tile(np.eye(3, dtype="float32"), (steps, 1, 1))
    errors = np.full(steps, np.inf, dtype="float32")
    for index in range(steps):
        pose = solve(
            world[index],
            image[index],
            intrinsics,
            max_reprojection=max_reprojection,
            robust=robust,
        )
        if not pose.ok:
            continue
        positions[index] = pose.position
        rotations[index] = pose.rotation
        errors[index] = pose.reprojection
    return positions, rotations, errors


def rotations_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """3x3 rotations to wxyz, with a degenerate frame becoming identity."""
    out = np.zeros((len(matrices), 4), dtype="float64")
    out[:, 0] = 1.0
    for index, matrix in enumerate(np.asarray(matrices, dtype=float)):
        if not np.isfinite(matrix).all() or np.allclose(matrix, 0.0):
            continue
        trace = float(np.trace(matrix))
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            out[index] = [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            j, k = (axis + 1) % 3, (axis + 2) % 3
            s = np.sqrt(1.0 + matrix[axis, axis] - matrix[j, j] - matrix[k, k]) * 2.0
            quaternion = np.zeros(4)
            quaternion[0] = (matrix[k, j] - matrix[j, k]) / s
            quaternion[axis + 1] = 0.25 * s
            quaternion[j + 1] = (matrix[j, axis] + matrix[axis, j]) / s
            quaternion[k + 1] = (matrix[k, axis] + matrix[axis, k]) / s
            out[index] = quaternion
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 1e-9).astype("float32")


def plausible(positions: np.ndarray, *, near: float = 0.10, far: float = 1.20) -> float:
    """Fraction of solved positions at a distance a working hand could be at.

    A blunt sanity check with a specific job: catching a wrong focal length. Every
    distance scales with it, so a 2x error puts the whole trajectory at two metres
 -- still smooth, still consistent, and obviously wrong to anybody who looks.
    Nothing else in the pipeline would notice.
    """
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    solved = np.any(points != 0.0, axis=1)
    if not solved.any():
        return 0.0
    distance = np.linalg.norm(points[solved], axis=1)
    return float(((distance >= near) & (distance <= far)).mean())


def intrinsics_from(raw: Intrinsics | Mapping[str, Any] | None) -> Intrinsics | None:
    """Build intrinsics from plain data, for a manifest."""
    if raw is None or isinstance(raw, Intrinsics):
        return raw
    data = dict(raw)
    if "horizontal_fov_deg" in data:
        return Intrinsics.from_fov(**data)
    return Intrinsics(**data)


#: Known ego rigs and their horizontal field of view, for the common case where
#: somebody knows what camera was used and not what its focal length is. A lookup
#: a caller may reach for, never a default applied on their behalf.
RIGS: dict[str, float] = {
    "gopro_hero5_wide": 94.4,
    "gopro_hero5_linear": 64.4,
    "gopro_hero11_wide": 92.0,
    "aria_rgb": 110.0,
    "quest3_passthrough": 100.0,
    "iphone_ultrawide": 106.0,
    "iphone_wide": 69.0,
}


def for_rig(name: str, *, width: int, height: int) -> Intrinsics:
    """Intrinsics for a known rig, with the assumption recorded on the object."""
    try:
        fov = RIGS[name]
    except KeyError:
        raise ConfigError(
            f"unknown rig {name!r}; known: {sorted(RIGS)}. A rig not listed is "
            "described with Intrinsics.from_fov, or calibrated properly"
        ) from None
    return Intrinsics.from_fov(
        width=width,
        height=height,
        horizontal_fov_deg=fov,
        note=f"{name}: manufacturer field of view, not calibrated",
    )
