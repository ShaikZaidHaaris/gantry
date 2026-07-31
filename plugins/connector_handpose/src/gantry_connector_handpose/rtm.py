"""RTMPose for the 2D hand, which is the half MediaPipe is weakest at.

MediaPipe's contribution to this pipeline is two things: 2D keypoints, and a
metric hand model. It is genuinely good at the second — the hand it reconstructs
is a plausible hand at a plausible size, which is what makes the
perspective-n-point path work at all. It is mediocre at the first, especially
under motion blur and partial occlusion, which is most of an ego video: measured
on real kitchen footage it found a hand in 63% and 71% of frames, and the two
hands together in 8%.

RTMPose is a stronger 2D detector, it is Apache-2.0, and — the practical part —
it is reachable through ``rtmlib``, which ships ONNX weights and needs neither
mmcv nor a matching CUDA toolchain. That last point is why this is buildable and
HaWoR was not.

What it does not give you
-------------------------
3D. RTMPose returns pixels and confidences, so on its own it cannot place a hand
in space at all. It has to be paired with something that supplies the third
dimension:

``rtmpose + depth``  — a metric depth model reads the distance directly. Best
when the depth model is metric and permissively licensed; see :mod:`.depth`.

``rtmpose + mediapipe`` — RTMPose finds the hand accurately in 2D, MediaPipe's
world landmarks supply the metric hand shape, and perspective-n-point does the
rest. This is the combination that improves the existing path without adding a
depth model, and it is the one to reach for first.

The point of the module is that these are compositions rather than forks. Each
piece declares what it produces, and the estimator that combines them declares
the union — so a report can say which detector found the hand and which model
supplied the scale, separately, because they are separately wrong.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gantry.errors import ConfigError

from .handpose import HANDS, Track
from .pnp import MAX_REPROJECTION, Intrinsics, rotations_to_quaternions, solve_sequence

#: What rtmlib exposes that is worth pointing at a hand. Whole-body gives 133
#: keypoints including 21 per hand, which is more robust in ego video than a
#: hand-only model because the arm gives context when the hand is half out of
#: frame.
BACKENDS = ("wholebody", "hand")

#: Where the 21 hand keypoints sit inside COCO-WholeBody's 133.
WHOLEBODY_LEFT = slice(91, 112)
WHOLEBODY_RIGHT = slice(112, 133)

#: Below this the keypoint is a guess and is treated as absent.
CONFIDENT = 0.3

#: How close two hands may be, as a fraction of a hand's own width, before at
#: most one of them is real.
#:
#: This exists because of what a whole-body model does in egocentric video, and
#: it is worth stating plainly: COCO-WholeBody assumes a third-person view of a
#: whole person. Ego footage has no body and no face — just hands, close up — so
#: the model fits its 133-point skeleton to whatever it can and puts *both* hand
#: sets on the one visible hand. Measured on real EPIC footage: in 54% of frames
#: the two predicted hands were less than half a hand-width apart, and in 28%
#: they were on top of each other, while a proper handedness classifier saw one
#: hand in 59 frames and two in 6.
#:
#: Two hands cannot occupy the same pixels. That is not a heuristic, it is a fact
#: about hands, and it is the cheapest reliable way to catch this.
APART = 0.5


def centroid(points: np.ndarray) -> np.ndarray:
    """Middle of the detected keypoints, ignoring the ones that are absent."""
    live = np.asarray(points, dtype=float).reshape(-1, 2)
    live = live[np.any(live != 0.0, axis=1)]
    return live.mean(axis=0) if len(live) else np.array([np.nan, np.nan])


def separate(
    found: Mapping[str, tuple[np.ndarray, np.ndarray]], *, apart: float = APART
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Drop the weaker of two hands predicted in the same place.

    Whole-body models hallucinate the hand that is not there, on top of the one
    that is. Where the two predictions sit closer together than :data:`APART`,
    the lower-confidence one is dropped — keeping both would put two skeletons on
    one hand and, worse, feed a phantom trajectory into training as though a
    second arm had been doing something.
    """
    from .pnp import extent

    out = {hand: (points.copy(), scores.copy()) for hand, (points, scores) in found.items()}
    if len(out) < 2:
        return out
    (left_name, (left, left_score)), (right_name, (right, right_score)) = list(out.items())[:2]
    for t in range(len(left)):
        if left_score[t] <= 0 or right_score[t] <= 0:
            continue
        gap = np.linalg.norm(centroid(left[t]) - centroid(right[t]))
        size = max(extent(left[t]), extent(right[t]), 1e-6)
        if np.isfinite(gap) and gap / size < apart:
            loser = right_name if left_score[t] >= right_score[t] else left_name
            out[loser][0][t] = 0.0
            out[loser][1][t] = 0.0
    return out


def rtmpose(
    *,
    backend: str = "wholebody",
    device: str = "cuda",
    mode: str = "balanced",
    confident: float = CONFIDENT,
    model: Any = None,
) -> Any:
    """A 2D hand detector, Apache-2.0, through ONNX.

    Returns an object with ``detect(frames) -> (keypoints, scores)`` per hand,
    keypoints in pixels. Deliberately not a full :class:`Estimator` — it cannot
    be, because it produces no 3D — so it composes rather than substitutes.
    """
    if backend not in BACKENDS:
        raise ConfigError(f"unknown rtmlib backend {backend!r}; known: {list(BACKENDS)}")

    class RTMPose:
        licence = "Apache-2.0 (RTMPose / rtmlib)"

        def __init__(self) -> None:
            self._model = model

        @property
        def net(self) -> Any:  # pragma: no cover - needs the weights
            if self._model is None:
                try:
                    import rtmlib
                except ImportError as error:
                    raise ConfigError(
                        "RTMPose needs rtmlib: pip install 'gantry-connector-handpose[rtmpose]'"
                    ) from error
                builder = rtmlib.Wholebody if backend == "wholebody" else rtmlib.Hand
                self._model = builder(mode=mode, backend="onnxruntime", device=device)
            return self._model

        def detect(self, frames: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            points: dict[str, list[np.ndarray]] = {h: [] for h in HANDS}
            scores: dict[str, list[float]] = {h: [] for h in HANDS}
            for frame in np.asarray(frames):
                keypoints, confidence = self.net(np.asarray(frame))
                for hand, span in (
                    ("left", WHOLEBODY_LEFT),
                    ("right", WHOLEBODY_RIGHT),
                ):
                    if len(keypoints) == 0:
                        points[hand].append(np.zeros((21, 2), dtype="float32"))
                        scores[hand].append(0.0)
                        continue
                    person = 0  # the wearer's hands are the largest detection
                    block = np.asarray(keypoints[person])[span]
                    weight = float(np.mean(np.asarray(confidence[person])[span]))
                    points[hand].append(
                        block.astype("float32")
                        if weight >= confident
                        else np.zeros((21, 2), dtype="float32")
                    )
                    scores[hand].append(weight)
            return separate(
                {
                    hand: (
                        np.stack(points[hand]),
                        np.asarray(scores[hand], dtype="float32"),
                    )
                    for hand in HANDS
                }
            )

    return RTMPose()


def template_from(world: np.ndarray, *, minimum: int = 5) -> np.ndarray | None:
    """One rigid hand, averaged over the frames a shape model did solve.

    The fix for a real disappointment: pairing a strong 2D detector with a weak
    3D one gets you the *weak* one's recall, because a frame needs both. On real
    footage RTMPose found hands in every frame and the combination solved 6% of
    them, because MediaPipe supplied the metric shape and MediaPipe is what was
    missing.

    A hand does not change size. So the frames where the shape model worked
    measure *this person's* hand, and that measurement can carry the frames where
    it did not. Per-person rather than a generic template, which matters because
    hand size is exactly the quantity being used as a ruler.

    The cost is articulation: a rigid template cannot bend, so a fisted hand fits
    it badly. That shows up honestly as reprojection error and gets rejected by
    the usual threshold rather than becoming a confident wrong pose.
    """
    frames = np.asarray(world, dtype=float)
    if frames.ndim != 3:
        return None
    usable = np.any(frames.reshape(len(frames), -1) != 0.0, axis=1)
    if usable.sum() < minimum:
        return None
    # Median rather than mean: a badly-fitted frame is an outlier, not noise.
    return np.median(frames[usable], axis=0)


def rtm_with_mediapipe(
    detector: Any,
    shape_source: Any,
    *,
    intrinsics: Intrinsics,
    max_reprojection: float = MAX_REPROJECTION,
    use_template: bool = True,
) -> Any:
    """RTMPose for where the hand is, MediaPipe for how big it is, PnP for the rest.

    The composition that improves the existing metric path without introducing a
    depth model. Each half is used for the thing it is good at, and the resulting
    licence is the union of both — which is still Apache-2.0, which is the point.
    """

    class Combined:
        licence = "Apache-2.0 (RTMPose 2D + MediaPipe hand model + OpenCV solvePnP)"

        def estimate(self, frames: np.ndarray) -> Track:
            found = detector.detect(frames)
            base = shape_source.estimate(frames)
            height, width = np.asarray(frames).shape[1:3]
            # MediaPipe classifies handedness as its actual job; a whole-body
            # model infers it from body context that ego video does not contain.
            # Where the classifier saw only one hand, believe it.
            found = _confirm(found, base, width, height)

            poses: dict[str, np.ndarray] = {}
            residual: dict[str, np.ndarray] = {}
            confidence: dict[str, np.ndarray] = {}
            agreement: dict[str, np.ndarray] = {}
            keypoints: dict[str, np.ndarray] = {}
            templated: dict[str, int] = {}
            for hand in HANDS:
                if hand not in found or hand not in base.world:
                    continue
                pixels, scores = found[hand]
                world = np.asarray(base.world[hand], dtype=float)
                if not world.size:
                    continue
                # RTMPose says where; the shape model says how big. Where the
                # shape model is missing, this person's own averaged hand stands
                # in — otherwise the pair inherits the weaker recall of the two,
                # which measured 6% on real footage against RTMPose's 100%.
                shape = world.copy()
                filled = np.zeros(len(world), dtype=bool)
                if use_template:
                    template = template_from(world)
                    if template is not None:
                        empty = ~np.any(world.reshape(len(world), -1) != 0.0, axis=1)
                        shape[empty] = template
                        filled = empty
                usable = (scores > 0) & np.any(shape.reshape(len(shape), -1), axis=1)
                positions, rotations, errors = solve_sequence(
                    shape, pixels, intrinsics, max_reprojection=max_reprojection
                )
                positions[~usable] = 0.0
                errors[~usable] = np.inf
                templated[hand] = int((filled & usable).sum())
                poses[hand] = np.concatenate(
                    [positions, rotations_to_quaternions(rotations)], axis=1
                ).astype("float32")
                residual[hand] = errors
                # The detector's own rate, because that is what "the detector
                # found a hand" means and what the extraction report reads. The
                # agreement between the two is a different and stricter quantity
                # — reporting it under this name made a 100% detector look like
                # a 49% one, and pointed the fix at the wrong component.
                confidence[hand] = np.asarray(scores, dtype="float32")
                agreement[hand] = np.minimum(
                    scores, np.asarray(base.confidence.get(hand, scores))
                ).astype("float32")
                normalised = pixels.astype("float32").copy()
                normalised[..., 0] /= max(1, width)
                normalised[..., 1] /= max(1, height)
                keypoints[hand] = np.concatenate(
                    [normalised, np.zeros((*normalised.shape[:2], 1), dtype="float32")],
                    axis=-1,
                )
            return Track(
                keypoints=keypoints or base.keypoints,
                world=base.world,
                confidence=confidence or base.confidence,
                convention=base.convention,
                poses=poses,
                residual=residual,
                licence=self.licence,
                metadata={
                    **dict(base.metadata),
                    "detector": "rtmpose",
                    "shape_model": base.metadata.get("estimator", "mediapipe"),
                    "pose_method": "solvePnP(SQPNP) on RTMPose 2D + MediaPipe hand model",
                    "template_frames": {h: int(v) for h, v in templated.items()},
                    **intrinsics.as_dict(),
                },
            )

    return Combined()


def _confirm(
    found: Mapping[str, tuple[np.ndarray, np.ndarray]],
    base: Track,
    width: int,
    height: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Drop an RTMPose hand the handedness classifier did not also see.

    Only where the classifier saw *something* in that frame — if it found no
    hands at all, it has no opinion and RTMPose's detection stands. The point is
    to use a disagreement, not to inherit the weaker model's recall again.
    """
    out = {hand: (points.copy(), scores.copy()) for hand, (points, scores) in found.items()}
    any_seen = None
    seen: dict[str, np.ndarray] = {}
    for hand in out:
        world = np.asarray(base.world.get(hand, ()), dtype=float)
        image = np.asarray(base.keypoints.get(hand, ()), dtype=float)
        source = world if world.size else image
        seen[hand] = (
            np.any(source.reshape(len(source), -1) != 0.0, axis=1)
            if source.size
            else np.zeros(len(next(iter(out.values()))[1]), dtype=bool)
        )
        any_seen = seen[hand] if any_seen is None else (any_seen | seen[hand])
    if any_seen is None:
        return out
    for hand, (points, scores) in out.items():
        # Frames where the classifier looked, saw a hand, and it was not this one.
        contradicted = any_seen & ~seen[hand]
        points[contradicted] = 0.0
        scores[contradicted] = 0.0
    return out


def rtm_with_depth(
    detector: Any,
    depth: Any,
    shape_source: Any,
    *,
    intrinsics: Intrinsics,
) -> Any:
    """RTMPose for where, a metric depth model for how far, MediaPipe for orientation.

    The path that stops distance depending on the focal length. Orientation still
    comes from the hand model, because a depth map says nothing about which way a
    palm faces.
    """
    from .depth import lift

    class Combined:
        @property
        def licence(self) -> str:
            return (
                f"{getattr(detector, 'licence', 'unknown')} + "
                f"{getattr(depth, 'licence', 'unknown')} (depth) + "
                f"{getattr(shape_source, 'licence', 'Apache-2.0')} (hand model)"
            )

        def estimate(self, frames: np.ndarray) -> Track:
            found = detector.detect(frames)
            base = shape_source.estimate(frames)
            maps = depth.estimate(frames)
            height, width = np.asarray(frames).shape[1:3]
            scale_y = maps.values.shape[1] / max(1, height)
            scale_x = maps.values.shape[2] / max(1, width)

            poses: dict[str, np.ndarray] = {}
            residual: dict[str, np.ndarray] = {}
            for hand in HANDS:
                if hand not in found:
                    continue
                pixels, scores = found[hand]
                # Depth maps are often produced at a different resolution than
                # the frame; sampling at frame coordinates would read the wrong
                # pixel entirely.
                scaled = pixels.astype(float).copy()
                scaled[..., 0] *= scale_x
                scaled[..., 1] *= scale_y
                positions, solved = lift(scaled, maps, _scaled(intrinsics, scale_x, scale_y))
                world = np.asarray(base.world.get(hand, ()), dtype=float)
                rotations = (
                    _orientations(world)
                    if world.size
                    else np.tile(np.eye(3), (len(positions), 1, 1))
                )
                positions[~solved] = 0.0
                poses[hand] = np.concatenate(
                    [positions, rotations_to_quaternions(rotations)], axis=1
                ).astype("float32")
                residual[hand] = np.where(solved, 0.0, np.inf).astype("float32")
            return Track(
                keypoints=base.keypoints,
                world=base.world,
                confidence=base.confidence,
                convention=base.convention,
                poses=poses,
                residual=residual,
                licence=self.licence,
                metadata={
                    **dict(base.metadata),
                    "detector": "rtmpose",
                    "depth_model": maps.model,
                    "depth_metric": maps.metric,
                    "pose_method": "metric depth unprojection",
                    **intrinsics.as_dict(),
                },
            )

    return Combined()


def _scaled(intrinsics: Intrinsics, sx: float, sy: float) -> Intrinsics:
    """Intrinsics for a resized image. Focal and centre scale with the image."""
    return Intrinsics(
        fx=intrinsics.fx * sx,
        fy=intrinsics.fy * sy,
        cx=intrinsics.cx * sx,
        cy=intrinsics.cy * sy,
        width=max(1, int(round(intrinsics.width * sx))),
        height=max(1, int(round(intrinsics.height * sy))),
        source=intrinsics.source,
        note=f"{intrinsics.note} (rescaled x{sx:.3f})",
    )


def _orientations(world: np.ndarray) -> np.ndarray:
    """Palm frames from a metric hand, for the frames that have one."""
    from .handpose import wrist_from

    quaternions = wrist_from(np.asarray(world, dtype="float32"))[:, 3:]
    out = np.tile(np.eye(3), (len(quaternions), 1, 1))
    for index, (w, x, y, z) in enumerate(quaternions):
        out[index] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
    return out
