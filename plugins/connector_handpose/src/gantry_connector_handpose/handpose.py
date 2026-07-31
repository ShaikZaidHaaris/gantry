"""The same clips with hands estimated onto them, as a dataset derived from theirs.

Deliberately not a step inside the video connector. That connector's docstring
says why: a reader that quietly estimated hands would make the estimate
indistinguishable from a measurement, and for ego data that is precisely the
distinction everything else turns on. So this is a *different dataset*, derived
from the first, and it says so — ``derived_from`` on every episode, the estimator
named in the descriptor, and every hand channel declaring ``scale="unscaled"``.

The consequence is worth spelling out. Somebody later reading a run's provenance
can tell whether the hands in it were measured by a calibrated rig or guessed by
a monocular network, and can tell *which* network. Those runs are not comparable
and nothing else in the pipeline would have preserved the difference.

Also not a curation module
--------------------------
The curation plane prescribes: drop these episodes, weight that cohort, collect
this much more. Nothing here prescribes anything — it adds channels. Filing it
under curation would have put an enrichment behind an interface built for
proposals, and the ladder, the predictions and the leakage guard would all have
been noise.

What it writes for the report
-----------------------------
Three per-clip numbers that ``feedback_capture`` later turns into advice, and
they are written *here* because this is the only stage that ever looks at every
frame:

``hands_visible`` — the fraction of frames a hand was found in. The single most
expensive property of an ego upload, because a frame with no hand in it cannot
produce a pose and is dropped before anything trains.

``motion_ok`` — the fraction of frames whose wrist speed is something an arm
could follow. Human hands move several times faster than a robot; the fast parts
are simultaneously blurred and unreachable.

``usable_length`` — whether the clip is long enough to contain a whole attempt.

None of these is a judgement. They are counts, and the module that turns them
into sentences is somewhere else on purpose.

Confidence is kept, not thresholded away
----------------------------------------
The estimator's own per-frame confidence travels as a channel. A pipeline that
threw it away and kept only the poses would leave every downstream consumer
unable to distinguish a firmly-tracked hand from a guess — and monocular hand
estimators are confidently wrong often enough that the difference matters more
than the poses themselves in the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from gantry_semantics_ego import aperture_channel, hand_channel, wrist_channel

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ArraySource,
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
)

from .pnp import (
    Intrinsics,
    for_rig,
    intrinsics_from,
    plausible,
    rotations_to_quaternions,
    solve_sequence,
)

VERSION = "0.1.0.dev0"

#: Both hands, always in this order, so a channel name is enough to say which.
HANDS = ("left", "right")

#: What the estimator's confidence has to reach before a frame counts as one a
#: hand was found in. Not a filter — the poses are kept either way — only the
#: threshold for the ``hands_visible`` count.
FOUND = 0.5

#: Wrist speed above which an arm cannot follow, in metres per second of the
#: person's own (unscaled) units. Generous: the point is to catch the parts of a
#: clip where somebody was moving at conversational speed rather than
#: demonstrating.
FAST = 1.5

#: Shorter than this and a clip cannot hold a whole reach-grasp-release.
MIN_STEPS = 15


@dataclass(frozen=True)
class Track:
    """One estimator's output for one clip.

    ``keypoints`` is ``(steps, joints, 3)`` per hand and ``confidence`` is
    ``(steps,)``. Both are required: an estimator that returns poses without
    confidence is asking every downstream consumer to treat a guess and a firm
    detection identically.
    """

    keypoints: Mapping[str, np.ndarray]
    confidence: Mapping[str, np.ndarray]
    #: Hand-centred metric landmarks, where the estimator produces them.
    #:
    #: The distinction between this and ``keypoints`` is the single most
    #: important thing in this file, and getting it wrong produced smooth,
    #: confident, wrong trajectories on the first real video ever run through
    #: here. ``keypoints`` are image-normalised — x and y are fractions of the
    #: frame and z is a relative offset near zero. ``world`` is metres, centred
    #: on the hand itself.
    #:
    #: So the hand's *shape* is metric and its *position in the scene* is not,
    #: and no amount of arithmetic on image coordinates recovers the second.
    #: That is the whole Tier-1 versus Tier-2 distinction, and it is a property
    #: of monocular video rather than of any particular estimator.
    world: Mapping[str, np.ndarray] = field(default_factory=dict)
    #: Which convention the joints are in. Load-bearing — see the ego vocabulary
    #: on why MANO and MediaPipe are both 21 joints and not interchangeable.
    convention: str = "mediapipe"
    #: Metric wrist position in the camera frame, per hand, where the estimator
    #: recovered one. This is the quantity a retargeter actually needs and the
    #: one neither `keypoints` nor `world` is on its own.
    poses: Mapping[str, np.ndarray] = field(default_factory=dict)
    #: Per-step reprojection error in pixels, where poses were solved.
    residual: Mapping[str, np.ndarray] = field(default_factory=dict)
    #: What the weights behind this estimator permit.
    #:
    #: Load-bearing rather than decorative. The best monocular hand
    #: reconstructions available — HaWoR, WiLoR, anything built on MANO — are
    #: CC-BY-NC-ND and registration-gated, so a dataset produced through one is
    #: encumbered. That is discovered at legal review rather than at build time
    #: unless something carries it, and this is the something: it propagates onto
    #: every derived episode.
    licence: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [hand for hand in self.keypoints if hand not in self.confidence]
        if missing:
            raise ConfigError(
                f"the estimator returned poses for {missing} with no confidence. A "
                "consumer cannot then tell a firmly-tracked hand from a guess, and "
                "monocular estimators are confidently wrong often enough that the "
                "difference matters more than the poses in the tail"
            )


class Estimator(Protocol):
    """Whatever finds hands in frames."""

    def estimate(self, frames: np.ndarray) -> Track:
        """``(steps, h, w, 3)`` uint8 in, one :class:`Track` out."""


def mediapipe(model_path: str | None = None, **options: Any) -> Estimator:
    """MediaPipe's hand landmarker. Imported here, not at module load.

    One worked estimator so that the pipeline runs end to end without anybody
    writing an adapter first. It is not privileged: ``Estimator`` is two lines,
    and a lab with its own tracker passes theirs in.
    """

    class MediaPipe:
        def estimate(self, frames: np.ndarray) -> Track:  # pragma: no cover - needs the model
            try:
                import mediapipe as mp
            except ImportError as error:
                raise ConfigError(
                    "estimating hands needs a tracker: pip install "
                    "'gantry-connector-handpose[mediapipe]', or pass your own "
                    "estimator — the interface is one method"
                ) from error
            from mediapipe.tasks.python import vision

            landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                    num_hands=2,
                    **options,
                )
            )
            points = {hand: [] for hand in HANDS}
            world = {hand: [] for hand in HANDS}
            scores = {hand: [] for hand in HANDS}
            for frame in frames:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))
                result = landmarker.detect(image)
                found = {
                    handedness[0].category_name.lower(): index
                    for index, handedness in enumerate(result.handedness or [])
                }
                for hand in HANDS:
                    index = found.get(hand)
                    if index is None:
                        points[hand].append(np.zeros((21, 3), dtype="float32"))
                        world[hand].append(np.zeros((21, 3), dtype="float32"))
                        scores[hand].append(0.0)
                        continue
                    marks = result.hand_landmarks[index]
                    points[hand].append(
                        np.asarray([[m.x, m.y, m.z] for m in marks], dtype="float32")
                    )
                    metric = (getattr(result, "hand_world_landmarks", None) or [None])[
                        index if result.hand_world_landmarks else 0
                    ]
                    world[hand].append(
                        np.asarray([[m.x, m.y, m.z] for m in metric], dtype="float32")
                        if metric is not None
                        else np.zeros((21, 3), dtype="float32")
                    )
                    scores[hand].append(float(result.handedness[index][0].score))
            return Track(
                keypoints={hand: np.stack(points[hand]) for hand in HANDS},
                world={hand: np.stack(world[hand]) for hand in HANDS},
                confidence={hand: np.asarray(scores[hand], dtype="float32") for hand in HANDS},
                convention="mediapipe",
                metadata={"estimator": "mediapipe.HandLandmarker"},
            )

    return MediaPipe()


# -- turning keypoints into the things a retargeter reads ---------------------


def wrist_from(keypoints: np.ndarray, *, convention: str = "mediapipe") -> np.ndarray:
    """Wrist position and an orientation, as ``(steps, 7)`` position + wxyz.

    The orientation is built from the palm rather than reported: the wrist joint
    on its own is a point, and the frame it implies comes from where the fingers
    are. Index-knuckle-minus-wrist gives one axis, the cross with
    pinky-knuckle-minus-wrist gives another, and the third follows.

    Crude and standard. What matters is that it is written down here rather than
    happening implicitly inside an estimator wrapper, because a different choice
    of axes is a fixed rotation offset on every pose — exactly the error the
    retargeter's mount is there to absorb, and it can only absorb it if somebody
    knows to measure it.
    """
    points = np.asarray(keypoints, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ConfigError(f"expected (steps, joints, 3) keypoints, got {points.shape}")
    index_knuckle, pinky_knuckle = _knuckles(convention, points.shape[1])
    wrist = points[:, 0, :]

    forward = points[:, index_knuckle, :] - wrist
    sideways = points[:, pinky_knuckle, :] - wrist
    x = _unit(forward)
    z = _unit(np.cross(x, sideways))
    y = np.cross(z, x)
    matrices = np.stack([x, y, z], axis=2)
    return np.concatenate([wrist, _quaternions(matrices)], axis=1).astype("float32")


def aperture_from(keypoints: np.ndarray, *, convention: str = "mediapipe") -> np.ndarray:
    """Thumb tip to index tip, per step — what becomes a gripper command."""
    points = np.asarray(keypoints, dtype=float)
    thumb, index = _tips(convention, points.shape[1])
    return np.linalg.norm(points[:, thumb, :] - points[:, index, :], axis=1).astype("float32")


#: Joint indices per convention: (index knuckle, pinky knuckle, thumb tip, index
#: tip). Written out because this is exactly the table that gets silently wrong —
#: MANO and MediaPipe are both 21 joints in different orders.
JOINTS = {
    "mediapipe": (5, 17, 4, 8),
    "mano": (1, 13, 4, 8),
    "openpose": (5, 17, 4, 8),
}


def _knuckles(convention: str, joints: int) -> tuple[int, int]:
    index, pinky, _, _ = _joints(convention, joints)
    return index, pinky


def _tips(convention: str, joints: int) -> tuple[int, int]:
    _, _, thumb, index = _joints(convention, joints)
    return thumb, index


def _joints(convention: str, joints: int) -> tuple[int, int, int, int]:
    try:
        indices = JOINTS[convention]
    except KeyError:
        raise ConfigError(
            f"no joint indices for {convention!r}; known: {sorted(JOINTS)}. These are "
            "not interchangeable — MANO and MediaPipe are both 21 joints in "
            "different orders, and using one table for the other attaches every "
            "finger to the wrong knuckle with no error anywhere"
        ) from None
    if max(indices) >= joints:
        raise ConfigError(f"{convention!r} indexes joint {max(indices)} of a {joints}-joint hand")
    return indices


def _unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 1e-9)


def _quaternions(matrices: np.ndarray) -> np.ndarray:
    """Rotation matrices to wxyz, with a degenerate frame becoming identity.

    Degenerate happens: a frame where the hand was not found is all zeros, and
    the axes built from it are meaningless. Identity is the honest filler,
    because the confidence channel already says that frame is worthless and
    inventing a plausible rotation would hide it.
    """
    out = np.zeros((len(matrices), 4))
    out[:, 0] = 1.0
    trace = np.trace(matrices, axis1=1, axis2=2)
    for index, (matrix, tr) in enumerate(zip(matrices, trace)):
        if not np.isfinite(matrix).all() or np.allclose(matrix, 0.0):
            continue
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            out[index] = [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            j, k = (axis + 1) % 3, (axis + 2) % 3
            s = np.sqrt(1.0 + matrix[axis, axis] - matrix[j, j] - matrix[k, k]) * 2
            quaternion = np.zeros(4)
            quaternion[0] = (matrix[k, j] - matrix[j, k]) / s
            quaternion[axis + 1] = 0.25 * s
            quaternion[j + 1] = (matrix[j, axis] + matrix[axis, j]) / s
            quaternion[k + 1] = (matrix[k, axis] + matrix[axis, k]) / s
            out[index] = quaternion
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 1e-9)


def metric(
    model_path: str | None = None,
    *,
    intrinsics: Intrinsics | Mapping[str, Any],
    max_reprojection: float = 0.15,
    **options: Any,
) -> Estimator:
    """MediaPipe plus perspective-n-point: metric hand pose from one camera.

    The recommended wire, and the one that produces something a retargeter will
    accept. MediaPipe supplies a metric hand shape and its projection; PnP turns
    that pair into a pose in metres, using the hand's own measured size as the
    ruler. Nothing has to be assumed about the person.

    Apache-2.0 throughout, which is why it is the default rather than HaWoR or
    WiLoR. Those reconstruct better and are CC-BY-NC-ND on top of a
    registration-gated MANO, so a dataset built through them cannot be sold.
    """
    camera = intrinsics_from(intrinsics)
    if camera is None:
        raise ConfigError(
            "metric pose needs the camera's intrinsics. Every distance scales with "
            "the focal length, so a guessed one makes every number in the dataset "
            "wrong by the same unseen factor. Use Intrinsics.from_fov() or for_rig() "
            "if the camera was never calibrated — both record the assumption"
        )
    base = mediapipe(model_path, **options)

    class Metric:
        def estimate(self, frames: np.ndarray) -> Track:
            track = base.estimate(frames)
            height, width = np.asarray(frames).shape[1:3]
            poses: dict[str, np.ndarray] = {}
            residual: dict[str, np.ndarray] = {}
            for hand, image in track.keypoints.items():
                world = np.asarray(track.world.get(hand, ()), dtype=float)
                if not world.size:
                    continue
                pixels = np.asarray(image, dtype=float).copy()
                pixels[..., 0] *= width
                pixels[..., 1] *= height
                positions, rotations, errors = solve_sequence(
                    world, pixels[..., :2], camera, max_reprojection=max_reprojection
                )
                poses[hand] = np.concatenate(
                    [positions, rotations_to_quaternions(rotations)], axis=1
                ).astype("float32")
                residual[hand] = errors
            return Track(
                keypoints=track.keypoints,
                world=track.world,
                confidence=track.confidence,
                convention=track.convention,
                poses=poses,
                residual=residual,
                licence="Apache-2.0 (MediaPipe hand landmarker + OpenCV solvePnP)",
                metadata={
                    **dict(track.metadata),
                    "pose_method": "solvePnP(SQPNP) on metric hand landmarks",
                    **camera.as_dict(),
                },
            )

    return Metric()


#: Estimator wires addressable by name, so a manifest can choose one.
#:
#: Every argument that has to cross a manifest accepts plain data — the same
#: rule the gym bridge states and the pi0 layouts follow. A component whose key
#: argument is only constructible in Python is a component no manifest can use,
#: which means the pipeline that uses it is a script.
WIRES = ("mediapipe", "metric", "rtm+mediapipe", "rtm+depth")


def estimator_from(spec: Any) -> Any:
    """An estimator from plain data, a callable, or one already built."""
    if spec is None or hasattr(spec, "estimate"):
        return spec
    if callable(spec) and not isinstance(spec, Mapping):
        return spec
    config = dict(spec)
    wire = str(config.pop("wire", "metric"))
    if wire not in WIRES:
        raise ConfigError(f"unknown estimator wire {wire!r}; known: {list(WIRES)}")

    model_path = config.pop("model_path", None)
    rig = config.pop("rig", None)
    width, height = int(config.pop("width", 0)), int(config.pop("height", 0))
    camera = None
    if rig:
        if not (width and height):
            raise ConfigError(
                f"an estimator naming rig {rig!r} also needs the frame width and "
                "height, because intrinsics are in pixels and a focal length "
                "without a resolution is not a camera"
            )
        camera = for_rig(str(rig), width=width, height=height)
    elif "intrinsics" in config:
        camera = intrinsics_from(config.pop("intrinsics"))

    if wire == "mediapipe":
        return mediapipe(model_path, **config)
    if camera is None:
        raise ConfigError(
            f"the {wire!r} wire recovers metric pose and needs the camera's "
            "intrinsics: name a 'rig' with 'width' and 'height', or give "
            "'intrinsics' directly. Every distance scales with the focal length, "
            "so a guessed one makes every number wrong by the same unseen factor"
        )
    if wire == "metric":
        return metric(model_path, intrinsics=camera, **config)

    from .rtm import rtm_with_depth, rtm_with_mediapipe, rtmpose

    detector = rtmpose(**dict(config.pop("detector", {})))
    shape = mediapipe(model_path)
    if wire == "rtm+mediapipe":
        return rtm_with_mediapipe(detector, shape, intrinsics=camera, **config)

    from .depth import depth_anything, zoedepth

    depth_config = dict(config.pop("depth", {}))
    model = str(depth_config.pop("model", "zoedepth"))
    depth = zoedepth(**depth_config) if model == "zoedepth" else depth_anything(**depth_config)
    return rtm_with_depth(detector, depth, shape, intrinsics=camera, **config)


class HandPoseConnector(Connector):
    """Another connector's clips, with hands estimated onto them."""

    def __init__(
        self,
        source: Connector,
        *,
        estimator: Estimator | Callable[[], Estimator] | None = None,
        video: str = "ego_rgb",
        name: str = "handpose",
        found: float = FOUND,
        fast: float = FAST,
        min_steps: int = MIN_STEPS,
        keep_video: bool = True,
        cache: bool = True,
    ):
        self._source = source
        self._estimator = estimator_from(estimator)
        self._video = video
        self._name = name
        self._found = float(found)
        self._fast = float(fast)
        self._min_steps = int(min_steps)
        self._keep_video = bool(keep_video)
        # Off for a large build: every cached episode holds its full-resolution
        # frames, and two dozen 1080p clips is tens of gigabytes of resident
        # memory that nothing reads twice.
        self._cache_on = bool(cache)
        self._cache: dict[str, EpisodeRecord] = {}

    # -- contract ----------------------------------------------------------

    @property
    def estimator(self) -> Estimator:
        if self._estimator is None:
            raise ConfigError(
                f"{self._name}: no estimator. Pass one — the interface is a single "
                "estimate(frames) -> Track — or install a worked one with "
                "pip install 'gantry-connector-handpose[mediapipe]'"
            )
        if not hasattr(self._estimator, "estimate") and callable(self._estimator):
            self._estimator = self._estimator()
        return self._estimator  # type: ignore[return-value]

    def descriptor(self) -> Descriptor:
        upstream = self._source.descriptor()
        return connector_descriptor(
            name=self._name,
            version=VERSION,
            lazy=False,
            stage_events=False,
            outcomes=bool(upstream.provides.get("outcomes", False)),
            media=self._keep_video,
            writes=False,
            # Named so a run's provenance says which network guessed these hands.
            # Two runs through different estimators are not comparable and
            # nothing else here would have kept the difference.
            estimator=type(self._estimator).__name__ if self._estimator else "none",
            licence=getattr(self._estimator, "licence", "unknown"),
            derived_from=f"{upstream.name}@{upstream.version}",
            # Said once, loudly. Position is image-normalised and stays that
            # way; only the hand's own shape is metric.
            scale="normalized",
            metric_shape=True,
            found_threshold=self._found,
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(f"{self._name}/{identifier}" for identifier in self._source.episode_ids())

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        return self.open(episode_id).schema

    def open(self, episode_id: str) -> EpisodeRecord:
        if episode_id in self._cache:
            return self._cache[episode_id]
        derived = self._derive(episode_id)
        if self._cache_on:
            self._cache[episode_id] = derived
        return derived

    # -- deriving one episode ----------------------------------------------

    def _derive(self, episode_id: str) -> EpisodeRecord:
        upstream_id = self._upstream(episode_id)
        original = self._source.open(upstream_id)
        if not original.has(self._video):
            raise ConfigError(
                f"{episode_id}: no {self._video!r} channel to estimate from; it has "
                f"{list(original.channel_names)}"
            )
        frames = original.array(self._video)
        track = self.estimator.estimate(frames)

        arrays: dict[str, np.ndarray] = {}
        schema: list[ChannelSpec] = []
        if self._keep_video:
            arrays[self._video] = frames
            schema.append(original.channel(self._video))

        rate = original.channel(self._video).rate_hz
        for hand in HANDS:
            if hand not in track.keypoints:
                continue
            image = np.asarray(track.keypoints[hand], dtype="float32")
            metric = np.asarray(track.world.get(hand, np.zeros_like(image)), dtype="float32")
            has_world = bool(track.world) and np.any(metric)
            confidence = np.asarray(track.confidence[hand], dtype="float32")

            # Orientation comes from the metric hand where there is one: a
            # rotation needs no scale, so this part IS recoverable from a single
            # camera. Position does not come with it — the world landmarks are
            # centred on the hand, so they say what shape it is and nothing about
            # where it was.
            solved = np.asarray(track.poses.get(hand, ()), dtype="float32")
            if solved.size:
                # A real pose in metres, recovered by PnP from the hand's own
                # measured size. This is the only branch a retargeter accepts.
                wrist = solved
                wrist_scale, wrist_frame = "metric", "camera"
            else:
                shape_source = metric if has_world else image
                wrist = wrist_from(shape_source, convention=track.convention)
                # Position from the image, which is where the only positional
                # information exists, and declared normalized so nothing
                # downstream mistakes a pixel fraction for a metre.
                wrist[:, :3] = image[:, 0, :]
                wrist_scale, wrist_frame = "normalized", "camera"

            arrays[f"{hand}_hand"] = image
            arrays[f"{hand}_wrist"] = wrist
            arrays[f"{hand}_confidence"] = confidence
            if hand in track.residual:
                arrays[f"{hand}_reprojection"] = np.asarray(track.residual[hand], dtype="float32")
            schema += [
                hand_channel(
                    f"{hand}_hand",
                    hand=hand,
                    keypoints=track.convention,
                    scale="normalized",
                    frame="image",
                    rate_hz=rate,
                ),
                wrist_channel(
                    f"{hand}_wrist",
                    hand=hand,
                    # metric when PnP solved it; otherwise normalized, because
                    # image landmarks are pixel fractions and no arithmetic on
                    # them recovers metres. A hand span rescues an unscaled
                    # metric hand; it cannot rescue an image coordinate.
                    scale=wrist_scale,
                    rotation_repr="quat_wxyz",
                    frame=wrist_frame,
                    rate_hz=rate,
                ),
                ChannelSpec(
                    f"{hand}_confidence",
                    "scalar",
                    (),
                    "float32",
                    rate_hz=rate,
                    metadata={"estimator": track.metadata.get("estimator", "unknown")},
                ),
            ]
            if hand in track.residual:
                # The honest confidence for a solved pose. A detector's own score
                # says it found a hand; this says the pose it implies can
                # reproduce the pixels it was fitted to.
                schema.append(
                    ChannelSpec(
                        f"{hand}_reprojection",
                        "scalar",
                        (),
                        "float32",
                        units="px",
                        rate_hz=rate,
                        metadata={"method": track.metadata.get("pose_method", "")},
                    )
                )
            if has_world:
                # The half that genuinely is metric: hand shape, in metres,
                # centred on the hand. The aperture computed from it is a real
                # distance, which is what makes the gripper command trustworthy
                # even when the wrist trajectory is not.
                arrays[f"{hand}_shape"] = metric
                arrays[f"{hand}_aperture"] = aperture_from(metric, convention=track.convention)
                schema += [
                    hand_channel(
                        f"{hand}_shape",
                        hand=hand,
                        keypoints=track.convention,
                        scale="metric",
                        frame="head",
                        rate_hz=rate,
                    ),
                    aperture_channel(f"{hand}_aperture", hand=hand, scale="metric", rate_hz=rate),
                ]
            else:
                arrays[f"{hand}_aperture"] = aperture_from(image, convention=track.convention)
                schema.append(
                    aperture_channel(
                        f"{hand}_aperture", hand=hand, scale="normalized", rate_hz=rate
                    )
                )

        signals = self._signals(track, frames, rate)
        labels = original.labels
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._name,
                task=original.meta.task,
                embodiment=original.meta.embodiment,
                # The lineage that makes the estimate traceable back to the
                # footage it was guessed from.
                derived_from=(upstream_id,),
                extra={
                    **dict(original.meta.extra),
                    "estimator": track.metadata.get("estimator", type(self.estimator).__name__),
                    "keypoints": track.convention,
                    **{
                        k: v
                        for k, v in track.metadata.items()
                        if k.startswith(("intrinsics", "pose_", "fx", "fy", "resolution"))
                    },
                    # Two answers, because there are two quantities. The hand's
                    # shape is metres; where it was in the room is not.
                    "scale": "metric" if track.poses else "normalized",
                    "shape_scale": "metric" if track.world else "normalized",
                    # Travels onto every derived episode, because a dataset made
                    # with non-commercial weights is itself encumbered and
                    # nothing else in the pipeline would remember.
                    "estimator_licence": track.licence,
                },
            ),
            schema=tuple(schema),
            source=ArraySource(arrays),
            labels=EpisodeLabels(
                success=labels.success,
                stage_events=labels.stage_events,
                annotations={**dict(labels.annotations), **signals},
            ),
        )

    def _signals(self, track: Track, frames: np.ndarray, rate: float | None) -> dict[str, Any]:
        """The three counts ``feedback_capture`` turns into advice.

        Counts, not judgements. Which number is worth worrying about, and what to
        do, is decided somewhere else on purpose — a module that both measured
        and graded would be scoring its own homework.
        """
        steps = max(1, len(frames))
        confidences = [np.asarray(value) for value in track.confidence.values()]
        best = (
            np.max(np.stack(confidences), axis=0)
            if confidences
            else np.zeros(steps, dtype="float32")
        )
        visible = float((best >= self._found).mean())

        speeds: list[np.ndarray] = []
        for hand, points in track.keypoints.items():
            wrist = np.asarray(points, dtype=float)[:, 0, :]
            if len(wrist) < 2:
                continue
            step = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
            speeds.append(step * float(rate or 1.0))
        motion = float((np.concatenate(speeds) <= self._fast).mean()) if speeds else 1.0
        pose_quality = {}
        if track.poses:
            for hand, poses in track.poses.items():
                pose_quality[f"{hand}_pose_plausible"] = round(
                    plausible(np.asarray(poses)[:, :3]), 4
                )
        return {
            **pose_quality,
            "estimator_licence": track.licence,
            "hands_visible": round(visible, 4),
            "motion_ok": round(motion, 4),
            "usable_length": 1.0 if steps >= self._min_steps else 0.0,
            "steps": int(steps),
            "keypoints": track.convention,
            "scale": "normalized",
        }

    def _upstream(self, episode_id: str) -> str:
        prefix = f"{self._name}/"
        if not episode_id.startswith(prefix):
            raise KeyError(f"no episode {episode_id!r}")
        upstream = episode_id[len(prefix) :]
        if upstream not in self._source.episode_ids():
            raise KeyError(f"no episode {episode_id!r}")
        return upstream
