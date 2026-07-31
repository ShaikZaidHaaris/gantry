"""Camera-frame hands into a fixed world frame, given the camera's own motion.

The limitation the whole ego pipeline has carried until now: poses are measured
relative to a camera bolted to a moving head. That is genuinely fine when the
robot's camera sits where the person's eyes were, because then both sides are
learning hand-relative-to-viewpoint. It is wrong for everything else, and it is
wrong in a way that looks like noise rather than like an error — the target
swings whenever the person glances away, so a policy sees the same physical
reach labelled a dozen different ways.

Fixing it needs one thing this pipeline never had: where the camera was. Given
that, the composition is trivial —

    world_pose = camera_to_world[t] @ camera_frame_pose[t]

— and the whole difficulty is getting the trajectory honestly.

Three ways to get one, in descending order of how much you should trust it
--------------------------------------------------------------------------
**Recorded at capture.** ARKit, ARCore, Quest and Aria all track the device and
throw the result away when you export an MP4. If the capture app kept it, it is
metric, drift-corrected, and free. Nothing reconstructed will beat it.

**Structure from motion.** COLMAP (BSD, commercially usable) recovers camera
poses from the frames themselves. Good, slow, and **scaleless** — monocular SfM
determines the trajectory up to an unknown factor. Recovering that factor needs a
metric observation of something, and the only one available here is the hand.

The obvious version of that is wrong, and worth saying so plainly because it is
the version anybody would write first: a hand's *distance from the camera* is a
camera-frame quantity, so it is unchanged by how big the trajectory is and
constrains nothing. What does constrain it is a point that is stationary in the
world — its camera-frame position then moves only because the camera moved, and
the ratio recovers the factor. :func:`scale_to` does that, it rests on the hand
being still while the person moves, and it is off by default because that is true
some of the time and false the rest.

**Nothing.** Then this connector refuses, and the camera-frame data stays what it
is. That is a supported outcome — it is what the pipeline did before — rather
than a failure.

Drift is reported, never hidden
-------------------------------
Every trajectory drifts. A ten-second clip barely does; a ten-minute one can be
metres out by the end, which silently stretches every trajectory near the end of
the recording. There is no fixing that here, so it is measured where it can be —
against loop closures where the trajectory returns near a previous pose — and
reported per episode, so a downstream user can weight or drop the tail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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

VERSION = "0.1.0.dev0"

#: Where a trajectory came from, and whether its scale means anything.
SOURCES = {
    "device": ("recorded by the capture device (ARKit/ARCore/Quest/Aria)", True),
    "colmap": ("structure from motion; scaleless until fitted", False),
    "declared": ("supplied by the caller", True),
}


@dataclass(frozen=True)
class Trajectory:
    """Where the camera was, per frame.

    ``positions`` is ``(steps, 3)`` and ``rotations`` is ``(steps, 3, 3)``, both
    camera-to-world. ``metric`` is the field that decides whether this can be
    used at all: a scaleless trajectory composed with metric hand poses produces
    a hand that moves correctly relative to a room of the wrong size.
    """

    positions: np.ndarray
    rotations: np.ndarray
    metric: bool
    source: str = "declared"
    #: Frames the tracker itself could not solve.
    valid: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ConfigError(f"unknown trajectory source {self.source!r}; {sorted(SOURCES)}")
        positions = np.asarray(self.positions, dtype=float)
        rotations = np.asarray(self.rotations, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ConfigError(f"expected (steps, 3) positions, got {positions.shape}")
        if rotations.shape != (len(positions), 3, 3):
            raise ConfigError(f"expected ({len(positions)}, 3, 3) rotations, got {rotations.shape}")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(
            self,
            "valid",
            np.ones(len(positions), dtype=bool)
            if self.valid is None
            else np.asarray(self.valid, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.positions)

    def place(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame points into the world."""
        local = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.einsum("tij,tj->ti", self.rotations, local) + self.positions

    def turn(self, rotations: np.ndarray) -> np.ndarray:
        """Camera-frame orientations into the world."""
        return np.einsum("tij,tjk->tik", self.rotations, np.asarray(rotations, dtype=float))

    def scaled(self, factor: float) -> "Trajectory":
        """The same trajectory at a different size, now claiming to be metric."""
        return Trajectory(
            positions=self.positions * float(factor),
            rotations=self.rotations,
            metric=True,
            source=self.source,
            valid=self.valid,
            metadata={**dict(self.metadata), "scale_factor": float(factor)},
        )

    def drift(self) -> dict[str, Any]:
        """How far the trajectory returns from where it started, at the end.

        A crude loop-closure proxy and the only one available without the frames.
        Real drift is worse than this reports; a number that is already large
        here means the tail is unusable.
        """
        if len(self) < 2:
            return {"measured": False}
        travelled = float(np.linalg.norm(np.diff(self.positions[self.valid], axis=0), axis=1).sum())
        closure = float(np.linalg.norm(self.positions[-1] - self.positions[0]))
        return {
            "measured": True,
            "path_length_m": round(travelled, 4),
            "start_to_end_m": round(closure, 4),
            "unsolved_frames": round(1.0 - float(self.valid.mean()), 4),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_source": self.source,
            "trajectory_metric": self.metric,
            "trajectory_note": SOURCES[self.source][0],
            "steps": len(self),
            **{f"drift_{k}": v for k, v in self.drift().items() if k != "measured"},
            **dict(self.metadata),
        }


def scale_to(
    trajectory: Trajectory,
    camera_frame_positions: np.ndarray,
    *,
    solved: np.ndarray | None = None,
    quantile: float = 0.5,
) -> float:
    """The factor that makes a scaleless trajectory agree with metric observations.

    The physics, because the obvious version is wrong. A hand's *distance from the
    camera* is measured in the camera frame and is therefore invariant to how big
    the trajectory is — it constrains nothing. What does constrain it is a point
    that is stationary in the world: for such a point, the camera-frame position
    changes only because the camera moved, so

        |Δp_camera_frame|  =  s · |Δĉ|

    and the ratio recovers ``s``.

    The assumption is stated because it is doing all the work: it holds while the
    hand is resting or holding something still and the person moves, and fails
    while the hand is reaching. So the ratio is taken as a robust quantile across
    the clip rather than a mean, and even then this is a fit rather than a
    measurement.

    It is worth being blunt: this is the weakest link in the world-frame path,
    and it is why ``fit_scale`` defaults to off. A trajectory recorded by the
    capture device needs none of this, and is the reason to prefer one.
    """
    positions = np.asarray(camera_frame_positions, dtype=float).reshape(-1, 3)
    if len(positions) != len(trajectory):
        raise ConfigError(
            f"{len(positions)} camera-frame positions for {len(trajectory)} camera poses"
        )
    usable = np.any(positions != 0.0, axis=1) & trajectory.valid
    if solved is not None:
        usable &= np.asarray(solved, dtype=bool)

    pairs = usable[:-1] & usable[1:]
    if pairs.sum() < 8:
        raise ConfigError(
            "fewer than eight consecutive frames where both the hand and the camera "
            "were solved; there is nothing to fit a scale against"
        )
    hand_step = np.linalg.norm(np.diff(positions, axis=0), axis=1)[pairs]
    camera_step = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1)[pairs]
    moving = camera_step > 1e-9
    if not moving.any():
        raise ConfigError(
            "the trajectory does not move, so no factor relates it to anything. A "
            "clip filmed from a fixed head has no camera motion to scale"
        )
    ratio = hand_step[moving] / camera_step[moving]
    factor = float(np.quantile(ratio, quantile))
    if not np.isfinite(factor) or factor <= 0:
        raise ConfigError("the fitted scale is not a positive number; the fit failed")
    return factor


def from_device(path: str | Path) -> Trajectory:
    """A trajectory recorded by the capture device, from JSON.

    The shape ARKit and ARCore export: a list of ``{position, rotation}`` per
    frame, rotation as a wxyz quaternion. Metric by construction, which is the
    entire reason to prefer this route.
    """
    data = json.loads(Path(path).read_text())
    frames = data.get("frames", data) if isinstance(data, Mapping) else data
    if not isinstance(frames, list) or not frames:
        raise ConfigError(f"{path}: expected a list of per-frame poses")
    positions, rotations, valid = [], [], []
    for frame in frames:
        position = frame.get("position") or frame.get("t")
        rotation = frame.get("rotation") or frame.get("q")
        if position is None or rotation is None:
            positions.append([0.0, 0.0, 0.0])
            rotations.append(np.eye(3))
            valid.append(False)
            continue
        positions.append([float(v) for v in position])
        rotations.append(_matrix(rotation))
        valid.append(True)
    return Trajectory(
        positions=np.asarray(positions),
        rotations=np.asarray(rotations),
        metric=True,
        source="device",
        valid=np.asarray(valid),
        metadata={
            "file": str(path),
            "device": data.get("device", "unknown") if isinstance(data, Mapping) else "unknown",
        },
    )


def from_colmap(path: str | Path, *, frames: int | None = None) -> Trajectory:
    """A trajectory from COLMAP's text output. Scaleless, and says so.

    Reads ``images.txt``. COLMAP stores world-to-camera, so both the rotation and
    the translation are inverted here — getting that backwards produces a
    trajectory that is a plausible-looking mirror of the truth, which is the
    classic way to lose a day.
    """
    lines = [
        line
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for line in lines[::2]:  # COLMAP alternates pose lines and point lines
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = (float(v) for v in parts[1:5])
        tx, ty, tz = (float(v) for v in parts[5:8])
        name = parts[9]
        index = _index_of(name)
        rotation = _matrix([qw, qx, qy, qz])
        # world-to-camera -> camera-to-world
        poses[index] = (rotation.T, -rotation.T @ np.array([tx, ty, tz]))
    if not poses:
        raise ConfigError(f"{path}: no camera poses found; is this a COLMAP images.txt?")

    steps = frames if frames is not None else max(poses) + 1
    positions = np.zeros((steps, 3))
    rotations = np.tile(np.eye(3), (steps, 1, 1))
    valid = np.zeros(steps, dtype=bool)
    for index, (rotation, position) in poses.items():
        if 0 <= index < steps:
            rotations[index] = rotation
            positions[index] = position
            valid[index] = True
    return Trajectory(
        positions=positions,
        rotations=rotations,
        # The whole point: structure from motion cannot know how big the world
        # is, so this must be fitted before it can be composed with metres.
        metric=False,
        source="colmap",
        valid=valid,
        metadata={"file": str(path), "registered": int(valid.sum())},
    )


def _index_of(name: str) -> int:
    digits = "".join(c for c in Path(name).stem if c.isdigit())
    return int(digits) if digits else 0


def _matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion)
    norm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class WorldFrameConnector(Connector):
    """Camera-frame hands from upstream, placed in a fixed world frame."""

    def __init__(
        self,
        source: Connector,
        *,
        trajectories: Mapping[str, Trajectory],
        name: str = "worldframe",
        hands: Sequence[str] = ("left", "right"),
        fit_scale: bool = False,
    ):
        """``fit_scale`` is off by default, and that is a real recommendation.

        Fitting a scaleless trajectory against hand motion rests on the hand
        being world-stationary while the camera moves, which is true some of the
        time and false the rest. A trajectory recorded by the capture device
        needs none of it. See :func:`scale_to`.
        """
        self._source = source
        self._trajectories = dict(trajectories)
        self._name = name
        self._hands = tuple(hands)
        self._fit = bool(fit_scale)
        self._cache: dict[str, EpisodeRecord] = {}
        self._fitted: dict[str, float] = {}
        self._refuse_if_inert()

    def _refuse_if_inert(self) -> None:
        """Say at build time what would otherwise be said at the first read.

        A stage holding no trajectory for anything upstream cannot place a
        single episode in a world. Discovering that on the first ``open`` makes
        it a runtime fault halfway through a run; discovering it here makes it a
        buildable fact, which is what lets a chain declare the stage optional
        and carry on in camera frame without it.

        Partial coverage is not inertness and is not caught here. Asking for a
        world frame for episodes that have no trajectory is a real request that
        cannot be met, and :meth:`_derive` refuses it per episode.
        """
        upstream = self._source.episode_ids()
        if any(
            i in self._trajectories or f"{self._name}/{i}" in self._trajectories for i in upstream
        ):
            return
        raise ConfigError(
            f"{self._name}: no camera trajectory for any of the {len(upstream)} episodes "
            f"upstream, so there is no episode this stage could place in a world. Record "
            f"the trajectory at capture (ARKit/ARCore/Quest) or reconstruct it with COLMAP "
            f"— or drop the stage and keep the poses in camera frame, which is what they are"
        )

    def descriptor(self) -> Descriptor:
        upstream = self._source.descriptor()
        return connector_descriptor(
            name=self._name,
            version=VERSION,
            lazy=False,
            stage_events=False,
            outcomes=bool(upstream.provides.get("outcomes", False)),
            media=bool(upstream.provides.get("media", False)),
            writes=False,
            derived_from=f"{upstream.name}@{upstream.version}",
            frame="world",
            trajectories=len(self._trajectories),
            sources=sorted({t.source for t in self._trajectories.values()}),
            fit_scale=self._fit,
            estimator_licence=upstream.metadata.get("licence", "unknown"),
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(f"{self._name}/{i}" for i in self._source.episode_ids())

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        return self.open(episode_id).schema

    def open(self, episode_id: str) -> EpisodeRecord:
        if episode_id not in self._cache:
            self._cache[episode_id] = self._derive(episode_id)
        return self._cache[episode_id]

    def _derive(self, episode_id: str) -> EpisodeRecord:
        upstream_id = self._upstream(episode_id)
        original = self._source.open(upstream_id)
        trajectory = self._trajectories.get(upstream_id) or self._trajectories.get(episode_id)
        if trajectory is None:
            raise ConfigError(
                f"{episode_id}: no camera trajectory. Camera-frame poses cannot be "
                "placed in a world without knowing where the camera was — and a "
                "trajectory guessed from nothing would produce a hand that moves "
                "correctly through a room that does not exist. Record it at capture "
                "(ARKit/ARCore/Quest), or reconstruct it with COLMAP"
            )
        if len(trajectory) < len(original):
            raise ConfigError(
                f"{episode_id}: {len(trajectory)} camera poses for {len(original)} "
                "frames. A trajectory shorter than its footage cannot be aligned "
                "without guessing which frames it covers"
            )

        arrays: dict[str, np.ndarray] = {}
        schema: list[ChannelSpec] = []
        for spec in original.schema:
            if spec.name.endswith("_wrist"):
                continue
            arrays[spec.name] = original.array(spec.name)
            schema.append(spec)

        if not trajectory.metric:
            trajectory = self._fitted_trajectory(episode_id, original, trajectory)

        for hand in self._hands:
            key = f"{hand}_wrist"
            if not original.has(key):
                continue
            spec = original.channel(key)
            if spec.metadata.get("scale") != "metric":
                raise ConfigError(
                    f"{episode_id}: {key} is {spec.metadata.get('scale')!r}. Placing a "
                    "non-metric pose in a metric world produces a hand that moves "
                    "correctly relative to a room of the wrong size"
                )
            values = np.asarray(original.array(key), dtype=float)
            steps = len(values)
            positions = trajectory.place(values[:steps, :3])
            rotations = trajectory.turn(_matrices(values[:steps, 3:7]))
            solved = np.any(values[:, :3] != 0.0, axis=1) & trajectory.valid[:steps]
            placed = np.concatenate([positions, _quaternions(rotations)], axis=1).astype("float32")
            placed[~solved] = 0.0
            arrays[key] = placed
            schema.append(
                ChannelSpec(
                    key,
                    "vector",
                    (7,),
                    "float32",
                    frame="world",
                    rate_hz=spec.rate_hz,
                    semantics="ego.wrist_pose",
                    discriminators=("rotation_repr", "scale", "hand"),
                    metadata={
                        "rotation_repr": "quat_wxyz",
                        "scale": "metric",
                        "hand": hand,
                    },
                )
            )

        annotations = dict(original.labels.annotations)
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._name,
                task=original.meta.task,
                embodiment=original.meta.embodiment,
                derived_from=(upstream_id,),
                license=original.meta.license,
                extra={**dict(original.meta.extra), **trajectory.as_dict()},
            ),
            schema=tuple(schema),
            source=ArraySource(arrays),
            labels=EpisodeLabels(
                success=original.labels.success,
                annotations={**annotations, **trajectory.as_dict()},
            ),
        )

    def _fitted_trajectory(self, episode_id, original, trajectory) -> Trajectory:
        if not self._fit:
            raise ConfigError(
                f"{episode_id}: the trajectory is scaleless and fitting is off. "
                "Monocular structure from motion recovers a path up to one unknown "
                "multiplier; composing it with metric hands as-is places the hand in "
                "a room of the wrong size"
            )
        positions = None
        for hand in self._hands:
            key = f"{hand}_wrist"
            if original.has(key):
                positions = np.asarray(original.array(key), dtype=float)[:, :3]
                break
        if positions is None:
            raise ConfigError(f"{episode_id}: no metric hand positions to fit a scale against")
        factor = scale_to(trajectory, positions)
        self._fitted[episode_id] = factor
        return trajectory.scaled(factor)

    def fitted_scales(self) -> dict[str, float]:
        """What each scaleless trajectory had to be multiplied by.

        Worth looking at: these should be similar across clips from one capture
        session, and a wildly varying factor means the fit is picking up noise
        rather than a real relationship.
        """
        return dict(self._fitted)

    def _upstream(self, episode_id: str) -> str:
        prefix = f"{self._name}/"
        if not episode_id.startswith(prefix):
            raise KeyError(f"no episode {episode_id!r}")
        upstream = episode_id[len(prefix) :]
        if upstream not in self._source.episode_ids():
            raise KeyError(f"no episode {episode_id!r}")
        return upstream


def _matrices(quaternions: np.ndarray) -> np.ndarray:
    return np.stack([_matrix(q) for q in np.asarray(quaternions, dtype=float)])


def _quaternions(matrices: np.ndarray) -> np.ndarray:
    from gantry_connector_handpose import rotations_to_quaternions

    return rotations_to_quaternions(matrices)
