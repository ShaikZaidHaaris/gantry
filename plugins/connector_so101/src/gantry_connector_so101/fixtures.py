"""Write a synthetic SO-101 corpus to disk, so none of this needs hardware.

There is no real SO-101 dataset yet and there will not be one until the rig is
assembled, calibrated and burned in. A connector whose only proof is a dataset
its own author also wrote has proved that they agree with themselves — so what
this module writes is the *format*, exactly as ``lerobot-record`` and
``harness/so101/record_session.py`` produce it, and the trajectories inside it
come from :mod:`gantry.fixtures`, which nothing in this package wrote.

That matters most for ``make_duration_confound()``: a suite where nothing is
wrong and everything varies in length. Rendering it as an SO-101 corpus and
pushing it through this connector into the feedback plane is the test that the
connector adds no artefact of its own, and it is a harder test to pass than any
of the defective suites.

The joint rendering, and what it deliberately preserves
------------------------------------------------------
The fixture trajectories are abstract: a point per step, an engagement signal,
and a command. Turning them into SO-101 joint percentages is an **affine map
with a single fixed gain**, chosen so the properties the duration decoy rests on
survive it exactly:

* one gain for every episode, so no per-episode normalisation is applied. A
  per-episode scale would make a long episode's per-step motion smaller and turn
  the decoy into a real difference in step size;
* ratio statistics (path efficiency, direction-change *rate*) are invariant
  under a uniform positive scale, so they come through unchanged;
* values are **checked** against the embodiment's declared travel and the writer
  raises if any land outside it. A fixture that clamps would be writing a corpus
  the real refusals could never have produced.

``action`` is the next step's joint vector, because the SO-101 action space is an
*absolute* commanded position — where the leader was, which is where the
follower is told to go — and never a displacement.

The bimanual rendering pads: one arm gets the trajectory, the other is held at a
declared constant. That is not a bimanual demonstration and is not offered as
one (``SingleToBimanual`` in the embodiment package says the same thing at
greater length); it exists so the width guard can be tested in both directions.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gantry_evaluator_so101.bimanual import SIDE_BLOCKS
from gantry_evaluator_so101.embodiment import DOF, GRIPPER_INDEX, NORMALIZATION_RANGES

from gantry.errors import ConfigError
from gantry.spine import EpisodeRecord

from .binding import BIMANUAL_REF, FOLLOWER_REF, NORMALIZATION, get_binding

#: Mid-travel. Every joint's calibrated envelope contains it whatever that arm's
#: calibration says, which is why it is the origin of the map and the pose the
#: idle arm is held at.
HOME = 50.0

#: Percent of calibrated travel per metre of fixture displacement. Fixed for
#: every episode on purpose — see the module docstring — and sized so the
#: longest episode of ``make_duration_confound()`` still lands inside [0, 100].
GAIN = 40.0

#: Where the gripper sits open and closed, in percent of travel.
GRIPPER_OPEN, GRIPPER_CLOSED = 20.0, 80.0

#: The three streams the corpus factory records always and ablates at train
#: time. Declared in the dataset metadata; no mp4 is written, so the LeRobot
#: reader reports the cameras as unavailable rather than promising frames.
CAMERAS = ("top", "front", "wrist")

#: The recorder's CSV column order, mirrored here. This is a fixture of the
#: producer's format, so it is written out rather than imported: the producer
#: lives in another repository and importing it would make this package
#: unusable anywhere that repository is not checked out.
#: ``episode_index`` is the one column added — see :mod:`.sidecar` on the join.
BASE_COLUMNS = (
    "episode_index",
    "corpus_id",
    "condition_label",
    "treatment",
    "operator_id",
    "session_id",
    "block_id",
    "episode_index_within_session",
    "started_at",
    "ended_at",
    "wall_duration_s",
    "order_seed",
    "condition_order",
    "pose_seed",
    "object_start_x",
    "object_start_y",
    "object_start_yaw",
    "drawn_start_x",
    "drawn_start_y",
    "drawn_start_yaw",
    "pose_placement_err_m",
    "container_x",
    "container_y",
    "container_z",
    "container_yaw",
    "outcome",
    "episode_time_s",
    "control_hz",
    "frames_nominal",
    "num_frames",
    "frames_recorded",
    "frame_deficit",
    "early_exit",
    "declared_episode_time_s",
    "embodiment",
    "sim_commit",
    "lerobot_commit",
    "calib_sha256_leader",
    "calib_sha256_follower",
)

TAIL_COLUMNS = ("fatigue", "notes")


def calibration_hashes(tag: str) -> dict[str, str]:
    """A pair of plausible calibration digests, deterministic in ``tag``.

    Two different tags stand for two different calibration files, which is what
    the recalibration refusal is tested against.
    """
    return {
        arm: hashlib.sha256(f"{tag}/{arm}".encode("utf-8")).hexdigest()
        for arm in ("leader", "follower")
    }


def to_joints(episode: EpisodeRecord, ref: str = FOLLOWER_REF) -> np.ndarray:
    """One fixture episode as ``(steps, dof)`` of percent-of-calibrated-travel."""
    binding = get_binding(ref)
    position = np.asarray(episode.array("position"), dtype=float)
    engagement = np.asarray(episode.array("engagement"), dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ConfigError(
            f"{episode.meta.uid}: expected a 3-wide position channel, got {position.shape}"
        )

    arm = np.full((len(position), DOF), HOME, dtype=float)
    arm[:, 0:3] = HOME + GAIN * position
    # wrist_flex and wrist_roll stay at mid-travel: this synthetic rig does not
    # articulate the wrist, and a fabricated wrist motion would be invented
    # behaviour that a screen would then measure.
    arm[:, GRIPPER_INDEX] = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * engagement

    if ref == FOLLOWER_REF:
        joints = arm
    elif ref == BIMANUAL_REF:
        joints = np.full((len(position), 2 * DOF), HOME, dtype=float)
        left, _ = SIDE_BLOCKS["left"]
        joints[:, left : left + DOF] = arm
    else:  # pragma: no cover - get_binding already refuses anything else
        raise ConfigError(f"no fixture rendering for {ref!r}")

    low, high = NORMALIZATION_RANGES[NORMALIZATION]
    outside = np.argwhere((joints < low) | (joints > high))
    if outside.size:
        step, column = (int(value) for value in outside[0])
        raise ConfigError(
            f"{episode.meta.uid}: rendered joint {binding.joint_names[column]} at step "
            f"{step} is {joints[step, column]:.3f}, outside {NORMALIZATION} "
            f"[{low:g}, {high:g}]. Refusing rather than clamping: a clamped fixture is a "
            "corpus no rig could have produced, and the checks it then passes prove "
            "nothing about a corpus that could."
        )
    return joints


def _actions(joints: np.ndarray) -> np.ndarray:
    """The absolute target commanded at each step: where the arm goes next.

    The last step repeats its own pose, because there is no next one and
    inventing a further target would put a command in the record that no
    operator gave.
    """
    actions = np.empty_like(joints)
    actions[:-1] = joints[1:]
    actions[-1] = joints[-1]
    return actions


def write_corpus(
    episodes: Sequence[EpisodeRecord],
    root: str | os.PathLike[str],
    *,
    embodiment: str = FOLLOWER_REF,
    corpus_id: str = "C-FIX",
    condition_label: str = "E-PRAC",
    operator_id: str = "O1",
    task: str = "put the cube in the bin",
    fps: int = 30,
    episodes_per_session: int = 6,
    order_seed: int = 20260728,
    sidecar: bool = True,
    episode_index_column: bool = True,
    block_columns: bool = True,
    calibration: str | Callable[[int], str] = "cal-a",
    calibration_columns: bool = True,
    outcome: str | Callable[[int], str] = "success",
    drift: Mapping[str, Any] | None = None,
    cameras: Sequence[str] = (),
    dim_labels: Sequence[str] | None = None,
    widths: Mapping[str, int] | None = None,
) -> Path:
    """Write ``episodes`` as a LeRobot v2.1 dataset plus the recorder's sidecar.

    ``widths`` and ``dim_labels`` override what the dataset *declares* about its
    joint channels without changing what it contains. They exist so the width
    guard and the joint-order guard can be tested against a corpus that is
    wrong in exactly one way, which is the only kind of test that proves a guard
    fires for the reason it claims to.
    """
    root = Path(root)
    binding = get_binding(embodiment)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rendered = [to_joints(episode, embodiment) for episode in episodes]
    declared = dict(widths or {})
    labels = tuple(dim_labels) if dim_labels is not None else binding.joint_names

    features: dict[str, Any] = {}
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "rgb"],
        }
    for channel in ("observation.state", "action"):
        width = declared.get(channel, binding.dof)
        features[channel] = {
            "dtype": "float32",
            "shape": [width],
            "names": {"motors": list(labels[:width]) if len(labels) >= width else None},
        }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for column in ("frame_index", "episode_index", "index", "task_index"):
        features[column] = {"dtype": "int64", "shape": [1], "names": None}

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": binding.name,
                "total_episodes": len(episodes),
                "fps": fps,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                ),
                "features": features,
            },
            indent=2,
        )
    )
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": [task], "length": len(joints)}) + "\n"
            for index, joints in enumerate(rendered)
        )
    )

    for index, joints in enumerate(rendered):
        steps = len(joints)
        width = joints.shape[1]
        table = pa.table(
            {
                "observation.state": pa.array(
                    joints.astype("float32").tolist(), pa.list_(pa.float32(), width)
                ),
                "action": pa.array(
                    _actions(joints).astype("float32").tolist(), pa.list_(pa.float32(), width)
                ),
                "timestamp": pa.array(np.arange(steps) / float(fps), pa.float32()),
                "frame_index": pa.array(np.arange(steps), pa.int64()),
                "episode_index": pa.array([index] * steps, pa.int64()),
                "index": pa.array(np.arange(steps), pa.int64()),
                "task_index": pa.array([0] * steps, pa.int64()),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")

    if sidecar:
        _write_sidecar(
            root / "episodes.csv",
            rendered,
            binding=binding,
            corpus_id=corpus_id,
            condition_label=condition_label,
            operator_id=operator_id,
            fps=fps,
            episodes_per_session=episodes_per_session,
            order_seed=order_seed,
            episode_index_column=episode_index_column,
            block_columns=block_columns,
            calibration=calibration,
            calibration_columns=calibration_columns,
            outcome=outcome,
            drift=drift,
            cameras=cameras or CAMERAS,
        )
    return root


def _write_sidecar(
    path: Path,
    rendered: Sequence[np.ndarray],
    *,
    binding: Any,
    corpus_id: str,
    condition_label: str,
    operator_id: str,
    fps: int,
    episodes_per_session: int,
    order_seed: int,
    episode_index_column: bool,
    block_columns: bool,
    calibration: str | Callable[[int], str],
    calibration_columns: bool,
    outcome: str | Callable[[int], str],
    drift: Mapping[str, Any] | None,
    cameras: Sequence[str],
) -> Path:
    import csv

    columns = [column for column in BASE_COLUMNS]
    if not episode_index_column:
        columns.remove("episode_index")
    if not block_columns:
        for column in ("session_id", "block_id", "episode_index_within_session", "order_seed"):
            columns.remove(column)
    if not calibration_columns:
        for column in ("calib_sha256_leader", "calib_sha256_follower"):
            columns.remove(column)
    for joint in binding.joint_names:
        base = joint[:-4] if joint.endswith(".pos") else joint
        columns += [f"trk_mean_{base}", f"trk_max_{base}"]
    columns += [f"dropped_{camera}" for camera in cameras]
    columns += list(TAIL_COLUMNS)
    if drift:
        columns += sorted(drift)

    start = dt.datetime(2026, 7, 28, 9, 0, tzinfo=dt.timezone.utc)
    rows = []
    for index, joints in enumerate(rendered):
        steps = len(joints)
        session = index // max(1, episodes_per_session)
        began = start + dt.timedelta(minutes=3 * index)
        ended = began + dt.timedelta(seconds=steps / float(fps))
        tag = outcome(index) if callable(outcome) else outcome
        digests = calibration_hashes(calibration(index) if callable(calibration) else calibration)
        row = {
            "episode_index": index,
            "corpus_id": corpus_id,
            "condition_label": condition_label,
            "treatment": condition_label,
            "operator_id": operator_id,
            "session_id": "S%03d" % (session + 1),
            "block_id": "S%03d-B1" % (session + 1),
            "episode_index_within_session": index % max(1, episodes_per_session),
            "started_at": began.isoformat(),
            "ended_at": ended.isoformat(),
            "wall_duration_s": round(steps / float(fps), 3),
            "order_seed": order_seed,
            "condition_order": condition_label,
            "pose_seed": 300000 + index,
            "object_start_x": round(0.10 + 0.001 * index, 4),
            "object_start_y": round(-0.18 + 0.001 * index, 4),
            "object_start_yaw": 0.0,
            "drawn_start_x": round(0.10 + 0.001 * index, 4),
            "drawn_start_y": round(-0.18 + 0.001 * index, 4),
            "drawn_start_yaw": 0.0,
            "pose_placement_err_m": 0.0,
            "container_x": 0.05,
            "container_y": 0.20,
            "container_z": 0.02,
            "container_yaw": 0.0,
            "outcome": tag,
            "episode_time_s": round(steps / float(fps), 4),
            "control_hz": fps,
            "frames_nominal": steps,
            "num_frames": steps,
            "frames_recorded": steps,
            "frame_deficit": 0,
            "early_exit": False,
            "declared_episode_time_s": round(steps / float(fps), 4),
            "embodiment": binding.name,
            "sim_commit": "0" * 12,
            "lerobot_commit": "1" * 12,
            "calib_sha256_leader": digests["leader"],
            "calib_sha256_follower": digests["follower"],
            "fatigue": 2,
            "notes": "",
        }
        for joint in binding.joint_names:
            base = joint[:-4] if joint.endswith(".pos") else joint
            row[f"trk_mean_{base}"] = 0.4
            row[f"trk_max_{base}"] = 1.1
        for camera in cameras:
            row[f"dropped_{camera}"] = 0
        row.update(drift or {})
        rows.append({column: row.get(column, "") for column in columns})

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "BASE_COLUMNS",
    "CAMERAS",
    "GAIN",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "HOME",
    "calibration_hashes",
    "to_joints",
    "write_corpus",
]
