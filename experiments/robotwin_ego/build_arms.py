"""The three training sets the ablation needs.

    base       RoboTwin's own demonstrations
    ego        the same, plus the ego data
    shuffled   the same, plus the ego data with its actions detached

The first run of this experiment had only the last two, and both scored zero,
because a policy trained on kitchen video has no way to pick up a bottle in a
simulator it has never seen. Zero minus zero is not a measurement. ``base`` is
the arm that can actually do the task; the question then becomes whether adding
ego data moves it, and whether it moves it *more than adding scrambled ego data
does*. Only the second of those is attributable to the data.

Everything lands in one action space
------------------------------------
Two datasets that cannot be expressed in the same space cannot be mixed, so both
sources are written as RoboTwin's ``ee`` command: sixteen numbers, position and
scalar-first quaternion and gripper per arm, in RoboTwin's world frame.

RoboTwin's demonstrations are already there. The ego data needs two conversions,
and both are done with the same components the evaluator uses at run time rather
than with arithmetic written here:

    euler_xyz (14) -> quat_wxyz (16)     the rotation adapter
    camera frame   -> world frame        the frame shift, through RoboTwin's
                                         own published camera extrinsics

Doing it at build time rather than at inference means the policy is *trained* in
the frame it will be *run* in, which is the thing the first attempt got wrong.

The control is built from the converted data, not before it
------------------------------------------------------------
The shuffled arm detaches actions from the frames they belong to. That has to
happen after the conversions, so that treatment and control differ in exactly
one thing — the correspondence — and not also in what space they are in.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gantry_adapters_frame import invert, rigid, shift
from gantry_adapters_rotation.rotation import convert as reencode
from gantry_connector_lerobot import write_episodes
from gantry_connector_robotwin import ACTION, IMAGE, STATE, RoboTwinDemos, action_spec
from gantry_evaluator_robotwin import state_spec

from gantry.errors import ConfigError
from gantry.spine import ArraySource, ChannelSpec, EpisodeLabels, EpisodeMeta, EpisodeRecord

DEMOS = Path("/home/ubuntu/rt_data/demos")
EGO = Path("/home/ubuntu/.cache/huggingface/lerobot/gantry/ego")
OUT = Path("/home/ubuntu/.cache/huggingface/lerobot/gantry")
SIZE = (224, 224)
FPS = 25.0
TASK = "pick_dual_bottles"

#: What the ego pipeline emitted: a pose per arm with Euler angles, in the frame
#: the camera saw them in. Declared so the conversions are planned, not assumed.
EGO_LABELS = tuple(
    f"{arm}_{part}"
    for arm in ("left", "right")
    for part in ("x", "y", "z", "rx", "ry", "rz", "gripper")
)
EGO_POSE = ChannelSpec(
    "pose", "vector", (14,), "float32", frame="camera", semantics="action.eef_abs_pose",
    dim_labels=EGO_LABELS,
    metadata={"rotation_repr": "euler_xyz", "rotation_offset": (3, 10), "arms": 2},
)
RT_POSE = action_spec("pose")


def robotwin_episodes() -> list[EpisodeRecord]:
    connector = RoboTwinDemos(DEMOS, task=TASK, size=SIZE)
    episodes = [connector.open(name) for name in connector.episode_ids()]
    print(f"robotwin: {len(episodes)} episodes, {sum(len(e) for e in episodes)} frames", flush=True)
    return episodes


def camera_to_world() -> np.ndarray:
    """RoboTwin's own head-camera pose, from its own demonstrations.

    Taken from the data rather than from a live simulator so that building the
    training set does not need one, and so the transform is the same one the
    demonstrations were recorded under.
    """
    connector = RoboTwinDemos(DEMOS, task=TASK, size=(1, 1))
    first = connector.open(connector.episode_ids()[0])
    published = first.array("observation.head_camera.extrinsic_cv")[0]
    return invert(rigid(published))


def decode(path: Path, steps: int) -> np.ndarray:
    """The frames of one LeRobot video.

    LeRobot v2.1 stores images as h264 rather than in the parquet, so the pixel
    column simply is not there. Reading the table and finding no images looked,
    for one build, like a dataset with no cameras -- which would have trained
    the ego half of this ablation blind while the RoboTwin half could see.
    """
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) < steps:
        raise ConfigError(
            f"{path.name} decoded {len(frames)} frames for {steps} rows; the video and "
            "the table disagree about how long this episode is"
        )
    return np.stack(frames[:steps]).astype("uint8")


def ego_episodes(to_world: np.ndarray) -> list[EpisodeRecord]:
    """The ego data, in RoboTwin's action space and RoboTwin's frame."""
    out: list[EpisodeRecord] = []
    for path in sorted(glob.glob(str(EGO / "data" / "*" / "*.parquet"))):
        frame = pd.read_parquet(path)
        state = np.stack(frame["observation.state"].values).astype(float)
        action = np.stack(frame["action"].values).astype(float)

        video = EGO / "videos" / Path(path).parent.name / "observation.images.ego_rgb" / (
            Path(path).stem + ".mp4"
        )
        arrays = {IMAGE: decode(video, len(state))}
        for name, values in ((STATE, state), (ACTION, action)):
            wide = reencode(values, EGO_POSE, RT_POSE)   # euler -> quaternion
            arrays[name] = shift(wide, RT_POSE, to_world).astype("float32")  # camera -> world

        episode = EpisodeRecord(
            meta=EpisodeMeta(
                id=f"ego_{len(out):03d}",
                source="ego",
                embodiment="aloha-agilex",
                task=TASK,
                extra={"origin": "epic_kitchens", "converted": "euler->quat, camera->world"},
            ),
            schema=(
                ChannelSpec(IMAGE, "image", (*SIZE, 3), "uint8", rate_hz=FPS,
                            semantics="observation.image"),
                state_spec(STATE),
                action_spec(ACTION),
            ),
            source=ArraySource(arrays),
            labels=EpisodeLabels(success=None),
        )
        out.append(episode)
    print(f"ego: {len(out)} episodes, {sum(len(e) for e in out)} frames", flush=True)
    return out


def detach(episodes: list[EpisodeRecord], seed: int = 0) -> list[EpisodeRecord]:
    """The same frames and the same actions, with no relationship between them.

    Shuffled across the whole pool rather than within each episode: shuffling
    inside an episode leaves the actions still drawn from the same few seconds
    of the same scene, which is a much weaker scrambling than it looks.
    """
    rng = np.random.default_rng(seed)
    pool = np.concatenate([e.array(ACTION) for e in episodes])
    rng.shuffle(pool)

    out, cursor = [], 0
    for episode in episodes:
        steps = len(episode)
        arrays = {name: episode.array(name) for name in episode.channel_names}
        arrays[ACTION] = pool[cursor : cursor + steps]
        cursor += steps
        out.append(
            EpisodeRecord(
                meta=EpisodeMeta(
                    id=f"shuffled_{episode.meta.id}",
                    source="ego_shuffled",
                    embodiment=episode.meta.embodiment,
                    task=episode.meta.task,
                    extra={**dict(episode.meta.extra), "control": "actions detached"},
                ),
                schema=episode.schema,
                source=ArraySource(arrays),
                labels=episode.labels,
            )
        )
    return out


def main() -> None:
    to_world = camera_to_world()
    print("camera sits in world at", np.round(to_world[:3, 3], 3), flush=True)

    robotwin = robotwin_episodes()
    ego = ego_episodes(to_world)

    # Reported so the mixing ratio is a fact rather than an assumption. A
    # treatment arm that is 90% one source is measuring that source.
    rt_frames = sum(len(e) for e in robotwin)
    ego_frames = sum(len(e) for e in ego)
    print(f"mix: {rt_frames} robotwin + {ego_frames} ego = {ego_frames / (rt_frames + ego_frames):.0%} ego")

    arms = {
        "rt_base": robotwin,
        "rt_ego": robotwin + ego,
        "rt_shuffled": robotwin + detach(ego),
    }
    summary = {}
    for name, episodes in arms.items():
        report = write_episodes(
            episodes, OUT / name, fps=FPS, robot_type="aloha-agilex",
            accept_loss=True, videos=True,
        )
        summary[name] = {"episodes": len(episodes), "frames": sum(len(e) for e in episodes)}
        print(f"wrote {name}: {summary[name]}  {report}", flush=True)

    Path("/home/ubuntu/egorun/rt_arms.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
