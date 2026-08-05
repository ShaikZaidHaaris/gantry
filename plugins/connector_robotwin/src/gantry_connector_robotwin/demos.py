"""RoboTwin's own demonstrations, read as episodes.

The evaluator can run a policy in RoboTwin; this reads what RoboTwin's scripted
expert did there. Together they close the loop the ego experiment was missing:
a policy that has never seen a task cannot be asked whether extra data improved
it, because it scores zero either way and zero minus zero is not a measurement.

One file per episode, which is not how the other HDF5 connector here works
(RoboMimic packs every demo into one file). So this indexes a directory rather
than a file, and an episode's length is its own.

The action space is the pose, not the joints
--------------------------------------------
Each file carries both: ``joint_action`` for the arms' angles and ``endpose``
for where the grippers actually are. This reads the pose, for one reason -- it is
the only space the ego pipeline can also produce. ``retargeter_hands`` refuses to
turn hand poses into joint angles and says why. Two datasets that cannot be
expressed in one action space cannot be mixed, and mixing them is the whole
point of the ablation this feeds.

Actions are the next pose, not the current one
-----------------------------------------------
``endpose[t]`` is where the arms *were* at step t. The command that produced
step t+1 is ``endpose[t+1]``, so that is what the action at t must be. Reading
the file's rows straight across would train a policy to predict where it already
is -- a loss that falls beautifully and a policy that never moves.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from gantry_evaluator_robotwin import ARMS, CONTROL_HZ, labels_for, state_spec, width_of

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

STATE = "observation.state"
ACTION = "action"
IMAGE = "observation.images.head"
EXTRINSICS = "observation.head_camera.extrinsic_cv"


def action_spec(name: str = ACTION) -> ChannelSpec:
    """The pose command, in the layout and encoding RoboTwin executes."""
    spec = state_spec(name)
    from dataclasses import replace

    return replace(spec, semantics="action.eef_abs_pose", frame="world")


def _decode(raw: Any) -> np.ndarray:
    """One frame, from whatever the file stored it as.

    RoboTwin stores camera frames as encoded bytes rather than arrays -- a
    fixed-width ``|S`` column of PNG or JPEG. Reading it as an array gives a
    string of the right length and entirely the wrong thing.
    """
    from PIL import Image

    data = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype="uint8")


class RoboTwinDemos(Connector):
    """A directory of RoboTwin episode files."""

    def __init__(
        self,
        root: str | Path,
        *,
        task: str = "",
        embodiment: str = "aloha-agilex",
        camera: str = "head_camera",
        name: str = "robotwin_demos",
        size: tuple[int, int] | None = None,
        licence: str = "MIT (RoboTwin 2.0)",
    ):
        self._root = Path(root)
        files = sorted(self._root.rglob("episode*.hdf5"), key=lambda p: (len(p.stem), p.stem))
        if not files:
            raise ConfigError(
                f"{name}: no episode*.hdf5 under {self._root}. RoboTwin ships one file per "
                "episode; point this at the directory the archive unpacked into"
            )
        self._files = {path.stem: path for path in files}
        self._task = task or self._root.parent.name
        self._embodiment = embodiment
        self._camera = camera
        self._name = name
        self._size = size
        self._licence = licence

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return connector_descriptor(
            name=self._name,
            version=VERSION,
            episodes=len(self._files),
            # Everything is read into memory per episode: a RoboTwin file is one
            # episode of a couple of hundred small frames, not a corpus.
            lazy=False,
            # A demonstration has an outcome (it succeeded) and no milestones
            # along the way -- RoboTwin records the trajectory, not the stages.
            stage_events=False,
            outcomes=True,
            observations=[IMAGE, STATE, EXTRINSICS],
            actions=[ACTION],
            task=self._task,
            embodiment=self._embodiment,
            camera=self._camera,
            control_hz=CONTROL_HZ,
            action_type="ee",
            arms=len(ARMS),
            # A dataset inherits the licence of what it was built from, and this
            # one is permissive. Said here so lineage does not have to guess.
            licence=self._licence,
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._files)

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        self._require(episode_id)
        height, width = self._size or (None, None)
        return (
            ChannelSpec(
                IMAGE,
                "image",
                (height, width, 3),
                "uint8",
                rate_hz=CONTROL_HZ,
                semantics="observation.image",
                metadata={"camera": self._camera},
            ),
            state_spec(STATE),
            action_spec(ACTION),
            ChannelSpec(
                EXTRINSICS,
                "matrix",
                (3, 4),
                "float32",
                semantics="observation.camera_extrinsics",
                metadata={"convention": "opencv", "direction": "world_to_camera"},
            ),
        )

    def open(self, episode_id: str) -> EpisodeRecord:
        self._require(episode_id)
        with h5py.File(self._files[episode_id], "r") as handle:
            poses = self._poses(handle)
            frames = handle[f"observation/{self._camera}/rgb"]
            images = np.stack([self._frame(frames[index]) for index in range(len(poses))])
            extrinsics = np.asarray(
                handle[f"observation/{self._camera}/extrinsic_cv"], dtype="float32"
            )

        # The command that produced step t+1 is the pose at t+1. The last step
        # has no successor, so the episode is one shorter rather than padded
        # with a repeat that would teach the arm to stop.
        states, actions = poses[:-1], poses[1:]
        images, extrinsics = images[:-1], extrinsics[: len(states)]

        arrays = {
            IMAGE: images,
            STATE: states.astype("float32"),
            ACTION: actions.astype("float32"),
            EXTRINSICS: extrinsics,
        }
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._name,
                embodiment=self._embodiment,
                task=self._task,
                extra={"path": str(self._files[episode_id]), "licence": self._licence},
            ),
            schema=self.schema(episode_id),
            source=ArraySource(arrays),
            # RoboTwin ships demonstrations, not attempts: every one of these
            # succeeded, which is what makes them demonstrations.
            labels=EpisodeLabels(success=True),
        )

    # -- internals ---------------------------------------------------------

    def _require(self, episode_id: str) -> None:
        if episode_id not in self._files:
            raise KeyError(f"{self._name}: no episode {episode_id!r}")

    def _poses(self, handle: h5py.File) -> np.ndarray:
        """The two arms' poses, in the order an ``ee`` action is read.

        Assembled the same way the evaluator assembles them at run time, from
        the same four pieces, so a demonstration and a rollout describe the arms
        in the same layout. A dataset laid out differently from the thing it is
        evaluated against trains without complaint and is wrong by a fixed
        rotation.
        """
        pieces = []
        for arm in ARMS:
            pose = np.asarray(handle[f"endpose/{arm}_endpose"], dtype=float)
            grip = np.asarray(handle[f"endpose/{arm}_gripper"], dtype=float).reshape(-1, 1)
            pieces.extend([pose, grip])
        out = np.concatenate(pieces, axis=1)
        if out.shape[1] != width_of("ee"):
            raise ConfigError(
                f"{self._name}: assembled {out.shape[1]} numbers per step, expected "
                f"{width_of('ee')} for {labels_for('ee')}"
            )
        return out

    def _frame(self, raw: Any) -> np.ndarray:
        image = _decode(raw)
        if self._size is None:
            return image
        from PIL import Image

        height, width = self._size
        return np.asarray(Image.fromarray(image).resize((width, height)), dtype="uint8")
