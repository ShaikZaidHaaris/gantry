"""RoboTwin demonstrations, read without RoboTwin.

The fake writes files shaped like the real ones — one HDF5 per episode, poses
split across four datasets per arm, camera frames stored as encoded bytes rather
than arrays. Each of those is somewhere this could quietly read the wrong thing.
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest
from gantry_connector_robotwin import ACTION, EXTRINSICS, IMAGE, STATE, RoboTwinDemos

from gantry.errors import ConfigError

STEPS = 12
SIZE = 8


def write_episode(path, steps=STEPS, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as handle:
        for index, arm in enumerate(("left", "right")):
            pose = np.zeros((steps, 7))
            pose[:, 0] = np.arange(steps) * 0.01 + index  # x walks, per arm
            pose[:, 3] = 1.0  # identity quaternion
            handle.create_dataset(f"endpose/{arm}_endpose", data=pose)
            handle.create_dataset(f"endpose/{arm}_gripper", data=np.full(steps, 0.1 * (index + 1)))
        handle.create_dataset("joint_action/vector", data=np.zeros((steps, 14)))

        # RoboTwin stores frames as encoded bytes in a fixed-width |S column.
        blobs = []
        for step in range(steps):
            image = Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            blobs.append(buffer.getvalue())
        widest = max(len(b) for b in blobs)
        handle.create_dataset(
            "observation/head_camera/rgb",
            data=np.array([b.ljust(widest, b"\0") for b in blobs], dtype=f"|S{widest}"),
        )
        extrinsic = np.tile(
            np.array([[1.0, 0, 0, 0.032], [0, -0.8, -0.6, 0.45], [0, 0.6, -0.8, 1.35]]),
            (steps, 1, 1),
        )
        handle.create_dataset(
            "observation/head_camera/extrinsic_cv", data=extrinsic.astype("float32")
        )


def demos(tmp_path, count=3, steps=STEPS, **kwargs):
    root = tmp_path / "aloha-agilex_clean_50" / "data"
    root.mkdir(parents=True)
    for index in range(count):
        write_episode(root / f"episode{index}.hdf5", steps=steps, seed=index)
    return RoboTwinDemos(tmp_path, task="pick_dual_bottles", **kwargs)


# -- the shape of the thing -----------------------------------------------------


def test_one_file_per_episode_is_indexed_as_a_directory():
    """Unlike RoboMimic, which packs every demo into a single file."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        made = demos(Path(tmp), count=4)
        assert len(made.episode_ids()) == 4
        assert made.descriptor().metadata["episodes"] == 4


def test_a_directory_with_no_episodes_says_what_to_point_at(tmp_path):
    with pytest.raises(ConfigError, match="one file per episode"):
        RoboTwinDemos(tmp_path)


def test_the_poses_are_assembled_in_the_action_order(tmp_path):
    """Same layout the evaluator reads at run time, from the same four pieces.
    A dataset laid out differently from the thing it is evaluated against trains
    without complaint and is wrong by a fixed rotation."""
    episode = demos(tmp_path).open("episode0")
    state = episode.array(STATE)
    assert state.shape[1] == 16
    assert np.isclose(state[0, 7], 0.1)  # left gripper
    assert np.isclose(state[0, 15], 0.2)  # right gripper
    assert state[0, 0] < state[0, 8]  # left arm's x starts below the right's


# -- the mistake that trains a policy to stand still ----------------------------


def test_the_action_is_the_next_pose_not_the_current_one():
    """Reading the rows straight across would train a policy to predict where it
    already is: a loss that falls beautifully and an arm that never moves."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        episode = demos(Path(tmp)).open("episode0")
        state, action = episode.array(STATE), episode.array(ACTION)
        assert np.allclose(action[:-1], state[1:])
        assert not np.allclose(action, state)


def test_the_episode_is_one_shorter_rather_than_padded(tmp_path):
    """The last step has no successor. Repeating it would teach the arm to stop."""
    episode = demos(tmp_path, steps=STEPS).open("episode0")
    assert len(episode) == STEPS - 1
    for name in (IMAGE, STATE, ACTION, EXTRINSICS):
        assert len(episode.array(name)) == STEPS - 1


# -- the frames are bytes, not arrays -------------------------------------------


def test_encoded_frames_are_decoded_rather_than_read_as_arrays(tmp_path):
    """The column is fixed-width |S bytes. Reading it as an array gives a string
    of the right length and entirely the wrong thing."""
    episode = demos(tmp_path).open("episode0")
    images = episode.array(IMAGE)
    assert images.shape == (STEPS - 1, SIZE, SIZE, 3)
    assert images.dtype == np.dtype("uint8")
    assert images.std() > 0  # actual picture content, not padding


def test_frames_can_be_downscaled_on_the_way_out(tmp_path):
    """Storing 1080p was the 44 GB that wedged this box once already."""
    episode = demos(tmp_path, size=(4, 4)).open("episode0")
    assert episode.array(IMAGE).shape == (STEPS - 1, 4, 4, 3)


# -- what travels with it -------------------------------------------------------


def test_the_camera_pose_travels_with_the_episode(tmp_path):
    """Without it, ego data cannot be put into this data's frame, and the two
    cannot be mixed."""
    episode = demos(tmp_path).open("episode0")
    assert episode.array(EXTRINSICS).shape == (STEPS - 1, 3, 4)


def test_demonstrations_are_recorded_as_successes(tmp_path):
    """RoboTwin ships demonstrations, not attempts. That is what makes them
    demonstrations, and an unlabelled one would be read as an abstention."""
    assert demos(tmp_path).open("episode0").labels.success is True


def test_the_licence_is_declared_because_a_dataset_inherits_it(tmp_path):
    made = demos(tmp_path)
    assert "MIT" in made.descriptor().metadata["licence"]
    assert "MIT" in made.open("episode0").meta.extra["licence"]


def test_the_action_channel_says_it_is_a_world_frame_pose(tmp_path):
    """The ego data is camera-frame. If neither says, they mix silently."""
    spec = [s for s in demos(tmp_path).schema("episode0") if s.name == ACTION][0]
    assert spec.frame == "world"
    assert spec.metadata["rotation_repr"] == "quat_wxyz"
