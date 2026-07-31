"""Poses moved between frames, and the ways that goes wrong quietly.

Every failure here produces a well-formed array. A pose shifted by the wrong
transform is still a pose; an orientation translated along with its position is
still a valid rotation. None of it raises, and none of it is visible in a shape
check — which is the whole reason this is a declared component rather than three
lines in a run script.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_adapters_frame import PoseInFrame, blocks_of, invert, rigid, shift

from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec

# Two arms, [xyz, wxyz, gripper] each — the layout RoboTwin reads.
POSE = ChannelSpec(
    "action",
    "vector",
    (16,),
    "float32",
    semantics="action.eef_abs_pose",
    metadata={"rotation_repr": "quat_wxyz", "rotation_offset": (3, 11), "arms": 2},
)

# RoboTwin's own published head-camera extrinsics: world -> camera, OpenCV.
EXTRINSIC_CV = np.array([[1.0, 0.0, 0.0, 0.032], [0.0, -0.8, -0.6, 0.45], [0.0, 0.6, -0.8, 1.35]])


def pose_at(position, quaternion=(1.0, 0.0, 0.0, 0.0)):
    """One step of a two-armed command with both arms at the same place."""
    arm = np.concatenate([np.asarray(position, dtype=float), np.asarray(quaternion), [0.05]])
    return np.concatenate([arm, arm])[None, :]


# -- the transform itself -------------------------------------------------------


def test_a_three_by_four_is_completed_rather_than_refused():
    """Renderers publish [R | t]. Refusing it would mean every caller pastes on
    the same bottom row."""
    assert rigid(EXTRINSIC_CV).shape == (4, 4)
    assert np.allclose(rigid(EXTRINSIC_CV)[3], [0, 0, 0, 1])


def test_a_shape_that_is_not_a_transform_is_refused_not_reshaped():
    with pytest.raises(ConfigError, match="applies cleanly and means nothing"):
        rigid(np.zeros((3, 3)))


def test_the_inverse_is_exact_because_the_rotation_is_orthonormal():
    transform = rigid(EXTRINSIC_CV)
    assert np.allclose(invert(transform) @ transform, np.eye(4), atol=1e-12)


def test_a_transform_that_is_not_rigid_is_refused_rather_than_transposed():
    """Inverting a scaled or skewed transform by transpose is silently wrong."""
    scaled = np.eye(4)
    scaled[0, 0] = 2.0
    with pytest.raises(ConfigError, match="not a rigid motion"):
        invert(scaled)


# -- what a frame shift does, and does not, do ----------------------------------


def test_a_position_is_rotated_and_translated():
    """The camera sits at world (-0.032, -0.45, 1.35). A point at the camera's
    own origin must land there."""
    to_world = invert(rigid(EXTRINSIC_CV))
    out = shift(pose_at([0.0, 0.0, 0.0]), POSE, to_world)
    assert np.allclose(out[0, 0:3], [-0.032, -0.45, 1.35], atol=1e-9)


def test_an_orientation_is_rotated_but_not_translated():
    """The classic version of this mistake. Translating an orientation along
    with its position produces rotations that drift with how far the origin
    moved, and every one of them is still a unit quaternion."""
    far = np.eye(4)
    far[:3, 3] = [10.0, -5.0, 3.0]  # translation only
    out = shift(pose_at([0.0, 0.0, 0.0]), POSE, far)
    assert np.allclose(out[0, 0:3], [10.0, -5.0, 3.0])
    assert np.allclose(out[0, 3:7], [1.0, 0.0, 0.0, 0.0])  # untouched


def test_both_arms_move_not_just_the_first():
    to_world = invert(rigid(EXTRINSIC_CV))
    out = shift(pose_at([0.1, 0.2, 0.3]), POSE, to_world)
    assert np.allclose(out[0, 0:3], out[0, 8:11])
    assert not np.allclose(out[0, 8:11], [0.1, 0.2, 0.3])


def test_the_grippers_are_left_where_they_are():
    to_world = invert(rigid(EXTRINSIC_CV))
    out = shift(pose_at([0.1, 0.2, 0.3]), POSE, to_world)
    assert np.isclose(out[0, 7], 0.05) and np.isclose(out[0, 15], 0.05)


def test_there_and_back_is_the_identity():
    to_world = invert(rigid(EXTRINSIC_CV))
    values = pose_at([0.1, -0.2, 0.45], quaternion=(np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    back = shift(shift(values, POSE, to_world), POSE, rigid(EXTRINSIC_CV))
    assert np.allclose(back, values, atol=1e-9)


def test_it_reproduces_the_offsets_measured_on_the_box():
    """The reason this component exists. The ego training set puts the hands at
    these camera-frame means; RoboTwin's arms worked at these world-frame means.
    Untransformed the gap was 0.302 m and 0.649 m."""
    to_world = invert(rigid(EXTRINSIC_CV))
    trained = {"left": [-0.006, 0.107, 0.449], "right": [0.103, 0.094, 0.543]}
    worked = {"left": [-0.001, -0.164, 0.927], "right": [0.137, -0.050, 0.911]}

    values = np.concatenate(
        [
            np.concatenate([np.array(trained[arm]), [1.0, 0, 0, 0], [0.05]])
            for arm in ("left", "right")
        ]
    )[None, :]
    out = shift(values, POSE, to_world)

    for index, arm in enumerate(("left", "right")):
        gap = np.linalg.norm(out[0, index * 8 : index * 8 + 3] - np.array(worked[arm]))
        assert gap < 0.2, f"{arm} still {gap:.3f} m out"


# -- what has to be declared ----------------------------------------------------


def test_a_channel_that_does_not_say_where_its_rotations_are_is_refused():
    bare = ChannelSpec(
        "action", "vector", (16,), "float32", metadata={"rotation_repr": "quat_wxyz"}
    )
    with pytest.raises(ConfigError, match="does not say where its rotations start"):
        blocks_of(bare)


def test_a_channel_with_no_rotation_encoding_is_refused():
    """A frame shift applied to the wrong three numbers is undetectable
    downstream, so it will not guess which three they are."""
    bare = ChannelSpec("action", "vector", (16,), "float32", metadata={"rotation_offset": (3, 11)})
    with pytest.raises(ConfigError, match="no way to know"):
        blocks_of(bare)


def test_a_rotation_with_no_room_for_a_position_before_it_is_refused():
    cramped = ChannelSpec(
        "action",
        "vector",
        (8,),
        "float32",
        metadata={"rotation_repr": "quat_wxyz", "rotation_offset": (1,)},
    )
    with pytest.raises(ConfigError, match="no room for the position"):
        blocks_of(cramped)


# -- the wrapper ----------------------------------------------------------------


class Echo(Policy):
    """Commands whatever it was shown, so the round trip is visible."""

    def __init__(self):
        self.saw = None

    def descriptor(self):
        return policy_descriptor(name="echo", version="0.1", horizon=1, chunk=1, deterministic=True)

    def action_spec(self):
        return POSE

    def observes(self):
        return requires_channels("echo", "policy")

    def reset(self, context):
        pass

    def act(self, observation):
        channels = dict(getattr(observation, "channels", observation))
        self.saw = channels["endpose.vector"]
        return np.array(self.saw, copy=True)


def wrapped(policy=None):
    return PoseInFrame(
        policy or Echo(),
        extrinsics="observation.head_camera.extrinsic_cv",
        state_channels=("endpose.vector",),
    )


def test_the_state_arrives_in_the_policys_frame_and_the_command_leaves_in_the_worlds():
    """One transform and its exact inverse, so a policy that simply repeats what
    it saw commands the arms to stay exactly where they are."""
    inner = Echo()
    made = wrapped(inner)
    here = pose_at([0.2, -0.3, 0.9])[0]
    out = made.act({"endpose.vector": here, "observation.head_camera.extrinsic_cv": EXTRINSIC_CV})
    # the policy saw camera coordinates, not world ones
    assert not np.allclose(inner.saw[0:3], here[0:3])
    # and what came back is where the arms already were
    assert np.allclose(out[0:3], here[0:3], atol=1e-9)


def test_a_world_that_publishes_no_camera_pose_is_refused_with_the_reason():
    made = wrapped()
    with pytest.raises(ConfigError, match="publishes no"):
        made.act({"endpose.vector": pose_at([0, 0, 0])[0]})


def test_the_declared_action_frame_is_the_worlds():
    """A pose channel that does not say what it is relative to binds to anything
    of the same width."""
    assert wrapped().action_spec().frame == "world"


def test_the_frames_are_recorded_on_the_descriptor():
    metadata = wrapped().descriptor().metadata
    assert metadata["pose_frame_from"] == "camera"
    assert metadata["pose_frame_to"] == "world"
    assert metadata["pose_frame_source"].endswith("extrinsic_cv")
