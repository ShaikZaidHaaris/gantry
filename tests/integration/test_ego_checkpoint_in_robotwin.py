"""The exact wiring the real run uses, with the far ends faked.

Three real components composed the way the box composes them: a RoboTwin
evaluator reading 16-wide quaternion poses, a pi0.5 server emitting 14-wide
Euler poses, and the adapter plane between them in both directions. Only the
simulator and the socket are fakes.

Every failure this pins cost a real run somewhere: a state converted in one
direction but not the other, an arm order that survives a conversion, a chunk
lifted wrongly. None of them raise -- they produce well-formed arrays that mean
something else.
"""

from __future__ import annotations

import numpy as np
from gantry_adapters_core import adapt_policy
from gantry_adapters_frame import PoseInFrame
from gantry_evaluator_robotwin import RoboTwinEvaluator, width_of
from gantry_policy_pi0 import Layout, Pi0Policy

from gantry.contracts.evaluator import Protocol

SIZE = 8
EGO_LABELS = tuple(
    f"{arm}_{part}"
    for arm in ("left", "right")
    for part in ("x", "y", "z", "rx", "ry", "rz", "gripper")
)

LAYOUT = Layout(
    name="ego_bimanual",
    images={"observation.head_camera.rgb": "cam_high"},
    state=14,
    action=14,
    state_from="endpose.vector",
    images_key="images",
    channels_first=True,
    labels=EGO_LABELS,
    arms=2,
    metadata={
        "rotation_repr": "euler_xyz",
        "rotation_offset": (3, 10),
        "semantics": "action.eef_abs_pose",
    },
    discriminators=("rotation_repr", "rotation_offset"),
    state_semantics="observation.eef_abs_pose",
)


class FakeServer:
    """An openpi websocket client, as far as Pi0Policy uses one."""

    def __init__(self, horizon=10):
        self.horizon = horizon
        self.seen: list[dict] = []

    def infer(self, payload):
        self.seen.append(payload)
        actions = np.zeros((self.horizon, 14), dtype="float32")
        actions[:, 5] = np.pi / 2  # left arm, a quarter turn in z
        actions[:, 6] = 0.04  # left gripper
        actions[:, 13] = 0.09  # right gripper
        return {"actions": actions}


class FakeTwin:
    """RoboTwin's own API, including the endpose dict and its scalar grippers."""

    def __init__(self, win_at=2):
        self.win_at = win_at
        self.steps = 0
        self.sent: list[np.ndarray] = []
        self.eval_success = False

    def setup_demo(self, now_ep_num=0, seed=0, is_test=True, **kwargs):
        self.steps = 0
        self.eval_success = False

    def get_instruction(self):
        return "pick up both bottles"

    #: RoboTwin's own published head-camera extrinsics, world -> camera.
    EXTRINSIC_CV = np.array(
        [[1.0, 0.0, 0.0, 0.032], [0.0, -0.8, -0.6, 0.45], [0.0, 0.6, -0.8, 1.35]],
        dtype="float32",
    )

    def get_obs(self):
        pose = np.zeros(7, dtype="float32")
        pose[3] = 1.0  # identity quaternion, scalar first
        return {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((SIZE, SIZE, 3), dtype="uint8"),
                    "extrinsic_cv": self.EXTRINSIC_CV,
                }
            },
            "joint_action": {"vector": np.zeros(14, dtype="float32")},
            "endpose": {
                "left_endpose": pose,
                "left_gripper": 0.3,
                "right_endpose": pose,
                "right_gripper": 0.7,
            },
        }

    def take_action(self, action, action_type="qpos"):
        assert action_type == "ee", action_type
        self.sent.append(np.asarray(action))
        self.steps += 1
        if self.steps >= self.win_at:
            self.eval_success = True

    def check_success(self):
        return self.steps >= self.win_at

    def close_env(self):
        pass


def built(win_at=2):
    envs = []

    def factory(task, **_):
        made = FakeTwin(win_at=win_at)
        envs.append(made)
        return made

    evaluator = RoboTwinEvaluator("pick_dual_bottles", action_type="ee", factory=factory, horizon=6)
    server = FakeServer()
    policy = adapt_policy(
        Pi0Policy(layout=LAYOUT, client=server, variant="pi05"),
        evaluator.action(),
        reading=evaluator.provides(),
    )
    return evaluator, policy, server, envs


def test_the_whole_chain_runs_and_the_widths_change_where_they_should():
    evaluator, policy, server, envs = built()
    record = evaluator.run(policy, evaluator.task_for(scenes=2), Protocol())

    # in: RoboTwin publishes 16 quaternion numbers, the checkpoint reads 14 Euler
    assert server.seen[0]["state"].shape == (14,)
    # out: the checkpoint emits 14, RoboTwin executes 16
    assert envs[0].sent[0].shape == (16,)
    assert record.episodes[0].array("action").shape[1] == width_of("ee")


def test_the_rotation_really_is_re_encoded_in_both_directions():
    evaluator, policy, server, envs = built()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())

    # An identity quaternion arriving becomes zero Euler angles going in.
    state = server.seen[0]["state"]
    assert np.allclose(state[3:6], 0.0, atol=1e-9)

    # A quarter turn in z going out becomes the matching quaternion.
    sent = envs[0].sent[0]
    half = np.sqrt(0.5)
    assert np.allclose(sent[3:7], [half, 0, 0, half], atol=1e-6)


def test_the_grippers_survive_both_conversions_in_the_right_slots():
    """They sit between the two rotation blocks, which is exactly where an
    adapter that handled only the first block would have displaced them."""
    evaluator, policy, server, envs = built()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())

    assert np.isclose(server.seen[0]["state"][6], 0.3)  # left, on the way in
    assert np.isclose(server.seen[0]["state"][13], 0.7)  # right, on the way in
    assert np.isclose(envs[0].sent[0][7], 0.04)  # left, on the way out
    assert np.isclose(envs[0].sent[0][15], 0.09)  # right, on the way out


def test_the_camera_reaches_the_server_under_the_key_the_config_uses():
    evaluator, policy, server, envs = built()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())
    image = server.seen[0]["images"]["cam_high"]
    assert image.shape == (3, SIZE, SIZE)  # channels first, as the Aloha configs want


def test_the_instruction_the_environment_chose_is_what_the_server_was_prompted_with():
    """A policy scored against a sentence it was not given fails in a way that
    reads as a low number."""
    evaluator, policy, server, envs = built()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())
    assert server.seen[0]["prompt"] == "pick up both bottles"


def test_the_conversion_is_on_the_record_not_just_in_the_wiring():
    evaluator, policy, server, envs = built()
    metadata = policy.descriptor().metadata
    assert metadata["action_adapter_chain"]
    assert metadata["action_adapter_losses"] == []


def test_a_success_the_environment_latched_is_the_recorded_outcome():
    evaluator, policy, server, envs = built(win_at=2)
    record = evaluator.run(policy, evaluator.task_for(scenes=3), Protocol())
    assert all(e.labels.success for e in record.episodes)
    assert record.metrics["success_rate"].value == 1.0


# -- and now in the right frame -------------------------------------------------
#
# The first real run scored 0/10 with its commands 0.30 m (left) and 0.65 m
# (right) from where the arms actually work. The ego pipeline solves hand poses
# against the camera's own intrinsics, so its poses are camera-frame, and
# Mount.aligned() passes them through unchanged; RoboTwin executes in world
# frame. Nothing in the widths, the encodings or the labels disagreed.
#
# The transform is read from what RoboTwin publishes about its own camera, not
# fitted to the commands. Nesting matters: the frame shift is outermost, so it
# works in 16-wide quaternion space on both sides and the rotation adapter
# underneath never sees a frame it did not expect.


def framed(win_at=2):
    envs = []

    def factory(task, **_):
        made = FakeTwin(win_at=win_at)
        envs.append(made)
        return made

    evaluator = RoboTwinEvaluator(
        "pick_dual_bottles", action_type="ee", factory=factory, horizon=6
    )
    server = FakeServer()
    policy = PoseInFrame(
        adapt_policy(
            Pi0Policy(layout=LAYOUT, client=server, variant="pi05"),
            evaluator.action(),
            reading=evaluator.provides(),
        ),
        extrinsics="observation.head_camera.extrinsic_cv",
        state_channels=("endpose.vector",),
    )
    return evaluator, policy, server, envs


def test_the_whole_stack_runs_with_the_frame_shift_outermost():
    evaluator, policy, server, envs = framed()
    record = evaluator.run(policy, evaluator.task_for(scenes=2), Protocol())

    assert server.seen[0]["state"].shape == (14,)   # still Euler at the server
    assert envs[0].sent[0].shape == (16,)           # still quaternions at the simulator
    assert record.episodes[0].array("action").shape[1] == width_of("ee")


def test_the_camera_extrinsics_reach_the_wrapper_through_the_evaluator():
    """flatten() has to keep them: they arrive under the same head_camera key as
    the image, and a camera filter that dropped everything but rgb would take
    the transform with it."""
    evaluator, policy, server, envs = framed()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())
    assert policy._seen is not None
    assert policy._seen.shape == (4, 4)


def test_the_commands_land_in_world_coordinates_not_camera_ones():
    """The failure being fixed. The server returns zeros, which in the policy's
    own frame means the camera origin -- so the command must come out at where
    the camera is in the world, not at the world origin."""
    evaluator, policy, server, envs = framed()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())
    sent = envs[0].sent[0]
    assert np.allclose(sent[0:3], [-0.032, -0.45, 1.35], atol=1e-5)
    assert np.allclose(sent[8:11], [-0.032, -0.45, 1.35], atol=1e-5)


def test_the_state_handed_to_the_server_is_in_the_frame_it_trained_in():
    """RoboTwin's arms sit around z=0.92 in world. The ego data trained around
    z=0.45 in camera coordinates. Without the shift the server sees a state half
    a metre from anything it has ever seen."""
    evaluator, policy, server, envs = framed()
    evaluator.run(policy, evaluator.task_for(scenes=1), Protocol())
    state = server.seen[0]["state"]
    # the fake's arms are at the world origin, which is 1.35 m below the camera
    assert state[2] > 1.0
    assert not np.isclose(state[2], 0.0)


def test_the_frames_are_on_the_record():
    evaluator, policy, server, envs = framed()
    metadata = policy.descriptor().metadata
    assert metadata["pose_frame_from"] == "camera"
    assert metadata["pose_frame_to"] == "world"
    # and the conversion underneath is still recorded too
    assert metadata["action_adapter_chain"]
