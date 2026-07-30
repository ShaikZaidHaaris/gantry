"""ManiSkill's shape, checked without ManiSkill.

The fake copies the real API where it matters: a nested observation dict, a
leading batch axis of one, tensor-like values with ``.cpu()``/``.numpy()``, and
``info["success"]`` as a one-element array rather than a bool. Those four are
where an adapter goes wrong, and they cost nothing to reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_maniskill import (
    CONTROL_MODES,
    ManiskillEvaluator,
    flatten,
    numeric,
    unbatch,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec


class Tensorish:
    """What a torch CUDA tensor looks like to code that must not import torch.

    ``np.asarray`` on the real thing raises, so a fake that is secretly a numpy
    array would let a missing ``.cpu()`` pass the tests and fail on a GPU.
    """

    def __init__(self, array):
        self._array = np.asarray(array)
        self.moved = False

    def detach(self):
        return self

    def cpu(self):
        self.moved = True
        return self

    def numpy(self):
        if not self.moved:
            raise AssertionError("read without being moved off the device first")
        return self._array

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert a device tensor to numpy directly")


class Space:
    def __init__(self, width):
        self.shape = (1, width)


class FakeEnv:
    """One ManiSkill environment, batch of one."""

    def __init__(self, width=7, win_at=3, success_key="success"):
        self.single_action_space = Space(width)
        self.win_at = win_at
        self.success_key = success_key
        self.step_count = 0
        self.seeds: list[int | None] = []
        self.closed = False

    def _observation(self):
        return {
            "agent": {"qpos": Tensorish(np.zeros((1, 9), dtype="float32"))},
            "sensor_data": {
                "base_camera": {
                    "rgb": Tensorish(np.zeros((1, 8, 8, 3), dtype="uint8")),
                    "depth": Tensorish(np.zeros((1, 8, 8, 1), dtype="uint16")),
                }
            },
            "extra": {"tcp_pose": Tensorish(np.zeros((1, 7), dtype="float32"))},
        }

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.step_count = 0
        return self._observation(), {}

    def step(self, action):
        self.step_count += 1
        won = self.step_count >= self.win_at
        info = {self.success_key: Tensorish(np.asarray([won]))}
        return (
            self._observation(),
            Tensorish(np.asarray([1.0 if won else 0.0], dtype="float32")),
            Tensorish(np.asarray([won])),
            Tensorish(np.asarray([False])),
            info,
        )

    def close(self):
        self.closed = True


class Chunker(Policy):
    def __init__(self, width=7):
        self.width = width
        self.spec = ChannelSpec("action", "vector", (width,), "float32", semantics="actuation")

    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=4, chunk=True, deterministic=True
        )

    def action_spec(self):
        return self.spec

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((4, self.width), dtype="float32")


def evaluator(width=7, win_at=3, **kwargs):
    envs = []

    def factory(task, **_):
        env = FakeEnv(width=width, win_at=win_at)
        envs.append(env)
        return env

    made = ManiskillEvaluator("PickCube-v1", factory=factory, horizon=10, **kwargs)
    made.envs = envs
    return made


# -- the conversion, which is where this kind of adapter usually breaks -------


def test_device_tensors_are_moved_before_being_read():
    tensor = Tensorish(np.arange(3))
    assert numeric(tensor).tolist() == [0, 1, 2]
    with pytest.raises(TypeError):
        np.asarray(tensor)  # the fake behaves like the real thing


def test_unbatch_drops_the_batch_axis_and_not_a_real_dimension_of_one():
    """A squeeze would collapse a single-channel depth image too, quietly
    changing the shape a policy is handed."""
    assert unbatch(np.zeros((1, 8, 8, 1))).shape == (8, 8, 1)
    assert unbatch(np.zeros((4, 7))).shape == (4, 7)
    assert unbatch(np.zeros((1, 1))).shape == (1,)


def test_the_nested_observation_becomes_dotted_channel_names():
    flat = flatten(FakeEnv()._observation())
    assert sorted(flat) == [
        "agent.qpos",
        "extra.tcp_pose",
        "sensor_data.base_camera.depth",
        "sensor_data.base_camera.rgb",
    ]
    assert flat["sensor_data.base_camera.rgb"].shape == (8, 8, 3)
    assert flat["agent.qpos"].shape == (9,)


# -- the evaluator -----------------------------------------------------------


def test_the_action_width_is_read_from_the_environment_not_guessed():
    """The mode-to-width mapping depends on the robot's DoF and has changed
    between releases; asking is the only version-proof answer."""
    assert evaluator(width=7).action().shape == (7,)
    assert evaluator(width=8, control_mode="pd_joint_delta_pos").action().shape == (8,)


def test_an_unknown_control_mode_is_refused_at_construction():
    with pytest.raises(ConfigError, match="control mode"):
        ManiskillEvaluator("PickCube-v1", control_mode="pd_vibes", factory=lambda *a, **k: None)
    assert "pd_ee_delta_pose" in CONTROL_MODES


def test_an_environment_with_no_action_space_is_refused_rather_than_assumed():
    class Spaceless(FakeEnv):
        def __init__(self):
            super().__init__()
            self.single_action_space = None
            self.action_space = None

    made = ManiskillEvaluator("PickCube-v1", factory=lambda *a, **k: Spaceless())
    with pytest.raises(ConfigError, match="no.*action space"):
        made.action()


def test_success_comes_out_of_info_as_a_bool_and_ends_the_trial():
    made = evaluator(win_at=3)
    record = made.run(Chunker(), made.task_for(scenes=4), Protocol())
    assert all(episode.labels.success for episode in record.episodes)
    assert record.metrics["success_rate"].value == 1.0
    assert record.episodes[0].array("action").shape[0] == 3


def test_a_timeout_here_really_is_a_failure_because_the_suite_says_so():
    """Not the same call as on a real bench, and the difference is the suite's to
    make. ManiSkill evaluates its success predicate every step, so an attempt
    that ran out of horizon has been checked and found wanting — a genuine
    False. A world with no predicate returns None instead and the trial is an
    abstention. The loop records whichever the suite actually reports."""
    made = evaluator(win_at=999)
    record = made.run(Chunker(), made.task_for(scenes=2, horizon=6), Protocol())
    assert [e.labels.success for e in record.episodes] == [False, False]
    assert record.metrics["success_rate"].value == 0.0
    assert record.metrics["success_rate"].n == 2


def test_scenes_are_seeds_and_the_seed_reaches_the_environment():
    made = evaluator()
    made.run(Chunker(), made.task_for(scenes=3), Protocol())
    assert made.envs[0].seeds == [0, 1, 2]


def test_the_body_is_one_argument_and_lands_in_the_record():
    """The claim this plugin exists for. Every other suite here has one arm
    welded in, so 'does it transfer' is a question they cannot be asked."""
    panda = evaluator()
    assert panda.descriptor().provides["hosts_embodiment"] is True

    widow = panda.for_embodiment("widowx250s")
    assert widow.descriptor().metadata["robot"] == "widowx250s"

    record = widow.run(Chunker(), widow.task_for(scenes=1), Protocol())
    assert record.episodes[0].meta.embodiment == "widowx250s"


def test_an_unlisted_robot_is_passed_through_rather_than_rejected():
    """ROBOTS is documentation. A ManiSkill release adding an arm must not need
    this file edited before it can be evaluated."""
    made = evaluator().for_embodiment("some_new_arm_v2")
    assert made.descriptor().metadata["robot"] == "some_new_arm_v2"


def test_it_does_not_claim_a_throughput_it_cannot_deliver():
    """The parallelism needs a batched policy contract. Until that exists the
    descriptor says so rather than letting 'GPU-parallel simulator' be read as a
    property of this harness."""
    metadata = evaluator().descriptor().metadata
    assert metadata["num_envs"] == 1
    assert "one observation at a time" in metadata["batched"]


def test_observes_narrows_the_recorded_channels():
    made = evaluator(observes=("sensor_data.base_camera.rgb", "agent.qpos"))
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    assert set(record.episodes[0].channel_names) == {
        "sensor_data.base_camera.rgb",
        "agent.qpos",
        "action",
        "reward",
    }


def test_a_maniskill_evaluator_conforms():
    made = evaluator()
    verdict = check_evaluator(made, Chunker(), made.task_for(scenes=2))
    assert verdict.ok, verdict.explain()


def test_close_releases_the_environment():
    made = evaluator()
    made.run(Chunker(), made.task_for(scenes=1), Protocol())
    env = made.envs[0]
    made.close()
    assert env.closed


@pytest.mark.skip(reason="needs the ManiSkill simulator and a CUDA-capable torch")
def test_against_the_real_simulator():  # pragma: no cover
    made = ManiskillEvaluator("PickCube-v1", robot="panda")
    assert made.action().shape == (7,)
