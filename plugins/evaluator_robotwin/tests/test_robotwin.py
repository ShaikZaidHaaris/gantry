"""RoboTwin's shape, checked without RoboTwin.

The fake copies the API where it differs from everything else here: get_obs /
take_action(action, action_type) / check_success / get_instruction rather than a
gym five-tuple, and a nested observation with three cameras.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_robotwin import (
    ACTION_TYPES,
    TASKS,
    RoboTwinEvaluator,
    flatten,
    for_ego,
    labels_for,
    width_of,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec

SIZE = 8


class FakeTask:
    """What RoboTwin's class_decorator(task) returns, as far as this plugin sees."""

    def __init__(self, win_at=3, instruction="pick up both bottles"):
        self.win_at = win_at
        self._instruction = instruction
        self.steps = 0
        self.seeds: list[int] = []
        self.sent: list[tuple[np.ndarray, str]] = []
        self.closed = False

    def setup_demo(self, now_ep_num=0, seed=0, is_test=True, **kwargs):
        self.seeds.append(seed)
        self.steps = 0

    def get_instruction(self):
        return self._instruction

    def get_obs(self):
        return {
            "observation": {
                "head_camera": {"rgb": np.zeros((SIZE, SIZE, 3), dtype="uint8")},
                "left_camera": {"rgb": np.zeros((SIZE, SIZE, 3), dtype="uint8")},
                "right_camera": {"rgb": np.zeros((SIZE, SIZE, 3), dtype="uint8")},
                "wrist_camera": {"rgb": np.zeros((SIZE, SIZE, 3), dtype="uint8")},
            },
            "joint_action": np.zeros(14, dtype="float32"),
            "endpose": np.zeros(16, dtype="float32"),
            "task_name": "dual_bottles_pick_easy",  # a string, not a channel
        }

    def take_action(self, action, action_type="qpos"):
        self.sent.append((np.asarray(action), action_type))
        self.steps += 1

    def check_success(self):
        return self.steps >= self.win_at

    def close_env(self):
        self.closed = True


class Chunker(Policy):
    def __init__(self, width=16):
        self.width = width

    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=4, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ChannelSpec("action", "vector", (self.width,), "float32", semantics="actuation")

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((4, self.width), dtype="float32")


def evaluator(action_type="ee", win_at=3, **kwargs):
    built = []

    def factory(task, **_):
        made = FakeTask(win_at=win_at)
        built.append(made)
        return made

    made = RoboTwinEvaluator(
        "dual_bottles_pick_easy", action_type=action_type, factory=factory,
        horizon=20, **kwargs
    )
    made.built = built
    return made


# -- the reason this backend exists ------------------------------------------


def test_end_effector_mode_takes_what_the_ego_retargeter_produces():
    """retargeter_hands refuses to produce joint positions and says why. That
    refusal has blocked the ego path from every ALOHA-family config since it was
    written; RoboTwin's ee mode takes exactly what the retargeter does emit."""
    made = for_ego(factory=lambda task, **_: FakeTask())
    assert made.descriptor().metadata["action_type"] == "ee"
    assert made.descriptor().metadata["accepts_end_effector"] is True
    assert made.action().shape == (16,)


def test_it_is_the_first_dual_arm_backend_here():
    made = evaluator()
    assert made.descriptor().metadata["arms"] == 2
    labels = made.action().dim_labels
    assert labels[0].startswith("left_")
    assert labels[8].startswith("right_")


# -- three action spaces on one arm ------------------------------------------


def test_the_action_type_is_required_and_has_no_default():
    """Three spaces, invisible in the array, and a policy trained for one
    evaluated under another produces numbers that are accepted and mean
    something else."""
    with pytest.raises(TypeError):
        RoboTwinEvaluator("dual_bottles_pick_easy")  # no action_type


def test_the_width_follows_from_the_action_type():
    assert width_of("qpos") == 14
    assert width_of("ee") == 16
    assert width_of("delta_ee") == 16
    assert evaluator(action_type="qpos").action().shape == (14,)
    assert evaluator(action_type="ee").action().shape == (16,)


def test_an_unknown_action_type_is_refused_with_the_reason():
    with pytest.raises(ConfigError, match="mean something else"):
        evaluator(action_type="whatever")


def test_the_action_type_is_a_discriminator_so_a_mismatch_is_caught_by_name():
    from gantry.spine import compatible

    joints = evaluator(action_type="qpos").action()
    poses = evaluator(action_type="ee").action()
    verdict = compatible(joints, poses)
    assert not verdict.ok
    assert "action_type" in verdict.explain() or "shape" in verdict.explain()


def test_the_labels_say_which_space_as_well_as_which_arm():
    assert labels_for("qpos")[:2] == ("left_j1", "left_j2")
    assert labels_for("ee")[:4] == ("left_x", "left_y", "left_z", "left_qw")
    assert len(set(labels_for("ee"))) == 16


def test_the_action_type_reaches_the_simulator():
    made = evaluator(action_type="delta_ee")
    made.run(Chunker(16), made.task_for(scenes=1), Protocol())
    assert {t for _, t in made.built[0].sent} == {"delta_ee"}


# -- the rollout --------------------------------------------------------------


def test_success_comes_from_the_environments_own_check():
    made = evaluator(win_at=2)
    record = made.run(Chunker(), made.task_for(scenes=4), Protocol())
    assert all(e.labels.success for e in record.episodes)
    assert record.metrics["success_rate"].value == 1.0
    assert record.episodes[0].array("action").shape[0] == 2


def test_a_timeout_is_a_genuine_failure_because_the_suite_checks_every_step():
    """Unlike a real bench, where nobody looked."""
    made = evaluator(win_at=999)
    record = made.run(Chunker(), made.task_for(scenes=2, horizon=6), Protocol())
    assert [e.labels.success for e in record.episodes] == [False, False]
    assert record.metrics["success_rate"].n == 2


def test_scenes_are_seeds_and_reach_the_environment():
    made = evaluator(win_at=2)
    made.run(Chunker(), made.task_for(scenes=3), Protocol())
    assert made.built[0].seeds == [0, 1, 2]


def test_the_instruction_the_environment_chose_is_recorded():
    """RoboTwin varies the sentence. A policy given the nominal one while the
    environment scores another is tested on a mismatch that reads as a low
    number."""
    made = evaluator(win_at=2)
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.annotations["instruction_given"] == "pick up both bottles"


def test_the_space_the_actions_were_read_in_is_on_every_episode():
    made = evaluator(action_type="ee", win_at=2)
    record = made.run(Chunker(16), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.annotations["action_type"] == "ee"


# -- the nested observation ---------------------------------------------------


def test_the_nested_observation_becomes_dotted_names_without_its_strings():
    flat = flatten(FakeTask().get_obs(), keep=("head_camera", "left_camera", "right_camera"))
    assert "observation.head_camera.rgb" in flat
    assert "joint_action" in flat and "endpose" in flat
    # a task name is a string, not a channel
    assert "task_name" not in flat
    # and a camera nobody asked for is dropped rather than recorded
    assert "observation.wrist_camera.rgb" not in flat


def test_asking_for_every_camera_keeps_them_all():
    flat = flatten(FakeTask().get_obs(), keep=())
    assert "observation.wrist_camera.rgb" in flat


# -- the rest ------------------------------------------------------------------


def test_the_task_list_is_available_without_the_simulator():
    assert len(TASKS) >= 15
    assert "dual_bottles_pick_easy" in TASKS
    assert set(ACTION_TYPES) == {"qpos", "ee", "delta_ee"}


def test_it_does_not_claim_to_host_other_bodies():
    """A task is built around the configuration it was given and cannot be
    rebuilt mid-run."""
    assert evaluator().descriptor().provides["hosts_embodiment"] is False


def test_the_licence_is_declared_because_a_dataset_inherits_it():
    assert "MIT" in evaluator().descriptor().metadata["licence"]


def test_a_robotwin_evaluator_conforms():
    made = evaluator(win_at=2)
    verdict = check_evaluator(made, Chunker(16), made.task_for(scenes=2))
    assert verdict.ok, verdict.explain()


def test_close_releases_the_environment():
    made = evaluator(win_at=2)
    made.run(Chunker(16), made.task_for(scenes=1), Protocol())
    env = made.built[0]
    made.close()
    assert env.closed


@pytest.mark.skip(reason="needs RoboTwin 2.0, SAPIEN and its object dataset")
def test_against_the_real_simulator():  # pragma: no cover
    made = for_ego("dual_bottles_pick_easy")
    assert made.action().shape == (16,)
