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
    endpose_vector,
    flatten,
    for_ego,
    labels_for,
    state_spec,
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
        self.settings: list[dict] = []
        self.sent: list[tuple[np.ndarray, str]] = []
        self.closed = False

    def setup_demo(self, now_ep_num=0, seed=0, is_test=True, **kwargs):
        self.seeds.append(seed)
        self.settings.append(kwargs)
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
            # The real shape: a dict per arm, and the grippers are bare floats.
            "joint_action": {"vector": np.zeros(14, dtype="float32")},
            "endpose": {
                "left_endpose": np.zeros(7, dtype="float32"),
                "left_gripper": 0.3,
                "right_endpose": np.zeros(7, dtype="float32"),
                "right_gripper": 0.7,
            },
            "task_name": "pick_dual_bottles",  # a string, not a channel
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
        "pick_dual_bottles", action_type=action_type, factory=factory, horizon=20, **kwargs
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
    """Two spaces, invisible in the array, and a policy trained for one
    evaluated under another produces numbers that are accepted and mean
    something else."""
    with pytest.raises(TypeError):
        RoboTwinEvaluator("pick_dual_bottles")  # no action_type


def test_the_width_follows_from_the_action_type():
    assert width_of("qpos") == 14
    assert width_of("ee") == 16
    assert evaluator(action_type="qpos").action().shape == (14,)
    assert evaluator(action_type="ee").action().shape == (16,)


def test_the_delta_mode_the_docs_imply_is_not_offered_because_it_does_not_exist():
    """RoboTwin's own signature is Literal['qpos', 'ee']. A third value falls
    through both branches of take_action to an unbound local, so advertising it
    would trade a clear refusal for a crash deep in the simulator."""
    assert set(ACTION_TYPES) == {"qpos", "ee"}
    with pytest.raises(ConfigError, match="mean something else"):
        width_of("delta_ee")


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
    made = evaluator(action_type="qpos")
    made.run(Chunker(14), made.task_for(scenes=1), Protocol())
    assert {t for _, t in made.built[0].sent} == {"qpos"}


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


def test_every_reset_reuses_the_settings_the_environment_was_built_with():
    """A partial dict on reset would quietly change the cameras or the
    randomisation partway through a run, and between the screen and the run it
    was screening for."""

    def factory(task, **_):
        made = FakeTask(win_at=2)
        made.gantry_settings = {"domain_randomization": {"random_light": False}}
        return made

    made = RoboTwinEvaluator("pick_dual_bottles", action_type="ee", factory=factory, horizon=20)
    made.built = []
    made.run(Chunker(16), made.task_for(scenes=2), Protocol())
    seen = made.world.env.settings
    assert len(seen) == 2
    assert all(s["domain_randomization"] == {"random_light": False} for s in seen)


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
    assert "joint_action.vector" in flat
    assert flat["endpose.left_endpose"].shape == (7,)
    # a task name is a string, not a channel
    assert "task_name" not in flat
    # and a camera nobody asked for is dropped rather than recorded
    assert "observation.wrist_camera.rgb" not in flat


def test_asking_for_every_camera_keeps_them_all():
    flat = flatten(FakeTask().get_obs(), keep=())
    assert "observation.wrist_camera.rgb" in flat


def test_a_bare_float_gripper_survives_flattening():
    """RoboTwin's endpose carries each gripper as a scalar, not an array.
    Dropping zero-rank values would discard both and leave a pose-only state
    that still looks well formed."""
    flat = flatten(FakeTask().get_obs(), keep=())
    assert flat["endpose.left_gripper"].shape == (1,)
    assert float(flat["endpose.right_gripper"][0]) == 0.7


# -- the rest ------------------------------------------------------------------


def test_the_task_list_is_the_installed_versions_not_the_papers():
    """RoboTwin 2.0 renamed every task. The 1.0 ids read like valid arguments
    and fail only when the import misses, one run in."""
    assert len(TASKS) == 50
    assert "pick_dual_bottles" in TASKS
    assert "dual_bottles_pick_easy" not in TASKS  # the 1.0 name for it


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
    made = for_ego("pick_dual_bottles")
    assert made.action().shape == (16,)


# -- the seed screen -----------------------------------------------------------
#
# RoboTwin randomises object placement per seed and not every arrangement is
# solvable. Scoring a policy on one that is not charges it for the sampler, and
# the unsolvable fraction moves with the seed range — so two runs over different
# ranges are not comparable even on the same task. RoboTwin's own loop screens
# with the scripted expert for exactly this reason.


class Screenable(FakeTask):
    """A fake whose expert solves only some arrangements."""

    def __init__(self, solvable=(0, 2, 5), explodes=(), **kwargs):
        super().__init__(**kwargs)
        self.solvable = set(solvable)
        self.explodes = set(explodes)
        self.seed = None
        self.plan_success = True
        self.played = []

    def setup_demo(self, now_ep_num=0, seed=0, is_test=True, **kwargs):
        super().setup_demo(now_ep_num, seed, is_test, **kwargs)
        self.seed = seed
        self.eval_success = False

    def play_once(self):
        self.played.append(self.seed)
        if self.seed in self.explodes:
            raise RuntimeError("the expert's planner gave up")
        self.plan_success = self.seed in self.solvable
        self.eval_success = self.seed in self.solvable


def screening(**kwargs):
    built = []

    def factory(task, **_):
        made = Screenable(**kwargs)
        built.append(made)
        return made

    made = RoboTwinEvaluator("pick_dual_bottles", action_type="ee", factory=factory)
    made.built = built
    return made


def test_screening_returns_only_the_seeds_the_expert_solved():
    made = screening(solvable=(0, 2, 5))
    assert made.screen(3, limit=10) == (0, 2, 5)


def test_screened_seeds_are_the_ones_the_run_uses():
    made = screening(solvable=(1, 4))
    task = made.task_for(seeds=made.screen(2, limit=10))
    assert [scene.seed for scene in task.scenes] == [1, 4]


def test_an_expert_that_throws_is_a_property_of_the_arrangement_not_an_error():
    """RoboTwin raises UnStableError when the scene settles badly. That is a
    fact about the seed, and it must not take the run down with it."""
    made = screening(solvable=(0, 3), explodes=(1, 2))
    assert made.screen(2, limit=10) == (0, 3)


def test_screening_stops_at_the_limit_rather_than_searching_forever():
    made = screening(solvable=())
    assert made.screen(5, limit=4) == ()
    assert made.built[0].played == [0, 1, 2, 3]


def test_an_unscreened_run_says_so_on_every_episode():
    """So a rate that includes unsolvable arrangements cannot be read as one
    that does not."""
    made = evaluator(win_at=2)
    record = made.run(Chunker(16), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.annotations["expert_screened"] is False


def test_a_screened_run_records_the_experts_own_rate():
    """The ceiling. A policy at 40% where the expert managed 50% is a different
    claim from one at 40% where the expert managed 100%."""
    made = screening(solvable=(0, 2))
    made.screen(2, limit=4)
    record = made.run(Chunker(16), made.task_for(seeds=(0, 2)), Protocol())
    labels = record.episodes[0].labels.annotations
    assert labels["expert_screened"] is True
    # Seeds 0, 1 and 2 were tried and two were solved. The denominator is what
    # was actually examined, not the limit — screening stops as soon as it has
    # enough, so the limit says nothing about how hard the task was.
    assert made.built[0].played == [0, 1, 2]
    assert labels["expert_solve_rate"] == round(2 / 3, 4)


def test_screening_needs_an_expert_and_says_so_when_there_is_none():
    made = evaluator()
    with pytest.raises(ConfigError, match="no scripted expert"):
        made.screen(2)


# -- success is latched, not re-asked ------------------------------------------


def test_success_detected_mid_motion_is_not_lost_when_the_object_settles_back():
    """RoboTwin interpolates one command into hundreds of physics steps and
    latches eval_success inside that loop. Asking check_success() afterwards can
    disagree — and the trial that succeeded would be recorded as a failure."""

    class Settles(FakeTask):
        def take_action(self, action, action_type="qpos"):
            super().take_action(action, action_type)
            self.eval_success = True  # latched during the motion

        def check_success(self):
            return False  # and no longer true once everything stopped moving

    made = RoboTwinEvaluator(
        "pick_dual_bottles", action_type="ee", factory=lambda task, **_: Settles(), horizon=20
    )
    record = made.run(Chunker(16), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.success is True


# -- the state, in the space being commanded ------------------------------------


def test_the_arms_current_pose_is_published_in_the_actions_own_layout():
    """RoboTwin publishes joint_action.vector -- the qpos state in the order
    qpos actions are read -- but nothing equivalent for poses. A policy
    controlling in ee space otherwise has no single channel saying where the
    arms are, in the space it is commanding."""
    made = evaluator(win_at=2)
    record = made.run(Chunker(16), made.task_for(scenes=1), Protocol())
    state = record.episodes[0].array("endpose.vector")
    assert state.shape[1] == 16


def test_the_state_is_assembled_in_the_action_order_not_a_convenient_one():
    """A state laid out differently from the action trains and evaluates without
    complaint and is wrong by a fixed rotation."""
    flat = flatten(FakeTask().get_obs(), keep=())
    flat["endpose.left_endpose"] = np.arange(7, dtype=float)
    flat["endpose.left_gripper"] = np.array([0.5])
    flat["endpose.right_endpose"] = np.arange(100, 107, dtype=float)
    flat["endpose.right_gripper"] = np.array([0.9])
    vector = endpose_vector(flat)
    assert np.allclose(vector[:7], np.arange(7))
    assert vector[7] == 0.5
    assert np.allclose(vector[8:15], np.arange(100, 107))
    assert vector[15] == 0.9
    assert list(state_spec().dim_labels) == list(labels_for("ee"))


def test_no_endpose_data_gives_nothing_rather_than_a_vector_of_zeros():
    """Which would read as 'the arms are at the origin'."""
    assert endpose_vector({}) is None


def test_the_state_and_the_action_share_an_encoding_so_one_conversion_serves_both():
    action = evaluator(action_type="ee").action()
    state = state_spec()
    assert state.metadata["rotation_repr"] == action.metadata["rotation_repr"]
    assert state.metadata["rotation_offset"] == action.metadata["rotation_offset"]
