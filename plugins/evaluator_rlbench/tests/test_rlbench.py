"""RLBench's shape, and the perturbation discipline on top of it.

The fake copies the two things an adapter gets wrong here: ``task.reset()``
returning a *list* of instruction paraphrases alongside the observation, and
``task.step()`` returning ``(obs, reward, terminate)`` where terminate fires on
failure as well as success.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_rlbench import (
    BASELINE,
    COLOSSEUM_TASKS,
    FACTORS,
    RlbenchEvaluator,
    sweep,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec

SIZE = 8
PARAPHRASES = [
    "open the drawer",
    "pull the drawer open",
    "grip the handle and slide the drawer out",
]


class FakeObservation:
    def __init__(self):
        self.front_rgb = np.zeros((SIZE, SIZE, 3), dtype="uint8")
        self.wrist_rgb = np.zeros((SIZE, SIZE, 3), dtype="uint8")
        self.joint_positions = np.zeros(7, dtype="float32")
        self.gripper_open = np.asarray(1.0, dtype="float32")  # rank 0, skipped
        self.misc = {"front_camera_intrinsics": np.eye(3)}
        self.task_low_dim_state = None


class FakeTask:
    def __init__(self, solve_at=None, terminate_at=None, descriptions=None):
        self.solve_at = solve_at
        self.terminate_at = terminate_at
        self.descriptions = PARAPHRASES if descriptions is None else descriptions
        self.variations: list[int] = []
        self.step_count = 0
        self.closed = False

    def set_variation(self, index):
        self.variations.append(index)

    def reset(self):
        self.step_count = 0
        return list(self.descriptions), FakeObservation()

    def step(self, action):
        self.step_count += 1
        solved = self.solve_at is not None and self.step_count >= self.solve_at
        stopped = self.terminate_at is not None and self.step_count >= self.terminate_at
        return FakeObservation(), 1.0 if solved else 0.0, bool(solved or stopped)

    def close(self):
        self.closed = True


class Chunker(Policy):
    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=2, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ChannelSpec("action", "vector", (8,), "float32", semantics="actuation")

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((2, 8), dtype="float32")


def evaluator(solve_at=None, terminate_at=None, descriptions=None, **kwargs):
    tasks = []

    def loader(env, name):
        task = FakeTask(solve_at, terminate_at, descriptions)
        tasks.append(task)
        return task

    made = RlbenchEvaluator(
        "open_drawer",
        factory=lambda **_: object(),
        task_loader=loader,
        image_size=SIZE,
        horizon=10,
        **kwargs,
    )
    made.tasks = tasks
    return made


# -- one factor at a time ----------------------------------------------------


def test_perturbing_several_factors_at_once_is_refused():
    """A drop measured with three things moved attributes the loss to nothing in
    particular — the same error as varying two planes in one comparison, which
    this framework already refuses on the other side."""
    with pytest.raises(ConfigError, match="attributes the loss to nothing"):
        evaluator(factors=("light_color", "object_texture"))


def test_an_unknown_factor_is_refused():
    with pytest.raises(ConfigError, match="unknown perturbation factor"):
        evaluator(factors=("vibes",))


def test_everything_at_once_is_allowed_and_records_itself_as_such():
    """A legitimate stress test. What it must not do is pass as a per-factor claim."""
    made = evaluator(factors="all")
    assert made.condition == "all"
    assert made.perturbed is True
    assert made.descriptor().metadata["condition"] == "all"


def test_the_baseline_condition_is_named_rather_than_empty():
    """So a record can tell 'the unperturbed condition' from 'somebody forgot'."""
    made = evaluator()
    assert made.condition == BASELINE
    assert made.perturbed is False
    assert made.descriptor().metadata["perturbed"] is False


def test_the_condition_is_in_every_scene_id_so_runs_cannot_be_pooled():
    """Pooling a baseline run with a perturbed one is the specific mistake that
    turns a robustness experiment into an unexplained average."""
    baseline = evaluator().task_for(scenes=2)
    perturbed = evaluator(factors=("light_color",)).task_for(scenes=2)
    assert all(BASELINE in scene.id for scene in baseline.scenes)
    assert all("light_color" in scene.id for scene in perturbed.scenes)
    assert not set(s.id for s in baseline.scenes) & set(s.id for s in perturbed.scenes)


def test_the_sweep_is_the_whole_experiment_with_a_baseline_first():
    """One evaluator per factor plus the unperturbed condition, so every
    difference is attributable to exactly one moved thing."""
    made = sweep("open_drawer", factory=lambda **_: object(), task_loader=lambda e, n: FakeTask())
    assert list(made)[0] == BASELINE
    assert set(made) == set(FACTORS)
    assert made["light_color"].condition == "light_color"
    assert made[BASELINE].perturbed is False


def test_the_colosseum_task_list_is_available_without_the_simulator():
    assert len(COLOSSEUM_TASKS) == 20
    assert "open_drawer" in COLOSSEUM_TASKS


# -- success is a reward of one, not a termination ---------------------------


def test_terminating_without_the_solved_reward_is_a_failure():
    """`terminate` also fires on failure conditions. Reading it as success
    inflates every rate this suite produces."""
    made = evaluator(terminate_at=3)
    record = made.run(Chunker(), made.task_for(scenes=2), Protocol())
    assert [episode.labels.success for episode in record.episodes] == [False, False]
    assert record.metrics["success_rate"].value == 0.0


def test_a_reward_of_one_is_success():
    made = evaluator(solve_at=3)
    record = made.run(Chunker(), made.task_for(scenes=2), Protocol())
    assert all(episode.labels.success for episode in record.episodes)


def test_the_success_rule_is_stated_rather_than_left_to_the_reader():
    assert "not terminate" in evaluator().descriptor().metadata["success_rule"]


# -- the instruction is a list, and which one was used is a variable ---------


def test_the_paraphrase_used_is_recorded_and_chosen_explicitly():
    """RLBench returns several sentences for the same goal. A benchmark that
    resampled silently would have language variation as an uncontrolled variable
    in every result it ever produced."""
    made = evaluator(solve_at=2)
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    annotations = record.episodes[0].labels.annotations
    assert annotations["instruction_given"] == PARAPHRASES[0]
    assert annotations["instruction_options"] == 3

    other = evaluator(solve_at=2, instruction_index=2)
    record = other.run(Chunker(), other.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.annotations["instruction_given"] == PARAPHRASES[2]


def test_an_out_of_range_paraphrase_index_wraps_rather_than_crashing():
    made = evaluator(solve_at=2, instruction_index=7)
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.annotations["instruction_given"] == PARAPHRASES[1]


def test_a_task_with_no_descriptions_records_no_instruction():
    made = evaluator(solve_at=2, descriptions=[])
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].labels.annotations["instruction_given"] is None


# -- the rest ----------------------------------------------------------------


def test_an_unknown_action_mode_is_refused_with_the_reason():
    with pytest.raises(ConfigError, match="no safe default"):
        evaluator(action_mode="mind_control")


def test_the_action_width_follows_the_mode():
    assert evaluator().action().shape == (8,)
    assert evaluator(action_mode="joint_velocity").action().shape == (8,)


def test_the_observation_object_becomes_named_arrays_without_its_metadata():
    made = evaluator(solve_at=2)
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    names = set(record.episodes[0].channel_names)
    assert {"front_rgb", "wrist_rgb", "joint_positions"} <= names
    # misc is a dict of camera intrinsics, gripper_open is rank zero, and
    # task_low_dim_state is None — none of the three is a channel.
    assert not {"misc", "gripper_open", "task_low_dim_state"} & names


def test_scenes_are_rlbench_variations_and_reach_the_task():
    made = evaluator(solve_at=2)
    made.run(Chunker(), made.task_for(scenes=3), Protocol())
    assert made.tasks[0].variations == [0, 1, 2]


def test_the_camera_channels_are_declared_as_rgb():
    made = evaluator(solve_at=2)
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    assert record.episodes[0].channel("front_rgb").semantics == "rgb"


def test_an_rlbench_evaluator_conforms():
    made = evaluator(solve_at=2)
    verdict = check_evaluator(made, Chunker(), made.task_for(scenes=2))
    assert verdict.ok, verdict.explain()


def test_close_releases_the_task():
    made = evaluator(solve_at=2)
    made.run(Chunker(), made.task_for(scenes=1), Protocol())
    task = made.tasks[0]
    made.close()
    assert task.closed


@pytest.mark.skip(reason="needs RLBench, PyRep and a CoppeliaSim install")
def test_against_the_real_simulator():  # pragma: no cover
    made = RlbenchEvaluator("open_drawer")
    assert made.action().shape == (8,)
