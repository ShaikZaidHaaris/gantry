"""Meta-World's shape, and the flickering success flag.

The flag is the reason this file exists. A fake that reports success once and then
stops is not a contrived case -- it is what the real thing does, because success is
a distance threshold and a policy that reaches the goal drifts off it.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_metaworld import MT10, TASKS, MetaworldEvaluator, suite

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec


class FakeSawyer:
    """Meta-World's API where it matters: set_task, a 39-vector, info['success']."""

    def __init__(self, succeed_at=None, flicker=False, five_value=True, tasks=4):
        self.succeed_at = succeed_at
        self.flicker = flicker
        self.five_value = five_value
        self.tasks = [f"task-{index}" for index in range(tasks)]
        self.chosen: list[str] = []
        self.step_count = 0
        self.closed = False

    def set_task(self, task):
        self.chosen.append(task)

    def reset(self):
        self.step_count = 0
        return np.zeros(39, dtype="float32"), {}

    def step(self, action):
        self.step_count += 1
        hit = self.succeed_at is not None and self.step_count == self.succeed_at
        after = self.succeed_at is not None and self.step_count > self.succeed_at
        # The flicker: True once, then False again as the arm drifts off the
        # distance threshold.
        flag = hit or (after and not self.flicker)
        info = {"success": float(bool(flag))}
        observation = np.zeros(39, dtype="float32")
        if self.five_value:
            return observation, 0.0, False, False, info
        return observation, 0.0, False, info

    def close(self):
        self.closed = True


class Chunker(Policy):
    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=2, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ChannelSpec("action", "vector", (4,), "float32", semantics="actuation")

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((2, 4), dtype="float32")


def evaluator(**fake):
    envs = []

    def factory(task, **_):
        env = FakeSawyer(**fake)
        envs.append(env)
        return env

    made = MetaworldEvaluator("reach-v2", factory=factory, horizon=10)
    made.envs = envs
    return made


def test_a_flickering_success_flag_still_counts_as_a_success():
    """The bug this adapter is written around. Meta-World thresholds a distance,
    so a policy that reaches the goal and drifts a centimetre reports True then
    False -- and reading the final flag scores it as a failure, penalising exactly
    the near-miss policies a benchmark is meant to separate."""
    made = evaluator(succeed_at=3, flicker=True)
    record = made.run(Chunker(), made.task_for(scenes=2), Protocol())
    assert all(episode.labels.success for episode in record.episodes)
    # The trial stops at the first True rather than running on to see it flicker.
    assert record.episodes[0].array("action").shape[0] == 3


def test_never_succeeding_within_the_horizon_is_a_genuine_failure():
    """Safe to call False here, unlike on a real bench: the predicate was
    evaluated on every step and answered no each time. The trial was checked, not
    merely cut short."""
    made = evaluator(succeed_at=None)
    record = made.run(Chunker(), made.task_for(scenes=3), Protocol())
    assert [episode.labels.success for episode in record.episodes] == [False, False, False]
    assert record.metrics["success_rate"].value == 0.0
    assert record.metrics["success_rate"].n == 3


def test_both_step_dialects_are_accepted():
    for five_value in (True, False):
        made = evaluator(succeed_at=2, five_value=five_value)
        record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
        assert record.episodes[0].labels.success is True


def test_scenes_are_stored_goal_placements_not_random_draws():
    """A list, so scene k is the same arrangement on any machine -- which is what
    a paired comparison rests on."""
    made = evaluator(succeed_at=2)
    made.run(Chunker(), made.task_for(scenes=6), Protocol())
    # Four stored tasks, six scenes: the indices wrap rather than going out of range.
    assert made.envs[0].chosen == [
        "task-0",
        "task-1",
        "task-2",
        "task-3",
        "task-0",
        "task-1",
    ]


def test_a_mistyped_task_id_is_refused_with_a_suggestion():
    """The ids are easy to get wrong -- the -v2 suffix, the side/wall variants --     and the alternative to refusing is running an environment nobody meant."""
    with pytest.raises(ConfigError, match="did you mean"):
        MetaworldEvaluator("door-open")
    with pytest.raises(ConfigError, match="did you mean"):
        MetaworldEvaluator("reach-v9")


def test_a_task_from_a_newer_release_can_be_run_by_asking():
    made = MetaworldEvaluator(
        "brand-new-task-v3", strict=False, factory=lambda *a, **k: FakeSawyer()
    )
    assert made.descriptor().metadata["task"] == "brand-new-task-v3"


def test_the_fifty_are_listed_so_a_run_can_be_planned_with_nothing_installed():
    assert len(TASKS) == 50
    assert len(MT10) == 10
    assert set(MT10) <= set(TASKS)


def test_the_suite_is_one_evaluator_per_task():
    """Not one evaluator with a task list. A hidden loop makes 'vary the task'
    invisible to the framework -- the axis ends up inside a component instead of
    between components, and that is a comparison nobody can check."""
    made = suite(MT10, factory=lambda *a, **k: FakeSawyer())
    assert set(made) == set(MT10)
    assert len({id(value) for value in made.values()}) == 10
    assert made["push-v2"].descriptor().metadata["task"] == "push-v2"


def test_it_does_not_claim_to_host_other_bodies():
    """One Sawyer, welded in -- which is why this is a breadth backend and not a
    transfer one, and the descriptor should not let anyone read it otherwise."""
    assert evaluator().descriptor().provides["hosts_embodiment"] is False


def test_the_observation_vector_is_declared_rather_than_inferred():
    """Inference would give (39,) float32 and no idea what it means. It is
    current and previous end-effector state plus the goal, which is not guessable."""
    made = evaluator(succeed_at=2)
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    spec = record.episodes[0].channel("observation")
    assert spec.semantics == "proprioception"
    assert spec.shape == (39,)


def test_the_success_rule_is_stated_in_the_descriptor():
    assert "ever succeeded" in evaluator().descriptor().metadata["success_rule"]


def test_a_metaworld_evaluator_conforms():
    made = evaluator(succeed_at=2)
    verdict = check_evaluator(made, Chunker(), made.task_for(scenes=2))
    assert verdict.ok, verdict.explain()


def test_close_releases_the_environment():
    made = evaluator(succeed_at=2)
    made.run(Chunker(), made.task_for(scenes=1), Protocol())
    env = made.envs[0]
    made.close()
    assert env.closed


@pytest.mark.skip(reason="needs Meta-World and MuJoCo")
def test_against_the_real_simulator():  # pragma: no cover
    made = MetaworldEvaluator("reach-v2")
    assert made.action().shape == (4,)
