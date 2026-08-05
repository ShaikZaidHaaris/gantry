"""SimplerEnv's shape, and the discipline around the paired real column.

Most of these tests are about the real rates rather than the rollout, because the
rollout is the shared loop and the real rates are where this plugin can do
damage: a published number that quietly becomes something this benchmark claims
to have measured is a much worse failure than a crash.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_simpler import (
    VARIANT_AGGREGATION,
    VISUAL_MATCHING,
    RealResult,
    SimplerEvaluator,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec
from gantry.spine.inference import mmrv


class Space:
    def __init__(self, width):
        self.shape = (width,)


class FakeSimpler:
    def __init__(self, width=7, win_at=3, instruction="pick coke can"):
        self.action_space = Space(width)
        self.win_at = win_at
        self._instruction = instruction
        self.step_count = 0
        self.seeds = []
        self.closed = False

    def _observation(self):
        return {
            "image": np.zeros((8, 8, 3), dtype="uint8"),
            "agent": {"qpos": np.zeros(8, dtype="float32")},
        }

    def get_language_instruction(self):
        return self._instruction

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.step_count = 0
        return self._observation(), {}

    def step(self, action):
        self.step_count += 1
        won = self.step_count >= self.win_at
        return self._observation(), 1.0 if won else 0.0, won, False, {"success": won}

    def close(self):
        self.closed = True


class Chunker(Policy):
    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=4, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ChannelSpec("action", "vector", (7,), "float32", semantics="actuation")

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((4, 7), dtype="float32")


REAL = RealResult(
    task="google_robot_pick_coke_can",
    policy="rt-1-x",
    rate=0.567,
    trials=300,
    source="Li et al. 2024, SIMPLER, table 2",
    platform="Google Robot",
)


def evaluator(win_at=3, instruction="pick coke can", **kwargs):
    envs = []

    def factory(task, **_):
        env = FakeSimpler(win_at=win_at, instruction=instruction)
        envs.append(env)
        return env

    made = SimplerEvaluator("google_robot_pick_coke_can", factory=factory, horizon=10, **kwargs)
    made.envs = envs
    return made


# -- the paired real column --------------------------------------------------


def test_a_real_rate_without_a_source_is_refused():
    """The paired column's whole value is that a sim number can be checked
    against something checkable. An unsourced rate looks identical, in every
    table it appears in, to one that can be traced."""
    with pytest.raises(ConfigError, match="source named"):
        RealResult(task="t", policy="p", rate=0.5, trials=100, source="  ")


def test_a_real_rate_that_is_not_a_proportion_is_refused():
    with pytest.raises(ConfigError, match="not a proportion"):
        RealResult(task="t", policy="p", rate=56.7, trials=300, source="table 2")


def test_a_real_rate_with_no_trial_count_is_refused():
    """The trial count is what makes the real rate's own interval computable.
    Without it the paired column is a point with no uncertainty, which will get
    compared against a sim interval as though it were exact."""
    with pytest.raises(ConfigError, match="carries no information"):
        RealResult(task="t", policy="p", rate=0.5, trials=0, source="table 2")


def test_the_real_rate_is_an_annotation_and_never_this_runs_metric():
    """Somebody else's measurement, on their robot, on their day. It must stay
    distinguishable from what this run measured, forever."""
    made = evaluator(real=REAL)
    record = made.run(Chunker(), made.task_for(scenes=4), Protocol())

    assert set(record.metrics) == {"success_rate"}
    assert record.metrics["success_rate"].value == 1.0

    annotations = record.episodes[0].labels.annotations
    assert annotations["real_rate"] == 0.567
    assert annotations["real_trials"] == 300
    assert "table 2" in annotations["real_source"]


def test_the_paired_columns_feed_the_statistic_core_already_has():
    """This is why the plugin exists: mmrv has been sitting in core with nothing
    to feed it, because no backend produced a real column."""
    # One entry per policy, in the same order in both columns: what the real
    # robot did, and what this simulator said.
    real = [0.567, 0.170, 0.340]  # rt-1-x, octo-base, rt-2-x
    faithful = [0.600, 0.200, 0.380]  # same ordering, different absolute level
    inverted = [0.200, 0.600, 0.380]  # top two swapped

    # A proxy that is uniformly optimistic violates no ordering, and that is the
    # right answer: a benchmark's job is to rank, not to predict the level.
    assert mmrv(real, faithful) == 0.0
    assert mmrv(real, inverted) > 0.0


def test_a_run_with_no_real_pairing_says_nothing_rather_than_zero():
    made = evaluator()
    assert "real" not in made.descriptor().metadata
    annotations = (
        made.run(Chunker(), made.task_for(scenes=1), Protocol()).episodes[0].labels.annotations
    )
    assert "real_rate" not in annotations


def test_a_real_rate_can_come_from_plain_data_for_a_manifest():
    made = evaluator(
        real={
            "task": "google_robot_pick_coke_can",
            "policy": "octo-base",
            "rate": 0.17,
            "trials": 300,
            "source": "SIMPLER table 2",
        }
    )
    assert made.real.policy == "octo-base"
    assert made.descriptor().metadata["real"]["real_rate"] == 0.17


# -- the two conditions ------------------------------------------------------


def test_an_unknown_condition_is_refused_with_the_reason():
    with pytest.raises(ConfigError, match="not two measurements of one quantity"):
        evaluator(variant="whatever")


def test_the_condition_is_in_every_scene_id_and_every_annotation():
    """So a record holding both can be split apart. A table that averaged the
    two would be reporting a quantity that does not exist."""
    for variant in (VISUAL_MATCHING, VARIANT_AGGREGATION):
        made = evaluator(variant=variant)
        task = made.task_for(scenes=2)
        assert all(variant in scene.id for scene in task.scenes)
        assert task.name.endswith(variant)
        record = made.run(Chunker(), task, Protocol())
        assert record.episodes[0].labels.annotations["variant"] == variant


# -- the rollout -------------------------------------------------------------


def test_the_instruction_the_environment_chose_is_the_one_recorded():
    """SimplerEnv resamples the instruction on reset for some tasks. A policy
    given the nominal one while the environment scores another is being tested on
    a mismatch that shows up only as poor performance."""
    made = evaluator(instruction="move the coke can near the apple")
    record = made.run(Chunker(), made.task_for(scenes=1), Protocol())
    annotations = record.episodes[0].labels.annotations
    assert annotations["instruction_given"] == "move the coke can near the apple"


def test_it_does_not_claim_to_host_other_bodies():
    """The scene approximates one specific real platform; rebuilding it around a
    different arm would destroy the only property it has."""
    assert evaluator().descriptor().provides["hosts_embodiment"] is False


def test_the_platform_is_named_from_the_task_id():
    assert "Google Robot" in evaluator().descriptor().metadata["platform"]
    made = SimplerEvaluator("widowx_spoon_on_towel", factory=lambda *a, **k: FakeSimpler())
    assert "WidowX" in made.platform
    unknown = SimplerEvaluator("some_new_task", factory=lambda *a, **k: FakeSimpler())
    assert unknown.platform == "unknown"


def test_scenes_are_seeds_and_reach_the_environment():
    made = evaluator()
    made.run(Chunker(), made.task_for(scenes=3), Protocol())
    assert made.envs[0].seeds == [0, 1, 2]


def test_the_action_width_comes_from_the_environment():
    assert evaluator().action().shape == (7,)


def test_an_environment_with_no_action_space_is_refused():
    made = SimplerEvaluator("google_robot_pick_coke_can", factory=lambda *a, **k: object())
    with pytest.raises(ConfigError, match="no action space"):
        made.action()


def test_a_simpler_evaluator_conforms():
    made = evaluator(real=REAL)
    verdict = check_evaluator(made, Chunker(), made.task_for(scenes=2))
    assert verdict.ok, verdict.explain()


def test_close_releases_the_environment():
    made = evaluator()
    made.run(Chunker(), made.task_for(scenes=1), Protocol())
    env = made.envs[0]
    made.close()
    assert env.closed


@pytest.mark.skip(reason="needs SimplerEnv and ManiSkill2-real2sim")
def test_against_the_real_simulator():  # pragma: no cover
    made = SimplerEvaluator("google_robot_pick_coke_can")
    assert made.action().shape[0] == 7
