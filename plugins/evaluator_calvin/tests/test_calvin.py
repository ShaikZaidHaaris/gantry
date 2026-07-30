"""CALVIN's chain logic, checked without pybullet.

The fake oracle is the interesting part. CALVIN decides a subtask is solved by
diffing two ``get_info()`` snapshots, so the fake takes a script — which subtask
index is solved after how many steps — and the tests then assert on the thing
that actually matters: that the chain does not reset between subtasks, and that a
chain broken at three is recorded as three rather than as a failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_calvin import CalvinEvaluator, all_five, at_least

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec

CHAINS = [
    ["open_drawer", "lift_blue_block", "push_red_block", "turn_on_lightbulb", "close_drawer"],
    ["turn_on_led", "lift_pink_block", "place_in_slider", "open_drawer", "push_blue_block"],
]


class FakeOracle:
    """Solves a subtask after ``after`` steps of it, up to ``solves`` of them."""

    def __init__(self, solves=5, after=2):
        self.solves = solves
        self.after = after

    def get_task_info_for_set(self, start, now, wanted):
        index = int(now["subtask_index"])
        elapsed = int(now["step"]) - int(start["step"])
        return set(wanted) if index < self.solves and elapsed >= self.after else set()


class FakeEnv:
    def __init__(self):
        self.step_count = 0
        self.resets = 0
        self.subtask_index = 0
        self.closed = False

    def _observation(self):
        return {
            "rgb_obs": {"rgb_static": np.zeros((8, 8, 3), dtype="uint8")},
            "robot_obs": np.zeros(15, dtype="float32"),
        }

    def reset(self, **kwargs):
        self.resets += 1
        self.step_count = 0
        return self._observation()

    def get_info(self):
        return {"step": self.step_count, "subtask_index": self.subtask_index}

    def step(self, action):
        self.step_count += 1
        return self._observation(), 0.0, False, {}

    def close(self):
        self.closed = True


class Chunker(Policy):
    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=2, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ChannelSpec("action", "vector", (7,), "float32", semantics="actuation")

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((2, 7), dtype="float32")


def evaluator(solves=5, after=2, subtask_steps=8, **kwargs):
    envs = []

    def factory(split, **_):
        env = FakeEnv()
        envs.append(env)
        return env

    made = CalvinEvaluator(
        chains=kwargs.pop("chains", CHAINS),
        factory=factory,
        oracle=FakeOracle(solves=solves, after=after),
        subtask_steps=subtask_steps,
        **kwargs,
    )
    made.envs = envs
    # The fake oracle keys off the world's own subtask index, so the world has to
    # tell the env which subtask it is on.
    return made


class Tracking(CalvinEvaluator):
    """Keeps the env's subtask index in step with the chain's, for the fake oracle."""

    def world_for(self, scene):
        world = super().world_for(scene)
        original = world._advance_subtask

        def advance(info):
            original(info)
            world.env.subtask_index = world._index

        world._advance_subtask = advance
        world.env.subtask_index = 0
        return world


def tracking(solves=5, after=2, subtask_steps=8, **kwargs):
    envs = []

    def factory(split, **_):
        env = FakeEnv()
        envs.append(env)
        return env

    made = Tracking(
        chains=kwargs.pop("chains", CHAINS),
        factory=factory,
        oracle=FakeOracle(solves=solves, after=after),
        subtask_steps=subtask_steps,
        **kwargs,
    )
    made.envs = envs
    return made


# -- the refusals that keep this from being a single-goal benchmark -----------


def test_a_one_deep_chain_is_refused():
    """The most misleading thing this plugin could do is report a single-goal
    run under a long-horizon name."""
    with pytest.raises(ConfigError, match="single-goal benchmark"):
        evaluator(chains=[["open_drawer"], ["lift_block"]])


def test_no_chains_at_all_is_refused():
    with pytest.raises(ConfigError, match="nothing to measure"):
        evaluator(chains=[])


def test_an_unknown_split_is_refused():
    with pytest.raises(ConfigError, match="unknown CALVIN split"):
        evaluator(split="Z_Z")


def test_a_scene_with_no_chain_in_it_is_refused_at_run_time():
    from gantry.contracts.evaluator import Scene

    made = evaluator()
    world = made.world_for(made.task_for().scenes[0])
    with pytest.raises(ConfigError, match="no instruction chain"):
        world.begin(Scene(id="bare"))


# -- the chain itself --------------------------------------------------------


def test_the_chain_does_not_reset_between_subtasks():
    """The entire reason CALVIN measures something the single-goal suites cannot.
    An adapter that reset per subtask would produce five independent single-goal
    trials wearing a chain's clothing, and its numbers would look plausible."""
    made = tracking()
    made.run(Chunker(), made.task_for(chains=1), Protocol())
    # One reset for the whole chain of five, not five.
    assert made.envs[0].resets == 1


def test_every_completed_subtask_is_a_milestone_at_the_step_it_landed():
    made = tracking()
    record = made.run(Chunker(), made.task_for(chains=1), Protocol())
    events = record.episodes[0].labels.stage_events
    assert [event.name for event in events] == CHAINS[0]
    assert [event.step for event in events] == [1, 3, 5, 7, 9]


def test_a_chain_broken_at_three_is_recorded_as_three_and_names_what_broke():
    """Not a failure — a length. A policy that reaches three is meaningfully
    better than one that reaches zero, and a boolean cannot say so."""
    made = tracking(solves=3)
    record = made.run(Chunker(), made.task_for(chains=1), Protocol())
    annotations = record.episodes[0].labels.annotations
    assert annotations["chain_length"] == 3
    assert annotations["chain_of"] == 5
    assert annotations["broke_on"] == CHAINS[0][3]
    assert record.episodes[0].labels.success is False


def test_average_chain_length_is_reported_beside_the_success_rate():
    """The headline CALVIN number, and the one that behaves interestingly: 5/5 is
    zero both for a policy that reaches four every time and for one that does
    nothing, and chain length tells them apart."""
    made = tracking(solves=3)
    record = made.run(Chunker(), made.task_for(), Protocol())
    length = record.metrics["chain_length"]
    assert length.value == 3.0
    assert length.n == 2
    assert length.detail["of"] == 5
    assert length.detail["distribution"] == {"3": 2}
    assert record.metrics["success_rate"].value == 0.0


def test_a_full_chain_counts_as_success_under_the_default():
    made = tracking(solves=5)
    record = made.run(Chunker(), made.task_for(), Protocol())
    assert all(episode.labels.success for episode in record.episodes)
    assert record.metrics["chain_length"].value == 5.0


def test_running_out_of_time_on_a_subtask_stops_the_chain_there():
    made = tracking(solves=5, after=999, subtask_steps=6)
    record = made.run(Chunker(), made.task_for(chains=1), Protocol())
    assert record.episodes[0].labels.annotations["chain_length"] == 0
    assert record.episodes[0].labels.annotations["broke_on"] == CHAINS[0][0]


# -- the collapse, which must never be implicit ------------------------------


def test_how_success_was_defined_is_in_the_descriptor_and_the_record():
    """A run scored at_least(2) and one scored 5/5 are not comparable, and the
    difference must not live only in somebody's memory of how they invoked it."""
    strict = tracking()
    assert strict.descriptor().metadata["success_means"] == "all_five"

    lenient = tracking(succeeded=at_least(2))
    assert lenient.descriptor().metadata["success_means"] == "at_least_2"

    record = lenient.run(Chunker(), lenient.task_for(), Protocol())
    assert record.episodes[0].labels.annotations["success_means"] == "at_least_2"


def test_at_least_two_calls_a_three_chain_a_success_and_all_five_does_not():
    three = dict(solves=3)
    assert (
        tracking(**three)
        .run(Chunker(), tracking(**three).task_for(chains=1), Protocol())
        .episodes[0]
        .labels.success
        is False
    )

    lenient = tracking(succeeded=at_least(2), **three)
    record = lenient.run(Chunker(), lenient.task_for(chains=1), Protocol())
    assert record.episodes[0].labels.success is True


def test_at_least_refuses_a_k_outside_the_chain():
    for bad in (0, 6, -1):
        with pytest.raises(ConfigError, match="chain is 5 long"):
            at_least(bad)


def test_all_five_is_exactly_what_it_says():
    assert all_five(5) is True
    assert all_five(4) is False


# -- the rest ----------------------------------------------------------------


def test_the_horizon_covers_the_whole_chain_by_default():
    """A horizon that cut a chain short would report a policy as having broken on
    subtask three when in fact the harness ran out."""
    made = evaluator(subtask_steps=100)
    assert made.task_for().horizon == 500


def test_the_stages_a_funnel_would_use_are_declared_on_the_task():
    made = evaluator()
    task = made.task_for()
    assert made.descriptor().provides["stage_events"] is True
    assert set(task.stages) == set(CHAINS[0]) | set(CHAINS[1])


def test_a_calvin_evaluator_conforms():
    made = tracking()
    verdict = check_evaluator(made, Chunker(), made.task_for())
    assert verdict.ok, verdict.explain()


def test_close_releases_the_environment():
    made = tracking()
    made.run(Chunker(), made.task_for(chains=1), Protocol())
    env = made.envs[0]
    made.close()
    assert env.closed


@pytest.mark.skip(reason="needs CALVIN, pybullet and the CALVIN dataset")
def test_against_the_real_simulator():  # pragma: no cover
    made = CalvinEvaluator("D_D", chains=CHAINS)
    assert made.action().shape == (7,)
