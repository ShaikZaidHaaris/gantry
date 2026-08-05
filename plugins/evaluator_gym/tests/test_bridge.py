"""Driving a step/reset environment, in both dialects, with nothing installed.

The environments here are twenty lines of arithmetic. That is the point: this
plugin's claim is that it needs no environment library, and a test suite that
imported one would be testing that library's packaging rather than this bridge.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_gym import (
    GymEvaluator,
    Readers,
    Transition,
    accepts_seed,
    as_channels,
    channels_of,
    imported,
    infer_spec,
    milestones_from_flags,
    milestones_from_info,
    read_step,
    reset_env,
    success_from_info,
    terminated_is_success,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol, Scene, TaskSpec
from gantry.contracts.policy import EpisodeContext, Observation, Policy, policy_descriptor
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec, Descriptor

ACTION = ChannelSpec("action", "vector", (3,), "float32", units="m", semantics="actuation")
DELTA = ChannelSpec("delta", "vector", (3,), "float32", units="m", semantics="position")

MILESTONES = ("near", "closer", "arrived")


# -- two environments, one arithmetic ---------------------------------------


class Reach:
    """Move to a point. Current dialect: ``reset(seed=)`` and a five-tuple."""

    def __init__(self, tolerance: float = 0.15, near: float = 0.4, spread: float = 2.0):
        self.tolerance, self.near, self.spread = tolerance, near, spread
        self.closed = False

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(0 if seed is None else seed)
        direction = rng.normal(size=3)
        self.goal = direction / np.linalg.norm(direction) * rng.uniform(self.near, self.spread)
        self.state = np.zeros(3)
        return self._observation(), {}

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "state": self.state.astype("float32"),
            "delta": (self.goal - self.state).astype("float32"),
        }

    def step(self, action):
        self.state = self.state + np.asarray(action, dtype=float)
        distance = float(np.linalg.norm(self.goal - self.state))
        terminated = distance <= self.tolerance
        info = {
            "success": terminated,
            "distance": distance,
            "milestones": [
                name
                for name, threshold in zip(MILESTONES, (1.0, 0.5, self.tolerance))
                if distance <= threshold
            ],
        }
        return self._observation(), -distance, terminated, False, info

    def close(self) -> None:
        self.closed = True


class LegacyReach(Reach):
    """The older dialect: seeded on the side, bare reset, four-tuple."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seed: int | None = None

    def seed(self, seed: int) -> None:
        self._seed = seed

    def reset(self):
        return super().reset(seed=self._seed)[0]

    def step(self, action):
        observation, reward, terminated, _, info = super().step(action)
        return observation, reward, terminated, info


class ArrayReach(Reach):
    """Some environments observe one array rather than a dict."""

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        return (self.goal - self.state).astype("float32"), {}

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return (self.goal - self.state).astype("float32"), reward, terminated, truncated, info


# -- something to drive them ------------------------------------------------


class Mover(Policy):
    """Steps toward the goal, at most ``limit`` at a time."""

    def __init__(self, limit: float = 0.3, chunk: int = 1, raises: bool = False):
        self.limit, self._chunk, self._raises = limit, chunk, raises
        self.calls = 0

    def descriptor(self) -> Descriptor:
        return policy_descriptor("mover", "0.1", chunk=self._chunk, deterministic=True)

    def action_spec(self) -> ChannelSpec:
        return ACTION

    def observes(self):
        return requires_channels("mover", "policy", DELTA)

    def reset(self, context: EpisodeContext) -> None:
        self.context = context

    def act(self, observation: Observation) -> np.ndarray:
        self.calls += 1
        if self._raises:
            raise RuntimeError("the model fell over")
        delta = np.asarray(observation["delta"], dtype=float)
        chunk = []
        for _ in range(self._chunk):
            distance = float(np.linalg.norm(delta))
            step = delta if distance <= self.limit else delta / distance * self.limit
            chunk.append(step)
            delta = delta - step  # a chunk is executed open-loop
        return np.asarray(chunk, dtype="float32")


class Fixed(Policy):
    """Emits the same action forever, and reads nothing at all."""

    def __init__(self, value: float = 0.05):
        self._value = np.full(3, value, dtype="float32")

    def descriptor(self) -> Descriptor:
        return policy_descriptor("fixed", "0.1", chunk=1, deterministic=True)

    def action_spec(self) -> ChannelSpec:
        return ACTION

    def observes(self):
        return requires_channels("fixed", "policy")

    def act(self, observation: Observation) -> np.ndarray:
        return self._value[None, :]


def evaluator(env=Reach, **kwargs) -> GymEvaluator:
    kwargs.setdefault("readers", Readers(success_from_info(), milestones_from_info(), MILESTONES))
    kwargs.setdefault("scenes", 6)
    kwargs.setdefault("horizon", 25)
    return GymEvaluator(env, ACTION, **kwargs)


# -- the contract -----------------------------------------------------------


def test_conforms():
    gym = evaluator()
    verdict = check_evaluator(gym, Mover(), gym.task_for("reach"), Protocol(), strict=True)
    assert verdict.ok, verdict.explain()


def test_it_conforms_in_the_older_dialect_too():
    gym = evaluator(LegacyReach)
    verdict = check_evaluator(gym, Mover(), gym.task_for("reach"), Protocol())
    assert verdict.ok, verdict.explain()


def test_it_declares_only_what_its_readers_can_deliver():
    bare = GymEvaluator(Reach, ACTION)
    assert bare.descriptor().provides["outcomes"] is False
    assert bare.descriptor().provides["stage_events"] is False
    assert evaluator().descriptor().provides == {
        "stage_events": True,
        "outcomes": True,
        "seedable": True,
        "closed_loop": True,
        # A gym env brings its own world, so it is the same world whichever
        # dataset is under test and never needs binding to one.
        "needs_dataset": False,
        # And its body is whatever it was written as; it cannot be handed
        # another one.
        "hosts_embodiment": False,
    }


def test_it_asks_for_the_action_the_environment_takes():
    requirement = evaluator().requires()
    assert [spec.name for spec in requirement.channels] == ["action"]
    assert requirement.plane == "evaluation"


# -- running ----------------------------------------------------------------


def test_a_run_produces_one_episode_per_trial():
    gym = evaluator()
    record = gym.evaluate(Mover(), gym.task_for("reach", scenes=4), Protocol(epochs=2))
    assert len(record) == 8
    assert len({episode.meta.uid for episode in record.episodes}) == 8


def test_the_trajectory_that_comes_back_is_the_one_that_happened():
    gym = evaluator()
    record = gym.evaluate(Mover(), gym.task_for("reach", scenes=1), Protocol())
    episode = record.episodes[0]
    assert set(episode.channel_names) == {"state", "delta", "action", "reward"}
    steps = len(episode)
    assert episode.array("action").shape == (steps, 3)
    assert episode.array("reward").shape == (steps,)
    # State advances by exactly the action that was executed.
    state, action = episode.array("state"), episode.array("action")
    assert np.allclose(state[1:], state[:-1] + action[:-1], atol=1e-5)


def test_success_and_milestones_come_from_where_the_caller_said():
    gym = evaluator()
    record = gym.evaluate(Mover(), gym.task_for("reach"), Protocol())
    assert record.has_outcomes and record.has_stage_events
    assert set(record.stages) <= set(MILESTONES)
    succeeded = [e for e in record.episodes if e.labels.success]
    assert succeeded, "a policy that walks to the goal should reach some of them"
    assert all(e.labels.reached("arrived") for e in succeeded)


def test_a_milestone_is_recorded_once_at_first_arrival():
    gym = evaluator()
    episode = gym.evaluate(Mover(), gym.task_for("reach", scenes=1), Protocol()).episodes[0]
    names = [event.name for event in episode.labels.stage_events]
    assert len(names) == len(set(names))
    assert names == sorted(names, key=lambda name: episode.labels.step_of(name))


def test_the_horizon_stops_a_trial_that_is_not_getting_there():
    gym = evaluator(horizon=3)
    episode = gym.evaluate(Mover(limit=0.01), gym.task_for("reach", scenes=1), Protocol()).episodes[
        0
    ]
    assert len(episode) == 3
    assert episode.labels.annotations["truncated"] is True
    assert episode.labels.success is False


def test_execute_decides_how_much_of_a_chunk_is_played():
    """The lever that moved this project's own benchmark by fourteen points."""
    gym = evaluator(lambda: Reach(near=1.5), horizon=8)
    task = gym.task_for("reach", scenes=1)
    open_loop, closed = Mover(limit=0.05, chunk=4), Mover(limit=0.05, chunk=4)
    whole = gym.evaluate(open_loop, task, Protocol(execute=4)).episodes[0]
    every = gym.evaluate(closed, task, Protocol(execute=1)).episodes[0]

    assert len(whole) == len(every) == 8
    # The setting decides how often the world is consulted, which is the whole
    # of what it costs and the whole of what it buys.
    assert (open_loop.calls, closed.calls) == (2, 8)


def test_metrics_report_the_rate_and_the_return():
    gym = evaluator()
    record = gym.evaluate(Mover(), gym.task_for("reach"), Protocol())
    assert 0.0 <= record.metrics["success_rate"].value <= 1.0
    assert record.metrics["success_rate"].n == len(record)
    assert record.metrics["mean_return"].value < 0, "reward here is negative distance"


def test_the_same_seed_gives_the_same_run():
    gym = evaluator()
    task = gym.task_for("reach")
    first = gym.evaluate(Mover(), task, Protocol(seed_base=7))
    second = gym.evaluate(Mover(), task, Protocol(seed_base=7))
    assert first.outcomes() == second.outcomes()
    other = gym.evaluate(Mover(), task, Protocol(seed_base=99))
    assert other.provenance.digest != first.provenance.digest


# -- the vocabulary a funnel needs ------------------------------------------


def test_the_milestones_looked_for_are_recorded_even_when_none_are_reached():
    """A policy that never leaves the start still defines the funnel's rungs.

    Inferring the vocabulary from what happened would make a policy failing at
    the first rung look like a one-rung task at a hundred percent.
    """
    gym = evaluator(lambda: Reach(near=1.5), horizon=2)
    record = gym.evaluate(Mover(limit=0.0), gym.task_for("reach", scenes=2), Protocol())
    assert not record.has_stage_events
    assert record.provenance.protocol["stages"] == list(MILESTONES)


def test_reading_milestones_without_naming_them_is_refused():
    with pytest.raises(ValueError, match="naming them"):
        Readers(milestones=milestones_from_info())


def test_milestones_can_come_from_separate_flags():
    read = milestones_from_flags("grasped", "lifted")
    transition = Transition(0, {}, 0.0, False, False, {"grasped": True, "lifted": False})
    assert read(transition) == ("grasped",)


def test_a_mapping_of_flags_is_read_the_same_way():
    read = milestones_from_info("stages")
    transition = Transition(0, {}, 0.0, False, False, {"stages": {"a": True, "b": False}})
    assert read(transition) == ("a",)


# -- reading outcomes -------------------------------------------------------


def test_an_unreported_outcome_stays_unknown_rather_than_becoming_a_failure():
    read = success_from_info("success")
    assert read(Transition(0, {}, 0.0, False, False, {})) is None
    assert read(Transition(0, {}, 0.0, False, False, {"success": False})) is False


def test_terminated_as_success_is_available_but_never_the_default():
    read = terminated_is_success()
    assert read(Transition(0, {}, 0.0, True, False, {})) is True
    assert read(Transition(0, {}, 0.0, False, True, {})) is None
    assert GymEvaluator(Reach, ACTION)._readers.success is None


def test_an_environment_that_reports_nothing_yields_no_outcomes():
    gym = GymEvaluator(Reach, ACTION, scenes=2, horizon=5)
    record = gym.evaluate(Mover(), gym.task_for("reach"), Protocol())
    assert not record.has_outcomes
    assert "success_rate" not in record.metrics


# -- dialects and shapes ----------------------------------------------------


def test_both_step_dialects_unpack_the_same_way():
    five = read_step(({"a": 1}, 1.0, True, False, {"k": 1}))
    four = read_step(({"a": 1}, 1.0, True, {"k": 1}))
    assert five == four


def test_a_step_return_of_the_wrong_length_says_so():
    with pytest.raises(ComponentError, match="3 value"):
        read_step((1, 2, 3))
    with pytest.raises(ComponentError, match="expected a tuple"):
        read_step({"obs": 1})


def test_reset_is_seeded_however_the_environment_accepts_it():
    modern, legacy = Reach(), LegacyReach()
    assert accepts_seed(modern.reset) and not accepts_seed(legacy.reset)
    first, info = reset_env(modern, 5)
    second, _ = reset_env(legacy, 5)
    assert info == {}
    assert np.allclose(first["delta"], second["delta"]), "same seed, same goal"


def test_an_array_observation_is_given_a_name():
    assert list(as_channels(np.zeros(3))) == ["observation"]
    gym = GymEvaluator(ArrayReach, ACTION, scenes=1, horizon=4)
    episode = gym.evaluate(Fixed(), gym.task_for("reach"), Protocol()).episodes[0]
    assert "observation" in episode.channel_names


def test_a_policy_expecting_another_name_fails_visibly_rather_than_being_renamed():
    """The bridge renames nothing, and the mismatch is recorded, not hidden.

    A policy asking for a channel the environment does not observe is a wiring
    problem. It costs the trial, it appears in the record as the error it was,
    and nothing quietly substitutes the one array that happened to be there.
    """
    gym = GymEvaluator(ArrayReach, ACTION, scenes=1, horizon=2)
    episode = gym.evaluate(Mover(), gym.task_for("reach"), Protocol()).episodes[0]
    assert "delta" in episode.labels.annotations["error"]
    assert episode.channel_names == ()


def test_shape_and_dtype_are_observed_and_nothing_else_is_invented():
    spec = infer_spec("state", np.zeros((3,), dtype="float64"))
    assert (spec.kind, spec.shape, spec.dtype) == ("vector", (3,), "float64")
    assert (spec.units, spec.frame, spec.semantics) == (None, None, None)
    assert infer_spec("pixels", np.zeros((4, 4))).kind == "matrix"
    assert infer_spec("volume", np.zeros((2, 2, 2))).kind == "tensor"
    assert infer_spec("scalar", np.float32(1.0)).kind == "scalar"


def test_a_declared_spec_wins_over_what_could_be_inferred():
    described = ChannelSpec("delta", "vector", (3,), "float32", units="m", semantics="position")
    gym = evaluator(observation_specs=[described], scenes=1, horizon=3)
    episode = gym.evaluate(Mover(), gym.task_for("reach"), Protocol()).episodes[0]
    assert episode.channel("delta").units == "m"
    assert episode.channel("state").units is None


def test_channels_of_says_what_an_environment_observes():
    names = {spec.name for spec in channels_of(Reach())}
    assert names == {"state", "delta"}


# -- narrowing and refusing -------------------------------------------------


def test_recording_can_be_narrowed_to_what_matters():
    gym = evaluator(observations=["delta"], scenes=1, horizon=3)
    episode = gym.evaluate(Mover(), gym.task_for("reach"), Protocol()).episodes[0]
    assert set(episode.channel_names) == {"delta", "action", "reward"}


def test_narrowing_to_something_absent_lists_what_is_there():
    gym = evaluator(observations=["gripper"], scenes=1, horizon=3)
    with pytest.raises(ConfigError, match="delta"):
        gym.evaluate(Mover(), gym.task_for("reach"), Protocol())


def test_a_factory_that_will_not_build_fails_before_the_run():
    def broken():
        raise ImportError("no such simulator")

    gym = GymEvaluator(broken, ACTION, scenes=1)
    with pytest.raises(ConfigError, match="no such simulator"):
        gym.evaluate(Mover(), gym.task_for("reach"), Protocol())


def test_something_that_is_not_an_environment_is_refused():
    gym = GymEvaluator(lambda: object(), ACTION, scenes=1)
    with pytest.raises(ConfigError, match="reset"):
        gym.evaluate(Mover(), gym.task_for("reach"), Protocol())


def test_an_unusable_action_spec_is_refused_at_construction():
    with pytest.raises(ConfigError, match="action spec"):
        GymEvaluator(Reach, ChannelSpec("action", "vector", (3, 3), "float32"))


def test_a_task_with_no_scenes_is_refused():
    from gantry.spine import IncompatibleError

    gym = evaluator()
    with pytest.raises(IncompatibleError):
        gym.evaluate(Mover(), TaskSpec("empty", scenes=()), Protocol())


# -- one trial failing is not the run failing -------------------------------


def test_a_policy_that_raises_loses_its_trial_and_no_more():
    gym = evaluator(scenes=3, horizon=4)
    record = gym.evaluate(Mover(raises=True), gym.task_for("reach"), Protocol())
    assert len(record) == 3
    assert all(episode.labels.success is None for episode in record.episodes)
    assert all("fell over" in episode.labels.annotations["error"] for episode in record.episodes)
    assert record.provenance.notes == ("3 trial(s) failed to complete",)


def test_a_scene_may_pin_its_own_seed():
    gym = evaluator(scenes=1, horizon=4)
    pinned = TaskSpec("pinned", scenes=(Scene(id="fixed", seed=11),), horizon=4, stages=MILESTONES)
    first = gym.evaluate(Mover(), pinned, Protocol(seed_base=0))
    second = gym.evaluate(Mover(), pinned, Protocol(seed_base=500))
    assert np.allclose(first.episodes[0].array("delta"), second.episodes[0].array("delta"))


# -- constructible from a file ---------------------------------------------


def test_everything_that_has_to_cross_a_manifest_is_plain_data():
    """A run that cannot be written down cannot be committed or re-run."""
    from gantry.serial import spec_to_dict

    config = {
        # In a manifest the factory is a dotted path; naming one here would
        # only test this file's own importability, so it is covered on its own
        # in test_an_environment_can_be_named_rather_than_imported.
        "action": spec_to_dict(ACTION),
        "observation_specs": [spec_to_dict(DELTA)],
        "readers": {
            "success": {"info": ["success"]},
            "milestones": {"info": "milestones"},
            "names": list(MILESTONES),
        },
        "scenes": 2,
        "horizon": 5,
    }
    gym = GymEvaluator(Reach, **config)
    record = gym.evaluate(Mover(), gym.task_for("reach"), Protocol())
    assert gym.descriptor().provides["outcomes"] is True
    assert record.provenance.protocol["stages"] == list(MILESTONES)
    assert record.episodes[0].channel("delta").units == "m"


def test_a_reader_nobody_defined_is_refused_by_name():
    with pytest.raises(ConfigError, match="unknown success reader"):
        Readers.from_config({"success": "vibes"})
    with pytest.raises(ConfigError, match="single-key object"):
        Readers.from_config({"success": {"info": [], "terminated": []}})


def test_readers_default_to_reporting_nothing():
    readers = Readers.from_config({})
    assert readers.success is None and readers.milestones is None


def test_an_environment_can_be_named_rather_than_imported():
    assert imported("gantry_evaluator_gym:GymEvaluator") is GymEvaluator
    with pytest.raises(ConfigError, match="does not name an attribute"):
        imported("gantry_evaluator_gym")
    with pytest.raises(ConfigError, match="cannot import"):
        imported("no_such_module:thing")
    with pytest.raises(ConfigError, match="has no"):
        imported("gantry_evaluator_gym:NoSuchThing")


def test_closing_reaches_the_environment():
    gym = evaluator(scenes=1, horizon=2)
    gym.evaluate(Mover(), gym.task_for("reach"), Protocol())
    env = gym.env
    gym.close()
    assert env.closed
