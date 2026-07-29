"""Closed-loop robosuite, checked without MuJoCo.

The environment is injected and copies robomimic's ``EnvBase`` surface exactly —
``reset``, ``reset_to({"states": ...})``, ``step`` returning four values, and
``is_success()`` returning a mapping. What is tested is this plugin's use of
that API, not a paraphrase of it.

The fake world is a one-dimensional reach: the state is a position, the action
moves it, and the task is solved when it passes a threshold. That is enough to
tell a policy that acts from one that does not, which is all a harness test
needs to establish.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_robosuite import OSC_POSE, RobosuiteEvaluator

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol, TaskSpec
from gantry.contracts.policy import Observation, Policy, policy_descriptor
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec, Descriptor, IncompatibleError

ENV_META = {
    "env_name": "Lift",
    "type": 1,
    "env_kwargs": {
        "robots": ["Panda"],
        "controller_configs": {"type": "OSC_POSE"},
        "control_freq": 20,
    },
}

STATES = tuple(np.array([float(i), 0.0, 0.0]) for i in range(4))


class FakeEnv:
    """robomimic's EnvBase surface, backed by arithmetic."""

    def __init__(self, env_meta, *, goal: float = 0.5, **kwargs):
        self.env_meta = env_meta
        self.kwargs = kwargs
        self._goal = goal
        self.position = np.zeros(3)
        self.restored: list[np.ndarray] = []
        self.resets = 0
        self.closed = False

    def _obs(self):
        return {
            "robot0_eef_pos": self.position.astype("float32"),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype="float32"),
            "robot0_gripper_qpos": np.zeros(2, dtype="float32"),
            "object": np.zeros(10, dtype="float32"),
            "robot0_joint_vel": np.zeros(7),  # not in PROPRIO; should be ignored
        }

    def reset(self):
        self.resets += 1
        self.position = np.zeros(3)
        return self._obs()

    def reset_to(self, state):
        restored = np.asarray(state["states"])
        self.restored.append(restored)
        self.position = restored.copy()
        return self._obs()

    def step(self, action):
        self.position = self.position + np.asarray(action[:3], dtype=float)
        return self._obs(), 1.0 if self._solved() else 0.0, False, {}

    def _solved(self) -> bool:
        return bool(self.position[0] >= self._goal)

    def is_success(self):
        return {"task": self._solved()}

    def close(self):
        self.closed = True


class Pusher(Policy):
    """Moves +x by a fixed amount. Zero means it never solves anything."""

    def __init__(self, step: float = 1.0, chunk: int = 1, raises: bool = False):
        self._step, self._chunk, self._raises = step, chunk, raises
        self.calls = 0

    def descriptor(self) -> Descriptor:
        return policy_descriptor("pusher", "0.1", chunk=self._chunk, deterministic=True)

    def action_spec(self) -> ChannelSpec:
        return OSC_POSE

    def observes(self):
        return requires_channels("pusher", "policy")

    def act(self, observation: Observation) -> np.ndarray:
        self.calls += 1
        if self._raises:
            raise RuntimeError("the model fell over")
        one = np.array([self._step, 0, 0, 0, 0, 0, 0], dtype="float32")
        return np.tile(one, (self._chunk, 1))


def evaluator(**kwargs) -> RobosuiteEvaluator:
    built: list[FakeEnv] = []
    goal = kwargs.pop("goal", 0.5)

    def factory(env_meta, **kw):
        env = FakeEnv(env_meta, goal=goal, **kw)
        built.append(env)
        return env

    kwargs.setdefault("factory", factory)
    ev = RobosuiteEvaluator(ENV_META, STATES, **kwargs)
    ev.built = built  # type: ignore[attr-defined]
    return ev


# -- the contract ----------------------------------------------------------


def test_conforms():
    ev = evaluator()
    verdict = check_evaluator(ev, Pusher(), ev.task_for(scenes=2, horizon=20), Protocol(), strict=True)
    assert verdict.ok, verdict.explain()


def test_it_takes_its_identity_from_the_recording():
    descriptor = evaluator().descriptor()
    assert descriptor.metadata["task"] == "Lift"
    assert descriptor.metadata["robots"] == ["Panda"]
    assert descriptor.metadata["scenes"] == 4
    assert descriptor.provides["closed_loop"] is True
    assert descriptor.provides["outcomes"] is True
    # Sub-goals exist for some robosuite tasks but are not read, so no claim.
    assert descriptor.provides["stage_events"] is False


def test_it_asks_for_the_controller_the_data_was_collected_with():
    spec = evaluator().requires().channels[0]
    assert spec.shape == (7,)
    assert spec.dim_labels == ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def test_a_world_with_no_name_is_refused():
    with pytest.raises(ConfigError, match="env_name"):
        RobosuiteEvaluator({"env_kwargs": {}}, STATES)


def test_no_starting_states_is_refused():
    with pytest.raises(ConfigError, match="initial states"):
        RobosuiteEvaluator(ENV_META, [])


# -- scenes are the demonstrations' starting states ------------------------


def test_a_scene_is_a_recorded_demonstrations_starting_state():
    task = evaluator().task_for()
    assert len(task.scenes) == 4
    assert task.scenes[2].id == "demo_2"
    assert task.scenes[2].metadata == {"initial_state": 2}


def test_the_starting_state_is_actually_restored():
    """The whole point: the policy faces the scene the demonstrator faced."""
    ev = evaluator(goal=99.0)
    ev.evaluate(Pusher(), ev.task_for(scenes=3, horizon=2), Protocol())
    restored = ev.built[0].restored
    assert len(restored) == 3
    assert np.array_equal(restored[2], STATES[2])


def test_asking_for_more_scenes_than_there_are_states_takes_what_exists():
    assert len(evaluator().task_for(scenes=99).scenes) == 4


def test_one_world_is_built_and_reused():
    """Constructing a robosuite env starts MuJoCo; per-trial would dominate."""
    ev = evaluator(goal=99.0)
    ev.evaluate(Pusher(), ev.task_for(scenes=4, horizon=3), Protocol())
    assert len(ev.built) == 1


# -- running ---------------------------------------------------------------


def test_success_comes_from_the_task_not_the_reward():
    """A shaped reward is positive for being close, which is not the same thing."""
    solving = evaluator(goal=0.5)
    won = solving.evaluate(Pusher(step=1.0), solving.task_for(scenes=2, horizon=10), Protocol())
    assert [e.labels.success for e in won.episodes] == [True, True]
    assert won.metrics["success_rate"].value == 1.0

    stuck = evaluator(goal=99.0)
    lost = stuck.evaluate(Pusher(step=0.0), stuck.task_for(scenes=2, horizon=10), Protocol())
    assert [e.labels.success for e in lost.episodes] == [False, False]
    assert lost.metrics["success_rate"].n == 2


def test_a_solved_trial_stops_there():
    ev = evaluator(goal=2.5)
    record = ev.evaluate(Pusher(step=1.0), ev.task_for(scenes=1, horizon=50), Protocol())
    # Starts at x=0, needs 3 pushes of 1.0 to pass 2.5.
    assert record.episodes[0].labels.annotations["steps"] == 3


def test_the_recorded_episode_is_what_happened():
    ev = evaluator(goal=99.0)
    record = ev.evaluate(Pusher(), ev.task_for(scenes=1, horizon=5), Protocol())
    episode = record.episodes[0]
    assert set(episode.channel_names) == {
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object",
        "actions", "reward",
    }
    assert episode.array("actions").shape == (5, 7)
    assert episode.validate(deep=True).ok
    # The position advances by exactly the action that was executed.
    positions = episode.array("robot0_eef_pos")[:, 0]
    assert np.allclose(np.diff(positions), 1.0)


def test_keys_outside_the_requested_set_are_left_out():
    ev = evaluator(goal=99.0)
    record = ev.evaluate(Pusher(), ev.task_for(scenes=1, horizon=3), Protocol())
    assert "robot0_joint_vel" not in record.episodes[0].channel_names


def test_one_episode_per_trial_across_epochs():
    ev = evaluator(goal=99.0)
    record = ev.evaluate(Pusher(), ev.task_for(scenes=2, horizon=3), Protocol(epochs=2))
    assert len(record) == 4
    assert len({e.meta.uid for e in record.episodes}) == 4


def test_execute_decides_how_often_the_policy_is_asked():
    ev = evaluator(goal=99.0)
    task = ev.task_for(scenes=1, horizon=8)
    whole, closed = Pusher(chunk=4), Pusher(chunk=4)
    ev.evaluate(whole, task, Protocol(execute=4))
    ev.evaluate(closed, task, Protocol(execute=1))
    assert (whole.calls, closed.calls) == (2, 8)


def test_the_world_and_protocol_are_recorded():
    ev = evaluator(goal=99.0)
    record = ev.evaluate(Pusher(), ev.task_for(scenes=1, horizon=4), Protocol(execute=2))
    assert record.provenance.protocol["env"] == "Lift"
    assert record.provenance.protocol["execute"] == 2
    assert record.episodes[0].meta.embodiment == "Panda"


def test_a_policy_that_raises_costs_one_trial():
    ev = evaluator()
    record = ev.evaluate(Pusher(raises=True), ev.task_for(scenes=2, horizon=4), Protocol())
    assert len(record) == 2
    assert all(e.labels.success is None for e in record.episodes)
    assert "fell over" in record.episodes[0].labels.annotations["error"]
    assert record.provenance.notes == ("2 trial(s) failed to complete",)


def test_a_badly_shaped_chunk_halts():
    class Flat(Pusher):
        def act(self, observation):
            return np.zeros(7, dtype="float32")

    ev = evaluator()
    with pytest.raises(ComponentError, match="expected"):
        ev.evaluate(Flat(), ev.task_for(scenes=1, horizon=4), Protocol())


def test_a_task_with_no_scenes_is_refused():
    ev = evaluator()
    with pytest.raises(IncompatibleError):
        ev.evaluate(Pusher(), TaskSpec("empty", scenes=()), Protocol())


def test_closing_reaches_the_world():
    ev = evaluator(goal=99.0)
    ev.evaluate(Pusher(), ev.task_for(scenes=1, horizon=2), Protocol())
    env = ev.env
    ev.close()
    assert env.closed


# -- the simulator ---------------------------------------------------------


def test_the_simulator_is_not_imported_by_installing_this():
    """A laptop should be able to read the descriptor and plan a run.

    Only the simulator is asserted here. ``h5py`` cannot be checked this way —
    ``sys.modules`` is process-global, so a sibling plugin's tests importing it
    would fail an assertion about *this* one. That claim is instead carried by
    tools/isolation_check.py, which installs only what this plugin declares and
    runs the suite with h5py genuinely absent.
    """
    import sys

    assert "robosuite" not in sys.modules
    assert "robomimic" not in sys.modules
    assert evaluator().descriptor().metadata["task"] == "Lift"


def test_without_the_simulator_the_refusal_says_how_to_get_it():
    import importlib.util

    if importlib.util.find_spec("robomimic") is not None:  # pragma: no cover
        pytest.skip("robomimic is installed here")
    ev = RobosuiteEvaluator(ENV_META, STATES)
    with pytest.raises(ConfigError, match=r"evaluator-robosuite\[sim\]"):
        ev.env
