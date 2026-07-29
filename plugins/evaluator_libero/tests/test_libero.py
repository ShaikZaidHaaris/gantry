"""LIBERO's shape, checked without LIBERO.

The suite and the environment are both injected, so everything here runs on a
laptop with no MuJoCo. The fakes copy the real API exactly as the vendored
source defines it — ``get_task``/``get_task_init_states`` on a suite,
``reset``/``set_init_state``/``step``/``check_success`` on an environment — so
what is tested is this plugin's use of that API rather than a convenient
paraphrase of it.

The one test that needs the real simulator is marked and skips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from gantry_evaluator_libero import (
    ACTION,
    SUITES,
    LiberoEvaluator,
    success_from_check,
    success_from_done,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol, TaskSpec
from gantry.contracts.policy import Observation, Policy, policy_descriptor
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec, Descriptor, IncompatibleError

SIZE = 8


@dataclass
class FakeTask:
    name: str
    language: str
    bddl_file: str
    problem_folder: str = "libero_spatial"


class FakeSuite:
    """What ``benchmark.get_benchmark_dict()[name]()`` returns."""

    def __init__(self, tasks: int = 3, inits: int = 2):
        self.n_tasks = tasks
        self._inits = inits
        self.name = "libero_spatial"

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task(self, i: int) -> FakeTask:
        return FakeTask(f"task_{i}", f"do thing {i}", __file__)

    def get_task_init_states(self, i: int):
        return np.arange(self._inits * 4).reshape(self._inits, 4) + i * 100


class FakeEnv:
    """What ``OffScreenRenderEnv`` gives you: robosuite's four-tuple, no seed."""

    def __init__(self, bddl_file: str, *, solves_at: int | None = 3, **kwargs):
        self.bddl_file = bddl_file
        self.kwargs = kwargs
        self._solves_at = solves_at
        self.steps = 0
        self.init_state = None
        self.closed = False
        self.resets = 0

    def _obs(self):
        return {
            "agentview_image": np.full((SIZE, SIZE, 3), self.steps % 255, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            "robot0_eef_pos": np.full(3, float(self.steps), dtype="float32"),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype="float32"),
            "robot0_gripper_qpos": np.zeros(2, dtype="float32"),
            "unwanted_key": np.zeros(9),
        }

    def reset(self):
        self.resets += 1
        self.steps = 0
        return self._obs()

    def set_init_state(self, state):
        self.init_state = np.asarray(state)
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.steps += 1
        done = self.steps >= 50
        return self._obs(), -1.0, done, {}

    def check_success(self) -> bool:
        return self._solves_at is not None and self.steps >= self._solves_at

    def close(self):
        self.closed = True


class Nudger(Policy):
    """Emits a fixed OSC_POSE delta. Enough to drive the loop."""

    def __init__(self, chunk: int = 1, raises: bool = False):
        self._chunk, self._raises = chunk, raises
        self.calls = 0

    def descriptor(self) -> Descriptor:
        return policy_descriptor("nudger", "0.1", chunk=self._chunk, deterministic=True)

    def action_spec(self) -> ChannelSpec:
        return ACTION

    def observes(self):
        return requires_channels("nudger", "policy")

    def act(self, observation: Observation) -> np.ndarray:
        self.calls += 1
        if self._raises:
            raise RuntimeError("the model fell over")
        return np.tile(np.array([0.1] * 6 + [-1.0], dtype="float32"), (self._chunk, 1))


def evaluator(**kwargs) -> LiberoEvaluator:
    envs: list[FakeEnv] = []
    solves_at = kwargs.pop("solves_at", 3)

    def factory(bddl, **kw):
        env = FakeEnv(bddl, **{**kw, "solves_at": solves_at})
        envs.append(env)
        return env

    kwargs.setdefault("loader", lambda name: FakeSuite())
    kwargs.setdefault("factory", factory)
    ev = LiberoEvaluator(**kwargs)
    ev.built = envs  # type: ignore[attr-defined]
    return ev


# -- the contract ----------------------------------------------------------


def test_conforms():
    ev = evaluator()
    task = ev.task_for(tasks=2, inits=1, horizon=20)
    verdict = check_evaluator(ev, Nudger(), task, Protocol(), strict=True)
    assert verdict.ok, verdict.explain()


def test_it_declares_what_libero_can_actually_report():
    provides = evaluator().descriptor().provides
    assert provides["outcomes"] is True
    assert provides["closed_loop"] is True
    # One rung: solved or not. A funnel over it would say nothing, so it does
    # not claim milestones it cannot produce.
    assert provides["stage_events"] is False


def test_an_unknown_suite_is_refused_by_name():
    with pytest.raises(ConfigError, match="libero_spatial"):
        LiberoEvaluator("libero_kitchen")
    assert "libero_90" in SUITES


def test_it_asks_for_an_osc_pose_action():
    spec = evaluator().requires().channels[0]
    assert spec.shape == (7,)
    assert spec.dim_labels[-1] == "gripper"


# -- scenes are (task, initial state) --------------------------------------


def test_a_scene_is_a_task_and_one_of_its_stored_initial_states():
    """LIBERO's reset takes no seed, so this is what pins a trial."""
    task = evaluator().task_for(tasks=3, inits=2)
    assert len(task.scenes) == 6
    assert task.scenes[0].id == "task_0#0"
    assert task.scenes[1].id == "task_0#1"
    assert task.scenes[0].instruction == "do thing 0"
    assert task.scenes[0].metadata == {"task_index": 0, "init_index": 0}


def test_asking_for_more_initial_states_than_exist_takes_what_there_is():
    task = evaluator(loader=lambda n: FakeSuite(tasks=1, inits=2)).task_for(inits=99)
    assert len(task.scenes) == 2


def test_the_initial_state_actually_reaches_the_environment():
    ev = evaluator()
    ev.evaluate(Nudger(), ev.task_for(tasks=1, inits=2, horizon=5), Protocol())
    env = ev.built[0]
    assert env.init_state is not None
    assert np.array_equal(env.init_state, np.arange(8).reshape(2, 4)[1])


def test_one_environment_is_built_per_task_and_reused_across_its_states():
    """Building one parses a scene and starts a renderer; per-trial would dominate."""
    ev = evaluator()
    ev.evaluate(Nudger(), ev.task_for(tasks=2, inits=3, horizon=5), Protocol())
    assert len(ev.built) == 2, "two tasks, six trials"


# -- running ---------------------------------------------------------------


def test_a_run_gives_one_episode_per_trial():
    ev = evaluator()
    record = ev.evaluate(Nudger(), ev.task_for(tasks=2, inits=2, horizon=10), Protocol())
    assert len(record) == 4
    assert len({e.meta.uid for e in record.episodes}) == 4


def test_the_recorded_episode_is_what_happened():
    ev = evaluator(solves_at=None)
    record = ev.evaluate(Nudger(), ev.task_for(tasks=1, inits=1, horizon=6), Protocol())
    episode = record.episodes[0]
    assert set(episode.channel_names) == {
        "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos", "action", "reward",
    }
    assert episode.array("action").shape == (6, 7)
    assert episode.array("agentview_image").shape == (6, SIZE, SIZE, 3)
    assert episode.validate(deep=True).ok


def test_keys_the_caller_did_not_ask_for_are_left_out():
    ev = evaluator(solves_at=None)
    record = ev.evaluate(Nudger(), ev.task_for(tasks=1, inits=1, horizon=4), Protocol())
    assert "unwanted_key" not in record.episodes[0].channel_names


def test_success_is_read_from_check_success_not_from_done():
    """``done`` also fires on the horizon, so reading it would score timeouts as wins."""
    solving = evaluator(solves_at=2)
    won = solving.evaluate(Nudger(), solving.task_for(tasks=1, inits=1, horizon=30), Protocol())
    assert won.episodes[0].labels.success is True
    assert won.metrics["success_rate"].value == 1.0

    never = evaluator(solves_at=None)
    lost = never.evaluate(Nudger(), never.task_for(tasks=1, inits=1, horizon=30), Protocol())
    assert lost.episodes[0].labels.success is False
    assert lost.metrics["success_rate"].value == 0.0


def test_the_done_reader_is_available_when_a_suite_wants_it():
    ev = evaluator(success=success_from_done, solves_at=None)
    record = ev.evaluate(Nudger(), ev.task_for(tasks=1, inits=1, horizon=80), Protocol())
    assert record.episodes[0].labels.success is True, "done fires at step 50"
    assert success_from_check is not success_from_done


def test_execute_decides_how_often_the_policy_is_asked():
    ev = evaluator(solves_at=None)
    task = ev.task_for(tasks=1, inits=1, horizon=8)
    whole, closed = Nudger(chunk=4), Nudger(chunk=4)
    ev.evaluate(whole, task, Protocol(execute=4))
    ev.evaluate(closed, task, Protocol(execute=1))
    assert (whole.calls, closed.calls) == (2, 8)


def test_the_suite_and_protocol_are_recorded():
    ev = evaluator()
    record = ev.evaluate(Nudger(), ev.task_for(tasks=1, inits=1, horizon=5), Protocol(execute=2))
    assert record.provenance.protocol["suite"] == "libero_spatial"
    assert record.provenance.protocol["execute"] == 2


def test_a_policy_that_raises_costs_one_trial():
    ev = evaluator()
    record = ev.evaluate(Nudger(raises=True), ev.task_for(tasks=2, inits=1, horizon=5), Protocol())
    assert len(record) == 2
    assert all(e.labels.success is None for e in record.episodes)
    assert "fell over" in record.episodes[0].labels.annotations["error"]


def test_a_badly_shaped_chunk_halts_rather_than_being_reshaped():
    class Flat(Nudger):
        def act(self, observation):
            return np.zeros(7, dtype="float32")

    ev = evaluator()
    with pytest.raises(ComponentError, match="expected"):
        ev.evaluate(Flat(), ev.task_for(tasks=1, inits=1, horizon=5), Protocol())


def test_a_task_with_no_scenes_is_refused():
    ev = evaluator()
    with pytest.raises(IncompatibleError):
        ev.evaluate(Nudger(), TaskSpec("empty", scenes=()), Protocol())


def test_closing_reaches_every_environment():
    ev = evaluator()
    ev.evaluate(Nudger(), ev.task_for(tasks=2, inits=1, horizon=4), Protocol())
    ev.close()
    assert all(env.closed for env in ev.built)


# -- the simulator itself --------------------------------------------------


def test_the_simulator_is_not_imported_just_by_installing_this():
    """A laptop should be able to read the descriptor and plan a run."""
    import sys

    assert "libero" not in sys.modules
    assert "robosuite" not in sys.modules
    assert LiberoEvaluator("libero_object").descriptor().provides["outcomes"] is True


def test_asking_for_a_real_suite_without_the_simulator_says_how_to_get_it():
    pytest.importorskip  # noqa: B018 - documented below
    try:
        import libero  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigError, match=r"evaluator-libero\[sim\]"):
            LiberoEvaluator("libero_spatial").suite


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("libero") is None,
    reason="LIBERO is not installed here",
)
def test_against_the_real_suite():  # pragma: no cover - needs the simulator
    ev = LiberoEvaluator("libero_spatial")
    task = ev.task_for(tasks=1, inits=1)
    assert task.scenes and task.scenes[0].instruction
