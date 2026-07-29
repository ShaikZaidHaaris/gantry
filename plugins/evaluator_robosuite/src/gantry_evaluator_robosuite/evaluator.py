"""robosuite as an evaluator, with the scenes taken from the demonstrations.

This is the piece that turns "how close were the predicted actions" into "did
the arm actually pick up the cube". Open-loop action error is cheap and honest
about what it measures, and it cannot tell you whether a policy recovers from
its own mistakes — the moment its error puts the arm somewhere the recording
never went, the recording stops being an answer key.

Where the scenes come from
--------------------------
A robomimic file carries two things this needs. ``env_args`` is the exact recipe
for the world it was recorded in — task, robot, controller, control rate. And
each demonstration stores the simulator's state at step zero. Reconstruct the
one, reset to the other, and a policy faces precisely the scene a human faced.

So a scene here is a recorded demonstration's starting state, which makes the
comparison the strongest kind available: the policy is judged on the same
layouts the demonstrator was given, not on a fresh random draw that nobody has
a reference for.

Neither library is imported
---------------------------
``robomimic`` brings robosuite, which brings MuJoCo and a GL context. None of it
is imported at module load and none is a hard dependency, so this plugin
installs on a laptop and its task list can be planned against without a
simulator anywhere. Install ``gantry-evaluator-robosuite[sim]`` to run one.

Success is the simulator's own answer
-------------------------------------
robomimic's environment wrapper exposes ``is_success()``, which asks the task
whether it is solved. That is used rather than the reward or the ``done`` flag,
both of which also fire for reasons that are not success — a shaped reward is
positive for being close, and ``done`` fires at the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import (
    Evaluator,
    Protocol,
    Scene,
    TaskSpec,
    evaluator_descriptor,
)
from gantry.contracts.policy import EpisodeContext, Observation
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import Requirement, requires_channels
from gantry.spine import (
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeRecord,
    Measurement,
    Provenance,
    RunRecord,
    episode_from_arrays,
    episode_from_labels,
    proportion,
)

VERSION = "0.1.0.dev0"

#: Observation keys worth recording by default. robosuite emits a great many;
#: these are the ones a screening or a funnel can actually use.
PROPRIO = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)

#: OSC_POSE, which is what the robomimic datasets were collected with: three
#: position deltas, three orientation deltas, one gripper command.
OSC_POSE = ChannelSpec(
    "actions", "vector", (7,), "float32", semantics="actuation",
    dim_labels=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
)


def success_from_is_success(env: Any, observation: Mapping[str, Any], done: bool) -> bool:
    """Ask the task whether it is solved.

    robomimic returns a mapping — ``{"task": True}`` — because some
    environments report sub-goals too. Only the overall answer is read here;
    a sub-goal is a milestone and this evaluator does not claim to emit those.
    """
    verdict = env.is_success()
    if isinstance(verdict, Mapping):
        return bool(verdict.get("task", False))
    return bool(verdict)


def build_env(env_meta: Mapping[str, Any], *, use_image_obs: bool = False, **kwargs: Any) -> Any:
    """Rebuild the world a robomimic file was recorded in.

    Goes through robomimic's own factory rather than calling robosuite
    directly, because ``env_args`` is robomimic's format and its factory is what
    knows how to read it — including the ``type`` field that distinguishes a
    robosuite environment from a gym or MOMART one.
    """
    try:
        import robomimic.utils.env_utils as EnvUtils
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running robosuite needs the simulator: pip install "
            "'gantry-evaluator-robosuite[sim]' (it pulls in robomimic and robosuite, "
            "which want MuJoCo and a GL context)"
        ) from error
    return EnvUtils.create_env_from_metadata(
        env_meta=dict(env_meta),
        render=False,
        render_offscreen=use_image_obs,
        use_image_obs=use_image_obs,
        **kwargs,
    )


class _Native:
    """robosuite on its own, presenting the small surface this evaluator uses.

    robomimic's factory is the documented path and the default, and it is not
    always available: robomimic 0.3 imports ``mujoco_py``, which robosuite 1.4
    replaced with the current bindings, so the two do not install together.
    That is a version fight nobody should have to win to run a simulator they
    already have.

    So this speaks to robosuite directly. It is not a reimplementation of
    robomimic — it is four methods, each one line, wrapping what robosuite
    already exposes: restore a flattened MuJoCo state, step, ask the task
    whether it is solved.
    """

    def __init__(self, env: Any):
        self._env = env

    def reset(self) -> Any:
        return self._env.reset()

    def reset_to(self, state: Mapping[str, Any]) -> Any:
        self._env.sim.set_state_from_flattened(np.asarray(state["states"]))
        self._env.sim.forward()
        return self._env._get_observations()

    def step(self, action: Any) -> Any:
        return self._env.step(action)

    def is_success(self) -> bool:
        return bool(self._env._check_success())

    def close(self) -> None:
        if callable(getattr(self._env, "close", None)):
            self._env.close()


def build_native_env(env_meta: Mapping[str, Any], *, use_image_obs: bool = False, **kwargs: Any) -> Any:
    """Rebuild the world with robosuite alone, no robomimic.

    Pass as ``factory=build_native_env`` where robomimic is absent or pinned
    against a different MuJoCo generation. The recipe is the same ``env_args``
    the dataset carries; only who reads it changes.
    """
    try:
        import robosuite
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running robosuite needs the simulator: pip install "
            "'gantry-evaluator-robosuite[sim]'"
        ) from error
    settings = dict(env_meta.get("env_kwargs") or {})
    settings.update(
        has_renderer=False,
        has_offscreen_renderer=use_image_obs,
        use_camera_obs=use_image_obs,
        # The evaluator applies the horizon and reads success itself, so the
        # environment must not end an episode on its own timer.
        ignore_done=True,
    )
    settings.update(kwargs)
    return _Native(robosuite.make(env_name=str(env_meta["env_name"]), **settings))


@dataclass
class _Trial:
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    success: bool = False
    steps: int = 0


class RobosuiteEvaluator(Evaluator):
    """Rolls a policy out in the world a robomimic dataset was recorded in."""

    def __init__(
        self,
        env_meta: Mapping[str, Any],
        initial_states: Sequence[np.ndarray],
        *,
        name: str = "robosuite",
        observations: Sequence[str] = PROPRIO,
        action: ChannelSpec = OSC_POSE,
        success: Callable[..., bool] = success_from_is_success,
        factory: Callable[..., Any] = build_env,
        use_image_obs: bool = False,
        instruction: str | None = None,
    ):
        """``env_meta`` and ``initial_states`` are what a robomimic connector
        exposes as ``.env`` and ``.initial_states()``. Passed in rather than
        read here, so this plugin needs no reader and no HDF5 library."""
        if not env_meta.get("env_name"):
            raise ConfigError("env_meta names no env_name, so there is no world to build")
        if len(initial_states) == 0:
            raise ConfigError(
                "no initial states; a scene here is a demonstration's starting state"
            )
        self.env_meta = dict(env_meta)
        self.initial_states = tuple(np.asarray(state) for state in initial_states)
        self._name = name
        self._observations = tuple(observations)
        self._action = action
        self._success = success
        self._factory = factory
        self._use_image_obs = use_image_obs
        self._instruction = instruction
        self._env: Any = None

    # -- contract ----------------------------------------------------------

    @property
    def task_name(self) -> str:
        return str(self.env_meta["env_name"])

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            # robosuite answers solved-or-not. Sub-goals exist for some tasks
            # but are not read here, so claiming milestones would overstate it.
            stage_events=False,
            outcomes=True,
            # By restored simulator state, which is stronger than a seed: the
            # same scene index is the same scene, exactly.
            seedable=True,
            closed_loop=True,
            task=self.task_name,
            robots=self.env_meta.get("env_kwargs", {}).get("robots"),
            scenes=len(self.initial_states),
        )

    def requires(self) -> Requirement:
        return requires_channels(
            self._name, "evaluation", self._action,
            description=f"a policy commanding {self.task_name} through its controller",
        )

    def task_for(self, name: str = "", scenes: int | None = None, horizon: int = 400) -> TaskSpec:
        """One scene per recorded demonstration's starting state."""
        count = len(self.initial_states) if scenes is None else min(scenes, len(self.initial_states))
        return TaskSpec(
            name=name or self.task_name,
            scenes=tuple(
                Scene(
                    id=f"demo_{index}",
                    instruction=self._instruction,
                    metadata={"initial_state": index},
                )
                for index in range(count)
            ),
            horizon=horizon,
        )

    # -- the world ---------------------------------------------------------

    @property
    def env(self) -> Any:
        """Built once. Constructing a robosuite env starts MuJoCo, which is
        seconds; per-trial would dominate a run of any size."""
        if self._env is None:
            self._env = self._factory(self.env_meta, use_image_obs=self._use_image_obs)
        return self._env

    def close(self) -> None:
        if self._env is not None and callable(getattr(self._env, "close", None)):
            self._env.close()
        self._env = None

    # -- running -----------------------------------------------------------

    def run(self, policy: Any, task: TaskSpec, protocol: Protocol) -> RunRecord:
        trials = [
            (scene, epoch) for scene in task.scenes for epoch in range(protocol.epochs)
        ]
        if protocol.max_trials is not None:
            trials = trials[: protocol.max_trials]

        episodes: list[EpisodeRecord] = []
        failures = 0
        for scene, epoch in trials:
            trial, error = self._one(policy, scene, task, protocol)
            failures += error is not None
            episodes.append(self._record(trial, scene, epoch, task, protocol, error))

        return RunRecord(
            provenance=self._provenance(policy, task, protocol, failures),
            episodes=tuple(episodes),
            metrics=self._metrics(episodes),
        )

    def _one(self, policy: Any, scene: Scene, task: TaskSpec, protocol: Protocol):
        index = int(scene.metadata["initial_state"])
        env = self.env
        env.reset()
        observation = env.reset_to({"states": self.initial_states[index]})
        policy.reset(EpisodeContext(scene.id, scene.instruction, seed=index))

        trial = _Trial()
        while trial.steps < task.horizon:
            try:
                chunk = np.asarray(policy.act(Observation(trial.steps, dict(observation))))
            except Exception as error:  # noqa: BLE001 - one trial, not the run
                return trial, f"{type(error).__name__}: {error}"
            if chunk.ndim != 2 or len(chunk) < 1:
                raise ComponentError(
                    f"{getattr(policy, 'name', 'policy')} returned a chunk of shape "
                    f"{chunk.shape}; expected (horizon, {self._action.shape[0]})"
                )
            played = len(chunk) if protocol.execute is None else min(protocol.execute, len(chunk))
            for action in chunk[: max(1, played)]:
                if trial.steps >= task.horizon:
                    break
                self._keep(trial, observation, action)
                observation, reward, done, _info = env.step(np.asarray(action, dtype=float))
                trial.rewards.append(float(reward))
                trial.steps += 1
                if self._success(env, observation, bool(done)):
                    trial.success = True
                    return trial, None
                if done:
                    return trial, None
        return trial, None

    def _keep(self, trial: _Trial, observation: Mapping[str, Any], action) -> None:
        for key in self._observations:
            if key in observation:
                trial.frames.setdefault(key, []).append(np.asarray(observation[key]))
        trial.actions.append(np.asarray(action, dtype="float32"))

    def _schema(self, trial: _Trial) -> tuple[ChannelSpec, ...]:
        specs = []
        for key, values in trial.frames.items():
            sample = np.asarray(values[0])
            shape = tuple(int(d) for d in sample.shape)
            kind = "image" if key.endswith("image") else "vector"
            specs.append(ChannelSpec(key, kind, shape, str(sample.dtype)))
        specs.append(self._action)
        specs.append(ChannelSpec("reward", "scalar", (), "float32", semantics="reward"))
        return tuple(specs)

    def _record(self, trial, scene, epoch, task, protocol, error) -> EpisodeRecord:
        eid = f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id
        annotations = {
            "epoch": epoch,
            "steps": trial.steps,
            "initial_state": scene.metadata["initial_state"],
            **({"error": error} if error else {}),
        }
        if not trial.actions:
            return episode_from_labels(
                id=eid, source=task.name,
                labels=EpisodeLabels(success=None, annotations=annotations),
                task=task.name, embodiment=self._embodiment(),
            )
        arrays = {key: np.asarray(values) for key, values in trial.frames.items()}
        arrays[self._action.name] = np.asarray(trial.actions, dtype="float32")
        arrays["reward"] = np.asarray(trial.rewards[: len(trial.actions)], dtype="float32")
        return episode_from_arrays(
            arrays, self._schema(trial), id=eid, source=task.name,
            labels=EpisodeLabels(
                success=None if error else trial.success,
                annotations={**annotations, "return": float(sum(trial.rewards))},
            ),
            task=task.name, embodiment=self._embodiment(),
        )

    def _embodiment(self) -> str | None:
        robots = self.env_meta.get("env_kwargs", {}).get("robots")
        if isinstance(robots, str):
            return robots
        if isinstance(robots, Sequence) and robots:
            return str(robots[0])
        return None

    def _provenance(self, policy, task, protocol, failures) -> Provenance:
        return Provenance(
            components=(
                policy.descriptor().component_ref(),
                self.descriptor().component_ref(),
            ),
            protocol={
                "task": task.name,
                "env": self.task_name,
                "horizon": task.horizon,
                "stages": [],
                **protocol.as_dict(),
            },
            notes=(f"{failures} trial(s) failed to complete",) if failures else (),
        )

    def _metrics(self, episodes) -> dict[str, Measurement]:
        scored = [e for e in episodes if e.labels.success is not None]
        if not scored:
            return {}
        wins = sum(1 for e in scored if e.labels.success)
        return {"success_rate": proportion(wins, len(scored))}
