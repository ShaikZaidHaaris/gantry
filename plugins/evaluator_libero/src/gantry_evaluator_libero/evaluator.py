"""LIBERO as an evaluator, without LIBERO being a dependency of anything else.

LIBERO is a standard closed-loop benchmark: four task suites plus a ninety-task
set, each task a language instruction over a tabletop scene, each with a fixed
list of initial states. It is the natural first sim evaluator to plug in, and it
is the one this project's own checkpoints were fine-tuned against.

Why this is not the gym bridge
------------------------------
The gym bridge already speaks ``reset``/``step``, and LIBERO's environment does
too — but LIBERO's ``reset()`` takes no seed. A trial is pinned by calling
``set_init_state`` with one of the task's stored initial states, and the list of
those states *is* the scene list. Passing a seed would do nothing, and a run that
believes it is seeded and is not produces paired comparisons over unrelated
scenes, which is worse than an unseeded run that says so.

So the scene here is a ``(task, init state)`` pair, which is both what LIBERO
actually offers and what makes a rerun land on the same conditions.

Nothing is imported until it is used
------------------------------------
``libero`` brings robosuite, MuJoCo and a rendering context. None of it is
imported at module load, and none of it is a hard dependency, so this plugin can
be installed and inspected — its descriptor read, its task list planned against —
on a laptop with no simulator. Install ``gantry-evaluator-libero[sim]`` to
actually run one.

What is declared rather than assumed
------------------------------------
Whether a trial succeeded. LIBERO's wrapper exposes ``check_success()`` and its
``step`` also returns a ``done`` flag, and those are not always the same thing —
``done`` fires on the horizon too. The default asks ``check_success()`` because
that is the question being asked, but the reader is a named argument, so a suite
that reports success differently is one argument away rather than a fork.
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

#: The suites LIBERO ships. Names as its own benchmark registry spells them.
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

#: What LIBERO's cameras are called, and the size its wrapper defaults to.
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_SIZE = 128

#: The proprioceptive keys robosuite puts in every observation.
PROPRIO = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

#: OSC_POSE: six pose deltas and a gripper command.
ACTION = ChannelSpec(
    "action",
    "vector",
    (7,),
    "float32",
    semantics="actuation",
    dim_labels=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
)

#: Its control loop runs at 20 Hz.
CONTROL_HZ = 20.0


def success_from_check(env: Any, observation: Mapping[str, Any], done: bool, info: Mapping) -> bool:
    """Ask the environment whether the task is solved.

    The default because it answers the question being asked. ``done`` is not the
    same question: it also fires when the horizon runs out, which would score
    every timeout as a win.
    """
    return bool(env.check_success())


def success_from_done(env: Any, observation: Mapping[str, Any], done: bool, info: Mapping) -> bool:
    """Treat the ``done`` flag as success. Correct only where the horizon is
    handled elsewhere; offered so the choice is visible at the call site."""
    return bool(done)


def build_env(bddl_file: str, *, camera_size: int = CAMERA_SIZE, **kwargs: Any) -> Any:
    """Construct LIBERO's off-screen environment. Imported here, not at load."""
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running LIBERO needs the simulator: pip install "
            "'gantry-evaluator-libero[sim]' (it pulls in libero and robosuite, "
            "which want MuJoCo and a GL context)"
        ) from error
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
        **kwargs,
    )


def load_suite(name: str) -> Any:
    """One of LIBERO's task suites, by the name its registry uses."""
    try:
        from libero.libero import benchmark
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "listing LIBERO tasks needs the package: pip install 'gantry-evaluator-libero[sim]'"
        ) from error
    registry = benchmark.get_benchmark_dict()
    if name not in registry:
        raise ConfigError(f"unknown LIBERO suite {name!r}; it has {sorted(registry)}")
    return registry[name]()


@dataclass
class _Trial:
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    success: bool = False
    steps: int = 0


class LiberoEvaluator(Evaluator):
    """Runs a policy over LIBERO tasks and initial states."""

    def __init__(
        self,
        suite: str = "libero_spatial",
        *,
        name: str = "libero",
        cameras: Sequence[str] = CAMERAS,
        proprio: Sequence[str] = PROPRIO,
        camera_size: int = CAMERA_SIZE,
        success: Callable[..., bool] = success_from_check,
        loader: Callable[[str], Any] = load_suite,
        factory: Callable[..., Any] = build_env,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        if suite not in SUITES:
            raise ConfigError(f"unknown LIBERO suite {suite!r}; known: {list(SUITES)}")
        self.suite_name = suite
        self._name = name
        self._cameras = tuple(cameras)
        self._proprio = tuple(proprio)
        self._camera_size = camera_size
        self._success = success
        self._loader = loader
        self._factory = factory
        self._env_kwargs = dict(env_kwargs or {})
        self._suite: Any = None
        self._envs: dict[int, Any] = {}

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            # LIBERO reports whether a task was solved and nothing in between.
            # A funnel over it would have one rung, so the honest answer is no.
            stage_events=False,
            outcomes=True,
            # Not by seed — by initial state, which is stronger: the same scene
            # index is the same scene, not merely the same random draw.
            seedable=True,
            closed_loop=True,
            suite=self.suite_name,
            cameras=list(self._cameras),
            control_hz=CONTROL_HZ,
        )

    def requires(self) -> Requirement:
        return requires_channels(
            self._name,
            "evaluation",
            ACTION,
            description="a policy emitting OSC_POSE deltas and a gripper command",
        )

    @property
    def suite(self) -> Any:
        if self._suite is None:
            self._suite = self._loader(self.suite_name)
        return self._suite

    def task_for(
        self, name: str = "", tasks: int | None = None, inits: int = 1, horizon: int = 520
    ) -> TaskSpec:
        """One scene per (task, initial state).

        ``inits`` is how many of each task's stored initial states to use. They
        are a list, not a seed, so asking for two gives the first two — the same
        two on any machine, which is what makes a rerun comparable.
        """
        suite = self.suite
        count = suite.n_tasks if tasks is None else min(int(tasks), suite.n_tasks)
        scenes: list[Scene] = []
        for index in range(count):
            task = suite.get_task(index)
            available = len(suite.get_task_init_states(index))
            for init in range(min(inits, available)):
                scenes.append(
                    Scene(
                        id=f"{task.name}#{init}",
                        instruction=task.language,
                        metadata={"task_index": index, "init_index": init},
                    )
                )
        return TaskSpec(name=name or self.suite_name, scenes=tuple(scenes), horizon=horizon)

    # -- the environment ---------------------------------------------------

    def _env_for(self, task_index: int) -> Any:
        """One environment per task, reused across its initial states.

        Building a LIBERO env parses a scene description and starts a renderer,
        which costs seconds. Rebuilding per trial would dominate a run.
        """
        if task_index not in self._envs:
            task = self.suite.get_task(task_index)
            bddl = getattr(task, "bddl_file", None)
            if bddl is None:
                raise ConfigError(f"LIBERO task {task_index} names no bddl file")
            from pathlib import Path

            if not Path(str(bddl)).exists():
                from libero.libero import get_libero_path  # pragma: no cover

                bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / bddl)
            self._envs[task_index] = self._factory(
                str(bddl), camera_size=self._camera_size, **self._env_kwargs
            )
        return self._envs[task_index]

    def close(self) -> None:
        for env in self._envs.values():
            if callable(getattr(env, "close", None)):
                env.close()
        self._envs.clear()

    # -- running -----------------------------------------------------------

    def run(self, policy: Any, task: TaskSpec, protocol: Protocol) -> RunRecord:
        trials = [(scene, epoch) for scene in task.scenes for epoch in range(protocol.epochs)]
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
        task_index = int(scene.metadata["task_index"])
        init_index = int(scene.metadata["init_index"])
        env = self._env_for(task_index)
        env.reset()
        states = self.suite.get_task_init_states(task_index)
        observation = env.set_init_state(states[init_index])
        policy.reset(EpisodeContext(scene.id, scene.instruction, seed=init_index))

        trial = _Trial()
        while trial.steps < task.horizon:
            try:
                chunk = np.asarray(policy.act(Observation(trial.steps, dict(observation))))
            except Exception as error:  # noqa: BLE001 - one trial, not the run
                return trial, f"{type(error).__name__}: {error}"
            if chunk.ndim != 2 or len(chunk) < 1:
                raise ComponentError(
                    f"{getattr(policy, 'name', 'policy')} returned a chunk of shape "
                    f"{chunk.shape}; expected (horizon, 7)"
                )
            played = len(chunk) if protocol.execute is None else min(protocol.execute, len(chunk))
            for action in chunk[: max(1, played)]:
                if trial.steps >= task.horizon:
                    break
                self._record_step(trial, observation, action)
                observation, reward, done, info = env.step(np.asarray(action, dtype=float))
                trial.rewards.append(float(reward))
                trial.steps += 1
                if self._success(env, observation, bool(done), info or {}):
                    trial.success = True
                    return trial, None
                if done:
                    return trial, None
        return trial, None

    def _record_step(self, trial: _Trial, observation: Mapping[str, Any], action) -> None:
        for key in (*self._cameras, *self._proprio):
            if key in observation:
                trial.frames.setdefault(key, []).append(np.asarray(observation[key]))
        trial.actions.append(np.asarray(action, dtype="float32"))

    def _schema(self, trial: _Trial) -> tuple[ChannelSpec, ...]:
        specs: list[ChannelSpec] = []
        for key, values in trial.frames.items():
            sample = np.asarray(values[0])
            shape = tuple(int(d) for d in sample.shape)
            kind = "image" if key in self._cameras else "vector"
            specs.append(ChannelSpec(key, kind, shape, str(sample.dtype), rate_hz=CONTROL_HZ))
        specs.append(ACTION)
        specs.append(ChannelSpec("reward", "scalar", (), "float32", semantics="reward"))
        return tuple(specs)

    def _record(self, trial, scene, epoch, task, protocol, error) -> EpisodeRecord:
        eid = f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id
        if not trial.actions:
            return episode_from_labels(
                id=eid,
                source=task.name,
                labels=EpisodeLabels(
                    success=None,
                    annotations={"epoch": epoch, **({"error": error} if error else {})},
                ),
                task=task.name,
                embodiment="panda",
            )
        arrays = {key: np.asarray(values) for key, values in trial.frames.items()}
        arrays["action"] = np.asarray(trial.actions, dtype="float32")
        arrays["reward"] = np.asarray(trial.rewards[: len(trial.actions)], dtype="float32")
        return episode_from_arrays(
            arrays,
            self._schema(trial),
            id=eid,
            source=task.name,
            labels=EpisodeLabels(
                success=None if error else trial.success,
                annotations={
                    "epoch": epoch,
                    "steps": trial.steps,
                    "instruction": scene.instruction,
                    **({"error": error} if error else {}),
                },
            ),
            task=task.name,
            embodiment="panda",
        )

    def _provenance(self, policy, task, protocol, failures) -> Provenance:
        return Provenance(
            components=(
                policy.descriptor().component_ref(),
                self.descriptor().component_ref(),
            ),
            protocol={
                "task": task.name,
                "suite": self.suite_name,
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
