"""Meta-World, for breadth: fifty tasks that share one arm and one action space.

Its value here is narrow and real. Every task in Meta-World runs on the same
Sawyer with the same four-number action space, which makes it the cleanest place
to ask a question the richer suites make expensive: *how many different things
can this policy do at all?* Fifty tasks, one body, no per-task rebuild, no asset
pack. A thirteen-task table over robosuite variants took this project days; the
equivalent breadth here is a loop.

What it is not is diverse. One arm, one table, one camera arrangement, low-fidelity
scenes. A policy that scores well across Meta-World has demonstrated broad *skill
coverage* and nothing about visual robustness or embodiment transfer — those are
what RoboCasa and ManiSkill are for. Reporting a Meta-World average as a general
capability number is the mistake this docstring exists to prevent.

The success flag flickers, and that has to be handled
----------------------------------------------------
Meta-World decides success by thresholding a distance. So a policy that reaches
the goal and drifts a centimetre reports ``success=True`` and then ``False`` on
the next step, and an adapter that reads the last step's flag scores it as a
failure. The convention in every published Meta-World result is *ever* succeeded
during the episode, and that is what this does: the first ``True`` ends the trial.

This is not a detail. Reading the final flag instead of the first one silently
lowers every rate, by an amount that depends on how twitchy the policy is — so it
penalises exactly the near-miss policies a benchmark is meant to distinguish.

Tasks are separate evaluators, deliberately
-------------------------------------------
One evaluator per task, not one evaluator with a task list. A Meta-World task is a
different *task-plane* object with a different goal and a different rubric, and
folding fifty into one evaluator would make "vary the task" invisible to the
framework — the axis would be inside a component instead of between components,
which is exactly the arrangement that stops a comparison being checkable.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: The benchmark families Meta-World ships. MT is multi-task (train on all), ML
#: is meta-learning (held-out tasks); this evaluator runs single tasks from either.
FAMILIES = ("MT1", "MT10", "MT50", "ML1", "ML10", "ML45")

#: All fifty, by the ids Meta-World's own registry uses. Listed rather than
#: discovered so that a task list can be planned against with nothing installed —
#: which is the whole point of a plugin whose simulator is optional.
TASKS = (
    "assembly-v2",
    "basketball-v2",
    "bin-picking-v2",
    "box-close-v2",
    "button-press-topdown-v2",
    "button-press-topdown-wall-v2",
    "button-press-v2",
    "button-press-wall-v2",
    "coffee-button-v2",
    "coffee-pull-v2",
    "coffee-push-v2",
    "dial-turn-v2",
    "disassemble-v2",
    "door-close-v2",
    "door-lock-v2",
    "door-open-v2",
    "door-unlock-v2",
    "drawer-close-v2",
    "drawer-open-v2",
    "faucet-close-v2",
    "faucet-open-v2",
    "hammer-v2",
    "hand-insert-v2",
    "handle-press-side-v2",
    "handle-press-v2",
    "handle-pull-side-v2",
    "handle-pull-v2",
    "lever-pull-v2",
    "peg-insert-side-v2",
    "peg-unplug-side-v2",
    "pick-out-of-hole-v2",
    "pick-place-v2",
    "pick-place-wall-v2",
    "plate-slide-back-side-v2",
    "plate-slide-back-v2",
    "plate-slide-side-v2",
    "plate-slide-v2",
    "push-back-v2",
    "push-v2",
    "push-wall-v2",
    "reach-v2",
    "reach-wall-v2",
    "shelf-place-v2",
    "soccer-v2",
    "stick-pull-v2",
    "stick-push-v2",
    "sweep-into-v2",
    "sweep-v2",
    "window-close-v2",
    "window-open-v2",
)

#: The ten of MT10, for a run that wants breadth without fifty of everything.
MT10 = (
    "reach-v2",
    "push-v2",
    "pick-place-v2",
    "door-open-v2",
    "drawer-open-v2",
    "drawer-close-v2",
    "button-press-topdown-v2",
    "peg-insert-side-v2",
    "window-open-v2",
    "window-close-v2",
)

#: End-effector delta plus a gripper. The same four for every one of the fifty,
#: which is the property that makes the breadth cheap.
ACTION = ChannelSpec(
    "action",
    "vector",
    (4,),
    "float32",
    semantics="actuation",
    dim_labels=("dx", "dy", "dz", "gripper"),
)

#: Meta-World's own episode cap.
MAX_PATH_LENGTH = 500
CONTROL_HZ = 80.0

#: The single observation vector Meta-World reports. Declared rather than
#: inferred because inference cannot say what it means, and it is not obvious:
#: current and previous end-effector state plus goal, concatenated.
OBSERVATION = ChannelSpec(
    "observation", "vector", (39,), "float32", semantics="proprioception", rate_hz=CONTROL_HZ
)


def make_env(task: str, *, seed: int | None = None, **kwargs: Any) -> Any:
    """Construct one Meta-World task environment. Imported here, not at load."""
    try:
        import metaworld
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running Meta-World needs the simulator: pip install "
            "'gantry-evaluator-metaworld[sim]' (it wants MuJoCo)"
        ) from error
    benchmark = metaworld.MT1(task, seed=seed)
    env = benchmark.train_classes[task](**kwargs)
    env.tasks = list(benchmark.train_tasks)
    return env


class Bench:
    """One Meta-World task, as a world.

    The per-scene variation is Meta-World's own ``set_task``: each stored task is a
    different goal and object placement, and they are a list rather than a seed —
    so scene ``k`` is the same arrangement on any machine, which is what a paired
    comparison needs.
    """

    def __init__(self, env: Any, *, success_key: str = "success"):
        self.env = env
        self.success_key = success_key

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        tasks = list(getattr(self.env, "tasks", ()) or ())
        if tasks and callable(getattr(self.env, "set_task", None)):
            index = int(scene.metadata.get("task_index", scene.seed or 0)) % len(tasks)
            self.env.set_task(tasks[index])
        result = self.env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        return {"observation": np.asarray(observation, dtype="float32")}

    def advance(self, action: np.ndarray) -> Step:
        result = self.env.step(np.asarray(action, dtype=float))
        # Both dialects: five-value gymnasium and the older four-value form.
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated)
        else:
            observation, reward, done, info = result
            done = bool(done)
        info = info or {}
        flag = info.get(self.success_key)
        return Step(
            observation={"observation": np.asarray(observation, dtype="float32")},
            reward=float(reward),
            done=done,
            # True the first time it happens, and never False. Meta-World's flag
            # is a distance threshold that flickers; a `False` here would let a
            # policy that reached the goal and drifted be scored as a failure,
            # which penalises exactly the near-miss policies worth separating.
            success=True if flag is not None and bool(np.asarray(flag).reshape(-1)[0]) else None,
            info=dict(info),
        )

    def verdict(self, trial: Any) -> bool | None:
        """Never succeeded within the horizon, so it failed.

        Safe to state as failure here — unlike a real bench — because Meta-World's
        predicate was evaluated on every one of those steps and answered no each
        time. The trial has been checked, not merely cut short.
        """
        return False

    def close(self) -> None:
        if callable(getattr(self.env, "close", None)):
            self.env.close()


class MetaworldEvaluator(ClosedLoop):
    """Runs a policy over one Meta-World task, across its stored goal placements."""

    rate_hz = CONTROL_HZ
    embodiment = "sawyer"
    observes = ("observation",)

    def __init__(
        self,
        task: str = "reach-v2",
        *,
        name: str = "metaworld",
        factory: Callable[..., Any] = make_env,
        success_key: str = "success",
        horizon: int = MAX_PATH_LENGTH,
        strict: bool = True,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        """``strict`` checks the task id against :data:`TASKS`.

        On by default because a typo in a task name is otherwise a run of an
        environment nobody meant, and Meta-World's ids are easy to get wrong —
        the ``-v2`` suffix, the side/wall variants. Turn it off to run a task from
        a release newer than this list.
        """
        if strict and task not in TASKS:
            close = [known for known in TASKS if known.split("-")[0] == task.split("-")[0]]
            raise ConfigError(
                f"unknown Meta-World task {task!r}"
                + (f"; did you mean one of {close}?" if close else f"; it ships {len(TASKS)} tasks")
                + ". Pass strict=False to run an id from a newer release"
            )
        self._task = task
        self._name = name
        self._factory = factory
        self._success_key = success_key
        self._horizon = int(horizon)
        self._env_kwargs = dict(env_kwargs or {})
        self._world: Bench | None = None

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            stage_events=False,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            # One arm, welded in. Which is precisely why this is a breadth
            # backend and not a transfer one.
            hosts_embodiment=False,
            task=self._task,
            success_rule="ever succeeded during the episode, per Meta-World convention",
            control_hz=CONTROL_HZ,
        )

    def action(self) -> ChannelSpec:
        return ACTION

    def declares(self) -> Mapping[str, ChannelSpec]:
        return {OBSERVATION.name: OBSERVATION, ACTION.name: ACTION}

    @property
    def world(self) -> Bench:
        if self._world is None:
            self._world = Bench(
                self._factory(self._task, **self._env_kwargs), success_key=self._success_key
            )
        return self._world

    def task_for(self, name: str = "", scenes: int = 50, horizon: int | None = None) -> TaskSpec:
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(
                Scene(
                    id=f"{self._task}#{index:03d}",
                    seed=index,
                    metadata={"task_index": index},
                )
                for index in range(max(1, int(scenes)))
            ),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    def world_for(self, scene: Scene) -> Bench:
        return self.world

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None


def suite(tasks: Sequence[str] = MT10, **kwargs: Any) -> dict[str, MetaworldEvaluator]:
    """One evaluator per task, ready to run as a matrix.

    A plain dict rather than a combined evaluator, because a Meta-World task is a
    genuinely separate task-plane object. Keeping them separate is what lets the
    framework see "vary the task" as an axis between components instead of a loop
    hidden inside one — and a hidden loop is a comparison nobody can check.
    """
    return {task: MetaworldEvaluator(task, **kwargs) for task in tasks}
