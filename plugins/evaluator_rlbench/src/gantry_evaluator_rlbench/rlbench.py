"""RLBench, and THE COLOSSEUM on top of it: perturbation as a controlled variable.

RLBench on its own is a hundred tasks on a Franka, which is breadth this project
otherwise buys from Meta-World more cheaply. The reason to carry the heavier
dependency is what sits on top of it.

THE COLOSSEUM takes twenty RLBench tasks and adds fourteen perturbation factors —
object colour, object size, object texture, table colour, table texture,
distractor count, background texture, lighting colour, camera pose, and physical
properties like mass and friction — each independently controllable and each with
a graded magnitude. That turns "is this policy robust" from an adjective into an
experiment with one factor varied at a time.

Why that matters more than another hundred tasks
------------------------------------------------
This project has already measured the thing informally. Lift-narrow against
lift-wide separated two checkpoints that a narrow evaluation could not tell apart,
and the conclusion — that the difference was spatial generalisation rather than
skill — was only available because one variable had been moved and the rest held.
That was one factor, hand-built, on one task.

Fourteen factors across twenty tasks makes it systematic: the output is not "this
policy is more robust" but "this policy loses thirty points to lighting and two to
distractors", which is a prescription rather than a verdict. That is the shape the
feedback plane wants, and no other backend here can produce it.

One factor at a time, or the result means nothing
-------------------------------------------------
The refusal that matters in this plugin: perturbing several factors at once and
reporting a single drop attributes the loss to nothing in particular. It is the
same error as varying two planes in one comparison, and the framework already
refuses that; this makes the sim-side version of it refusable too. A caller who
genuinely wants everything moved at once says ``factors="all"`` and gets a run
whose id and record say so, which is a legitimate stress test and not a
per-factor claim.

Two things RLBench does that will bite an adapter
------------------------------------------------
**The instruction is a list.** ``task.reset()`` returns several paraphrases of the
goal, and which one a policy is given changes its behaviour. So the choice is
explicit — index zero by default, seeded per scene if asked — and whichever
sentence was used is recorded on the episode. A benchmark that resampled it
silently would have language variation as an uncontrolled variable in every
result.

**Success is a reward of one at termination.** There is no ``check_success``.
``task.step`` returns ``(obs, reward, terminate)``, and the convention is that a
reward of 1.0 means solved. That means ``terminate`` alone is not success — it
also fires on failure conditions — and reading it as success inflates every rate.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: THE COLOSSEUM's fourteen factors. Names as its own configuration spells them.
FACTORS = (
    "object_color",
    "object_size",
    "object_texture",
    "table_color",
    "table_texture",
    "distractor_object",
    "background_texture",
    "light_color",
    "camera_pose",
    "object_friction",
    "object_mass",
    "receptacle_scale",
    "shadow",
    "rlbench_default",
)

#: The twenty tasks THE COLOSSEUM perturbs. RLBench itself has about a hundred;
#: these are the ones with perturbation variants defined.
COLOSSEUM_TASKS = (
    "basketball_in_hoop",
    "close_box",
    "close_laptop_lid",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "insert_onto_square_peg",
    "meat_on_grill",
    "move_hanger",
    "wipe_desk",
    "open_drawer",
    "slide_block_to_target",
    "reach_and_drag",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "stack_cups",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula",
)

#: RLBench action modes, and the fact that matters about them: each is a
#: different action space, so a policy trained for one cannot be evaluated under
#: another. There is no default here that is safe to leave unstated.
ACTION_MODES = {
    # 3 position deltas + 4 quaternion + 1 gripper
    "delta_ee_pose": 8,
    # absolute pose, planned
    "abs_ee_pose_plan": 8,
    # 7 joint velocities + gripper
    "joint_velocity": 8,
    "joint_position": 8,
}

CAMERAS = ("front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb")
CONTROL_HZ = 20.0

#: An unperturbed run. Named rather than expressed as an empty tuple so that a
#: record can distinguish "the baseline condition" from "somebody forgot".
BASELINE = "rlbench_default"


def make_env(
    *,
    action_mode: str = "delta_ee_pose",
    cameras: Sequence[str] = CAMERAS,
    image_size: int = 128,
    headless: bool = True,
    factors: Sequence[str] = (BASELINE,),
    variation: int = 0,
    **kwargs: Any,
) -> Any:
    """Construct an RLBench environment. Imported here, not at module load."""
    try:
        from rlbench.environment import Environment
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running RLBench needs the simulator: pip install "
            "'gantry-evaluator-rlbench[sim]', and CoppeliaSim installed "
            "separately with COPPELIASIM_ROOT set. THE COLOSSEUM's perturbation "
            "configs come from its own repository"
        ) from error
    return Environment(action_mode=action_mode, headless=headless, **kwargs)


class Bench:
    """One RLBench task, as a world.

    Holds the sampled instruction so it can be recorded — the language a policy
    was actually given is otherwise unrecoverable from the run.
    """

    def __init__(
        self,
        task: Any,
        *,
        instruction_index: int = 0,
        solved_reward: float = 1.0,
    ):
        self.task = task
        self.instruction_index = int(instruction_index)
        self.solved_reward = float(solved_reward)
        self.instruction: str | None = None
        self.descriptions: tuple[str, ...] = ()

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        metadata = scene.metadata or {}
        variation = metadata.get("variation")
        if variation is not None and callable(getattr(self.task, "set_variation", None)):
            self.task.set_variation(int(variation))
        descriptions, observation = self.task.reset()
        self.descriptions = tuple(str(text) for text in descriptions or ())
        # Explicit rather than random. A benchmark that resampled the paraphrase
        # silently would have language variation as an uncontrolled variable in
        # every result it ever produced.
        index = int(metadata.get("instruction_index", self.instruction_index))
        self.instruction = (
            self.descriptions[index % len(self.descriptions)] if self.descriptions else None
        )
        return _channels(observation)

    def advance(self, action: np.ndarray) -> Step:
        observation, reward, terminate = self.task.step(np.asarray(action, dtype=float))
        value = float(reward)
        # A reward of one means solved. `terminate` alone does not — it also
        # fires on failure conditions, and reading it as success inflates every
        # rate this suite produces.
        solved = value >= self.solved_reward
        return Step(
            observation=_channels(observation),
            reward=value,
            done=bool(terminate) or solved,
            success=True if solved else None,
        )

    def verdict(self, trial: Any) -> bool | None:
        """Terminated or ran out without ever paying the solved reward."""
        return False

    def close(self) -> None:
        if callable(getattr(self.task, "close", None)):
            self.task.close()


def _channels(observation: Any) -> dict[str, np.ndarray]:
    """RLBench's Observation object as named arrays.

    An attribute bag rather than a dict, so this reads the ones that exist and
    ignores the rest — including ``misc``, which is a dict of metadata and not a
    channel.
    """
    if isinstance(observation, Mapping):
        source: Mapping[str, Any] = observation
    else:
        source = {
            name: getattr(observation, name)
            for name in dir(observation)
            if not name.startswith("_") and not callable(getattr(observation, name, None))
        }
    out: dict[str, np.ndarray] = {}
    for name, value in source.items():
        if value is None or name == "misc" or isinstance(value, Mapping):
            continue
        array = np.asarray(value)
        if array.dtype.kind in "OUSV" or array.ndim == 0:
            continue
        out[name] = array
    return out


class RlbenchEvaluator(ClosedLoop):
    """Runs a policy over one RLBench task, optionally under COLOSSEUM perturbation."""

    rate_hz = CONTROL_HZ
    embodiment = "franka"

    def __init__(
        self,
        task: str = "open_drawer",
        *,
        action_mode: str = "delta_ee_pose",
        name: str = "rlbench",
        factors: Sequence[str] | str = (BASELINE,),
        cameras: Sequence[str] = CAMERAS,
        image_size: int = 128,
        instruction_index: int = 0,
        factory: Callable[..., Any] = make_env,
        task_loader: Callable[[Any, str], Any] | None = None,
        solved_reward: float = 1.0,
        horizon: int = 200,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        """``factors`` is the perturbation condition, and it is checked.

        More than one named factor is refused. A drop measured with three things
        moved at once attributes the loss to nothing in particular — the same
        error as varying two planes in one comparison, which this framework
        already refuses on the other side. ``"all"`` is the deliberate stress
        test and says so in the record.
        """
        if action_mode not in ACTION_MODES:
            raise ConfigError(
                f"unknown RLBench action mode {action_mode!r}; known: "
                f"{sorted(ACTION_MODES)}. Each is a different action space, so a "
                "policy trained for one cannot be evaluated under another and "
                "there is no safe default to leave unstated"
            )
        self._factors = _factors(factors, name=name)
        self._task = task
        self._action_mode = action_mode
        self._name = name
        self._cameras = tuple(cameras)
        self._image_size = int(image_size)
        self._instruction_index = int(instruction_index)
        self._factory = factory
        self._task_loader = task_loader
        self._solved_reward = float(solved_reward)
        self._horizon = int(horizon)
        self._env_kwargs = dict(env_kwargs or {})
        self._env: Any = None
        self._world: Bench | None = None

    # -- contract ----------------------------------------------------------

    @property
    def condition(self) -> str:
        """The perturbation condition as one string, for ids and records."""
        return "all" if self._factors == ("all",) else self._factors[0]

    @property
    def perturbed(self) -> bool:
        return self.condition not in (BASELINE,)

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            stage_events=False,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            hosts_embodiment=False,
            task=self._task,
            action_mode=self._action_mode,
            # In the descriptor because a perturbed run and a baseline run are
            # the two halves of one experiment, and a record that cannot tell
            # them apart is a record of nothing.
            condition=self.condition,
            perturbed=self.perturbed,
            cameras=list(self._cameras),
            success_rule=f"reward >= {self._solved_reward:g}, not terminate",
            control_hz=CONTROL_HZ,
        )

    def action(self) -> ChannelSpec:
        width = ACTION_MODES[self._action_mode]
        return ChannelSpec(
            "action",
            "vector",
            (width,),
            "float32",
            semantics="actuation",
            rate_hz=CONTROL_HZ,
        )

    def declares(self) -> Mapping[str, ChannelSpec]:
        return {
            name: ChannelSpec(
                name,
                "image",
                (self._image_size, self._image_size, 3),
                "uint8",
                rate_hz=CONTROL_HZ,
                semantics="rgb",
            )
            for name in self._cameras
        }

    def task_for(self, name: str = "", scenes: int = 25, horizon: int | None = None) -> TaskSpec:
        """One scene per variation, with the perturbation condition in every id.

        The condition is in the id so a baseline run and a perturbed run of the
        same task cannot be silently pooled. That pooling is the specific mistake
        that turns a robustness experiment into an unexplained average.
        """
        return TaskSpec(
            name=name or f"{self._task}/{self.condition}",
            scenes=tuple(
                Scene(
                    id=f"{self._task}/{self.condition}#{index:03d}",
                    seed=index,
                    metadata={
                        "variation": index,
                        "condition": self.condition,
                        "instruction_index": self._instruction_index,
                    },
                )
                for index in range(max(1, int(scenes)))
            ),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    def world_for(self, scene: Scene) -> Bench:
        if self._world is None:
            self._env = self._factory(
                action_mode=self._action_mode,
                cameras=self._cameras,
                image_size=self._image_size,
                factors=self._factors,
                **self._env_kwargs,
            )
            loader = self._task_loader or _load_task
            self._world = Bench(
                loader(self._env, self._task),
                instruction_index=self._instruction_index,
                solved_reward=self._solved_reward,
            )
        return self._world

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None
        if self._env is not None and callable(getattr(self._env, "shutdown", None)):
            self._env.shutdown()
            self._env = None

    def assemble(self, trial, scene, epoch, task, protocol):
        """Record the condition and the exact sentence the policy was given."""
        episode = super().assemble(trial, scene, epoch, task, protocol)
        labels = episode.labels
        extra: dict[str, Any] = {
            "condition": self.condition,
            "perturbed": self.perturbed,
            "action_mode": self._action_mode,
        }
        if self._world is not None:
            extra["instruction_given"] = self._world.instruction
            extra["instruction_options"] = len(self._world.descriptions)
        return episode.with_labels(
            type(labels)(
                success=labels.success,
                stage_events=labels.stage_events,
                annotations={**dict(labels.annotations), **extra},
            )
        )


def _factors(raw: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    values = (raw,) if isinstance(raw, str) else tuple(str(value) for value in raw)
    if not values:
        return (BASELINE,)
    if values == ("all",):
        return values
    unknown = [value for value in values if value not in FACTORS]
    if unknown:
        raise ConfigError(
            f"{name}: unknown perturbation factor(s) {unknown}; THE COLOSSEUM "
            f"defines {list(FACTORS)}"
        )
    if len(values) > 1:
        raise ConfigError(
            f"{name}: {list(values)} perturbs {len(values)} factors at once, and a "
            "drop measured that way attributes the loss to nothing in particular. "
            "Run one factor per condition and compare each against the baseline, "
            "or pass factors='all' for a deliberate stress test that records "
            "itself as one"
        )
    return values


def _load_task(env: Any, task: str) -> Any:  # pragma: no cover - needs the simulator
    """Resolve a task name through RLBench's own task module and open it."""
    from importlib import import_module

    module = import_module("rlbench.tasks")
    class_name = "".join(part.capitalize() for part in task.split("_"))
    target = getattr(module, class_name, None)
    if target is None:
        raise ConfigError(
            f"RLBench has no task {task!r} (looked for {class_name} in rlbench.tasks)"
        )
    if not callable(getattr(env, "launch", None)) or env.get_task is None:
        raise ConfigError("the RLBench environment was not launched")
    env.launch()
    return env.get_task(target)


def sweep(
    task: str = "open_drawer", factors: Sequence[str] = FACTORS, **kwargs: Any
) -> dict[str, RlbenchEvaluator]:
    """One evaluator per factor, plus the baseline — the whole robustness experiment.

    This is what the plugin is for. Each evaluator moves exactly one thing, the
    baseline moves nothing, and the differences between them are attributable.
    The output is not "this policy is more robust" but "this policy loses thirty
    points to lighting and two to distractors", which is a prescription rather
    than a verdict.
    """
    wanted = [BASELINE, *[factor for factor in factors if factor != BASELINE]]
    return {factor: RlbenchEvaluator(task, factors=(factor,), **kwargs) for factor in wanted}
