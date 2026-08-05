"""ManiSkill as an evaluator, and the reason it is worth having: many robots.

Almost every suite in this field has one body welded in. LIBERO is a Panda, the
robomimic tasks are a Panda, Meta-World is a Sawyer -- which means that "does this
policy transfer to a different arm" is not a question those suites can be asked,
and a benchmark built only on them measures one embodiment while sounding general.

ManiSkill is the exception that matters. The same task is constructed with a
different ``robot_uids`` and the world rebuilds around it: Panda, xArm, WidowX,
Fetch, SO-100, a humanoid, a dexterous hand. That is what makes
``hosts_embodiment=True`` truthful here, and it is the only backend in this
project where varying the body is a one-argument change rather than a fork.

What it does not give you, and this plugin will not pretend otherwise
--------------------------------------------------------------------
Throughput. ManiSkill's headline feature is thousands of environments stepping
together on a GPU, and none of that helps here yet, because Gantry's trial loop
asks one policy for one action at a time. Making the parallelism real needs a
policy contract that accepts a batch of observations and returns a batch of
chunks -- a genuine piece of work on the policy plane, not a flag on this
evaluator. So ``num_envs`` is fixed at one, said out loud in the descriptor, and
the speedup is not claimed anywhere.

Two things that are easy to get quietly wrong
---------------------------------------------
**The action space follows the control mode.** ``pd_ee_delta_pose`` is seven
numbers; ``pd_joint_delta_pos`` on a Panda is eight; an arm with a different DoF
count is different again. Nothing here guesses: the control mode is an argument,
the action spec is derived from the live environment's own action space, and a
mismatch is a refusal at construction rather than a run of failures.

**Everything comes back as GPU tensors.** ManiSkill 3 returns torch tensors with a
leading batch dimension of one, including ``info["success"]``. ``bool()`` of a
one-element tensor happens to work, which is exactly why it is dangerous: the
same code silently reads only the first element once anybody raises ``num_envs``.
Conversion is explicit and in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: Bodies ManiSkill ships that are worth pointing a manipulation policy at.
#: Not exhaustive and not a whitelist -- an unknown uid is passed straight
#: through, because a suite adding a robot should not need this file edited.
ROBOTS = (
    "panda",
    "panda_wristcam",
    "xarm6_robotiq",
    "widowx250s",
    "fetch",
    "so100",
    "koch-v1.1",
    "unitree_g1_simplified_upper_body",
    "allegro_hand_right",
)

#: Control modes, and the shape each implies. The shape is *checked* against the
#: environment rather than trusted, because these change between releases.
CONTROL_MODES = (
    "pd_ee_delta_pose",
    "pd_ee_delta_pos",
    "pd_joint_delta_pos",
    "pd_joint_pos",
    "pd_ee_pose",
)

#: A sample of the task ids. ManiSkill has many more; ``task_for`` accepts any id
#: its registry knows.
TASKS = (
    "PickCube-v1",
    "StackCube-v1",
    "PushCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PickSingleYCB-v1",
    "PullCubeTool-v1",
    "LiftPegUpright-v1",
    "PokeCube-v1",
    "AssemblingKits-v1",
)

#: ManiSkill's default control frequency.
CONTROL_HZ = 20.0


def numeric(value: Any) -> np.ndarray:
    """A torch tensor, numpy array or scalar as a plain host-side numpy array.

    In one place on purpose. ManiSkill returns CUDA tensors with a leading batch
    axis of one, and ``np.asarray`` on such a tensor raises rather than copying --     so every caller that skipped this would need its own ``.cpu().numpy()``, and
    the ones that forgot would work anyway on the CPU backend and break on GPU.
    """
    for attribute in ("detach", "cpu"):
        if hasattr(value, attribute):
            value = getattr(value, attribute)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def unbatch(array: np.ndarray) -> np.ndarray:
    """Drop a leading axis of length one.

    Not a squeeze: ``squeeze()`` would also collapse a genuine dimension of one
    further in -- a single-channel depth image, say -- and silently change the
    shape a policy is handed.
    """
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """ManiSkill's nested observation dict as flat ``a.b.c`` channel names.

    Flattened rather than kept nested because a channel name is what the adapter
    plane matches on, and a path is the only name for a leaf that stays unique
    across two cameras with the same key.
    """
    out: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, prefix=f"{name}."))
            continue
        array = numeric(value)
        if array.dtype.kind in "OUSV":
            continue
        out[name] = unbatch(array)
    return out


def make_env(
    task: str,
    *,
    robot: str = "panda",
    control_mode: str = "pd_ee_delta_pose",
    obs_mode: str = "rgb",
    render_mode: str | None = None,
    **kwargs: Any,
) -> Any:
    """Construct a ManiSkill environment. Imported here, not at module load."""
    try:
        import gymnasium
        import mani_skill.envs  # noqa: F401 - registers the task ids
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running ManiSkill needs the simulator: pip install "
            "'gantry-evaluator-maniskill[sim]' (it wants a CUDA-capable torch "
            "and SAPIEN)"
        ) from error
    return gymnasium.make(
        task,
        num_envs=1,
        robot_uids=robot,
        control_mode=control_mode,
        obs_mode=obs_mode,
        render_mode=render_mode,
        **kwargs,
    )


class Maniskill:
    """One ManiSkill environment, as a world.

    ManiSkill reports success per step in ``info``, so there is no end-of-episode
    verdict: the trial ends the moment the task is solved, and a trial that runs
    out of horizon has established nothing rather than failed.
    """

    def __init__(self, env: Any, *, success_key: str = "success"):
        self.env = env
        self.success_key = success_key

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        observation, _info = self.env.reset(seed=scene.seed)
        return flatten(observation)

    def advance(self, action: np.ndarray) -> Step:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = info or {}
        solved = self.success_key in info
        return Step(
            observation=flatten(observation),
            reward=float(numeric(reward).reshape(-1)[0]),
            # `truncated` is ManiSkill running out of its own horizon, which is
            # not an outcome. Only `terminated` ends the attempt with a result.
            done=bool(numeric(terminated).reshape(-1)[0]),
            success=bool(numeric(info[self.success_key]).reshape(-1)[0]) if solved else None,
        )

    def close(self) -> None:
        if callable(getattr(self.env, "close", None)):
            self.env.close()


class ManiskillEvaluator(ClosedLoop):
    """Runs a policy over one ManiSkill task, on a chosen body."""

    rate_hz = CONTROL_HZ

    def __init__(
        self,
        task: str = "PickCube-v1",
        *,
        robot: str = "panda",
        control_mode: str = "pd_ee_delta_pose",
        obs_mode: str = "rgb",
        name: str = "maniskill",
        observes: Sequence[str] | None = None,
        factory: Callable[..., Any] = make_env,
        success_key: str = "success",
        horizon: int = 100,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        if control_mode not in CONTROL_MODES:
            raise ConfigError(
                f"unknown control mode {control_mode!r}; ManiSkill's manipulation "
                f"modes are {list(CONTROL_MODES)}. The action width follows from "
                "this, so it is not a detail that can be left to a default"
            )
        self._task = task
        self._robot = robot
        self._control_mode = control_mode
        self._obs_mode = obs_mode
        self._name = name
        self.observes = tuple(observes) if observes is not None else None
        self._factory = factory
        self._success_key = success_key
        self._horizon = horizon
        self._env_kwargs = dict(env_kwargs or {})
        self._world: Maniskill | None = None
        self._action: ChannelSpec | None = None

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            # ManiSkill reports solved-or-not and nothing between, so a funnel
            # over it would have a single rung.
            stage_events=False,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            # The claim this plugin exists for: the same task, a different arm,
            # one argument.
            hosts_embodiment=True,
            task=self._task,
            robot=self._robot,
            control_mode=self._control_mode,
            obs_mode=self._obs_mode,
            # Said out loud so nobody reads a GPU-parallel simulator into a
            # throughput claim this harness cannot yet make good on.
            num_envs=1,
            batched="no: the policy contract is one observation at a time",
            control_hz=CONTROL_HZ,
        )

    @property
    def world(self) -> Maniskill:
        """Built once. Constructing a ManiSkill scene loads assets and a renderer."""
        if self._world is None:
            env = self._factory(
                self._task,
                robot=self._robot,
                control_mode=self._control_mode,
                obs_mode=self._obs_mode,
                **self._env_kwargs,
            )
            self._world = Maniskill(env, success_key=self._success_key)
        return self._world

    def action(self) -> ChannelSpec:
        """Read off the live environment rather than assumed from the mode name.

        The mapping from control mode to action width depends on the robot's DoF
        and has changed between ManiSkill releases. Asking the environment is the
        only version-proof answer, and it costs one construction that was going
        to happen anyway.
        """
        if self._action is None:
            space = getattr(self.world.env, "single_action_space", None) or getattr(
                self.world.env, "action_space", None
            )
            shape = getattr(space, "shape", None)
            if not shape:
                raise ConfigError(
                    f"{self._name}: the environment for {self._task!r} exposes no "
                    "action space shape, so what a policy must emit cannot be stated"
                )
            width = int(shape[-1])
            self._action = ChannelSpec(
                "action",
                "vector",
                (width,),
                "float32",
                semantics="actuation",
                rate_hz=CONTROL_HZ,
            )
        return self._action

    def for_embodiment(self, embodiment: Any) -> "ManiskillEvaluator":
        """The same task on a different body.

        ``embodiment`` may be a name or anything with one. The uid is passed
        through unchecked against :data:`ROBOTS` -- that tuple is documentation,
        not a gate, and a ManiSkill release adding an arm should not need this
        file edited to be usable.
        """
        uid = (
            embodiment
            if isinstance(embodiment, str)
            else getattr(embodiment, "name", None) or str(embodiment)
        )
        return ManiskillEvaluator(
            self._task,
            robot=str(uid),
            control_mode=self._control_mode,
            obs_mode=self._obs_mode,
            name=self._name,
            observes=self.observes,
            factory=self._factory,
            success_key=self._success_key,
            horizon=self._horizon,
            env_kwargs=self._env_kwargs,
        )

    def embodiment_for(self, scene: Scene) -> str:
        return self._robot

    def task_for(self, name: str = "", scenes: int = 25, horizon: int | None = None) -> TaskSpec:
        """Scenes are seeds, which for ManiSkill genuinely re-randomise the layout.

        Worth stating because it is not true of every suite: LIBERO's scenes are a
        fixed list of stored states, so "more scenes" there means more of a finite
        set. Here a seed draws a fresh arrangement, so the scene count is a real
        knob and the same seed is the same arrangement on any machine.
        """
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(
                Scene(id=f"{self._task}#{index:03d}", seed=index) for index in range(scenes)
            ),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    def world_for(self, scene: Scene) -> Maniskill:
        return self.world

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None
