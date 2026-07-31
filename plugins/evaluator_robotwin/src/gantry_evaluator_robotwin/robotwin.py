"""RoboTwin as an evaluator — the dual-arm world this project has been missing.

Twelve evaluators were already here and not one of them could receive the policy
this project actually trains. Everything installed is single-arm and tabletop;
the ego pipeline produces bimanual end-effector commands for a kitchen. So the
report has been correctly refusing to say whether ego data helped, on the
grounds that no evaluation could execute the thing being asked about.

RoboTwin closes that. Fifty dual-arm tasks over a 731-object dataset, MIT
licensed, built on SAPIEN. It is the first backend here whose *shape* matches
what the ego chain emits.

The part that matters more than the task count
----------------------------------------------
RoboTwin accepts actions in **end-effector space**, not only joint space:

    qpos      left arm joints + left gripper + right arm joints + right gripper
    ee        left (xyz + quaternion) + gripper, then right — absolute pose
    delta_ee  the same, as a change from the current pose

That third column is why this plugin is worth more than another benchmark.
``retargeter_hands`` refuses to produce joint positions and says why — inverse
kinematics needs link lengths, joint limits and a choice of elbow configuration,
none of which a retargeter has. That refusal has blocked the ego path from every
ALOHA-family config since it was written. RoboTwin's ``ee`` mode takes exactly
what the retargeter *does* produce, so the chain runs end to end with no IK step
and no pretending.

Three action spaces, one arm, and the widths differ
---------------------------------------------------
Fourteen numbers in ``qpos``, sixteen in ``ee``. Same robot, same task, same
method name, and a policy trained for one cannot be evaluated under another —
the numbers would be accepted and mean something else. So ``action_type`` is a
constructor argument with no safe default, the width follows from it, and a
mismatch is refused at plan time rather than discovered as poor performance.

The instruction comes from the environment
------------------------------------------
``get_instruction()`` returns the language goal, and RoboTwin varies it. A
policy handed a different sentence than the one the environment is scoring
against is being tested on a mismatch that shows up only as a low number, so
whatever sentence was actually used is recorded on every episode.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step
from gantry.spine import ChannelSpec, Descriptor

#: The rotation-encoding keys, from the adapter that reads them. Named here so a
#: pose command declares its encoding the way every other pose channel does.
KEY_ROTATION = "rotation_repr"
KEY_OFFSET = "rotation_offset"

VERSION = "0.1.0.dev0"

#: The three action spaces RoboTwin accepts, and the width each implies *per
#: arm*. Doubling happens in :func:`width_of` — a dual-arm command is the two
#: arms concatenated, left first.
#:
#: These are not interchangeable and the difference is invisible in the array:
#: a fourteen-wide qpos vector and the first fourteen numbers of a sixteen-wide
#: ee vector are both plausible float32, and executing one as the other drives
#: the arms somewhere arbitrary.
ACTION_TYPES: dict[str, int] = {
    # six joints and a gripper
    "qpos": 7,
    # position, wxyz quaternion, gripper
    "ee": 8,
    # the same, as a delta from where the arm is now
    "delta_ee": 8,
}

#: Arms, in the order they are concatenated. The only written record of which
#: half of the vector is which arm — the same discipline policy_pi0 and
#: retargeter_hands already keep, for the same reason.
ARMS = ("left", "right")

#: A sample of RoboTwin 2.0's task ids. Not a whitelist: any id the installed
#: version knows is accepted, and this exists so a run can be planned against a
#: task list on a laptop with no simulator.
TASKS = (
    "block_hammer_beat",
    "block_handover",
    "blocks_stack_easy",
    "blocks_stack_hard",
    "bottle_adjust",
    "container_place",
    "diverse_bottles_pick",
    "dual_bottles_pick_easy",
    "dual_bottles_pick_hard",
    "dual_shoes_place",
    "empty_cup_place",
    "mug_hanging_easy",
    "mug_hanging_hard",
    "pick_apple_messy",
    "put_apple_cabinet",
    "shoe_place",
    "tool_adjust",
    "beat_block_hammer",
    "handover_block",
    "place_shoe",
)

#: RoboTwin's own control frequency.
CONTROL_HZ = 25.0


def width_of(action_type: str, arms: int = 2) -> int:
    """How wide a command is. Follows from the action type; never assumed."""
    try:
        return ACTION_TYPES[action_type] * int(arms)
    except KeyError:
        raise ConfigError(
            f"unknown action type {action_type!r}; RoboTwin accepts "
            f"{sorted(ACTION_TYPES)}. These are different action spaces on the same "
            "arm — a policy trained for one cannot be evaluated under another, and "
            "the numbers would be accepted and mean something else"
        ) from None


def labels_for(action_type: str) -> tuple[str, ...]:
    """Per-dimension names, left arm then right.

    Written down because a sixteen-wide float32 is the same object whichever arm
    comes first, and a swap produces valid commands sent to the wrong arm.
    """
    if action_type == "qpos":
        parts = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
    else:
        parts = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
    return tuple(f"{arm}_{part}" for arm in ARMS for part in parts)


def make_env(
    task: str,
    *,
    embodiment: str = "aloha-agilex",
    seed: int = 0,
    head_camera: str = "D435",
    **kwargs: Any,
) -> Any:
    """Construct one RoboTwin task environment. Imported here, not at load."""
    try:
        from envs import CONFIGS_PATH  # noqa: F401 - RoboTwin's own package layout
        from envs.utils import class_decorator
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running RoboTwin needs the simulator: install RoboTwin 2.0 (MIT) and "
            "its object dataset, then run from the repository root so its `envs` "
            "package is importable. See "
            "https://robotwin-platform.github.io/doc/usage/robotwin-install.html"
        ) from error
    environment = class_decorator(task)
    environment.setup_demo(
        now_ep_num=0,
        seed=seed,
        is_test=True,
        embodiment=[embodiment],
        head_camera_type=head_camera,
        **kwargs,
    )
    return environment


class DualArm:
    """One RoboTwin task, as a world.

    RoboTwin's loop is ``get_obs`` / ``take_action`` / ``check_success`` rather
    than the gym five-tuple, which is why this is a small adapter rather than a
    manifest over the step/reset bridge.
    """

    def __init__(
        self,
        env: Any,
        *,
        action_type: str = "ee",
        cameras: Sequence[str] = ("head_camera", "left_camera", "right_camera"),
    ):
        self.env = env
        self.action_type = action_type
        self.cameras = tuple(cameras)
        self.instruction: str | None = None

    # -- the world ---------------------------------------------------------

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        seed = scene.seed if scene.seed is not None else 0
        if callable(getattr(self.env, "setup_demo", None)):
            self.env.setup_demo(now_ep_num=seed, seed=seed, is_test=True)
        elif callable(getattr(self.env, "reset", None)):  # pragma: no cover
            self.env.reset()
        # Asked rather than assumed: RoboTwin varies the sentence, and a policy
        # given the nominal instruction while the environment scores a different
        # one is being tested on a mismatch that reads as poor performance.
        getter = getattr(self.env, "get_instruction", None)
        self.instruction = str(getter()) if callable(getter) else scene.instruction
        return self._observe()

    def advance(self, action: np.ndarray) -> Step:
        self.env.take_action(np.asarray(action, dtype=float), action_type=self.action_type)
        solved = self._success()
        return Step(
            observation=self._observe(),
            reward=1.0 if solved else 0.0,
            done=bool(solved),
            # Tri-state on purpose: RoboTwin answers every step, so a trial that
            # ran out of horizon has genuinely been checked and found wanting —
            # unlike a real bench, where nobody looked.
            success=bool(solved),
        )

    def verdict(self, trial: Any) -> bool | None:  # pragma: no cover - per-step already answers
        return self._success()

    def close(self) -> None:
        if callable(getattr(self.env, "close_env", None)):
            self.env.close_env()
        elif callable(getattr(self.env, "close", None)):
            self.env.close()

    # -- internals ---------------------------------------------------------

    def _success(self) -> bool:
        checker = getattr(self.env, "check_success", None)
        return bool(checker()) if callable(checker) else False

    def _observe(self) -> dict[str, np.ndarray]:
        raw = self.env.get_obs()
        return flatten(raw, keep=self.cameras)


def flatten(observation: Any, *, keep: Sequence[str] = ()) -> dict[str, np.ndarray]:
    """RoboTwin's nested observation as flat ``a.b`` channel names.

    Its cameras arrive as ``obs["observation"]["head_camera"]["rgb"]`` and its
    proprioception as ``obs["joint_action"]``, so a path is the only name for a
    leaf that stays unique across three cameras with the same key.
    """
    out: dict[str, np.ndarray] = {}

    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{prefix}{key}.")
            return
        name = prefix.rstrip(".")
        if not name:
            return
        array = np.asarray(node)
        if array.dtype.kind in "OUSV" or array.ndim == 0:
            return
        if keep and not any(camera in name for camera in keep) and "camera" in name:
            return
        out[name] = array

    walk(observation)
    return out


class RoboTwinEvaluator(ClosedLoop):
    """Runs a policy over one RoboTwin dual-arm task."""

    rate_hz = CONTROL_HZ

    def __init__(
        self,
        task: str = "dual_bottles_pick_easy",
        *,
        action_type: str,
        embodiment: str = "aloha-agilex",
        name: str = "robotwin",
        cameras: Sequence[str] = ("head_camera", "left_camera", "right_camera"),
        factory: Callable[..., Any] = make_env,
        horizon: int = 400,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        """``action_type`` is required and has no default.

        Three action spaces on one arm, invisible in the array, and a policy
        trained for one evaluated under another produces numbers that are
        accepted and mean something else. There is nothing safe to choose on the
        caller's behalf, so the caller chooses.
        """
        self._width = width_of(action_type)
        self._task = task
        self._action_type = action_type
        self._embodiment = embodiment
        self._name = name
        self._cameras = tuple(cameras)
        self.observes = None
        self._factory = factory
        self._horizon = int(horizon)
        self._env_kwargs = dict(env_kwargs or {})
        self._world: DualArm | None = None

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            # RoboTwin reports solved-or-not and nothing between.
            stage_events=False,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            # It ships several dual-arm configurations, but a task is built
            # around the one it was given and cannot be rebuilt mid-run.
            hosts_embodiment=False,
            task=self._task,
            embodiment=self._embodiment,
            arms=len(ARMS),
            action_type=self._action_type,
            action_width=self._width,
            # The reason this backend exists, said where a report will find it.
            accepts_end_effector=self._action_type in ("ee", "delta_ee"),
            cameras=list(self._cameras),
            control_hz=CONTROL_HZ,
            licence="MIT (RoboTwin 2.0)",
        )

    def action(self) -> ChannelSpec:
        metadata: dict[str, Any] = {"arms": len(ARMS), "action_type": self._action_type}
        discriminators = ["arms", "action_type"]
        if self._action_type != "qpos":
            # A pose per arm, so a rotation per arm: at 3 in the left block and
            # at 11 in the right. Written down because the encoding is invisible
            # in the shape — sixteen floats is sixteen floats whether the four in
            # the middle are scalar-first, scalar-last, or a 6D tangent — and
            # because a converter that assumed one block would leave the right
            # arm's rotation to be read as the start of the next one.
            metadata[KEY_ROTATION] = "quat_wxyz"
            metadata[KEY_OFFSET] = tuple(
                index * ACTION_TYPES[self._action_type] + 3 for index in range(len(ARMS))
            )
            discriminators.append(KEY_ROTATION)
        return ChannelSpec(
            "action",
            "vector",
            (self._width,),
            "float32",
            semantics="action.eef_abs_pose" if self._action_type == "ee" else "actuation",
            rate_hz=CONTROL_HZ,
            dim_labels=labels_for(self._action_type),
            # All invisible in the shape and all change what the numbers mean.
            discriminators=tuple(discriminators),
            metadata=metadata,
        )

    def embodiment_for(self, scene: Scene) -> str:
        return self._embodiment

    def task_for(self, name: str = "", scenes: int = 25, horizon: int | None = None) -> TaskSpec:
        """One scene per seed. RoboTwin re-randomises object placement per seed,
        so scene ``k`` is the same arrangement on any machine — which is what a
        paired comparison rests on."""
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(
                Scene(id=f"{self._task}#{index:03d}", seed=index) for index in range(max(1, scenes))
            ),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    @property
    def world(self) -> DualArm:
        if self._world is None:
            self._world = DualArm(
                self._factory(
                    self._task,
                    embodiment=self._embodiment,
                    **self._env_kwargs,
                ),
                action_type=self._action_type,
                cameras=self._cameras,
            )
        return self._world

    def world_for(self, scene: Scene) -> DualArm:
        return self.world

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None

    def assemble(self, trial, scene, epoch, task, protocol):
        """Record the sentence the environment actually used, and the space the
        actions were interpreted in."""
        episode = super().assemble(trial, scene, epoch, task, protocol)
        labels = episode.labels
        extra: dict[str, Any] = {
            "action_type": self._action_type,
            "arms": len(ARMS),
            "robotwin_task": self._task,
        }
        if self._world is not None and self._world.instruction:
            extra["instruction_given"] = self._world.instruction
        return episode.with_labels(
            type(labels)(
                success=labels.success,
                stage_events=labels.stage_events,
                annotations={**dict(labels.annotations), **extra},
            )
        )


def for_ego(task: str = "dual_bottles_pick_easy", **kwargs: Any) -> RoboTwinEvaluator:
    """The configuration the ego pipeline can actually drive.

    ``ee`` because that is what ``retargeter_hands`` produces and refuses to turn
    into joint targets. Named rather than defaulted so that "we evaluated in
    end-effector space" is something somebody wrote down.
    """
    return RoboTwinEvaluator(task, action_type="ee", **kwargs)
