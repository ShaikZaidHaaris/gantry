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

    qpos  left arm joints + left gripper, then right — 14 numbers
    ee    left (xyz + wxyz quaternion) + gripper, then right — 16, absolute pose

That second row is why this plugin is worth more than another benchmark.
``retargeter_hands`` refuses to produce joint positions and says why — inverse
kinematics needs link lengths, joint limits and a choice of elbow configuration,
none of which a retargeter has. That refusal has blocked the ego path from every
ALOHA-family config since it was written. RoboTwin's ``ee`` mode takes exactly
what the retargeter *does* produce, so the chain runs end to end with no IK step
and no pretending.

Two action spaces, one arm, and the widths differ
-------------------------------------------------
Fourteen numbers in ``qpos``, sixteen in ``ee``. Same robot, same task, same
method name, and a policy trained for one cannot be evaluated under another —
the numbers would be accepted and mean something else. So ``action_type`` is a
constructor argument with no safe default, the width follows from it, and a
mismatch is refused at plan time rather than discovered as poor performance.

There is no delta mode. An earlier draft of this file listed one, on the
strength of the documentation; the simulator's own signature is
``Literal['qpos', 'ee']`` and a third value falls through both branches to an
unbound local. A backend that advertises a control mode its simulator does not
implement is worse than one that offers fewer.

Seeds are screened by the expert first
--------------------------------------
RoboTwin's own evaluation loop runs the scripted expert on a seed before
scoring a policy on it, and skips the seed if the expert fails. Without that,
a success rate mixes "the policy could not do it" with "this arrangement was
not solvable", and the two move independently as the seed range changes.

Screening is a separate, recorded step here rather than something hidden inside
the rollout: :meth:`RoboTwinEvaluator.screen` returns the seeds the expert
solved, and they are passed to :meth:`task_for` explicitly. What was screened
out is then a fact somebody can look at, not a silent filter.

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

#: The two action spaces RoboTwin accepts, and the width each implies *per
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
    # position, wxyz quaternion, gripper. The quaternion is scalar-first because
    # the pose goes to mplib, which takes SAPIEN's convention.
    "ee": 8,
}

#: Arms, in the order they are concatenated. The only written record of which
#: half of the vector is which arm — the same discipline policy_pi0 and
#: retargeter_hands already keep, for the same reason.
ARMS = ("left", "right")

#: RoboTwin 2.0's fifty task ids, read off the installed ``envs`` package. Not a
#: whitelist: any id the installed version knows is accepted, and this exists so
#: a run can be planned against a task list on a laptop with no simulator.
#:
#: These are the 2.0 names. The 1.0 paper's ids (``dual_bottles_pick_easy`` and
#: friends) were renamed wholesale and none of them resolve.
TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
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


def config_for(
    task: str,
    *,
    config: str = "demo_clean",
    embodiment: str = "aloha-agilex",
    root: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """RoboTwin's own settings for a task, assembled the way RoboTwin assembles them.

    ``setup_demo(**args)`` wants the whole thing — the task YAML, plus resolved
    embodiment file paths, per-robot configs and the head camera's resolution,
    all of which its eval script derives at start-up and none of which are
    optional. An abbreviated dict raises a KeyError somewhere inside scene
    construction, a long way from anything that names the cause.

    So this reads the same files rather than reproducing their contents: task
    config, ``_embodiment_config.yml`` and ``_camera_config.yml``. Anything the
    installed version changes is picked up instead of drifting.
    """
    import os

    import yaml

    try:
        from envs import CONFIGS_PATH
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running RoboTwin needs the simulator: install RoboTwin 2.0 (MIT) and its "
            "object dataset, and run from the repository root so its `envs` package is "
            "importable. See "
            "https://robotwin-platform.github.io/doc/usage/robotwin-install.html"
        ) from error

    base = root or os.path.join(os.path.dirname(str(CONFIGS_PATH).rstrip("/")), "")
    task_config = os.path.join(base, "task_config", f"{config}.yml")
    if not os.path.isfile(task_config):  # pragma: no cover - needs the simulator
        raise ConfigError(
            f"no RoboTwin task config at {task_config!r}; the shipped ones are "
            "'demo_clean' and 'demo_randomized'"
        )
    with open(task_config, encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    args["task_name"] = task
    args["task_config"] = config
    args["embodiment"] = [embodiment]

    with open(
        os.path.join(str(CONFIGS_PATH), "_embodiment_config.yml"), encoding="utf-8"
    ) as handle:
        bodies = yaml.load(handle.read(), Loader=yaml.FullLoader)
    if embodiment not in bodies:  # pragma: no cover - needs the simulator
        raise ConfigError(f"unknown embodiment {embodiment!r}; installed: {sorted(bodies)}")
    robot_file = bodies[embodiment]["file_path"]
    if robot_file is None:  # pragma: no cover - needs the simulator
        raise ConfigError(f"embodiment {embodiment!r} has no files; download the assets")

    # One entry means the same robot on both sides, which is what makes it a
    # single dual-arm body rather than two independent ones facing each other.
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    for side in ("left", "right"):
        with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as handle:
            args[f"{side}_embodiment_config"] = yaml.load(handle.read(), Loader=yaml.FullLoader)

    with open(os.path.join(str(CONFIGS_PATH), "_camera_config.yml"), encoding="utf-8") as handle:
        cameras = yaml.load(handle.read(), Loader=yaml.FullLoader)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head]["h"]
    args["head_camera_w"] = cameras[head]["w"]

    args.update(overrides)
    return args


def make_env(
    task: str,
    *,
    embodiment: str = "aloha-agilex",
    seed: int = 0,
    head_camera: str = "D435",
    **kwargs: Any,
) -> Any:
    """Construct one RoboTwin task environment. Imported here, not at load.

    RoboTwin has no public constructor: its own scripts each carry a private copy
    of a ``class_decorator`` that imports ``envs.<task>`` and instantiates the
    class of the same name. That convention is reproduced rather than imported,
    because the copies live in ``script/`` next to ``__main__`` guards and
    importing one of them runs an argument parser.
    """
    import importlib

    try:
        module = importlib.import_module(f"envs.{task}")
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            f"cannot load RoboTwin task {task!r}: install RoboTwin 2.0 (MIT) and its "
            "object dataset, and run from the repository root so its `envs` package "
            "is importable. See "
            "https://robotwin-platform.github.io/doc/usage/robotwin-install.html"
        ) from error
    try:
        environment = getattr(module, task)()
    except AttributeError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            f"RoboTwin's `envs.{task}` has no class named {task!r}; the installed "
            f"version may use different task ids. Known here: {len(TASKS)}"
        ) from error
    settings = config_for(task, embodiment=embodiment, **kwargs)
    settings.setdefault("camera", {})["head_camera_type"] = head_camera
    environment.setup_demo(now_ep_num=0, seed=seed, is_test=True, **settings)
    # Kept so screen() and every later reset use the same settings rather than
    # a fresh partial dict that happens to differ.
    environment.gantry_settings = settings
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
            # The settings the environment was built with. Resetting with a
            # partial dict would quietly change the cameras or the randomisation
            # between the screen and the run being screened for.
            self.env.setup_demo(
                now_ep_num=seed,
                seed=seed,
                is_test=True,
                **dict(getattr(self.env, "gantry_settings", {}) or {}),
            )
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
        """The latched flag first, the predicate second.

        RoboTwin interpolates each command into hundreds of physics steps and
        checks success *inside* that loop, latching ``eval_success``. Asking
        ``check_success()`` afterwards can disagree: an object knocked into place
        mid-motion may settle back out of it, and the trial that succeeded would
        be recorded as a failure.
        """
        latched = getattr(self.env, "eval_success", None)
        if latched is not None:
            return bool(latched)
        checker = getattr(self.env, "check_success", None)
        return bool(checker()) if callable(checker) else False

    def _observe(self) -> dict[str, np.ndarray]:
        raw = self.env.get_obs()
        flat = flatten(raw, keep=self.cameras)
        vector = endpose_vector(flat)
        if vector is not None:
            flat["endpose.vector"] = vector
        return flat


def endpose_vector(flat: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """The arms' current poses, laid out exactly like an ``ee`` action.

    RoboTwin publishes ``joint_action.vector`` — the whole qpos state in the
    order ``qpos`` actions are read — but nothing equivalent for poses. Its
    endpose arrives as four separate pieces, so a policy controlling in ee space
    has no single channel describing where the arms currently are, in the space
    it is commanding.

    This is that channel and it is assembled in the action's own order rather
    than a convenient one: ``left(xyz+quat) + left gripper``, then right. A state
    laid out differently from the action is the kind of mismatch that trains and
    evaluates without complaint and is wrong by a fixed rotation.

    ``None`` when the environment was configured without endpose data, rather
    than a zero vector that would read as "the arms are at the origin".
    """
    pieces: list[np.ndarray] = []
    for arm in ARMS:
        pose = flat.get(f"endpose.{arm}_endpose")
        grip = flat.get(f"endpose.{arm}_gripper")
        if pose is None or grip is None:
            return None
        pieces.append(np.asarray(pose, dtype=float).reshape(-1))
        pieces.append(np.asarray(grip, dtype=float).reshape(-1))
    out = np.concatenate(pieces)
    return out if out.size == width_of("ee") else None


def state_spec(name: str = "observation.state") -> ChannelSpec:
    """What :func:`endpose_vector` produces, described.

    Same encoding and same offsets as an ``ee`` action, because it is the same
    layout — which is what lets one conversion serve both.
    """
    return ChannelSpec(
        name,
        "vector",
        (width_of("ee"),),
        "float32",
        semantics="observation.eef_abs_pose",
        rate_hz=CONTROL_HZ,
        dim_labels=labels_for("ee"),
        discriminators=("arms", KEY_ROTATION),
        metadata={
            "arms": len(ARMS),
            KEY_ROTATION: "quat_wxyz",
            KEY_OFFSET: tuple(index * ACTION_TYPES["ee"] + 3 for index in range(len(ARMS))),
        },
    )


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
        if array.dtype.kind in "OUSV":
            return
        if array.ndim == 0:
            # A gripper opening arrives as a bare float — RoboTwin's endpose is
            # {left_endpose: [7], left_gripper: 0.3, ...}. Dropping zero-rank
            # values would discard both grippers and leave a pose-only state
            # that still looks well formed.
            array = array.reshape(1)
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
        task: str = "pick_dual_bottles",
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
        #: (first seed tried, one past the last, the ones the expert solved).
        self._screened: tuple[int, int, tuple[int, ...]] | None = None

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
            accepts_end_effector=self._action_type == "ee",
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

    def provides(self) -> tuple[ChannelSpec, ...]:
        """What the world publishes, for anything that has to bind against it.

        Assembled here rather than by the caller because the caller would have
        to know that a camera arrives at ``observation.<name>.rgb``, that the
        pose state is a derived channel, and what encoding it is in — three
        facts about this simulator that belong with this simulator.
        """
        cameras = tuple(
            ChannelSpec(
                f"observation.{camera}.rgb",
                "image",
                (None, None, 3),
                "uint8",
                rate_hz=CONTROL_HZ,
                semantics="observation.image",
                metadata={"camera": camera},
            )
            for camera in self._cameras
        )
        return (
            *cameras,
            state_spec(),
            state_spec("endpose.vector"),
            ChannelSpec(
                "joint_action.vector",
                "vector",
                (width_of("qpos"),),
                "float32",
                rate_hz=CONTROL_HZ,
                semantics="observation.joint_pos",
                dim_labels=labels_for("qpos"),
                discriminators=("arms",),
                metadata={"arms": len(ARMS)},
            ),
        )

    def embodiment_for(self, scene: Scene) -> str:
        return self._embodiment

    def task_for(
        self,
        name: str = "",
        scenes: int = 25,
        horizon: int | None = None,
        seeds: Sequence[int] | None = None,
    ) -> TaskSpec:
        """One scene per seed. RoboTwin re-randomises object placement per seed,
        so scene ``k`` is the same arrangement on any machine — which is what a
        paired comparison rests on.

        ``seeds`` takes the screened list from :meth:`screen`. Left unset, the
        seeds are ``0..scenes-1`` unscreened, and the resulting success rate
        mixes "the policy failed" with "this arrangement was not solvable".
        """
        chosen = (
            tuple(int(seed) for seed in seeds)
            if seeds is not None
            else tuple(range(max(1, scenes)))
        )
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(Scene(id=f"{self._task}#{seed:03d}", seed=seed) for seed in chosen),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    @staticmethod
    def _settings(env: Any) -> dict[str, Any]:
        return dict(getattr(env, "gantry_settings", {}) or {})

    def screen(self, count: int, *, start: int = 0, limit: int | None = None) -> tuple[int, ...]:
        """Seeds the scripted expert can solve, which are the fair ones to score.

        RoboTwin randomises object placement per seed and not every arrangement
        is solvable. Scoring a policy on one that is not charges it for the
        sampler, and because the unsolvable fraction moves with the seed range,
        two runs over different ranges are not comparable even on the same task.
        RoboTwin's own loop screens for exactly this reason.

        Returns fewer than ``count`` if ``limit`` seeds are exhausted first —
        the caller can then see the expert's own rate rather than being handed a
        short list with no explanation.
        """
        world = self.world
        env = world.env
        if not callable(getattr(env, "play_once", None)):
            raise ConfigError(
                f"{self._name}: this environment has no scripted expert (`play_once`), "
                "so seeds cannot be screened. Pass seeds to task_for() explicitly, or "
                "run unscreened and record that the success rate includes unsolvable "
                "arrangements"
            )
        ceiling = start + (limit if limit is not None else count * 10)
        solved: list[int] = []
        seed = start
        while len(solved) < count and seed < ceiling:
            try:
                env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **self._settings(env))
                env.play_once()
                if bool(getattr(env, "plan_success", True)) and world._success():
                    solved.append(seed)
            except Exception:
                # An expert failure is a property of the arrangement, which is
                # what is being measured here. It is not this run's error.
                pass
            finally:
                if callable(getattr(env, "close_env", None)):
                    env.close_env()
            seed += 1
        self._screened = (start, seed, tuple(solved))
        return tuple(solved)

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
            # Whether the arrangement was known solvable before the policy saw
            # it. Without this an unscreened rate reads like a screened one.
            "expert_screened": self._screened is not None,
        }
        if self._screened is not None:
            start, stop, solved = self._screened
            extra["expert_solve_rate"] = round(len(solved) / max(1, stop - start), 4)
        if self._world is not None and self._world.instruction:
            extra["instruction_given"] = self._world.instruction
        return episode.with_labels(
            type(labels)(
                success=labels.success,
                stage_events=labels.stage_events,
                annotations={**dict(labels.annotations), **extra},
            )
        )


def for_ego(task: str = "pick_dual_bottles", **kwargs: Any) -> RoboTwinEvaluator:
    """The configuration the ego pipeline can actually drive.

    ``ee`` because that is what ``retargeter_hands`` produces and refuses to turn
    into joint targets. Named rather than defaulted so that "we evaluated in
    end-effector space" is something somebody wrote down.
    """
    return RoboTwinEvaluator(task, action_type="ee", **kwargs)
