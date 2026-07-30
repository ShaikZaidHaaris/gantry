"""RoboCasa as an evaluator, for the axis nothing else here covers: scene variety.

The tasks this project has measured so far live on one table. Lift is a cube on a
tabletop; pick-place-can is a can on the same tabletop; every LIBERO scene is a
tabletop. So a policy that has learned *this table* — its lighting, its texture,
the height of the camera above it — scores well, and the number does not
distinguish that from having learned the skill. There is no way to tell from
inside a single-scene benchmark which of the two happened.

RoboCasa is a hundred-odd kitchen tasks across a hundred-odd generated kitchen
layouts, with the furniture, textures and object instances resampled per episode.
That makes the distinction measurable: the same policy, the same task, held vs.
varied scene, and the gap between them is a number rather than a suspicion. It is
the same experiment this project already ran by hand as "lift narrow" against
"lift wide", except that the wide condition is a hundred kitchens instead of a
bigger rectangle on one table.

Built on the same simulator as the robosuite evaluator, and still its own plugin
-------------------------------------------------------------------------------
RoboCasa subclasses robosuite's environments, so a first instinct is to add a
flag to the robosuite evaluator. That instinct is wrong twice.

Its construction is different in kind: an environment is made with a *layout id*
and a *style id* drawn from generators, and the scene identity is that pair plus
the object-instance seed — not a placement-sampler rectangle. Folding that into
the robosuite evaluator means one class with two mutually exclusive halves and a
flag deciding which is live, which is where the special cases live.

And its dependency is heavier: RoboCasa ships assets measured in gigabytes.
Someone evaluating on plain robosuite tasks should not acquire them, and a plugin
boundary is the only thing that actually enforces that.

Layout and style are the scene, and they are recorded
----------------------------------------------------
A run whose provenance says "robocasa, 50 trials" is unreproducible; a run whose
scenes say ``(layout 3, style 7, seed 12)`` can be re-created exactly. So each
scene carries its triple in metadata and its id spells it out, which is also what
lets a held-scene condition be *stated* rather than hoped for: hold the layout,
vary the seed, and the record shows which was which.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: A sample of RoboCasa's task ids, by the names its own registry uses. Not a
#: whitelist: any id RoboCasa knows is accepted.
TASKS = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPCounterToStove",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeePressButton",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
)

#: The bodies RoboCasa ships. Mobile bases, which is the other thing a kitchen
#: forces: the arm has to get to the counter before it can use it.
ROBOTS = ("PandaMobile", "PandaOmron", "GR1FixedLowerBody", "Panda")

#: How many layouts and styles the generators offer. ``-1`` and ``-2`` are
#: RoboCasa's own "random" and "random from a fixed set" sentinels; spelled out
#: here because a run that used them recorded no scene and cannot be repeated.
LAYOUTS = 10
STYLES = 12
RANDOM_SENTINELS = (-1, -2)

CAMERAS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
)
PROPRIO = ("robot0_base_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
CAMERA_SIZE = 128
CONTROL_HZ = 20.0

#: A mobile Panda takes twelve numbers: the arm's six pose deltas, the gripper,
#: and the base. Read from the environment rather than trusted — see ``action``.
DEFAULT_WIDTH = 12


def make_env(
    task: str,
    *,
    robot: str = "PandaMobile",
    layout: int = 0,
    style: int = 0,
    camera_size: int = CAMERA_SIZE,
    cameras: Sequence[str] = CAMERAS,
    seed: int | None = None,
    **kwargs: Any,
) -> Any:
    """Construct one RoboCasa kitchen. Imported here, not at module load."""
    try:
        import robocasa  # noqa: F401 - registers the environments
        import robosuite
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running RoboCasa needs the simulator and its assets: pip install "
            "'gantry-evaluator-robocasa[sim]', then fetch the asset pack "
            "(several gigabytes) with `python -m robocasa.scripts.download_kitchen_assets`"
        ) from error
    return robosuite.make(
        env_name=task,
        robots=robot,
        layout_ids=layout,
        style_ids=style,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(cameras),
        camera_heights=camera_size,
        camera_widths=camera_size,
        control_freq=int(CONTROL_HZ),
        seed=seed,
        **kwargs,
    )


class Kitchen:
    """One RoboCasa environment, as a world.

    ``_check_success`` is asked every step, the same convention robosuite uses,
    so a trial ends the moment the task is done and a trial that ran out of
    horizon has genuinely been checked and found wanting.
    """

    def __init__(self, env: Any):
        self.env = env

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        return dict(self.env.reset())

    def advance(self, action: np.ndarray) -> Step:
        observation, reward, done, info = self.env.step(np.asarray(action, dtype=float))
        return Step(
            observation=dict(observation),
            reward=float(reward),
            done=bool(done),
            success=bool(self.env._check_success()),
            info=dict(info or {}),
        )

    def close(self) -> None:
        if callable(getattr(self.env, "close", None)):
            self.env.close()


class RobocasaEvaluator(ClosedLoop):
    """Runs a policy over one RoboCasa task, across kitchen layouts and styles."""

    rate_hz = CONTROL_HZ

    def __init__(
        self,
        task: str = "PnPCounterToCab",
        *,
        robot: str = "PandaMobile",
        name: str = "robocasa",
        layouts: Sequence[int] = (0,),
        styles: Sequence[int] = (0,),
        cameras: Sequence[str] = CAMERAS,
        proprio: Sequence[str] = PROPRIO,
        camera_size: int = CAMERA_SIZE,
        factory: Callable[..., Any] = make_env,
        horizon: int = 500,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        """``layouts`` and ``styles`` are the scene axis, and both are explicit.

        RoboCasa accepts ``-1`` for "pick a random layout", which is convenient
        and produces a run nobody can reproduce: the record would say the task
        and not the kitchen. So the sentinels are refused here and a caller who
        wants variety lists the layouts they want varied over — which is also the
        only way a held-scene condition can be stated as held.
        """
        bad = [value for value in (*layouts, *styles) if int(value) in RANDOM_SENTINELS]
        if bad:
            raise ConfigError(
                f"{name}: layout/style {bad} asks RoboCasa to choose at random, which "
                "records no scene — the run cannot be repeated and a held-scene "
                "condition cannot be told apart from a varied one. List the layouts "
                f"and styles instead (layouts 0..{LAYOUTS - 1}, styles 0..{STYLES - 1})"
            )
        if not layouts or not styles:
            raise ConfigError(f"{name}: needs at least one layout and one style")
        self._task = task
        self._robot = robot
        self._name = name
        self._layouts = tuple(int(value) for value in layouts)
        self._styles = tuple(int(value) for value in styles)
        self._cameras = tuple(cameras)
        self._proprio = tuple(proprio)
        self.observes = (*self._cameras, *self._proprio)
        self._camera_size = camera_size
        self._factory = factory
        self._horizon = horizon
        self._env_kwargs = dict(env_kwargs or {})
        self._worlds: dict[tuple[int, int], Kitchen] = {}
        self._action: ChannelSpec | None = None

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            stage_events=False,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            # A kitchen is built around whichever robot it was given, and
            # RoboCasa ships mobile bases and a humanoid. Same claim ManiSkill
            # makes, for the same reason.
            hosts_embodiment=True,
            task=self._task,
            robot=self._robot,
            layouts=list(self._layouts),
            styles=list(self._styles),
            cameras=list(self._cameras),
            control_hz=CONTROL_HZ,
        )

    def action(self) -> ChannelSpec:
        """Read from the live environment: a mobile base changes the width.

        A fixed Panda is seven numbers, a mobile one twelve, the humanoid
        different again. Assuming any of them turns a body swap into a run of
        truncated commands rather than an error.
        """
        if self._action is None:
            env = self._kitchen(self._layouts[0], self._styles[0]).env
            dim = getattr(env, "action_dim", None)
            if dim is None:
                spec = getattr(env, "action_spec", None)
                dim = len(spec[0]) if spec else None
            if not dim:
                raise ConfigError(
                    f"{self._name}: the environment for {self._task!r} does not report "
                    "its action dimension, so what a policy must emit cannot be stated"
                )
            self._action = ChannelSpec(
                "action",
                "vector",
                (int(dim),),
                "float32",
                semantics="actuation",
                rate_hz=CONTROL_HZ,
            )
        return self._action

    def for_embodiment(self, embodiment: Any) -> "RobocasaEvaluator":
        uid = (
            embodiment
            if isinstance(embodiment, str)
            else getattr(embodiment, "name", None) or str(embodiment)
        )
        return RobocasaEvaluator(
            self._task,
            robot=str(uid),
            name=self._name,
            layouts=self._layouts,
            styles=self._styles,
            cameras=self._cameras,
            proprio=self._proprio,
            camera_size=self._camera_size,
            factory=self._factory,
            horizon=self._horizon,
            env_kwargs=self._env_kwargs,
        )

    def embodiment_for(self, scene: Scene) -> str:
        return self._robot

    def task_for(self, name: str = "", per_scene: int = 5, horizon: int | None = None) -> TaskSpec:
        """One scene per (layout, style, seed), in that nesting order.

        ``per_scene`` is how many object-instance draws to take within each
        kitchen. Held layout with many seeds and many layouts with one seed each
        are two genuinely different experiments — the first measures variation
        within a scene, the second across scenes — and both are expressible here
        because the triple is explicit.
        """
        scenes: list[Scene] = []
        for layout in self._layouts:
            for style in self._styles:
                for index in range(max(1, int(per_scene))):
                    scenes.append(
                        Scene(
                            id=f"{self._task}/l{layout}s{style}#{index:02d}",
                            seed=index,
                            metadata={
                                "layout": layout,
                                "style": style,
                                "draw": index,
                            },
                        )
                    )
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(scenes),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    def _kitchen(self, layout: int, style: int) -> Kitchen:
        """One environment per (layout, style), reused across its draws.

        Building a RoboCasa kitchen composes a scene from asset files and starts
        a renderer; it is seconds, not milliseconds. Rebuilding per trial would
        cost more than the trials.
        """
        key = (int(layout), int(style))
        if key not in self._worlds:
            self._worlds[key] = Kitchen(
                self._factory(
                    self._task,
                    robot=self._robot,
                    layout=key[0],
                    style=key[1],
                    camera_size=self._camera_size,
                    cameras=self._cameras,
                    **self._env_kwargs,
                )
            )
        return self._worlds[key]

    def world_for(self, scene: Scene) -> Kitchen:
        metadata = scene.metadata or {}
        return self._kitchen(
            int(metadata.get("layout", self._layouts[0])),
            int(metadata.get("style", self._styles[0])),
        )

    def close(self) -> None:
        for world in self._worlds.values():
            world.close()
        self._worlds.clear()

    def declares(self) -> Mapping[str, ChannelSpec]:
        return {
            name: ChannelSpec(
                name,
                "image",
                (self._camera_size, self._camera_size, 3),
                "uint8",
                rate_hz=CONTROL_HZ,
                semantics="rgb",
            )
            for name in self._cameras
        }
