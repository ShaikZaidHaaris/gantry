"""robosuite as an evaluator, with the scenes taken from the demonstrations.

This is the piece that turns "how close were the predicted actions" into "did
the arm actually pick up the cube". Open-loop action error is cheap and honest
about what it measures, and it cannot tell you whether a policy recovers from
its own mistakes -- the moment its error puts the arm somewhere the recording
never went, the recording stops being an answer key.

Where the scenes come from
--------------------------
A robomimic file carries two things this needs. ``env_args`` is the exact recipe
for the world it was recorded in -- task, robot, controller, control rate. And
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
both of which also fire for reasons that are not success -- a shaped reward is
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
    Verdict,
    episode_from_arrays,
    episode_from_labels,
    proportion,
    seed_from,
)

from .staging import (  # noqa: E402  (same package; keeps the import block ordered)
    REALISATION,
    accepts_placement,
    check_staging,
    env_meta_for,
    materialise,
    sampler_shapes,
    surface_origin,
    verify_honoured,
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

#: ``REALISATION`` is imported from :mod:`.staging`: the name of this world is
#: one fact, and an embodiment's realisation block and a task's staging block
#: are keyed by the same one. Two copies would be two things to keep in step.


def _with_body(env_meta: Mapping[str, Any], embodiment: Any) -> dict[str, Any]:
    """The same world, built around a different machine.

    Reads the embodiment's ``robosuite`` metadata block -- robot name, controller,
    gripper -- and falls back to the embodiment's own name, which is what most
    descriptions of a robosuite-supported arm will say anyway.
    """
    block = dict((getattr(embodiment, "metadata", {}) or {}).get(REALISATION, {}))
    # How many arms this body has, which changes the shape of what the
    # simulator wants rather than just its values.
    arms = int(block.pop("arms", 1))
    robot = block.pop("robots", None) or [getattr(embodiment, "name", None)]
    if not robot or robot == [None]:
        raise ConfigError(
            f"{getattr(embodiment, 'name', embodiment)!r} does not say which robosuite "
            f"robot it is; add a {REALISATION!r} block to its description"
        )
    out = dict(env_meta)
    kwargs = dict(out.get("env_kwargs") or {})
    kwargs["robots"] = list(robot) if isinstance(robot, (list, tuple)) else [robot]
    kwargs.update(block)
    if arms > 1:
        # A multi-armed body is configured per arm here: one controller each,
        # one gripper each. Handed a single arm's settings it does not say so --         # it fails inside its gripper factory with an error mentioning neither
        # the arms nor the controller, which is how a body that simply cannot
        # do a one-armed task ends up looking like a bug in the harness.
        # Shaped correctly, the simulator gives its own clear answer instead.
        for key in ("controller_configs", "gripper_types"):
            if key in kwargs and not isinstance(kwargs[key], (list, tuple)):
                kwargs[key] = [kwargs[key]] * arms
    out["env_kwargs"] = kwargs
    return out


#: OSC_POSE, which is what the robomimic datasets were collected with: three
#: position deltas, three orientation deltas, one gripper command.
OSC_POSE = ChannelSpec(
    "actions",
    "vector",
    (7,),
    "float32",
    semantics="actuation",
    dim_labels=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
)


def success_from_is_success(env: Any, observation: Mapping[str, Any], done: bool) -> bool:
    """Ask the task whether it is solved.

    robomimic returns a mapping -- ``{"task": True}`` -- because some
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
    knows how to read it -- including the ``type`` field that distinguishes a
    robosuite environment from a gym or MOMART one.
    """
    if env_meta.get("placement"):
        raise ConfigError(
            "this world carries placement from a task file, and robomimic's factory "
            "has nowhere to put it; build it with factory=build_native_env, or the "
            "run would report the task's regions and use the environment's"
        )
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
    robomimic -- it is four methods, each one line, wrapping what robosuite
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


def build_native_env(
    env_meta: Mapping[str, Any], *, use_image_obs: bool = False, **kwargs: Any
) -> Any:
    """Rebuild the world with robosuite alone, no robomimic.

    Pass as ``factory=build_native_env`` where robomimic is absent or pinned
    against a different MuJoCo generation. The recipe is the same ``env_args``
    the dataset carries; only who reads it changes.
    """
    try:
        import robosuite
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running robosuite needs the simulator: pip install 'gantry-evaluator-robosuite[sim]'"
        ) from error
    from robosuite.utils.errors import RandomizationError

    settings = dict(env_meta.get("env_kwargs") or {})
    # A machine description names a controller; robosuite wants the whole
    # config -- gains, limits, interpolation. Expanded here, through robosuite's
    # own loader, rather than transcribed into every embodiment file: a gain
    # copied by hand drifts from the arm it was tuned for. Done at build time
    # rather than at merge time so that describing a machine never needs a
    # simulator installed.
    controller = settings.get("controller_configs")
    if isinstance(controller, Mapping) and set(controller) <= {"type"}:
        from robosuite.controllers import load_controller_config

        settings["controller_configs"] = load_controller_config(
            default_controller=controller["type"]
        )
    settings.update(
        has_renderer=False,
        has_offscreen_renderer=use_image_obs,
        use_camera_obs=use_image_obs,
        # The evaluator applies the horizon and reads success itself, so the
        # environment must not end an episode on its own timer.
        ignore_done=True,
    )
    settings.update(kwargs)
    name = str(env_meta["env_name"])
    placements = tuple(env_meta.get("placement") or ())

    def make(**extra: Any) -> Any:
        """Build, turning this world's own objections into named refusals.

        An environment here rejects a configuration it cannot host by asserting
 -- a body with the wrong number of arms, a task that insists on its own
        gripper. That is the environment answering the question correctly, and
        it should read as a refusal beside the cells that ran rather than as a
        crash that stops the sweep. The original words are kept, because they
        are the accurate reason and this layer should not paraphrase them.
        """
        try:
            return robosuite.make(env_name=name, **settings, **extra)
        except AssertionError as error:
            raise ConfigError(
                f"{name} will not host this configuration: {str(error).splitlines()[0].strip()}"
            ) from error

    if not placements:
        return _Native(make())

    from robosuite.environments.base import REGISTERED_ENVS

    if not accepts_placement(REGISTERED_ENVS[name]):
        raise ConfigError(
            f"{name} takes no placement at all, so a task file's start regions cannot "
            "drive it; this environment lays its objects out its own way, and a task "
            "staged in it should map none of them"
        )

    # A task file's regions are relative to the surface, which is what makes
    # them printable onto a real table. Where this world puts that surface is
    # measured from a world rather than written into the task, so the two
    # cannot drift apart: build once bare to read it, then build for real.
    probe = make()
    try:
        reference = surface_origin(probe)
    finally:
        probe.close()
    # Environments bind objects to an injected sampler in two incompatible
    # ways and say which nowhere, so the shape is offered and the world either
    # takes it or rejects it. Only the rejection each convention raises is
    # caught; anything else is this task being wrong about this world and
    # should surface as itself.
    refusals = []
    for composite in sampler_shapes(placements):
        sampler = materialise(placements, reference, composite=composite)
        try:
            env = make(placement_initializer=sampler)
        except (AttributeError, KeyError) as error:
            refusals.append(f"{'composite' if composite else 'single'}: {error}")
            continue
        except RandomizationError as error:
            # The shape was right and the regions do not fit the objects. Named
            # here because it is a fact about the task file, not about this
            # world, and robosuite's own message ("Cannot place all objects")
            # does not say which regions or whose.
            raise ConfigError(
                f"{name} could not lay out "
                + ", ".join(f"{p.task_id!r} in x={list(p.x)} y={list(p.y)}" for p in placements)
                + ", the declared start regions are too small for the objects, or "
                "two of them overlap"
            ) from error
        verify_honoured(env, sampler, name)
        return _Native(env)
    raise ConfigError(
        f"{name} would not take the placement this task describes, in any shape "
        f"this world offers ({'; '.join(refusals)}); its objects cannot be laid "
        "out from a task file"
    )


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
        initial_states: Sequence[np.ndarray] = (),
        *,
        seeds: Sequence[int] = (),
        embodiment: Any = None,
        name: str = "robosuite",
        observations: Sequence[str] = PROPRIO,
        action: ChannelSpec = OSC_POSE,
        success: Callable[..., bool] = success_from_is_success,
        factory: Callable[..., Any] = build_env,
        observe: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        use_image_obs: bool = False,
        instruction: str | None = None,
        definition: Any = None,
    ):
        """``env_meta`` and ``initial_states`` are what a robomimic connector
        exposes as ``.env`` and ``.initial_states()``. Passed in rather than
        read here, so this plugin needs no reader and no HDF5 library."""
        if not env_meta.get("env_name"):
            raise ConfigError("env_meta names no env_name, so there is no world to build")
        if len(initial_states) == 0 and len(seeds) == 0:
            raise ConfigError(
                "no scenes: give initial_states (restored from a recording) or seeds "
                "(the simulator's own placement, drawn the same way every time)"
            )
        if initial_states and seeds:
            raise ConfigError(
                "initial_states and seeds are two different definitions of a scene; "
                "pick one, or a rerun is not comparable to this run"
            )
        self.env_meta = dict(env_meta)
        self.initial_states = tuple(np.asarray(state) for state in initial_states)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.embodiment = embodiment
        self._name = name
        self._observations = tuple(observations)
        self._action = action
        self._success = success
        self._factory = factory
        self._observe = observe
        self._use_image_obs = use_image_obs
        self._instruction = instruction
        self._env: Any = None
        self._checked = False
        self.definition = definition
        if embodiment is not None:
            self.env_meta = _with_body(self.env_meta, embodiment)

    @classmethod
    def for_task(
        cls,
        task: Any,
        *,
        trials: int | None = None,
        embodiment: Any = None,
        world: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> "RobosuiteEvaluator":
        """Build the world a declared task describes.

        The other constructor takes its world from a recording: the scenes are
        that dataset's, and nobody without the file can reproduce them. This one
        takes it from a task file, which is a few hundred bytes of text that
        says where things start and what counts as done -- so the same scenes can
        be built by anyone, on any of the arms this world hosts, and marked out
        on a real table by hand.

        Scenes are seeds rather than restored states for the same reason: a
        restored simulator state is one machine's joint configuration and cannot
        be given to a different body, which would make the embodiment axis
        unusable exactly where it matters most.

        ``world`` is settings for the simulator itself -- which cameras to
        render, at what size, at what rate. They are kept separate from the
        evaluator's own arguments rather than mixed into ``**options``, because
        a typo in one silently becomes a keyword the environment ignores while
        a typo in the other is an error here; and a task file's own staging
        block can carry the same keys, which these override for one run without
        editing the file.
        """
        check_staging(task).raise_if_refused(f"cannot stage {task.name!r} in {REALISATION}")
        count = task.trials if trials is None else trials
        return cls(
            env_meta_for(task, **dict(world or {})),
            seeds=[seed_from(task.name, i) for i in range(count)],
            embodiment=embodiment,
            instruction=task.instruction,
            # robomimic's factory has nowhere to put placement, and this world
            # is defined by its placement. Chosen here rather than left to the
            # caller, because the alternative silently ignores the task file.
            factory=options.pop("factory", build_native_env),
            definition=task,
            **options,
        )

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
            # It rebuilds its world from a machine description, which is what
            # makes the embodiment a real axis rather than a manifest key.
            hosts_embodiment=True,
            task=self.task_name,
            robots=self.env_meta.get("env_kwargs", {}).get("robots"),
            embodiment=getattr(self.embodiment, "name", None),
            scenes=len(self.initial_states or self.seeds),
            scene_source=self.restores,
        )

    def requires(self) -> Requirement:
        return requires_channels(
            self._name,
            "evaluation",
            self._action,
            description=f"a policy commanding {self.task_name} through its controller",
        )

    def for_embodiment(self, embodiment: Any) -> "RobosuiteEvaluator":
        """The same task and scenes, on a different machine.

        Refused when the scenes are restored states. Those are one machine's
        joint configuration written down; handing them to another body either
        errors on a shape mismatch or -- worse, where the widths happen to
        agree -- silently puts the arm somewhere meaningless. Seeded scenes have
        no such problem: the simulator draws the same layout and each robot
        starts at its own home.
        """
        if self.restores == "states":
            raise ConfigError(
                f"{self.name}'s scenes are MuJoCo states restored from a recording of "
                f"{self.env_meta.get('env_kwargs', {}).get('robots')}, and a state is a "
                "description of that machine's joints. Build with seeds= to compare "
                "bodies: the layout is drawn the same way and each starts at its own home"
            )
        return RobosuiteEvaluator(
            self.env_meta,
            seeds=self.seeds,
            embodiment=embodiment,
            name=self._name,
            observations=self._observations,
            action=self._action,
            success=self._success,
            factory=self._factory,
            observe=self._observe,
            use_image_obs=self._use_image_obs,
            instruction=self._instruction,
            definition=self.definition,
        )

    @property
    def restores(self) -> str:
        """``states`` or ``seeds`` -- how a scene here is defined."""
        return "states" if self.initial_states else "seeds"

    def task_for(
        self, name: str = "", scenes: int | None = None, horizon: int | None = None
    ) -> TaskSpec:
        """One scene per recorded starting state, or per seed.

        A world built from a task file already knows how long the task gets and
        what it is called; asking the caller to repeat those is how the number
        that ran stops matching the number in the file.
        """
        if horizon is None:
            horizon = self.definition.horizon if self.definition is not None else 400
        if not name and self.definition is not None:
            name = self.definition.name
        source = self.initial_states or self.seeds
        count = len(source) if scenes is None else min(scenes, len(source))
        if self.restores == "states":
            made = [
                Scene(id=f"demo_{i}", instruction=self._instruction, metadata={"initial_state": i})
                for i in range(count)
            ]
        else:
            made = [
                Scene(
                    id=f"seed_{self.seeds[i]}",
                    instruction=self._instruction,
                    seed=self.seeds[i],
                    metadata={"initial_state": self.seeds[i]},
                )
                for i in range(count)
            ]
        return TaskSpec(name=name or self.task_name, scenes=tuple(made), horizon=horizon)

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

    def fits(self, policy: Any, observation: Mapping[str, Any]) -> Verdict:
        """Can this policy read what this world emits?

        Asked once, before the first trial, because the alternative is what
        actually happened: a policy expecting a dataset's channel names got a
        simulator's, raised on every scene, and produced twenty identical
        errors that looked like twenty failures. A world and a recording of a
        world do not name things the same way, and there is no reason they
        should -- but somebody has to say so out loud before the run.
        """
        wanted = getattr(policy, "observes", None)
        if wanted is None:
            return Verdict.yes()
        missing = [
            spec.name
            for spec in wanted().channels
            if not spec.optional and spec.name not in observation
        ]
        if not missing:
            return Verdict.yes()
        return Verdict.no(
            "robosuite.observation_mismatch",
            f"{getattr(policy, 'name', 'the policy')} reads {missing}, which this world "
            f"does not emit; it emits {sorted(observation)}",
            hint="pass observe= to assemble what the policy wants from what the world "
            "gives, which channel of a simulator corresponds to which column of a "
            "recording is knowledge that lives with the caller, not with either of them",
            missing=missing,
        )

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
        env = self.env
        if self.restores == "states":
            env.reset()
            raw = env.reset_to(
                {"states": self.initial_states[int(scene.metadata["initial_state"])]}
            )
        else:
            # The simulator draws its own layout under a fixed seed, and the
            # robot starts at its own home pose. The only way a scene can mean
            # the same thing to two different bodies: a restored state is a
            # description of one machine's joints and cannot be given to
            # another.
            np.random.seed(int(scene.seed))
            raw = env.reset()
        observation = self._seen(raw)
        if not self._checked:
            # Once, on the first scene's own first observation. Outside the try
            # below on purpose: a policy that cannot read this world should stop
            # the run, not be recorded as twenty separate failures.
            self._checked = True
            self.fits(policy, observation).raise_if_refused(
                f"{self.name} cannot show {getattr(policy, 'name', 'this policy')} what it reads"
            )
        policy.reset(
            EpisodeContext(scene.id, scene.instruction, seed=int(scene.metadata["initial_state"]))
        )

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
                raw, reward, done, _info = env.step(np.asarray(action, dtype=float))
                observation = self._seen(raw)
                trial.rewards.append(float(reward))
                trial.steps += 1
                if self._success(env, observation, bool(done)):
                    trial.success = True
                    return trial, None
                if done:
                    return trial, None
        return trial, None

    def _seen(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """What the policy is shown: the world's own keys, or the caller's
        assembly of them."""
        return self._observe(observation) if self._observe else observation

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
                id=eid,
                source=task.name,
                labels=EpisodeLabels(success=None, annotations=annotations),
                task=task.name,
                embodiment=self._embodiment(),
            )
        arrays = {key: np.asarray(values) for key, values in trial.frames.items()}
        arrays[self._action.name] = np.asarray(trial.actions, dtype="float32")
        arrays["reward"] = np.asarray(trial.rewards[: len(trial.actions)], dtype="float32")
        return episode_from_arrays(
            arrays,
            self._schema(trial),
            id=eid,
            source=task.name,
            labels=EpisodeLabels(
                success=None if error else trial.success,
                annotations={**annotations, "return": float(sum(trial.rewards))},
            ),
            task=task.name,
            embodiment=self._embodiment(),
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
