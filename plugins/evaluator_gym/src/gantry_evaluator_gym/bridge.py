"""The seam that lets any step/reset environment be evaluated.

``reset()`` and ``step(action)`` is the most widely implemented interface in
this field. Simulators, benchmark suites, real-robot wrappers and toy examples
all speak it, which makes it the one bridge worth writing: everything on the far
side of it becomes an evaluator without another line of Gantry.

Nothing is imported
-------------------
No environment library appears anywhere in this plugin, in its dependencies, or
in its tests. The caller passes a factory that returns something with ``reset``
and ``step``, and duck typing does the rest. That is what keeps a suite with a
hard pin on an old numpy from dictating what the rest of the framework may
install, and it is why this bridge works with libraries written after it.

Both dialects
-------------
The five-value ``(obs, reward, terminated, truncated, info)`` form and the older
four-value ``(obs, reward, done, info)`` are both accepted and told apart by
length. ``reset`` is accepted with or without a seed and with or without the
info half of its return. None of that is guessing: each shape is unambiguous.

What is *not* inferred
----------------------
Whether a trial succeeded, and which milestones it passed. The step API has no
place for either, so different suites bury them in different corners of ``info``
and some do not record them at all. Reading one convention by default would mean
silently reporting zero successes for every environment that uses a different
one — a complete, well-formatted, wrong result. So a caller states how to read
them, in one argument, using a named helper; and until they do, the descriptor
says out loud that this evaluator reports no outcomes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from gantry.contracts.evaluator import (
    Evaluator,
    Scene,
    TaskSpec,
    evaluator_descriptor,
)
from gantry.contracts.evaluator import Protocol as RunProtocol
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
    StageEvent,
    episode_from_arrays,
)

VERSION = "0.1.0.dev0"

#: What the recorded reward channel is called.
REWARD = "reward"

#: Rank of one timestep -> the modality core knows it by.
KINDS = {0: "scalar", 1: "vector", 2: "matrix"}


class Env(Protocol):
    """The far side of the bridge. Deliberately the smallest possible surface."""

    def reset(self, *args: Any, **kwargs: Any) -> Any: ...

    def step(self, action: Any) -> Any: ...


@dataclass(frozen=True)
class Transition:
    """One environment response, in whichever dialect it arrived."""

    step: int
    observation: Mapping[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


# --------------------------------------------------------------------------
# how to read an outcome — named, so the assumption is visible at the call site
# --------------------------------------------------------------------------


def success_from_info(*keys: str) -> Callable[[Transition], bool | None]:
    """Read success from ``info``, which is where most suites put it.

    Returns ``None`` while the key is absent, so an environment that only
    reports success on the final step is not read as a failure on every step
    before it.
    """
    wanted = keys or ("success", "is_success")

    def read(transition: Transition) -> bool | None:
        for key in wanted:
            if key in transition.info:
                return bool(np.asarray(transition.info[key]).reshape(-1)[0])
        return None

    return read


def terminated_is_success() -> Callable[[Transition], bool | None]:
    """Treat a terminal state as success and a time limit as failure.

    True for environments that only end when the task is done, wrong for any
    that also terminate on an unrecoverable failure. Named rather than default
    precisely because it is a claim about one environment.
    """

    def read(transition: Transition) -> bool | None:
        return True if transition.terminated else None

    return read


def milestones_from_info(key: str = "milestones") -> Callable[[Transition], Sequence[str]]:
    """Read the milestones reached at this step from one ``info`` key."""

    def read(transition: Transition) -> Sequence[str]:
        value = transition.info.get(key, ())
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Mapping):
            return tuple(name for name, reached in value.items() if reached)
        return tuple(str(name) for name in value)

    return read


def milestones_from_flags(*names: str) -> Callable[[Transition], Sequence[str]]:
    """Read each milestone from its own boolean ``info`` key."""

    def read(transition: Transition) -> Sequence[str]:
        return tuple(name for name in names if bool(transition.info.get(name)))

    return read


SUCCESS_READERS = {"info": success_from_info, "terminated": terminated_is_success}
MILESTONE_READERS = {"info": milestones_from_info, "flags": milestones_from_flags}


@dataclass(frozen=True)
class Readers:
    """How this environment reports what happened.

    ``names`` is the milestone vocabulary the run was *looking for*, and it is
    recorded whether or not any trial reaches them. A funnel that inferred its
    rungs from the milestones that occurred would see a policy failing at the
    second one in every episode as a one-rung funnel at a hundred percent.
    """

    success: Callable[[Transition], bool | None] | None = None
    milestones: Callable[[Transition], Sequence[str]] | None = None
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.milestones is not None and not self.names:
            raise ValueError(
                "reading milestones without naming them leaves a funnel to infer its "
                "own rungs from whatever happened to be reached"
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Readers":
        """Build from plain data, so a manifest can say how to read an outcome.

        ``{"success": {"info": ["is_success"]}, "milestones": {"flags": [...]},
        "names": [...]}``. Each reader is named, never inferred: a manifest that
        omits ``success`` gets an evaluator that declares it reports none, which
        is the same honest default as the Python interface.
        """
        return cls(
            success=_reader(SUCCESS_READERS, config.get("success"), "success"),
            milestones=_reader(MILESTONE_READERS, config.get("milestones"), "milestones"),
            names=tuple(config.get("names") or ()),
        )


def _reader(known: Mapping[str, Callable], config: Any, what: str) -> Callable | None:
    """One named reader and its arguments, or nothing at all."""
    if config is None:
        return None
    if isinstance(config, str):
        kind, arguments = config, ()
    elif isinstance(config, Mapping) and len(config) == 1:
        ((kind, value),) = config.items()
        arguments = (value,) if isinstance(value, str) else tuple(value or ())
    else:
        raise ConfigError(f"{what}: expected a reader name or a single-key object, got {config!r}")
    if kind not in known:
        raise ConfigError(f"unknown {what} reader {kind!r}; known: {sorted(known)}")
    return known[kind](*arguments)


def channel(spec: ChannelSpec | Mapping[str, Any]) -> ChannelSpec:
    """A channel spec, or the JSON object a manifest can carry one as."""
    if isinstance(spec, ChannelSpec):
        return spec
    from gantry.serial import spec_from_dict

    return spec_from_dict(spec)


def imported(target: str) -> Any:
    """Resolve ``"package.module:attribute"`` to the thing it names.

    How an environment gets named in a file. The import happens here, in the
    plugin, at build time — core still imports nothing it was not installed
    with, and a typo is a refusal with the name in it rather than a traceback
    from inside importlib.
    """
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ConfigError(f"{target!r} does not name an attribute; expected 'package.module:name'")
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ConfigError(f"cannot import {module_name!r}: {error}") from error
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ConfigError(f"{module_name!r} has no {attribute!r}") from error


# --------------------------------------------------------------------------
# dialect handling
# --------------------------------------------------------------------------


def accepts_seed(reset: Callable[..., Any]) -> bool:
    """Whether ``reset`` takes a ``seed`` keyword.

    Asked of the signature rather than discovered by calling and catching
    ``TypeError`` — an environment whose reset raises ``TypeError`` for its own
    reasons would otherwise be silently reset unseeded, and every seeded
    comparison built on it would be comparing noise.
    """
    try:
        signature = inspect.signature(reset)
    except (TypeError, ValueError):  # a builtin or C callable: just try it
        return True
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return True
    return "seed" in signature.parameters


def reset_env(env: Env, seed: int | None) -> tuple[Any, Mapping[str, Any]]:
    """Reset, accepting every reset signature in circulation."""
    if seed is None:
        result = env.reset()
    elif accepts_seed(env.reset):
        result = env.reset(seed=seed)
    else:
        # Older environments seed on the side, then reset.
        if callable(getattr(env, "seed", None)):
            env.seed(seed)
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], result[1]
    return result, {}


def read_step(result: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    """Unpack either dialect, or say precisely what arrived instead."""
    if not isinstance(result, tuple):
        raise ComponentError(f"step() returned {type(result).__name__}, expected a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), info or {}
    if len(result) == 4:
        observation, reward, done, info = result
        # The older dialect cannot tell a goal from a time limit. Recording it
        # as terminated is the reading that loses nothing: a horizon cut is
        # applied by this evaluator, which knows when it applied one.
        return observation, float(reward), bool(done), False, info or {}
    raise ComponentError(
        f"step() returned {len(result)} value(s); expected 5 "
        "(obs, reward, terminated, truncated, info) or 4 (obs, reward, done, info)"
    )


def as_channels(observation: Any, name: str = "observation") -> dict[str, np.ndarray]:
    """One observation as named arrays, whether it arrived as a dict or not."""
    if isinstance(observation, Mapping):
        return {str(key): np.asarray(value) for key, value in observation.items()}
    return {name: np.asarray(observation)}


def infer_spec(name: str, sample: np.ndarray) -> ChannelSpec:
    """Describe a channel from what can actually be observed, and no further.

    Shape and dtype are facts. Units, frame and meaning are not: this interface
    carries no place to put them, so nothing here invents any. A caller who
    knows what the numbers mean passes a spec that says so.
    """
    array = np.asarray(sample)
    shape = tuple(int(dimension) for dimension in array.shape)
    kind = KINDS.get(len(shape), "tensor")
    return ChannelSpec(name, kind, shape, str(array.dtype))


# --------------------------------------------------------------------------
# the evaluator
# --------------------------------------------------------------------------


@dataclass
class _Trial:
    observations: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    events: list[StageEvent] = field(default_factory=list)
    reached: set[str] = field(default_factory=set)
    success: bool | None = None
    terminated: bool = False
    truncated: bool = False

    def record(self, transition: Transition, action: np.ndarray) -> None:
        for name, value in transition.observation.items():
            self.observations.setdefault(name, []).append(value)
        self.actions.append(np.asarray(action))
        self.rewards.append(transition.reward)


class GymEvaluator(Evaluator):
    """Runs a policy against anything with ``reset`` and ``step``."""

    def __init__(
        self,
        factory: Callable[[], Env] | str,
        action: ChannelSpec | Mapping[str, Any],
        *,
        name: str = "gym",
        task: str = "gym",
        embodiment: str | None = None,
        readers: Readers | Mapping[str, Any] = Readers(),
        observations: Sequence[str] | None = None,
        observation_specs: Sequence[ChannelSpec | Mapping[str, Any]] = (),
        seedable: bool = True,
        scenes: int = 10,
        horizon: int = 200,
    ):
        """``observations`` narrows what is recorded; ``observation_specs``
        describes it properly where the caller knows more than the interface
        carries. Anything not described is inferred from the first observation,
        which yields shape and dtype and nothing else.

        Every argument that has to cross a manifest accepts plain data: the
        factory as ``"package.module:make_env"``, the channels as JSON, the
        readers as a small object.
        """
        action = channel(action)
        verdict = action.validate()
        if not verdict.ok:
            raise ConfigError(f"the declared action spec is not usable\n{verdict.explain()}")
        if scenes < 1:
            raise ValueError(f"a task needs at least one scene, got {scenes}")
        if isinstance(readers, Mapping):
            readers = Readers.from_config(readers)
        self._factory = imported(factory) if isinstance(factory, str) else factory
        self._action = action
        self._name = name
        self._task = task
        self._embodiment = embodiment
        self._readers = readers
        self._wanted = tuple(observations) if observations is not None else None
        self._declared = {(spec := channel(raw)).name: spec for raw in observation_specs}
        self._seedable = seedable
        self._scenes = scenes
        self._horizon = horizon
        self._env: Env | None = None

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            stage_events=self._readers.milestones is not None,
            outcomes=self._readers.success is not None,
            seedable=self._seedable,
            closed_loop=True,
            stages=list(self._readers.names),
        )

    def requires(self) -> Requirement:
        return requires_channels(
            self._name,
            "evaluation",
            self._action,
            description=f"an action this environment accepts, as {self._action.name!r}",
        )

    def task_for(
        self, name: str = "", scenes: int | None = None, horizon: int | None = None
    ) -> TaskSpec:
        count = self._scenes if scenes is None else scenes
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(Scene(id=f"scene-{index:03d}") for index in range(count)),
            horizon=self._horizon if horizon is None else horizon,
            stages=self._readers.names,
        )

    # -- the environment ---------------------------------------------------

    @property
    def env(self) -> Env:
        """Built once, on first use, and reused. Construction is often the
        expensive part, and a per-trial rebuild would dominate a short run."""
        if self._env is None:
            try:
                self._env = self._factory()
            except Exception as error:  # noqa: BLE001 - a world that will not build
                raise ConfigError(
                    f"{self._name}: the environment factory raised {type(error).__name__}: {error}"
                ) from error
            for method in ("reset", "step"):
                if not callable(getattr(self._env, method, None)):
                    raise ConfigError(
                        f"{self._name}: the environment has no callable {method!r}; "
                        "this bridge needs reset() and step()"
                    )
        return self._env

    def close(self) -> None:
        if self._env is not None and callable(getattr(self._env, "close", None)):
            self._env.close()
        self._env = None

    def _observation(self, raw: Any) -> dict[str, np.ndarray]:
        channels = as_channels(raw)
        if self._wanted is None:
            return channels
        missing = [name for name in self._wanted if name not in channels]
        if missing:
            raise ConfigError(
                f"{self._name}: asked to record {missing}, which the environment does "
                f"not observe; it observes {sorted(channels)}"
            )
        return {name: channels[name] for name in self._wanted}

    def _schema(self, trial: _Trial) -> tuple[ChannelSpec, ...]:
        specs = []
        for name, values in trial.observations.items():
            specs.append(self._declared.get(name) or infer_spec(name, values[0]))
        specs.append(self._action)
        specs.append(ChannelSpec(REWARD, "scalar", (), "float32", semantics="reward"))
        return tuple(specs)

    # -- running -----------------------------------------------------------

    def run(self, policy: Any, task: TaskSpec, protocol: RunProtocol) -> RunRecord:
        trials = [
            (index, scene, epoch)
            for index, scene in enumerate(task.scenes)
            for epoch in range(protocol.epochs)
        ]
        if protocol.max_trials is not None:
            trials = trials[: protocol.max_trials]

        episodes: list[EpisodeRecord] = []
        failures = 0
        for index, scene, epoch in trials:
            seed = scene.seed if scene.seed is not None else protocol.seed_for(index, epoch)
            trial, error = self._one(policy, scene, task, protocol, seed)
            failures += error is not None
            episodes.append(self._record(trial, scene, epoch, task, protocol, error))

        return RunRecord(
            provenance=self._provenance(policy, task, protocol, failures),
            episodes=tuple(episodes),
            metrics=self._metrics(episodes),
        )

    def _one(
        self,
        policy: Any,
        scene: Scene,
        task: TaskSpec,
        protocol: RunProtocol,
        seed: int,
    ) -> tuple[_Trial, str | None]:
        trial = _Trial()
        raw, info = reset_env(self.env, seed)
        observation = self._observation(raw)
        policy.reset(EpisodeContext(scene.id, scene.instruction, seed))

        step = 0
        while step < task.horizon:
            try:
                chunk = np.asarray(policy.act(Observation(step, dict(observation))))
            except Exception as error:  # noqa: BLE001 - one trial, not the run
                return trial, f"{type(error).__name__}: {error}"
            if chunk.ndim != len(self._action.shape) + 1 or len(chunk) < 1:
                raise ComponentError(
                    f"{getattr(policy, 'name', 'policy')} returned a chunk of shape "
                    f"{chunk.shape}; expected (horizon, *{self._action.shape})"
                )
            played = len(chunk) if protocol.execute is None else min(protocol.execute, len(chunk))
            for action in chunk[: max(1, played)]:
                if step >= task.horizon:
                    break
                transition = self._step(action, step)
                trial.record(transition, action)
                self._read(trial, transition)
                observation = transition.observation
                step += 1
                if transition.done:
                    trial.terminated = transition.terminated
                    trial.truncated = transition.truncated
                    return trial, None
        trial.truncated = True
        return trial, None

    def _step(self, action: np.ndarray, step: int) -> Transition:
        raw, reward, terminated, truncated, next_info = read_step(self.env.step(action))
        return Transition(
            step=step,
            observation=self._observation(raw),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=next_info,
        )

    def _read(self, trial: _Trial, transition: Transition) -> None:
        if self._readers.success is not None:
            outcome = self._readers.success(transition)
            if outcome is not None:
                # Sticky: a suite reporting success on the step it happens must
                # not have it erased by the steps after it.
                trial.success = bool(outcome) or bool(trial.success)
        if self._readers.milestones is not None:
            for name in self._readers.milestones(transition):
                if name in trial.reached:
                    continue
                trial.reached.add(name)
                trial.events.append(StageEvent(name, transition.step))

    def _record(
        self,
        trial: _Trial,
        scene: Scene,
        epoch: int,
        task: TaskSpec,
        protocol: RunProtocol,
        error: str | None,
    ) -> EpisodeRecord:
        if not trial.actions:
            # A trial that failed before its first step is a record of that,
            # not an episode to be quietly dropped from the denominator.
            return self._empty(scene, epoch, task, protocol, error)
        arrays: dict[str, np.ndarray] = {
            name: np.asarray(values) for name, values in trial.observations.items()
        }
        arrays[self._action.name] = np.asarray(trial.actions, dtype=self._action.dtype)
        arrays[REWARD] = np.asarray(trial.rewards, dtype="float32")
        return episode_from_arrays(
            arrays,
            self._schema(trial),
            id=f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id,
            source=task.name,
            labels=EpisodeLabels(
                success=None if error else trial.success,
                stage_events=tuple(trial.events),
                annotations={
                    "epoch": epoch,
                    "steps": len(trial.actions),
                    "return": float(sum(trial.rewards)),
                    "terminated": trial.terminated,
                    "truncated": trial.truncated,
                    **({"error": error} if error else {}),
                },
            ),
            task=task.name,
            embodiment=self._embodiment,
        )

    def _empty(
        self,
        scene: Scene,
        epoch: int,
        task: TaskSpec,
        protocol: RunProtocol,
        error: str | None,
    ) -> EpisodeRecord:
        from gantry.spine import episode_from_labels

        return episode_from_labels(
            id=f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id,
            source=task.name,
            labels=EpisodeLabels(
                success=None,
                annotations={"epoch": epoch, "steps": 0, **({"error": error} if error else {})},
            ),
            task=task.name,
            embodiment=self._embodiment,
        )

    def _provenance(
        self, policy: Any, task: TaskSpec, protocol: RunProtocol, failures: int
    ) -> Provenance:
        return Provenance(
            components=(
                policy.descriptor().component_ref(),
                self.descriptor().component_ref(),
            ),
            protocol={
                "task": task.name,
                "horizon": task.horizon,
                "stages": list(task.stages or self._readers.names),
                **protocol.as_dict(),
            },
            notes=(f"{failures} trial(s) failed to complete",) if failures else (),
        )

    def _metrics(self, episodes: Sequence[EpisodeRecord]) -> dict[str, Measurement]:
        scored = [e for e in episodes if e.labels.success is not None]
        returns = [
            float(e.labels.annotations.get("return", 0.0))
            for e in episodes
            if "return" in e.labels.annotations
        ]
        metrics: dict[str, Measurement] = {}
        if scored:
            successes = sum(1 for e in scored if e.labels.success)
            metrics["success_rate"] = Measurement(
                successes / len(scored),
                n=len(scored),
                units="fraction",
                method=f"as read by {self._name}",
            )
        if returns:
            metrics["mean_return"] = Measurement(
                float(np.mean(returns)), n=len(returns), method="mean total reward per trial"
            )
        return metrics


def channels_of(env: Env, seed: int | None = 0) -> tuple[ChannelSpec, ...]:
    """Describe what an environment observes, by resetting it once.

    A convenience for wiring: it says what the channels are called and what
    shape they are, which is what a caller needs before declaring anything
    better about them.
    """
    raw, _ = reset_env(env, seed)
    return tuple(infer_spec(name, value) for name, value in as_channels(raw).items())
