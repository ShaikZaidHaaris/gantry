"""The closed-loop trial, written once.

Every simulator-backed evaluator in this project independently grew the same
hundred and fifty lines: build the world, reset it on a scene, ask the policy for
an action chunk, play some prefix of it, read whether the thing succeeded,
accumulate arrays, assemble an episode, tally a success rate. Four plugins, four
copies, four places for the same off-by-one to live — and the copies had already
begun to disagree about whether an observation is recorded before or after the
step that follows it.

That disagreement is not cosmetic. ``(observation, action)`` pairs offset by one
step are a silently corrupted dataset, and a policy trained on episodes exported
from the wrong copy would be learning to predict the action it *already* took.

So the loop lives here and the suites supply only what is genuinely theirs
-------------------------------------------------------------------------
A :class:`World` is three methods: start a scene, take an action, stop. That is
the entire surface a benchmark suite has to present, and it is small enough that
adding a suite is a morning rather than a project. Everything downstream of it —
chunk handling, horizons, event de-duplication, schema inference, provenance,
the success rate and its interval — is identical across suites and is written
once, here.

What this is not
----------------
Not a base class that knows about any particular simulator. Nothing in this
module imports anything a robot needs; it will happily drive a dictionary. That
is deliberate: the moment core can name a physics engine, "no plane gives
preference to any implementation" has stopped being true, and the test that
enforces it caught an earlier draft of this very paragraph.

Not the duck-typed step/reset bridge either — that one lives in a plugin, adapts
whatever it is handed, and guesses dialects and success conventions. This asks a
suite to say plainly what it means, which is the right shape when someone is
writing an adapter on purpose rather than wrapping an environment they were given.

Held conventions
----------------
1. An observation is recorded, then the action taken from it. ``arrays["action"]
   [t]`` is what the policy did having seen step ``t``.
2. A policy raising fails that trial and no other. A world raising halts the run,
   because a broken simulator makes every subsequent trial meaningless.
3. Success is whatever the world says, including ``None`` for "not established".
   A trial that ran out of horizon is not a failure unless the suite says so.
4. A milestone is recorded the first time it is reached and not again.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts.evaluator import Evaluator, Scene, TaskSpec
from .contracts.evaluator import Protocol as RunProtocol
from .contracts.policy import EpisodeContext, Observation
from .errors import ComponentError, ConfigError
from .resolve import Requirement, requires_channels
from .spine import (
    ChannelSpec,
    EpisodeLabels,
    EpisodeRecord,
    Measurement,
    Provenance,
    RunRecord,
    StageEvent,
    episode_from_arrays,
    episode_from_labels,
    proportion,
)

#: Name given to the reward channel, where a world reports one.
REWARD = "reward"


@dataclass(frozen=True)
class Step:
    """What a world reports back after one action.

    ``success`` is tri-state on purpose. A suite with a success predicate answers
    ``True``/``False``; one that only knows at the end answers ``None`` until it
    does; one with no notion of success at all answers ``None`` throughout and
    declares ``outcomes=False`` in its descriptor, so an analysis that needs
    outcomes is refused rather than handed zeros.
    """

    observation: Mapping[str, Any]
    reward: float = 0.0
    done: bool = False
    success: bool | None = None
    #: Milestones reached as of this step, by name. Repeats are ignored.
    reached: tuple[str, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class World(Protocol):
    """One scene of one suite, mid-attempt.

    Stateful by nature — ``advance`` depends on every ``advance`` before it — so
    a world is obtained per scene rather than shared. Suites for which
    construction is expensive return the same cached object with fresh state;
    that is the adapter's business and not visible here.
    """

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        """Put the world into this scene's starting condition; first observation."""

    def advance(self, action: np.ndarray) -> Step:
        """Apply one action."""

    def close(self) -> None:
        """Release whatever was held. Called once per run, not per trial."""


@dataclass
class Trial:
    """One attempt, accumulating."""

    channels: dict[str, list[np.ndarray]] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    events: list[StageEvent] = field(default_factory=list)
    success: bool | None = None
    steps: int = 0
    truncated: bool = False
    error: str | None = None

    @property
    def reached(self) -> tuple[str, ...]:
        return tuple(event.name for event in self.events)

    def note(self, reached: Sequence[str]) -> None:
        seen = set(self.reached)
        for name in reached:
            if name not in seen:
                self.events.append(StageEvent(name=str(name), step=self.steps))
                seen.add(str(name))

    def observe(self, observation: Mapping[str, Any], keep: Sequence[str] | None) -> None:
        for key, value in observation.items():
            if keep is not None and key not in keep:
                continue
            array = np.asarray(value)
            # Strings, bytes and nested dicts are things suites put in
            # observations that are not channels. A language instruction becomes
            # a '<U16' array rather than an object one, so the test is on dtype
            # kind: O(bject), U(nicode), S(bytes), V(oid).
            if array.dtype.kind in "OUSV":
                continue
            self.channels.setdefault(key, []).append(array)


def imported(target: str) -> Any:
    """Resolve ``"package.module:name"`` to the thing it names.

    Here because every suite adapter needs it for the same reason: a manifest is
    JSON, a world factory is code, and the only way to name the second from the
    first is a string. Kept strict — a target that does not resolve is a
    :class:`~gantry.errors.ConfigError` naming what was tried, not an
    ``AttributeError`` from inside somebody's ``__init__``.
    """
    if ":" not in target:
        raise ConfigError(
            f"{target!r} does not name something importable; expected 'package.module:name'"
        )
    module_name, _, attribute = target.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise ConfigError(f"cannot import {module_name!r}: {error}") from error
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ConfigError(f"{module_name!r} has no {attribute!r}") from error


def infer_spec(name: str, sample: np.ndarray, *, rate_hz: float | None = None) -> ChannelSpec:
    """A channel description from one observation, and no more than is really known.

    Shape and dtype are genuine; ``kind`` is a guess from rank that happens to be
    right for the cases that occur (an HxWx3 array is an image, a 1-D array is a
    vector). Semantics are deliberately absent — this function cannot know
    whether a seven-vector is a joint position or a pose delta, and inventing an
    answer is how a mislabelled channel gets adapted into the wrong slot.
    """
    array = np.asarray(sample)
    shape = tuple(int(dimension) for dimension in array.shape)
    if array.ndim == 3 and shape[-1] in (1, 3, 4):
        kind = "image"
    elif array.ndim == 0:
        kind = "scalar"
    else:
        kind = "vector"
    return ChannelSpec(name, kind, shape, str(array.dtype), rate_hz=rate_hz)


class ClosedLoop(Evaluator):
    """An evaluator over a :class:`World`. Supplies ``run``; you supply the world.

    A subclass owes four things: a ``descriptor`` stating what it can report, a
    ``task_for`` listing the scenes the suite offers, an ``action`` spec saying
    what the policy must emit, and ``world_for`` to build one. Everything else is
    inherited, which means a new suite cannot get the trial loop subtly wrong
    because it does not write one.
    """

    #: Recorded channels, or ``None`` for everything the world observes. A suite
    #: with forty observation keys and two cameras worth keeping sets this.
    observes: tuple[str, ...] | None = None

    #: Control rate, where the suite has a fixed one. Goes on every channel spec,
    #: because a chunk of eight actions means something different at 5 Hz and 50.
    rate_hz: float | None = None

    #: Which body, for the episode's metadata. Overridden per scene by suites
    #: that host more than one.
    embodiment: str = "unknown"

    # -- what a subclass supplies -----------------------------------------

    @abstractmethod
    def action(self) -> ChannelSpec:
        """What the policy must emit for this suite to accept it."""

    @abstractmethod
    def world_for(self, scene: Scene) -> World:
        """A world in this scene's starting condition, ready for ``begin``."""

    def embodiment_for(self, scene: Scene) -> str:
        """Which body this scene runs. Constant unless the suite varies it."""
        return self.embodiment

    # -- provided ----------------------------------------------------------

    def requires(self) -> Requirement:
        spec = self.action()
        return requires_channels(
            self.descriptor().name,
            "evaluation",
            spec,
            description=f"an action this world accepts, as {spec.name!r}",
        )

    def close(self) -> None:
        """Release held worlds. A subclass that caches them overrides this."""

    def run(self, policy: Any, task: TaskSpec, protocol: RunProtocol) -> RunRecord:
        trials = [(scene, epoch) for scene in task.scenes for epoch in range(protocol.epochs)]
        if protocol.max_trials is not None:
            trials = trials[: protocol.max_trials]

        episodes: list[EpisodeRecord] = []
        failures = 0
        for index, (scene, epoch) in enumerate(trials):
            seed = scene.seed if scene.seed is not None else protocol.seed_for(index, epoch)
            trial = self.play(policy, scene, task, protocol, seed)
            failures += trial.error is not None
            episodes.append(self.assemble(trial, scene, epoch, task, protocol))

        return RunRecord(
            provenance=self.provenance(policy, task, protocol, failures),
            episodes=tuple(episodes),
            metrics=self.metrics(episodes),
        )

    def play(
        self,
        policy: Any,
        scene: Scene,
        task: TaskSpec,
        protocol: RunProtocol,
        seed: int | None = None,
    ) -> Trial:
        """One attempt at one scene.

        The chunk arithmetic is the part worth reading. A policy returns several
        actions at once; ``protocol.execute`` says how many of them to actually
        play before asking again. That number changed a measured success rate by
        fourteen points in this project's own benchmark, which is why it is a
        protocol field in the record rather than a default someone chose.
        """
        world = self.world_for(scene)
        trial = Trial()
        observation = world.begin(scene)
        policy.reset(EpisodeContext(scene.id, scene.instruction, seed))
        expected = self.action().shape

        while trial.steps < task.horizon:
            try:
                chunk = np.asarray(policy.act(Observation(trial.steps, dict(observation))))
            except Exception as error:  # noqa: BLE001 - one trial, not the run
                trial.error = f"{type(error).__name__}: {error}"
                return trial
            # The width is checked as well as the rank. A policy emitting
            # six-wide actions into a seven-wide world passes a rank check and
            # then feeds the world a truncated command every step, which is a
            # run of plausible failures rather than an error.
            if (
                chunk.ndim != len(expected) + 1
                or len(chunk) < 1
                or tuple(int(d) for d in chunk.shape[1:]) != tuple(expected)
            ):
                raise ComponentError(
                    f"{getattr(policy, 'name', 'policy')} returned a chunk of shape "
                    f"{chunk.shape}; expected (horizon, *{expected})"
                )
            played = len(chunk) if protocol.execute is None else min(protocol.execute, len(chunk))
            for action in chunk[: max(1, played)]:
                if trial.steps >= task.horizon:
                    break
                # Observation first, then the action taken from it. Reversing
                # these is the corruption this module exists to make impossible.
                trial.observe(observation, self.observes)
                trial.actions.append(np.asarray(action, dtype="float32"))
                step = world.advance(np.asarray(action, dtype=float))
                trial.rewards.append(float(step.reward))
                trial.steps += 1
                trial.note(step.reached)
                observation = step.observation
                if step.success is not None:
                    trial.success = bool(step.success)
                if step.success or step.done:
                    return trial
        trial.truncated = True
        return trial

    # -- assembling --------------------------------------------------------

    def schema_for(self, trial: Trial) -> tuple[ChannelSpec, ...]:
        """Channel specs for what was recorded.

        Inferred from the first sample of each channel, with the suite's own
        declarations winning where it made any. A suite that knows a channel is
        joint positions should say so in ``declares``; nothing here can work it
        out, and semantics are what the adapter plane matches on.
        """
        declared = self.declares()
        specs: list[ChannelSpec] = []
        for name, values in trial.channels.items():
            specs.append(declared.get(name) or infer_spec(name, values[0], rate_hz=self.rate_hz))
        specs.append(self.action())
        specs.append(ChannelSpec(REWARD, "scalar", (), "float32", semantics="reward"))
        return tuple(specs)

    def declares(self) -> Mapping[str, ChannelSpec]:
        """Channels this suite describes properly rather than leaving to inference."""
        return {}

    def assemble(
        self,
        trial: Trial,
        scene: Scene,
        epoch: int,
        task: TaskSpec,
        protocol: RunProtocol,
    ) -> EpisodeRecord:
        identifier = f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id
        annotations: dict[str, Any] = {
            "epoch": epoch,
            "steps": trial.steps,
            "instruction": scene.instruction,
            "truncated": trial.truncated,
            **({"error": trial.error} if trial.error else {}),
        }
        labels = EpisodeLabels(
            success=None if trial.error else trial.success,
            stage_events=tuple(trial.events),
            annotations=annotations,
        )
        if not trial.actions:
            return episode_from_labels(
                id=identifier,
                source=task.name,
                labels=labels,
                task=task.name,
                embodiment=self.embodiment_for(scene),
            )
        arrays: dict[str, np.ndarray] = {
            name: np.asarray(values) for name, values in trial.channels.items()
        }
        arrays[self.action().name] = np.asarray(trial.actions, dtype="float32")
        arrays[REWARD] = np.asarray(trial.rewards[: len(trial.actions)], dtype="float32")
        return episode_from_arrays(
            arrays,
            self.schema_for(trial),
            id=identifier,
            source=task.name,
            labels=labels,
            task=task.name,
            embodiment=self.embodiment_for(scene),
        )

    def provenance(
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
                "stages": list(task.stages),
                **protocol.as_dict(),
            },
            notes=(f"{failures} trial(s) failed to complete",) if failures else (),
        )

    def metrics(self, episodes: Sequence[EpisodeRecord]) -> dict[str, Measurement]:
        """Success rate over the trials that established one.

        Abstentions are dropped rather than counted as failures — the same rule
        the scorer plane uses, for the same reason: a trial whose policy crashed
        is missing evidence, and folding it in as a loss reports a policy as
        worse than measured on the strength of a bug in the harness.
        """
        scored = [episode for episode in episodes if episode.labels.success is not None]
        if not scored:
            return {}
        wins = sum(1 for episode in scored if episode.labels.success)
        return {"success_rate": proportion(wins, len(scored))}
