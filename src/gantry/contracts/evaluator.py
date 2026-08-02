"""Plane 4: running a policy against something and recording what happened.

One interface covers three genuinely different situations, which is the point:

**Closed loop.** A simulator or a real machine. The policy acts, the world
responds, the next observation depends on what it did.

**Offline.** No world at all. Recorded observations are replayed to the policy
and its predictions are compared against what was actually done. Cheap, needs no
simulator, and honest about what it cannot tell you: a policy that predicts the
right action at every recorded state may still fail the moment its own error
puts it somewhere the recording never went.

**Real.** A person resets the scene, watches, and says what happened. The
interface is identical; everything hard about it is operational.

They differ in what they can *report*, not in how they are called, so an
evaluator declares its capabilities and the resolver refuses analyses it cannot
feed. That is why the offline evaluator saying "no stage events" is a feature:
without it, a funnel over an offline run returns an empty table that looks like
a finished analysis.

Protocol is a first-class input, not a setting. Chunk size alone moved a
measured policy by fourteen points in this project's own benchmark, so it lives
in the record with everything else that changes results.

Invariants
----------
1. ``run`` returns a :class:`~gantry.spine.run.RunRecord` whose provenance names
   the policy, this evaluator, and the protocol used.
2. One episode per trial, and a task run over several epochs yields one episode
   per attempt rather than one summary per scene.
3. Declared capabilities are true: an evaluator claiming stage events emits
   them, and one denying them does not.
4. An evaluator declaring ``seedable`` returns identical records for identical
   seeds, which is what a paired comparison rests on.
5. Failures follow :mod:`gantry.errors`: a policy that raises fails one trial,
   a faulted world halts the run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..errors import ConfigError
from ..resolve.requirement import Requirement
from ..spine import Descriptor, RunRecord, Verdict

#: 1.1 made ``task_for`` a required method. It had been a duck-typed hook the
#: runner probed for, which meant an evaluator could pass every conformance
#: check and still be unusable from a manifest -- the kit said yes and the runner
#: refused. A minor bump is the right shape: an evaluator predating it is
#: refused by name with ``contract.minor`` rather than failing later.
EVALUATOR_CONTRACT = "evaluator@1.1"

#: This evaluator's world comes from the data under test rather than from its
#: own configuration, so it must be bound to a cohort before it can run.
#:
#: The distinction is real and not a convenience. A simulator brings its own
#: scenes and is the same world whichever dataset you point at it. An offline
#: replay's world *is* the recording, and a sim rebuilt from a dataset's own
#: ``env_args`` is the world that dataset came from. Without somewhere to say
#: which kind it is, a manifest cannot build the second kind at all -- every
#: plane is constructed independently, so an evaluator that needs the dataset
#: would have to reach for it, and reaching across planes is the thing this
#: design exists to prevent.
CAP_NEEDS_DATASET = "needs_dataset"

#: This evaluator can host different bodies -- the same task, run on a different
#: robot, without a different evaluator.
#:
#: Distinct from merely naming an embodiment. Most evaluators have one body
#: welded in: a recording has whatever arm recorded it, a gym env is whatever it
#: was written as. An evaluator that says yes here can be handed a machine
#: description and rebuild its world around it, which is what makes "vary the
#: embodiment" a real axis rather than a manifest key nothing reads.
CAP_HOSTS_EMBODIMENT = "hosts_embodiment"

#: Emits milestones, so a funnel diagnosis is possible.
CAP_STAGE_EVENTS = "stage_events"
#: Emits a success label per trial.
CAP_OUTCOMES = "outcomes"
#: Identical seeds give identical records.
CAP_SEEDABLE = "seedable"
#: The world responds to what the policy did. False for offline replay.
CAP_CLOSED_LOOP = "closed_loop"


@dataclass(frozen=True)
class Scene:
    """One thing to attempt, and the starting conditions for it."""

    id: str
    instruction: str | None = None
    seed: int | None = None
    setup: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """What to evaluate, independent of who evaluates it.

    ``stages`` names the milestones a scorer is expected to report. Declaring
    them here rather than discovering them from output means a task can state
    that a funnel is *supposed* to be possible, and an evaluator that cannot
    supply them is caught at planning time.
    """

    name: str
    scenes: tuple[Scene, ...] = ()
    horizon: int = 100
    stages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scenes)

    def validate(self) -> Verdict:
        checks = []
        if not self.scenes:
            checks.append(Verdict.no("task.no_scenes", f"task {self.name!r} defines no scenes"))
        ids = [scene.id for scene in self.scenes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            checks.append(
                Verdict.no(
                    "task.duplicate_scene",
                    f"task {self.name!r} repeats scene id(s) {sorted(duplicates)}",
                )
            )
        if self.horizon < 1:
            checks.append(
                Verdict.no("task.horizon", f"task {self.name!r} has horizon {self.horizon}")
            )
        return Verdict.all(checks)


@dataclass(frozen=True)
class Protocol:
    """How to run the task. Every field here can change the answer.

    ``execute`` is how many actions of each predicted chunk are carried out
    before the policy is asked again. It is the single cheapest lever on a
    result and the one most often left implicit.
    """

    epochs: int = 1
    execute: int | None = None
    seed_base: int = 0
    max_trials: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "execute": self.execute,
            "seed_base": self.seed_base,
            "max_trials": self.max_trials,
            **dict(self.extra),
        }

    def seed_for(self, scene_index: int, epoch: int) -> int:
        """A seed per trial, derived so a rerun lands on the same conditions."""
        return self.seed_base + scene_index * 1000 + epoch

    def validate(self) -> Verdict:
        checks = []
        if self.epochs < 1:
            checks.append(Verdict.no("protocol.epochs", f"epochs={self.epochs} runs nothing"))
        if self.execute is not None and self.execute < 1:
            checks.append(
                Verdict.no("protocol.execute", f"execute={self.execute} executes nothing")
            )
        return Verdict.all(checks)


def evaluator_descriptor(
    name: str,
    version: str,
    *,
    stage_events: bool,
    outcomes: bool,
    seedable: bool,
    closed_loop: bool,
    needs_dataset: bool = False,
    hosts_embodiment: bool = False,
    isolation: str = "in-process",
    **metadata: Any,
) -> Descriptor:
    return Descriptor(
        plane="evaluation",
        name=name,
        version=version,
        contract=EVALUATOR_CONTRACT,
        provides={
            CAP_STAGE_EVENTS: stage_events,
            CAP_OUTCOMES: outcomes,
            CAP_SEEDABLE: seedable,
            CAP_CLOSED_LOOP: closed_loop,
            CAP_NEEDS_DATASET: needs_dataset,
            CAP_HOSTS_EMBODIMENT: hosts_embodiment,
        },
        isolation=isolation,
        metadata=metadata,
    )


class Evaluator(ABC):
    """Runs a policy against a task and returns what happened."""

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Identity and capabilities, read before anything is asked of it."""

    @abstractmethod
    def requires(self) -> Requirement:
        """What this evaluator needs from a policy paired with it."""

    @abstractmethod
    def task_for(self, name: str, **options: Any) -> TaskSpec:
        """A task this evaluator can actually run, built by the evaluator itself.

        Only the evaluator knows what its scenes are: an offline evaluator's are
        its held-out episodes, a simulator's are its layouts, a real rig's are
        whatever a person can set up. Nothing else can construct one, so nothing
        else should try.

        Required rather than optional because a runner has no other way to start
        a run. When this was a hook the runner probed for, an evaluator could
        satisfy every check in the conformance kit and still be unusable from a
        manifest -- which is the exact gap between "conforms" and "works" that
        contracts exist to close.
        """

    @abstractmethod
    def run(self, policy: Any, task: TaskSpec, protocol: Protocol) -> RunRecord:
        """Evaluate the policy and return one record of what happened."""

    # -- provided ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.descriptor().name

    @property
    def needs_dataset(self) -> bool:
        return bool(self.descriptor().provides.get(CAP_NEEDS_DATASET, False))

    def bind(self, episodes: Sequence[Any]) -> "Evaluator":
        """Return an evaluator for *this* cohort's data.

        Only meaningful when the descriptor says ``needs_dataset``. The default
        returns ``self``, because a simulator is the same world whichever
        dataset is under test and rebinding it would be a lie.

        Returns a new evaluator rather than mutating: a run compares cohorts, so
        two bound evaluators exist at once, and an evaluator that quietly
        rebound itself would score the second cohort against the first one's
        answer key.
        """
        return self

    @property
    def hosts_embodiment(self) -> bool:
        return bool(self.descriptor().provides.get(CAP_HOSTS_EMBODIMENT, False))

    def for_embodiment(self, embodiment: Any) -> "Evaluator":
        """Return an evaluator whose world is built around this machine.

        The default returns ``self``, because most worlds have one body welded
        in and quietly accepting a different one would be a lie: the run would
        report the machine it was handed and simulate the one it already had.

        Returns a new evaluator rather than mutating, for the same reason
        :meth:`bind` does: a cross-embodiment run holds several at once.
        """
        if not self.hosts_embodiment:
            raise ConfigError(
                f"{self.name} has one body welded into it and cannot host "
                f"{getattr(embodiment, 'name', embodiment)!r}; an evaluator that can "
                "declares hosts_embodiment"
            )
        return self

    def check_inputs(self, task: TaskSpec, protocol: Protocol) -> Verdict:
        checks = [task.validate(), protocol.validate()]
        if self.needs_dataset and not self.bound:
            checks.append(
                Verdict.no(
                    "evaluator.unbound",
                    f"{self.name} takes its world from the data under test and has "
                    "not been bound to any",
                    hint="call bind(episodes) first; a runner does this per cohort",
                )
            )
        return Verdict.all(checks)

    @property
    def bound(self) -> bool:
        """Whether this instance has the data it needs. True unless overridden."""
        return True

    def evaluate(self, policy: Any, task: TaskSpec, protocol: Protocol | None = None) -> RunRecord:
        """Check, then run. The path callers should use."""
        protocol = protocol or Protocol()
        self.check_inputs(task, protocol).raise_if_refused(f"cannot run {self.name}")
        return self.run(policy, task, protocol)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.descriptor().ref}>"
