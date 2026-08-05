"""Plane: what is being attempted, described so anything can stage it.

A task here names no robot and no simulator. It says which objects are involved,
where they start, what a person is being asked to do, and how anyone -- a physics
engine or a human watching a video -- decides whether it was done.

Why it is its own plane
-----------------------
It used to live inside the evaluator, which meant a task could not be authored
once and reused: the world invented its own tasks, so "the same task on a
different arm" and "the same task on real hardware" were both impossible to
state. The evaluator's job is to *stage* a task, not to be the only place one
can exist.

Why the success criterion is written twice
------------------------------------------
This is the part that decides whether an evaluation survives contact with real
hardware. In simulation success is a pose check -- free, exact, unlimited. On a
real bench it is a person watching, and people disagree with each other about
ambiguous criteria far more than they disagree with a physics engine.

So a criterion carries both: a machine-checkable form and a **rubric** written
for a human. They are the same claim in two languages, and holding them side by
side is what lets a task defined today be scored on a robot nobody has bought
yet -- and what lets the two be calibrated against each other on video, which is
work that needs no hardware at all.

A criterion with no rubric is refused. A sim-only success criterion produces
numbers that cannot be reproduced by anyone without your simulator, which is the
opposite of a benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..spine import Descriptor, Verdict

TASK_CONTRACT = "task@1.0"

#: Which criterion argument keys name objects the task places.
#:
#: Singular and plural forms of the same role, because "put the nut on the peg"
#: and "put each nut on its peg" are the same kind of claim and should not need
#: two vocabularies. A world staging a criterion needs this answer to map the
#: task's object ids onto its own, and an auditor needs it to catch a criterion
#: that refers to something the task never places -- so it belongs to the
#: criterion rather than to either of them separately.
OBJECT_ARGS = frozenset({"object", "objects", "target", "targets", "into", "onto"})

#: How many tasks this source offers.
CAP_TASKS = "tasks"
#: Whether every criterion carries a human-scorable rubric.
CAP_RUBRICS = "rubrics"


@dataclass(frozen=True)
class Region:
    """Where an object may start, as a rectangle on a surface.

    A rectangle rather than a point because a task with one fixed layout
    measures memorisation. And a rectangle rather than "anywhere" because a
    person has to be able to reproduce it: these are the numbers that become
    marks on a printed mat when this task moves to a real bench.
    """

    surface: str
    x: tuple[float, float]
    y: tuple[float, float]
    yaw: tuple[float, float] | None = None

    def validate(self) -> Verdict:
        checks = []
        for axis, span in (("x", self.x), ("y", self.y)):
            if span[1] < span[0]:
                checks.append(
                    Verdict.no(
                        "region.inverted",
                        f"{self.surface}: {axis} runs from {span[0]} to {span[1]}",
                    )
                )
        return Verdict.all(checks)

    @property
    def fixed(self) -> bool:
        return self.x[0] == self.x[1] and self.y[0] == self.y[1]


@dataclass(frozen=True)
class Thing:
    """One object the task involves.

    ``kind`` is a name from whatever asset vocabulary you are using, and this
    plane deliberately does not adjudicate it -- a simulator resolves it to a
    mesh, a shopping list resolves it to something you can buy. What matters
    here is that the same name means the same object in both.
    """

    id: str
    kind: str
    start: Region | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Criterion:
    """When the task counts as done, said twice.

    ``check`` is the machine-readable claim -- a predicate name and its
    arguments, which a world implements however it can. ``rubric`` is the same
    claim in a sentence, precise enough that two people watching the same video
    agree. Both are required.
    """

    check: str
    args: Mapping[str, Any] = field(default_factory=dict)
    rubric: str = ""

    def objects(self) -> tuple[str, ...]:
        """Every object this criterion names, in order, without duplicates.

        A value may be one name or several: ``{"object": "cube"}`` and
        ``{"objects": ["milk", "bread"]}`` both name objects, and a caller
        should not have to know which form a given check used. Anything that
        is not a string is not a name -- ``{"height": 0.04}`` says nothing about
        which object is involved.
        """
        found: list[str] = []
        for key, value in self.args.items():
            if key not in OBJECT_ARGS:
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                if isinstance(item, str) and item not in found:
                    found.append(item)
        return tuple(found)

    def validate(self) -> Verdict:
        checks = []
        if not self.check:
            checks.append(Verdict.no("criterion.no_check", "a criterion names no predicate"))
        if not self.rubric.strip():
            checks.append(
                Verdict.no(
                    "criterion.no_rubric",
                    f"criterion {self.check!r} has no rubric",
                    hint="a criterion only a simulator can evaluate produces numbers "
                    "nobody without that simulator can reproduce",
                )
            )
        elif len(self.rubric.split()) < 5:
            checks.append(
                Verdict.note(
                    "criterion.terse_rubric",
                    f"criterion {self.check!r} has a {len(self.rubric.split())}-word rubric",
                    hint="two people should reach the same verdict from it; test that "
                    "on video before trusting it",
                )
            )
        return Verdict.all(checks)


@dataclass(frozen=True)
class TaskDefinition:
    """One task, named, staged and judged -- with no robot in sight."""

    name: str
    instruction: str
    things: tuple[Thing, ...] = ()
    success: tuple[Criterion, ...] = ()
    horizon: int = 300
    trials: int = 20
    surfaces: tuple[str, ...] = ("table",)
    #: How a particular world stages this task, keyed by that world's name.
    #: Namespaced so several can coexist and none needs to know about the others.
    staging: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> Verdict:
        checks = [
            *(thing.start.validate() for thing in self.things if thing.start),
            *(criterion.validate() for criterion in self.success),
        ]
        if not self.instruction.strip():
            checks.append(
                Verdict.no("task.no_instruction", f"{self.name!r} tells nobody what to do")
            )
        if not self.success:
            checks.append(
                Verdict.no(
                    "task.no_success",
                    f"{self.name!r} defines no success criterion, so nothing it produces "
                    "can be scored",
                )
            )
        ids = [thing.id for thing in self.things]
        repeated = {i for i in ids if ids.count(i) > 1}
        if repeated:
            checks.append(
                Verdict.no("task.duplicate_object", f"{self.name!r} repeats {sorted(repeated)}")
            )
        unknown = [
            thing.start.surface
            for thing in self.things
            if thing.start and thing.start.surface not in self.surfaces
        ]
        if unknown:
            checks.append(
                Verdict.no(
                    "task.unknown_surface",
                    f"{self.name!r} places objects on {sorted(set(unknown))}, which it "
                    f"does not declare; it declares {list(self.surfaces)}",
                )
            )
        if self.horizon < 1 or self.trials < 1:
            checks.append(
                Verdict.no(
                    "task.empty_protocol",
                    f"{self.name!r} asks for {self.trials} trial(s) of {self.horizon} step(s)",
                )
            )
        return Verdict.all(checks)

    @property
    def rubrics(self) -> tuple[str, ...]:
        """What a person is asked to judge. The artifact that moves to hardware."""
        return tuple(criterion.rubric for criterion in self.success if criterion.rubric)

    def staged_by(self, world: str) -> Mapping[str, Any] | None:
        return self.staging.get(world)


def task_descriptor(
    name: str, version: str, *, tasks: int, rubrics: bool, **metadata: Any
) -> Descriptor:
    return Descriptor(
        plane="task",
        name=name,
        version=version,
        contract=TASK_CONTRACT,
        provides={CAP_TASKS: tasks, CAP_RUBRICS: rubrics},
        metadata=metadata,
    )


class TaskSource(ABC):
    """Somewhere tasks come from: a directory of files, a suite, a generator."""

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Identity and capabilities, read before anything is asked of it."""

    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Every task this source offers."""

    @abstractmethod
    def task(self, name: str) -> TaskDefinition:
        """One task, by name."""

    # -- provided ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.descriptor().name

    def __len__(self) -> int:
        return len(self.names())

    def __iter__(self):
        for name in self.names():
            yield self.task(name)

    def validate(self) -> Verdict:
        """Every task this source offers, checked."""
        return Verdict.all(self.task(name).validate() for name in self.names())

    def staged_by(self, world: str) -> tuple[str, ...]:
        """Which of these tasks a given world knows how to set up."""
        return tuple(name for name in self.names() if self.task(name).staged_by(world) is not None)
