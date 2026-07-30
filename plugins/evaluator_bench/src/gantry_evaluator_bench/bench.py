"""A real machine as an evaluator, with a person in the loop.

This is the one the rest of the design was for. Regions became samplers so a
task-file rectangle could also be a printed mat; criteria were written twice so
a predicate and a sentence could be the same standard; scorers became a plane so
a person watching could be compared against a simulator checking a pose. All of
that only pays off if a run on hardware is the *same kind of object* as a run in
simulation — one ``RunRecord``, one provenance block, comparable by the same
feedback modules — and that is what this plugin is.

What is genuinely different about hardware
-----------------------------------------
**You cannot seed a table.** So this declares ``seedable=False``, and it means
it. Every paired analysis in this framework rests on "same seed, same scene", and
on a bench that is simply false: the cube is where a person put it, to within a
centimetre and their attention. Declaring otherwise would let a paired test run
over unrelated scenes and report a tight interval around a comparison that was
never made. The honest consequence is that hardware comparisons need more trials
than sim ones, and the power module will say so out loud.

**The scene is an instruction to a human.** ``Scene.setup`` carries the sentence
the operator reads before each attempt, and it is part of the record. A run where
half the attempts started from a different arrangement is not a run over one
task, and the only way anybody can ever check that is if the arrangement was
written down per scene rather than remembered.

**Success arrives at the end, from a person.** There is no ``check_success()``.
So the world implements ``verdict`` — core's hook for worlds that only know once
it is over — and the answer may be "I could not tell", which is recorded as an
abstention rather than a failure.

Two ways to get an outcome, and the second is usually right
----------------------------------------------------------
:class:`Attending` asks the operator after each attempt. Fine for twenty trials,
and it puts the judgement at the moment of best information — the person was
standing there.

:class:`Recording` never asks. Every trial comes back ``success=None`` and the
footage is labelled afterwards, through the scorer plane, by a person or a model
or both, against the rubric. Slower to a number but far better evidence: the
labels are reviewable, two judges can be compared on the same trials, and nobody
is deciding a borderline case while also trying to catch a falling cube. A run
scored this way declares ``outcomes=False``, so anything that needs outcomes is
refused until the labels exist rather than reading an empty column as zero.

Safety is the operator's, and this does not pretend otherwise
------------------------------------------------------------
``Machine`` is a protocol with two methods, satisfied by whatever driver already
works on that arm. This plugin does not implement kinematics, limits, collision
checks or an estimop — a framework that appeared to provide those while actually
just forwarding arrays would be worse than one that clearly does not. What it
does provide is the halt path: an operator can stop a trial, and ``halt`` on the
machine is called when they do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step, Trial
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: What the operator may answer. ``skip`` is not a verdict — it means the
#: attempt did not happen (the arm faulted, somebody walked through the cell) and
#: is recorded as an abstention with a note, not as a failure.
ANSWERS = {
    "y": True,
    "yes": True,
    "n": False,
    "no": False,
    "?": None,
    "unclear": None,
    "skip": None,
}


class Machine(Protocol):
    """Whatever is bolted to the bench, as two methods.

    Deliberately the smallest surface that can drive a robot. Anybody evaluating
    on hardware already has a working driver; asking them to reimplement it
    against a framework interface is how a plugin goes unused.
    """

    def observe(self) -> Mapping[str, Any]:
        """The current cameras and proprioception, by channel name."""

    def command(self, action: np.ndarray) -> None:
        """Send one action. Blocking or not is the driver's business."""

    def halt(self) -> None:
        """Stop moving. Called when an operator interrupts, and at trial end."""


class Operator(Protocol):
    """The person, as three methods a test can fake."""

    def prepare(self, scene: Scene) -> None:
        """Block until the scene is set up as described."""

    def interrupt(self) -> bool:
        """Whether to stop this attempt now. Polled; must not block."""

    def judge(self, scene: Scene, trial: Trial) -> bool | None:
        """Whether the attempt met the rubric. ``None`` for 'could not tell'."""


@dataclass
class Recording:
    """An operator who resets scenes and never judges.

    The default, and the one to use for anything that will be published. Every
    trial comes back as an abstention and the footage is labelled afterwards
    through the scorer plane, which means the labels exist as reviewable data
    rather than as a number somebody remembers agreeing with.
    """

    prompt: Callable[[str], str] = input
    #: Where the videos are, so the sheet can say what to label later.
    footage: str | None = None

    def prepare(self, scene: Scene) -> None:
        setup = scene.setup or scene.instruction or scene.id
        self.prompt(f"[{scene.id}] set up: {setup}\n  press enter when ready > ")

    def interrupt(self) -> bool:
        return False

    def judge(self, scene: Scene, trial: Trial) -> bool | None:
        return None


@dataclass
class Attending:
    """An operator who answers after each attempt.

    Honest about its cost: the judgement happens while the person is also
    running the cell, and it produces one bit with no rationale attached. Use it
    for a pilot, and for anything load-bearing use :class:`Recording` plus a
    scorer, where the same trials can be judged twice and the two judges
    compared.
    """

    prompt: Callable[[str], str] = input
    #: The rubric sentence, shown verbatim at the moment of judging. Not a
    #: paraphrase and not a reminder — the same text a scorer would be given, so
    #: that an operator's answers and a scorer's are answers to one question.
    rubric: str = ""

    def prepare(self, scene: Scene) -> None:
        setup = scene.setup or scene.instruction or scene.id
        self.prompt(f"[{scene.id}] set up: {setup}\n  press enter when ready > ")

    def interrupt(self) -> bool:
        return False

    def judge(self, scene: Scene, trial: Trial) -> bool | None:
        standard = f"\n  standard: {self.rubric}" if self.rubric else ""
        while True:
            answer = self.prompt(
                f"[{scene.id}] {trial.steps} steps.{standard}\n"
                "  met the standard? [y/n/? = could not tell/skip] > "
            )
            key = str(answer).strip().lower()
            if key in ANSWERS:
                return ANSWERS[key]


@dataclass
class Bench:
    """One machine and one operator, driven as a :class:`gantry.rollout.World`."""

    machine: Machine
    operator: Operator
    observes: tuple[str, ...] | None = None
    _scene: Scene | None = field(default=None, init=False)

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        self._scene = scene
        self.operator.prepare(scene)
        return self.machine.observe()

    def advance(self, action: np.ndarray) -> Step:
        self.machine.command(action)
        stop = bool(self.operator.interrupt())
        if stop:
            self.machine.halt()
        return Step(observation=self.machine.observe(), done=stop)

    def verdict(self, trial: Trial) -> bool | None:
        """Asked once the attempt is over, which on a bench is the only time
        anybody can answer."""
        self.machine.halt()
        assert self._scene is not None
        return self.operator.judge(self._scene, trial)

    def close(self) -> None:
        self.machine.halt()


class BenchEvaluator(ClosedLoop):
    """Runs a policy on a real machine over a list of physically set-up scenes."""

    def __init__(
        self,
        machine: Machine,
        *,
        action: ChannelSpec | Mapping[str, Any],
        scenes: Sequence[Scene | Mapping[str, Any]] = (),
        operator: Operator | None = None,
        name: str = "bench",
        task: str = "bench",
        embodiment: str = "unknown",
        observes: Sequence[str] | None = None,
        rate_hz: float | None = None,
        horizon: int = 300,
        declares: Sequence[ChannelSpec | Mapping[str, Any]] = (),
    ):
        """``scenes`` is the list of physical arrangements, and it is required.

        A bench run with no declared scenes would be "we tried it a few times",
        which is not a measurement of anything. Each scene carries the sentence
        the operator reads, so the record can be checked afterwards against what
        was actually set up.
        """
        self._machine = machine
        self._action = _channel(action)
        self._operator = operator or Recording()
        self._name = name
        self._task = task
        self.embodiment = embodiment
        self.observes = tuple(observes) if observes is not None else None
        self.rate_hz = rate_hz
        self._horizon = horizon
        self._declared = {(spec := _channel(raw)).name: spec for raw in declares}
        self._scenes = tuple(_scene(raw) for raw in scenes)
        if not self._scenes:
            raise ConfigError(
                f"{name}: a bench run needs its scenes declared — each one the "
                "physical arrangement an operator will set up, in words. Without "
                "them a run is a number of attempts at an unrecorded situation, "
                "which nothing downstream can compare against anything"
            )
        self._world = Bench(machine, self._operator, self.observes)

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        judges = not isinstance(self._operator, Recording)
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            # A person can say whether the cube ended up in the bin. Nobody can
            # reliably say, live, which of five funnel rungs an attempt reached.
            stage_events=False,
            # False under Recording, and that is the point: the outcomes do not
            # exist yet. Anything needing them is refused until the footage has
            # been labelled, instead of reading an empty column as a row of zeros.
            outcomes=judges,
            # You cannot seed a table. Declaring otherwise would let a paired
            # comparison run over unrelated scenes and report a tight interval
            # around a comparison nobody made.
            seedable=False,
            closed_loop=True,
            operator=type(self._operator).__name__,
            machine=type(self._machine).__name__,
            scenes=len(self._scenes),
            control_hz=self.rate_hz,
        )

    def action(self) -> ChannelSpec:
        return self._action

    def declares(self) -> Mapping[str, ChannelSpec]:
        return self._declared

    def task_for(
        self, name: str = "", scenes: int | None = None, horizon: int | None = None
    ) -> TaskSpec:
        chosen = self._scenes if scenes is None else self._scenes[: max(1, int(scenes))]
        return TaskSpec(
            name=name or self._task,
            scenes=chosen,
            horizon=self._horizon if horizon is None else int(horizon),
        )

    def world_for(self, scene: Scene) -> Bench:
        return self._world

    def close(self) -> None:
        self._world.close()

    # -- the paper ---------------------------------------------------------

    def run_sheet(self, task: TaskSpec, path: str | Path, *, rubric: Sequence[str] = ()) -> Path:
        """Write the sheet the operator works from, and keep it with the run.

        This is the least glamorous artifact in the project and probably the most
        important one. A benchmark that exists only as code cannot be run by
        somebody else on a different bench, and "cross-lab reproducible" is
        exactly the claim a hardware result needs to survive. So: the scene list
        in order, the setup sentence for each, the rubric verbatim, and a grid
        with a row per attempt.

        The rubric is copied rather than summarised. Every time somebody
        shortens a criterion for the printed version, the printed version
        becomes a different standard than the one the simulator checked, and the
        sim-to-real number that follows is comparing two different questions.
        """
        target = Path(path)
        lines = [
            f"run sheet — {task.name}",
            f"machine: {type(self._machine).__name__}    body: {self.embodiment}",
            f"horizon: {task.horizon} steps" + (f" at {self.rate_hz:g} Hz" if self.rate_hz else ""),
            "",
            "the standard (copy verbatim into any judgement, do not paraphrase):",
        ]
        lines += [f"  - {line}" for line in (rubric or ("(no rubric supplied)",))]
        lines += [
            "",
            "an attempt that could not be judged is '?', not a failure.",
            "an attempt that did not happen is 'skip'.",
            "",
            f"{'#':>3}  {'scene':<24} {'set up':<44} {'met':>5}",
            f"{'-' * 3}  {'-' * 24} {'-' * 44} {'-' * 5}",
        ]
        for index, scene in enumerate(task.scenes, start=1):
            setup = (scene.setup or scene.instruction or "")[:44]
            lines.append(f"{index:>3}  {scene.id[:24]:<24} {setup:<44} {'':>5}")
        lines += ["", f"instruction given to the policy: {task.scenes[0].instruction or '(none)'}"]
        target.write_text("\n".join(lines) + "\n")
        return target


# -- plain data in, objects out -----------------------------------------------


def _channel(raw: ChannelSpec | Mapping[str, Any]) -> ChannelSpec:
    if isinstance(raw, ChannelSpec):
        return raw
    return ChannelSpec(**dict(raw))


def _scene(raw: Scene | Mapping[str, Any]) -> Scene:
    if isinstance(raw, Scene):
        return raw
    return Scene(**dict(raw))


def scenes_from_file(path: str | Path) -> tuple[Scene, ...]:
    """Read a scene list from JSON, so a bench protocol can be version-controlled.

    Expects a list of objects with at least ``id`` and ``setup``. Refuses a bare
    list of strings: a scene whose only content is a name has no arrangement
    written down, which is the thing this evaluator will not run without.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not all(isinstance(item, Mapping) for item in data):
        raise ConfigError(
            f"{path}: expected a list of scene objects with 'id' and 'setup'; a "
            "list of names would record no arrangement"
        )
    return tuple(_scene(item) for item in data)
