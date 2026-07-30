"""Plane: who decides whether a trial succeeded, and on what evidence.

Until now one thing decided: the simulator. Every success rate this project has
produced came from a physics engine checking a pose, and that was treated as the
definition of success rather than as one judge's opinion about it.

That is fine until it isn't. On a real bench there is no ``_check_success()`` —
a person watches and says. If the sentence a person is given differs from what
the predicate measured, then a policy validated in simulation was validated
against a different criterion than the one it will be held to, and nobody finds
out until the hardware disagrees.

So judging becomes an axis
--------------------------
A scorer takes evidence and returns a judgement. The simulator's predicate is
one; a person watching a video is another; a vision-language model reading the
same rubric is a third. Making them values of one plane rather than one
privileged path and two special cases buys three things:

They can be **compared** — ``varies: scorer`` is a legitimate experiment, and
chance-corrected agreement between two of them is a measurable quantity rather
than a hope.

They can be **swapped** — an evaluation scored by a person and one scored by a
model differ in a recorded field, so a claim can state which produced it.

They can be **refused** — a scorer whose agreement with people has not been
measured, or has been measured and is poor, can be blocked from contributing
findings. That is the check that makes cheap judgement safe to scale.

Abstention is a first-class answer
----------------------------------
"I cannot tell from this video" is honesty, and a scorer forced to guess instead
produces labels that look like data. So ``passed`` may be ``None``, agreement
calculations drop those pairs rather than scoring them as disagreement, and a
scorer that abstains constantly is caught by its abstention rate rather than by
looking accurate on the few it answered.

Invariants
----------
1. ``judge`` returns one judgement per criterion the task declares, in order.
2. A scorer declares the evidence it needs and refuses evidence lacking it,
   rather than guessing from what it was given.
3. Judging the same evidence twice returns the same judgement, for scorers that
   declare determinism. A human scorer does not declare it; a predicate does.
4. Declared capabilities are true.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..errors import ConfigError
from ..spine import Descriptor, Verdict

#: 1.1 added ``annotate`` and ``recorded_labels``. Producing a surface for a
#: judge to work on, and reading its answers back, are scorer responsibilities —
#: and until they were on the contract, anything that wanted to drive a scoring
#: workflow had to import a particular scorer by name, which gave core a
#: favourite judge. Both default to refusing, so a scorer that only judges
#: in-process is unaffected and says so.
SCORER_CONTRACT = "scorer@1.1"

#: Kinds of evidence a scorer may ask for. Open vocabulary — a scorer naming
#: something not listed here is asking for evidence nobody produces yet, which
#: is a refusal rather than an error.
VIDEO = "video"
FINAL_STATE = "final_state"
EVENTS = "events"
TRAJECTORY = "trajectory"

#: What the scorer needs to see.
CAP_EVIDENCE = "evidence"
#: Same evidence, same judgement. True for a predicate, false for a person.
CAP_DETERMINISTIC = "deterministic"
#: Cost per judgement, coarsely: "free", "cheap", "human".
CAP_COST = "cost"
#: Whether this scorer may abstain.
CAP_ABSTAINS = "abstains"

COSTS = ("free", "cheap", "human")


@dataclass(frozen=True)
class Evidence:
    """What a judge is given about one trial.

    Deliberately a bundle rather than the episode itself. A human scorer needs a
    video and nothing else; a predicate needs the simulator's final state and
    cannot use a video at all. Handing everything to everyone would mean a
    scorer could quietly depend on evidence a real bench cannot supply, which is
    the failure that makes a sim-validated rubric untransferable.
    """

    trial: str
    #: Path to a rendered video, when one exists.
    video: str | None = None
    #: The world's own end-of-episode state, for a predicate to read.
    final_state: Mapping[str, Any] | None = None
    #: Milestones reached, when a funnel emitted them.
    events: tuple[str, ...] = ()
    #: Per-step arrays, for a scorer that measures rather than watches.
    trajectory: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def has(self, kind: str) -> bool:
        return {
            VIDEO: self.video is not None,
            FINAL_STATE: self.final_state is not None,
            EVENTS: bool(self.events),
            TRAJECTORY: self.trajectory is not None,
        }.get(kind, False)

    def available(self) -> tuple[str, ...]:
        return tuple(k for k in (VIDEO, FINAL_STATE, EVENTS, TRAJECTORY) if self.has(k))


@dataclass(frozen=True)
class Judgement:
    """One scorer's verdict on one criterion of one trial."""

    criterion: str
    #: ``None`` is an abstention: the evidence did not settle it. Kept distinct
    #: from ``False`` because "it failed" and "I could not tell" support
    #: different conclusions and averaging them together loses both.
    passed: bool | None
    #: Why, in the scorer's own words. A predicate can say which threshold; a
    #: person or a model can say what they saw. Recorded because a disagreement
    #: between two scorers is only diagnosable if both said why.
    rationale: str = ""
    #: Which evidence was actually used, of what was offered.
    used: tuple[str, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.passed is None


def scorer_descriptor(
    name: str,
    version: str,
    *,
    evidence: Sequence[str],
    deterministic: bool,
    cost: str = "free",
    abstains: bool = False,
    isolation: str = "in-process",
    **metadata: Any,
) -> Descriptor:
    if cost not in COSTS:
        raise ValueError(f"unknown cost {cost!r}; expected one of {list(COSTS)}")
    return Descriptor(
        plane="scorer",
        name=name,
        version=version,
        contract=SCORER_CONTRACT,
        provides={
            CAP_EVIDENCE: tuple(evidence),
            CAP_DETERMINISTIC: deterministic,
            CAP_COST: cost,
            CAP_ABSTAINS: abstains,
        },
        isolation=isolation,
        metadata=metadata,
    )


class Scorer(ABC):
    """Something that looks at a trial and says whether it succeeded."""

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Identity and capabilities, read before anything is asked of it."""

    @abstractmethod
    def judge(self, evidence: Evidence, task: Any) -> tuple[Judgement, ...]:
        """One judgement per criterion the task declares, in the task's order.

        ``task`` is a :class:`~gantry.contracts.task.TaskDefinition`. The scorer
        reads its criteria — and, if it is a human or a model, the rubric text
        attached to each one, which is the sentence the criterion was written
        twice for.
        """

    # -- provided ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.descriptor().name

    @property
    def needs(self) -> tuple[str, ...]:
        return tuple(self.descriptor().provides.get(CAP_EVIDENCE, ()))

    @property
    def deterministic(self) -> bool:
        return bool(self.descriptor().provides.get(CAP_DETERMINISTIC, False))

    @property
    def cost(self) -> str:
        return str(self.descriptor().provides.get(CAP_COST, "free"))

    def check_evidence(self, evidence: Evidence) -> Verdict:
        """Whether this scorer can judge from what it was handed."""
        missing = [kind for kind in self.needs if not evidence.has(kind)]
        if missing:
            return Verdict.no(
                "scorer.evidence_missing",
                f"{self.name} judges from {list(self.needs)} and trial "
                f"{evidence.trial!r} offers {list(evidence.available())}",
                hint="a scorer that guessed from the wrong evidence would produce "
                "labels indistinguishable from real ones",
            )
        return Verdict.yes()

    def score(self, evidence: Evidence, task: Any) -> tuple[Judgement, ...]:
        """Check the evidence, then judge. The path callers should use."""
        self.check_evidence(evidence).raise_if_refused(f"{self.name} cannot judge")
        return self.judge(evidence, task)

    @classmethod
    def annotate(
        cls, trials: Sequence[Mapping[str, Any]], task: Any, path: str, **options: Any
    ) -> Any:
        """Produce whatever surface this judge needs in order to work.

        A person needs a page with videos and the rubric on it; a model needs a
        prompt; a predicate needs nothing at all. A classmethod because the
        surface is a property of the kind of judge rather than of any instance —
        the thing being built is what a judge will later be constructed *from*.

        Refuses by default. A scorer that judges in-process has no surface to
        produce, and silently returning nothing would leave a caller believing a
        page exists.
        """
        raise ConfigError(
            f"{cls.__name__} judges in-process and has no annotation surface to "
            "produce; a scorer that needs one implements annotate"
        )

    @classmethod
    def recorded_labels(cls, path: str) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
        """A judge's name and its recorded verdicts, read off disk.

        ``(judge, {trial: {criterion: passed}})`` — the shape agreement is
        computed from. Here rather than in whatever module wrote the file so
        that comparing two judges never requires importing either of them.
        """
        raise ConfigError(
            f"{cls.__name__} does not record its judgements to a file; a scorer "
            "that does implements recorded_labels"
        )

    def overall(self, judgements: Sequence[Judgement]) -> bool | None:
        """Did the trial succeed, given every criterion?

        All criteria must pass. One abstention makes the whole trial an
        abstention rather than a failure — a scorer that could not tell whether
        the cube was held has not established that the trial failed, and
        recording it as a failure would quietly convert uncertainty into
        evidence.
        """
        if not judgements:
            return None
        if any(judgement.abstained for judgement in judgements):
            return None
        return all(bool(judgement.passed) for judgement in judgements)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.descriptor().ref}>"
