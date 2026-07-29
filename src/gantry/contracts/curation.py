"""Plane: what to do to the data, said precisely enough to be wrong.

Every other plane in this framework describes or measures. This one *prescribes*
— it is the only plane whose output is an instruction to change something. That
asymmetry is why it is built the way it is.

Why a plan is an object and not a sentence
------------------------------------------
"Collect more grasping data" cannot be applied, cannot be checked, and cannot be
found to have been wrong. ``collect(n=40, task='lift_cube', seeds=[...],
stage='grasp')`` can be all three. A prescription that cannot be executed
mechanically is advice, and advice does not compound.

So a plan is a list of typed actions over identified episodes, and it carries
three things that advice never does: which signal produced it, at what claim
strength, and **what it predicts will happen**. The prediction is the part that
matters. A plan that cannot be wrong is a plan nobody should act on.

Why proposing and verifying are different planes
------------------------------------------------
A curator proposes; the feedback plane judges. Never the same component, and
this is not tidiness — a signal that scores its own effect is the Goodhart
machine in miniature, and every curation method in the literature reports its
own wins. Here the verdict comes from a retrain and a paired evaluation run by
machinery that has never heard of the signal.

Why the ladder
--------------
Four ways to decide an episode is bad, in ascending cost and descending
deniability: it is labelled a failure; it looks unlike the good ones; the cohort
containing it trains worse; removing it provably lowers closed-loop return.
These are not interchangeable, and the ways they get confused are expensive. So
every plan states its rung and the rung is checked against the language of the
claim — a screening correlation phrased as a cause is refused by name.

Why evidence has to name its rollouts
-------------------------------------
The strongest signals read evaluation rollouts to decide what to drop. If the
plan is then verified on those same scenes, the answer is guaranteed and
meaningless. Nothing in the published literature guards this mechanically. Here
a plan records the seeds its signal consumed, and verification refuses to score
itself on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..spine import Descriptor, Measurement, Provenance, Verdict

CURATION_CONTRACT = "curation@1.0"

#: How this curator decided, in ascending cost and descending deniability.
#:
#: ``screening``    — a property of the data alone (labels, smoothness, length).
#: ``unsupervised`` — a property of the data's distribution (how unlike the rest).
#: ``leave_out``    — the cohort containing it was measured to train worse.
#: ``influence``    — removing it was estimated to change closed-loop return.
#:
#: Ordered, and the order is load-bearing: a claim may always be reported as
#: weaker than its rung, never as stronger.
RUNGS = ("screening", "unsupervised", "leave_out", "influence")

#: Rungs whose evidence includes evaluation rollouts. Plans at these rungs must
#: name the seeds they consumed, or verification cannot tell whether it is about
#: to mark its own homework.
ROLLOUT_RUNGS = frozenset({"influence"})

#: What a curator offers, read before anything is asked of it.
CAP_PER_EPISODE = "per_episode"
#: Needs evaluation runs, not just a dataset.
CAP_NEEDS_RUNS = "needs_runs"
#: Needs to see inside training — gradients, losses, a cooperating backend.
CAP_NEEDS_GRADIENTS = "needs_gradients"
#: The rung this curator argues at.
CAP_RUNG = "rung"

#: Action kinds. ``drop``/``keep`` name episodes; ``weight`` names a cohort;
#: ``collect`` names data that does not exist yet.
ACTION_KINDS = ("drop", "keep", "weight", "collect")


@dataclass(frozen=True)
class Prediction:
    """What this plan claims will happen, in a form that can turn out false.

    Stated before the retrain, so it cannot be adjusted to fit the result. The
    magnitude matters as much as the direction: a plan predicting "some
    improvement" is unfalsifiable in practice, because any positive noise
    confirms it and any negative noise is explained away.
    """

    metric: str = "success_rate"
    #: ``+`` or ``-``. Which way the metric is expected to move.
    direction: str = "+"
    #: How far, in the metric's own units. Also the effect size verification
    #: powers itself for, so an honest small number costs real trials.
    magnitude: float = 0.0
    #: Where the claim applies. Empty means "wherever this is verified", which
    #: is weaker and is noted rather than refused.
    tasks: tuple[str, ...] = ()

    def validate(self) -> Verdict:
        checks = []
        if self.direction not in ("+", "-"):
            checks.append(
                Verdict.no(
                    "prediction.direction",
                    f"direction {self.direction!r} is neither '+' nor '-'",
                )
            )
        if not self.tasks:
            checks.append(
                Verdict.note(
                    "prediction.unscoped",
                    "predicts nothing in particular about where it applies",
                )
            )
        return Verdict.all(checks)

    def __str__(self) -> str:
        where = f" on {', '.join(self.tasks)}" if self.tasks else ""
        return f"{self.metric} {self.direction}{self.magnitude:g}{where}"


@dataclass(frozen=True)
class CollectionOrder:
    """Data that does not exist yet, described well enough to go and get.

    The difference between this and "collect more data" is that every field
    here answers a question a person standing at a robot actually has: how
    many, of what task, starting from where, practising which part.

    ``seeds`` are the ones that failed. A seeded scene can be staged again
    exactly, so in simulation this is executable as written; on a bench the
    task's own region is the mark on the mat. That is the same bridge the task
    plane already builds, used in the other direction.
    """

    task: str
    n: int
    #: Scenes the policy failed, by seed. Re-stageable, so the demonstrations
    #: are collected in exactly the situations that beat it.
    seeds: tuple[int, ...] = ()
    #: Which milestone it failed at, when a funnel knows. Grasping and placing
    #: are different skills and the demonstrations for them are different work.
    stage: str | None = None
    #: Anything a person needs that the fields above cannot carry — "failures
    #: cluster on the far edge", "the cube was upright in every failure".
    note: str = ""

    def validate(self) -> Verdict:
        checks = []
        if self.n < 1:
            checks.append(
                Verdict.no("order.empty", f"asks for {self.n} demonstrations of {self.task!r}")
            )
        if not self.task:
            checks.append(Verdict.no("order.no_task", "names no task to demonstrate"))
        if not self.seeds and not self.note:
            checks.append(
                Verdict.note(
                    "order.untargeted",
                    f"{self.task!r}: asks for {self.n} demonstrations without saying "
                    "which situations",
                    hint="targeted demonstrations are worth more than the same number "
                    "of random ones; failed seeds are the targets",
                )
            )
        return Verdict.all(checks)

    def __str__(self) -> str:
        bits = [f"collect {self.n} x {self.task}"]
        if self.stage:
            bits.append(f"at the {self.stage} stage")
        if self.seeds:
            bits.append(f"from {len(self.seeds)} failed scene(s)")
        return ", ".join(bits)


@dataclass(frozen=True)
class CurationAction:
    """One change to make to the data."""

    kind: str
    #: Episode uids, for ``drop`` and ``keep``.
    episodes: tuple[str, ...] = ()
    #: Cohort name, for ``weight``.
    cohort: str | None = None
    #: Relative sampling weight, for ``weight``. Zero excludes without deleting.
    weight: float | None = None
    #: For ``collect``.
    order: CollectionOrder | None = None
    #: Why these and not others — a score, a rank, a reason per episode. Kept
    #: because "drop these 431" is unreviewable and "drop these 431, here is
    #: each one's score" is an argument someone can disagree with.
    detail: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> Verdict:
        checks = []
        if self.kind not in ACTION_KINDS:
            return Verdict.no(
                "action.unknown_kind",
                f"{self.kind!r} is not one of {list(ACTION_KINDS)}",
            )
        if self.kind in ("drop", "keep") and not self.episodes:
            checks.append(
                Verdict.no("action.no_episodes", f"{self.kind!r} names no episodes")
            )
        if self.kind == "weight":
            if not self.cohort:
                checks.append(Verdict.no("action.no_cohort", "weight names no cohort"))
            if self.weight is None or self.weight < 0:
                checks.append(
                    Verdict.no(
                        "action.bad_weight",
                        f"weight {self.weight} on {self.cohort!r} is not a "
                        "non-negative number",
                    )
                )
        if self.kind == "collect":
            if self.order is None:
                checks.append(Verdict.no("action.no_order", "collect carries no order"))
            else:
                checks.append(self.order.validate())
        return Verdict.all(checks)


@dataclass(frozen=True)
class CurationPlan:
    """A proposal to change the data, and the claim that it will help.

    Not a finding. A finding says what was; this says what to do and what it
    expects, and is only a finding once something has tested it.
    """

    actions: tuple[CurationAction, ...]
    #: The signal that produced this, named so two proposals can be told apart.
    signal: str
    #: One of :data:`RUNGS`.
    rung: str
    predicted: Prediction = field(default_factory=Prediction)
    #: What the signal read. For rollout-consuming rungs this must include the
    #: seeds, or verification cannot check itself for leakage.
    evidence: Provenance | None = None
    #: Scenes the signal consumed, by seed. Verification refuses to score a
    #: plan on the same scenes that produced it.
    evidence_seeds: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> Verdict:
        checks = [action.validate() for action in self.actions]
        checks.append(self.predicted.validate())
        if not self.actions:
            checks.append(
                Verdict.no("plan.empty", f"{self.signal!r} proposes nothing to do")
            )
        if self.rung not in RUNGS:
            checks.append(
                Verdict.no(
                    "plan.unknown_rung",
                    f"{self.rung!r} is not one of {list(RUNGS)}",
                )
            )
        elif self.rung in ROLLOUT_RUNGS and not self.evidence_seeds:
            checks.append(
                Verdict.no(
                    "plan.unnamed_evidence",
                    f"{self.signal!r} argues at the {self.rung!r} rung, which reads "
                    "evaluation rollouts, but names none of the scenes it read",
                    hint="verification has to know them to avoid scoring this plan on "
                    "the scenes that produced it",
                )
            )
        if self.proposes_a_change and self.predicted.magnitude <= 0:
            checks.append(
                Verdict.no(
                    "plan.no_prediction",
                    f"{self.signal!r} proposes a change but predicts nothing a result "
                    "could contradict",
                    hint="state how much it will help; it is also the effect size "
                    "verification powers for",
                )
            )
        dropped = {e for a in self.actions if a.kind == "drop" for e in a.episodes}
        kept = {e for a in self.actions if a.kind == "keep" for e in a.episodes}
        both = sorted(dropped & kept)
        if both:
            checks.append(
                Verdict.no(
                    "plan.contradictory",
                    f"{self.signal!r} both drops and keeps {len(both)} episode(s), "
                    f"e.g. {both[0]!r}",
                )
            )
        return Verdict.all(checks)

    # -- reading it --------------------------------------------------------

    @property
    def drops(self) -> tuple[str, ...]:
        return tuple(e for a in self.actions if a.kind == "drop" for e in a.episodes)

    @property
    def keeps(self) -> tuple[str, ...]:
        return tuple(e for a in self.actions if a.kind == "keep" for e in a.episodes)

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            a.cohort: float(a.weight)
            for a in self.actions
            if a.kind == "weight" and a.cohort and a.weight is not None
        }

    @property
    def orders(self) -> tuple[CollectionOrder, ...]:
        return tuple(a.order for a in self.actions if a.kind == "collect" and a.order)

    @property
    def proposes_a_change(self) -> bool:
        """Whether acting on this would alter anything.

        A curator asked for its opinion and having none is a real answer, and
        one worth recording: it says the signal was consulted and did not
        object. Such a plan predicts nothing because there is nothing to
        predict, and requiring a falsifiable claim from it would force every
        clean dataset to be given an invented one.
        """
        return any(
            action.kind in ("drop", "weight", "collect") for action in self.actions
        )

    @property
    def touches_existing_data(self) -> bool:
        """Whether applying this changes the training set, or only adds to it.

        A plan that only orders collection needs no retrain to apply — it needs
        a person and a robot. The two are verified differently and the
        distinction is worth having up front.
        """
        return any(a.kind in ("drop", "keep", "weight") for a in self.actions)

    def summary(self) -> str:
        bits = []
        if self.drops:
            bits.append(f"drop {len(self.drops)}")
        if self.keeps:
            bits.append(f"keep {len(self.keeps)}")
        for cohort, weight in self.weights.items():
            bits.append(f"{cohort} x{weight:g}")
        bits.extend(str(order) for order in self.orders)
        return f"[{self.signal}@{self.rung}] " + "; ".join(bits) + f" -> {self.predicted}"


@dataclass(frozen=True)
class CurationOutcome:
    """One plan, tested. The unit that accumulates.

    Both runs are named rather than summarised, because the summary is the part
    most likely to be wrong later and the records are the part that can be
    re-read.
    """

    plan: CurationPlan
    #: Run-record references, in whatever form the store addresses them.
    baseline_run: str
    curated_run: str
    #: Observed change, paired, with its interval.
    delta: Measurement
    #: Paired test on the shared scenes.
    p: float | None = None
    verdict: Verdict = field(default_factory=Verdict.yes)
    #: What it cost to find out — gpu-minutes, trials, wall time. Recorded so
    #: "was this worth testing" is answerable from the ledger rather than from
    #: memory.
    cost: Mapping[str, Any] = field(default_factory=dict)

    @property
    def held(self) -> bool:
        """Whether the prediction survived: right direction, and significant."""
        if not self.verdict.ok or self.p is None:
            return False
        moved = self.delta.value >= self.plan.predicted.magnitude
        if self.plan.predicted.direction == "-":
            moved = -self.delta.value >= self.plan.predicted.magnitude
        return moved and self.p < 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.plan.signal,
            "rung": self.plan.rung,
            "predicted": str(self.plan.predicted),
            "summary": self.plan.summary(),
            "baseline_run": self.baseline_run,
            "curated_run": self.curated_run,
            "delta": self.delta.as_dict(),
            "p": self.p,
            "held": self.held,
            "verdict": [r.code for r in self.verdict.reasons],
            "cost": dict(self.cost),
        }


def curator_descriptor(
    name: str,
    version: str,
    *,
    rung: str,
    per_episode: bool,
    needs_runs: bool = False,
    needs_gradients: bool = False,
    isolation: str = "in-process",
    **metadata: Any,
) -> Descriptor:
    return Descriptor(
        plane="curation",
        name=name,
        version=version,
        contract=CURATION_CONTRACT,
        provides={
            CAP_RUNG: rung,
            CAP_PER_EPISODE: per_episode,
            CAP_NEEDS_RUNS: needs_runs,
            CAP_NEEDS_GRADIENTS: needs_gradients,
        },
        isolation=isolation,
        metadata=metadata,
    )


class Curator(ABC):
    """Something that reads data, and proposes what to do about it."""

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Identity and capabilities, read before anything is asked of it."""

    @abstractmethod
    def propose(
        self, episodes: Sequence[Any], runs: Sequence[Any] = ()
    ) -> CurationPlan:
        """Read the data, and say what to do about it.

        ``runs`` is empty for curators that do not need evaluation rollouts,
        which is most of them and is the point: the cheapest rungs work on a
        dataset nobody has trained on yet.
        """

    # -- provided ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.descriptor().name

    @property
    def rung(self) -> str:
        return str(self.descriptor().provides.get(CAP_RUNG, "screening"))

    @property
    def needs_runs(self) -> bool:
        return bool(self.descriptor().provides.get(CAP_NEEDS_RUNS, False))

    def check_inputs(self, episodes: Sequence[Any], runs: Sequence[Any]) -> Verdict:
        checks = []
        if not episodes:
            checks.append(
                Verdict.no("curation.no_data", f"{self.name} was given no episodes")
            )
        if self.needs_runs and not runs:
            checks.append(
                Verdict.no(
                    "curation.no_runs",
                    f"{self.name} decides from evaluation rollouts and was given none",
                    hint="run the policy first; this rung cannot be reached from a "
                    "dataset alone",
                )
            )
        return Verdict.all(checks)

    def plan(self, episodes: Sequence[Any], runs: Sequence[Any] = ()) -> CurationPlan:
        """Check, then propose. The path callers should use."""
        self.check_inputs(episodes, runs).raise_if_refused(f"cannot run {self.name}")
        proposal = self.propose(episodes, runs)
        proposal.validate().raise_if_refused(f"{self.name} produced an invalid plan")
        return proposal

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.descriptor().ref}>"
