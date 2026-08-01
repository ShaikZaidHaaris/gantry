"""Refusing an experiment that cannot answer its question, before it is run.

Every other feedback module reads records that already exist. This one runs
before anything does, and it is the only module whose most valuable output is a
refusal. The reason is arithmetic: at twenty trials a five-point difference is
invisible, so a twenty-trial comparison of two things that differ by five points
returns noise — and the noise gets read as a verdict, because it arrives in the
same shape a real answer would.

That happened in this project. A downstream comparison was run at twenty trials,
came back at zero, and the honest reading was not "the policy is bad" but "this
experiment was four hundred times too small to say". Discovering that afterwards
cost the whole run. Discovering it beforehand costs a function call.

Why it reads history rather than asking
---------------------------------------
Sizing needs a baseline rate, and a caller who is asked for one will invent one.
An invented rate produces an invented trial count and the invented count is
always comfortably below the budget already planned, which is how an
underpowered experiment gets approved by its own author. So the rate comes from
:class:`~gantry.history.History` — what this task has actually produced before —
and when there is no history the module says so instead of guessing.

What it refuses, and what it merely notes
-----------------------------------------
It refuses when the arithmetic is decisive: the effect asked about cannot be
separated from the noise at the planned budget, full stop. It notes when the
situation is judgement — no history to size from, or a budget large enough to
detect something much smaller than anybody claimed, which usually means the
claim was hedged rather than the budget generous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.history import History
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, Verdict, count_of, proportion
from gantry.spine.inference import trials_needed

VERSION = "0.1.0.dev0"

#: Below this many trials nothing is worth calling an experiment. Not a
#: statistical threshold — a floor below which the arithmetic stops being the
#: interesting problem.
FLOOR = 4


@dataclass(frozen=True)
class Budget:
    """What an experiment intends to spend, and what it hopes to see."""

    trials: int
    #: The effect the experiment is being run to detect, in the metric's units.
    #: Required: an experiment that cannot say what it is looking for cannot be
    #: sized, and "any improvement" is not an answer because any positive noise
    #: satisfies it.
    magnitude: float
    metric: str = "success_rate"
    alpha: float = 0.05
    #: Runs already made on this question. Read from history when available;
    #: passed only when there is no history to read.
    attempts: int = 0

    def corrected_alpha(self) -> float:
        """The threshold this comparison must actually clear.

        A first look gets ``alpha``. The k-th look at the same question gets
        ``alpha/k``, because reporting the best of several tries is a different
        claim from finding something once and the arithmetic should say so
        before the money is spent rather than after the result is liked.
        """
        return self.alpha / max(1, self.attempts + 1)


def plan_for(
    budget: Budget,
    *,
    baseline: float | None = None,
    history: History | None = None,
    task: str | None = None,
    embodiment: str | None = None,
) -> Verdict:
    """Whether this budget can answer the question it is being spent on.

    ``baseline`` may be given directly, but the intended path is ``history`` plus
    ``task``: then the rate is measured rather than assumed, and the refusal
    carries a number somebody can act on.
    """
    checks: list[Verdict] = []

    if budget.magnitude <= 0:
        checks.append(
            Verdict.no(
                "power.no_effect_named",
                f"a budget of {count_of(budget.trials, 'trial')} is being planned to detect a "
                f"change of {budget.magnitude} in {budget.metric}",
                hint="no number of trials distinguishes a thing from itself; state "
                "the effect worth detecting",
            )
        )
    if budget.trials < FLOOR:
        checks.append(
            Verdict.no(
                "power.below_floor",
                f"{count_of(budget.trials, 'trial')} is not an experiment",
            )
        )
    if checks:
        return Verdict.all(checks)

    measured = None
    if baseline is None and history is not None and task is not None:
        measured = history.rate_for(task, embodiment)
        baseline = None if measured is None else measured.value

    if baseline is None:
        # Sized against the hardest case rather than refused outright: a task
        # nobody has run yet still deserves an answer, and 0.5 is where a
        # binary outcome is noisiest, so the number that comes back is the
        # conservative one.
        checks.append(
            Verdict.note(
                "power.no_history",
                f"nothing has been recorded for {task!r}, so this is sized against "
                "the noisiest case rather than a measured rate",
                hint="run it once and the sizing stops being a guess",
            )
        )
        baseline = 0.5

    alpha = budget.corrected_alpha()
    needed = trials_needed(baseline, budget.magnitude, alpha=alpha)

    if budget.attempts:
        checks.append(
            Verdict.note(
                "power.selection",
                f"{count_of(budget.attempts, 'run')} already exist on this question, so this "
                f"one must clear p<{alpha:.4f} rather than {budget.alpha}",
                hint="reporting the best of several tries without correcting is how "
                "a method looks better than it is",
            )
        )

    if budget.trials < needed:
        detectable = _smallest_detectable(baseline, budget.trials, alpha)
        hint = (
            f"run {needed}, or ask about an effect this budget can see — at "
            f"{budget.trials} trials that is roughly {detectable:+.3f}"
            if detectable is not None
            else f"run {needed}. There is no effect size {budget.trials} trials can "
            "reliably separate at this baseline, so there is no smaller question "
            "to ask instead"
        )
        checks.append(
            Verdict.no(
                "power.underpowered",
                f"separating a {budget.magnitude:+.3f} change in {budget.metric} from "
                f"a baseline of {baseline:.0%} needs about {needed} paired trials, and "
                f"{budget.trials} are planned",
                hint=hint,
                needed=needed,
                planned=budget.trials,
                baseline=round(baseline, 4),
                smallest_detectable=None if detectable is None else round(detectable, 4),
            )
        )
    elif budget.trials > needed * 4:
        checks.append(
            Verdict.note(
                "power.generous",
                f"{budget.trials} trials would detect a change about "
                f"{(_smallest_detectable(baseline, budget.trials, alpha) or 0.0):+.3f} in size, "
                f"far smaller than the {budget.magnitude:+.3f} claimed",
                hint="usually means the claim was hedged rather than the budget "
                "generous; a smaller claim is the stronger one to make",
            )
        )
    return Verdict.all(checks)


def _smallest_detectable(
    baseline: float, trials: int, alpha: float, *, resolution: float = 0.005
) -> float | None:
    """The smallest effect this many trials could separate, by search.

    ``None`` when the answer is "none": a budget too small to reliably detect
    *any* effect. It used to return 1.0 in that case, which reads as "this
    budget can detect a hundred-point change" -- a reassuring sentence about a
    hopeless experiment, and precisely backwards from what this module is for.
    """
    magnitude = resolution
    while magnitude < 1.0:
        if trials_needed(baseline, magnitude, alpha=alpha, cap=trials + 1) <= trials:
            return magnitude
        magnitude += resolution
    return None


class PowerCheck(FeedbackModule):
    """Reports what a set of recorded runs was and was not able to see.

    The retrospective half. :func:`plan_for` refuses before the money is spent;
    this reads runs that already happened and says which of their comparisons
    were ever capable of separating anything — which is the check that turns "we
    measured 0/20 and 0/20" from a finding into a note about the budget.
    """

    def __init__(self, *, magnitude: float = 0.10, alpha: float = 0.05):
        self._magnitude = magnitude
        self._alpha = alpha

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="power",
            version=VERSION,
            min_cohorts=1,
            prescribes=True,
            # It holds nothing: it is a statement about sample sizes, which is
            # true whatever varied between the cohorts.
            holds=(),
            magnitude=self._magnitude,
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "power",
            "feedback",
            capabilities={"outcomes": True},
            description="whether a budget could ever have separated the effect claimed",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        for cohort in cohorts:
            scored = [
                bool(getattr(getattr(e, "labels", None), "success", None))
                for e in cohort.episodes
                if getattr(getattr(e, "labels", None), "success", None) is not None
            ]
            if not scored:
                findings.append(
                    Finding(
                        code="power.unscored",
                        summary=f"{cohort.name}: no trial carries an outcome, so there "
                        "is nothing to have been powered for",
                        severity="info",
                        cohorts=(cohort.name,),
                    )
                )
                continue
            rate = proportion(sum(scored), len(scored))
            measurements[f"{cohort.name}.success_rate"] = rate
            needed = trials_needed(rate.value, self._magnitude, alpha=self._alpha)
            detectable = _smallest_detectable(rate.value, len(scored), self._alpha)
            if len(scored) < needed:
                findings.append(
                    Finding(
                        code="power.underpowered",
                        summary=(
                            f"{cohort.name}: {count_of(len(scored), 'scored trial')} at "
                            f"{rate.value:.0%} could separate an effect of about "
                            f"{detectable:+.3f}; a {self._magnitude:+.3f} claim needs "
                            f"about {needed}"
                        ),
                        severity="weak",
                        measurements={"success_rate": rate},
                        evidence={
                            "scored": len(scored),
                            "needed": needed,
                            "smallest_detectable": round(detectable, 4),
                        },
                        prescription=(
                            f"Run {needed} trials before reading this as a comparison, "
                            f"or state the claim at {detectable:+.3f} where the evidence "
                            "supports it."
                        ),
                        cohorts=(cohort.name,),
                    )
                )
        return Report(
            module="power",
            findings=tuple(findings),
            measurements=measurements,
            cohorts=tuple(c.name for c in cohorts),
        )
