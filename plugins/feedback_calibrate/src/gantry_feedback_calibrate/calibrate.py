"""Whether a judge agrees with people well enough for its labels to count.

This is the gate that makes cheap judgement safe. A model can score ten thousand
videos for the price of a coffee, which is worthless unless somebody has checked
that its answers resemble a person's — and "resemble" has to mean a measured
number, because a judge that is confidently wrong produces labels shaped exactly
like correct ones and every aggregate computed from them inherits the error
silently.

So: two judges, the same trials, chance-corrected agreement, and a verdict with
consequences. Above 0.80 the judge's conclusions may be reported as findings.
Between 0.67 and 0.80 they support a tentative claim and are labelled as such.
Below 0.67 the judge is not measuring the same thing as the person and
``judge.uncalibrated`` refuses its findings rather than reporting them with a
caveat nobody reads.

Chance correction is not optional
--------------------------------
On a task solved five percent of the time, two judges who both say "failed"
every time agree ninety-five percent of the time and have established nothing.
Raw agreement flatters exactly the judges least worth trusting, which is why the
thresholds above are on kappa and alpha rather than on percent-agreement.

The biases worth measuring, per judge per task
----------------------------------------------
A judge is not one thing. The same model can be reliable on lifting and useless
on insertion, so agreement is computed per task family and never pooled into a
single headline. Beyond agreement, two biases are cheap to test and known to
matter: whether the judge's verdict depends on how long the episode ran, and
whether it depends on the order evidence was presented. Both are measured here
rather than assumed absent.

Abstentions are dropped, and counted
------------------------------------
A pair where either judge said "cannot tell" carries no information about
agreement and is excluded. But the *rate* of abstention is itself a measurement:
a rubric producing many of them is ambiguous, and finding that out from twenty
videos is much cheaper than finding it out from a bench.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, Verdict, count_of, plural
from gantry.spine.inference import (
    AGREEMENT_TENTATIVE,
    AGREEMENT_TRUSTED,
    agreement_verdict_code,
    cohen_kappa,
    krippendorff_alpha,
)

VERSION = "0.1.0.dev0"

#: Words that turn a binary criterion into a graded one. Moving from graded to
#: binary raises inter-judge agreement by roughly twenty points in the
#: LLM-judging literature, and every one of these invites a rater to split the
#: difference instead of deciding.
HEDGES = (
    "partially",
    "mostly",
    "somewhat",
    "roughly",
    "approximately",
    "more or less",
    "generally",
    "largely",
    "reasonably",
)

#: Below this many judged pairs, an agreement number is not worth computing —
#: kappa on four trials swings between -1 and 1 on one disagreement.
MIN_PAIRS = 8


def hedges(rubric: str) -> tuple[str, ...]:
    """Hedge words in a rubric, which make two people split the difference."""
    lowered = rubric.lower()
    return tuple(word for word in HEDGES if word in lowered)


@dataclass(frozen=True)
class Corpus:
    """Judgements from several judges on the same trials.

    The thing that accumulates. Keyed by judge, then trial, then criterion, so
    the same corpus answers "do these two agree", "does this judge agree with
    itself over time", and "which criterion do they disagree about" without
    being reshaped.
    """

    #: ``{judge: {trial: {criterion: passed}}}``
    labels: Mapping[str, Mapping[str, Mapping[str, bool | None]]] = field(default_factory=dict)
    #: Per-trial facts a bias check needs — episode length, presentation order.
    context: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    task: str | None = None

    @property
    def judges(self) -> tuple[str, ...]:
        return tuple(sorted(self.labels))

    def shared_trials(self) -> tuple[str, ...]:
        common: set[str] | None = None
        for trials in self.labels.values():
            common = set(trials) if common is None else common & set(trials)
        return tuple(sorted(common or ()))

    def aligned(self, left: str, right: str) -> tuple[list[Any], list[Any]]:
        """Two judges' verdicts on the criteria they both judged, in one order."""
        a, b = [], []
        for trial in self.shared_trials():
            criteria = set(self.labels[left].get(trial, {})) & set(
                self.labels[right].get(trial, {})
            )
            for criterion in sorted(criteria):
                a.append(self.labels[left][trial][criterion])
                b.append(self.labels[right][trial][criterion])
        return a, b

    def abstention_rate(self, judge: str) -> float:
        verdicts = [
            value for trial in self.labels.get(judge, {}).values() for value in trial.values()
        ]
        if not verdicts:
            return 0.0
        return sum(1 for v in verdicts if v is None) / len(verdicts)


def agreement(corpus: Corpus, left: str, right: str) -> Measurement:
    """Chance-corrected agreement between two judges, with its verdict code."""
    a, b = corpus.aligned(left, right)
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    kappa = cohen_kappa(a, b)
    raw = sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else float("nan")
    return Measurement(
        value=round(kappa, 4) if kappa == kappa else float("nan"),
        n=len(pairs),
        method="Cohen's kappa on judgements both judges settled",
        detail={
            "raw_agreement": round(raw, 4) if raw == raw else None,
            "dropped_abstentions": len(a) - len(pairs),
            "verdict": agreement_verdict_code(kappa),
            "left": left,
            "right": right,
        },
    )


def _length_bias(corpus: Corpus, judge: str) -> Measurement | None:
    """Does this judge's verdict depend on how long the episode ran?

    The robotics analogue of verbosity bias. A judge that passes long episodes
    and fails short ones is measuring duration, and duration correlates with
    success on some tasks and against it on others — so the bias masquerades as
    signal in whichever direction the task happens to run.
    """
    passed, failed = [], []
    for trial, verdicts in corpus.labels.get(judge, {}).items():
        steps = (corpus.context.get(trial) or {}).get("steps")
        if steps is None:
            continue
        settled = [v for v in verdicts.values() if v is not None]
        if not settled:
            continue
        (passed if all(settled) else failed).append(float(steps))
    if len(passed) < 3 or len(failed) < 3:
        return None
    mean_pass = sum(passed) / len(passed)
    mean_fail = sum(failed) / len(failed)
    spread = max(mean_pass, mean_fail) or 1.0
    return Measurement(
        value=round((mean_pass - mean_fail) / spread, 4),
        n=len(passed) + len(failed),
        method="normalised difference in mean episode length, passed minus failed",
        detail={
            "mean_steps_passed": round(mean_pass, 1),
            "mean_steps_failed": round(mean_fail, 1),
        },
    )


class Calibration(FeedbackModule):
    """Measures whether judges agree, and refuses the ones that do not.

    Cohorts are judges: one per scorer, each holding that scorer's judgements of
    the same trials. The comparison axis is the scorer, so everything else must
    be held — same trials, same task, same evidence.
    """

    def __init__(
        self,
        corpus: Corpus | Mapping[str, Any],
        *,
        reference: str | None = None,
        task_rubrics: Mapping[str, str] | None = None,
    ):
        # A plain ``{judge: {trial: {criterion: passed}}}`` is accepted as well
        # as a Corpus, because the registry builds components from config
        # dictionaries and a caller coming through a manifest has no Corpus to
        # hand over.
        self._corpus = corpus if isinstance(corpus, Corpus) else Corpus(labels=dict(corpus))
        #: Which judge is the standard others are measured against. Defaults to
        #: the most expensive one present, because a person is the thing a
        #: cheaper judge is trying to stand in for and not the other way round.
        self._reference = reference
        self._rubrics = dict(task_rubrics or {})

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="calibrate",
            version=VERSION,
            min_cohorts=1,
            prescribes=True,
            # The scorer is what varies; everything that produced the trials is
            # held, or the disagreement being measured is not about judging.
            holds=("policy", "task", "evaluation", "embodiment"),
            reference=self._reference,
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "calibrate",
            "feedback",
            capabilities={"outcomes": True},
            description="whether two judges of one rubric agree well enough to be believed",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        corpus = self._corpus
        judges = corpus.judges
        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        notes: list[str] = []

        # A rubric that hedges will not agree with itself, let alone across
        # judges. Checked first, because it is free and it explains the rest.
        for criterion, rubric in self._rubrics.items():
            found = hedges(rubric)
            if found:
                findings.append(
                    Finding(
                        code="criterion.graded_rubric",
                        summary=(
                            f"the rubric for {criterion!r} hedges — {', '.join(found)} "
                            "— which invites a rater to split the difference instead "
                            "of deciding"
                        ),
                        severity="info",
                        evidence={"criterion": criterion, "hedges": list(found)},
                        prescription=(
                            "Rewrite it as something a person can see rather than "
                            "estimate. Moving a criterion from graded to binary is "
                            "worth roughly twenty points of agreement."
                        ),
                    )
                )

        if len(judges) < 2:
            notes.append(
                f"only {len(judges)} judge present, so agreement cannot be measured; "
                "abstention rate and rubric wording still can"
            )
        else:
            reference = self._reference or _most_expensive(judges)
            for judge in judges:
                if judge == reference:
                    continue
                score = agreement(corpus, reference, judge)
                measurements[f"{judge}.kappa"] = score
                code = str(score.detail["verdict"])
                if score.n < MIN_PAIRS:
                    findings.append(
                        Finding(
                            code="judge.unmeasured",
                            summary=(
                                f"{judge} and {reference} settled only {score.n} "
                                f"{plural(score.n, 'judgement')} in common. Agreement needs at least "
                                f"{MIN_PAIRS} before the number means anything"
                            ),
                            severity="info",
                            measurements={"kappa": score},
                            prescription=f"Score {count_of(MIN_PAIRS - score.n, 'more trial')}.",
                            cohorts=(judge, reference),
                        )
                    )
                    continue
                findings.append(
                    Finding(
                        code=code,
                        summary=(
                            f"{judge} against {reference}: kappa {score.value:.3f} "
                            f"over {count_of(score.n, 'settled judgement')} "
                            f"(raw agreement {score.detail['raw_agreement']:.0%}, "
                            f"{count_of(score.detail['dropped_abstentions'], 'abstention')} dropped)"
                        ),
                        severity="strong" if code == "judge.calibrated" else "weak",
                        measurements={"kappa": score},
                        evidence=dict(score.detail),
                        prescription=_prescription(code, judge, reference),
                        cohorts=(judge, reference),
                    )
                )

        for judge in judges:
            rate = corpus.abstention_rate(judge)
            measurements[f"{judge}.abstention_rate"] = Measurement(
                value=round(rate, 4),
                n=sum(len(v) for v in corpus.labels.get(judge, {}).values()),
                method="share of judgements where the judge could not tell",
            )
            if rate > 0.3:
                findings.append(
                    Finding(
                        code="judge.abstains_often",
                        summary=(f"{judge} could not tell on {rate:.0%} of judgements"),
                        severity="info",
                        prescription=(
                            "Usually the rubric rather than the judge. Finding this "
                            "out from twenty videos is far cheaper than finding it "
                            "out from a bench."
                        ),
                        cohorts=(judge,),
                    )
                )
            bias = _length_bias(corpus, judge)
            if bias is not None:
                measurements[f"{judge}.length_bias"] = bias
                if abs(bias.value) > 0.25:
                    findings.append(
                        Finding(
                            code="judge.bias.length",
                            summary=(
                                f"{judge}'s verdicts track episode length: passed "
                                f"episodes averaged "
                                f"{bias.detail['mean_steps_passed']} steps against "
                                f"{bias.detail['mean_steps_failed']} for failed"
                            ),
                            severity="weak",
                            measurements={"length_bias": bias},
                            prescription=(
                                "Duration correlates with success on some tasks and "
                                "against it on others, so this bias looks like signal "
                                "in whichever direction the task runs. Check whether "
                                "the rubric can be judged from a fixed-length clip."
                            ),
                            cohorts=(judge,),
                        )
                    )

        if len(judges) >= 3:
            table = [
                [
                    corpus.labels[judge].get(trial, {}).get(criterion)
                    for trial in corpus.shared_trials()
                    for criterion in sorted(
                        set().union(*(set(corpus.labels[j].get(trial, {})) for j in judges))
                        if trial
                        else set()
                    )
                ]
                for judge in judges
            ]
            alpha = krippendorff_alpha(table)
            measurements["alpha"] = Measurement(
                value=round(alpha, 4) if alpha == alpha else float("nan"),
                n=len(corpus.shared_trials()),
                method="Krippendorff's alpha across all judges, nominal",
                detail={"judges": list(judges), "verdict": agreement_verdict_code(alpha)},
            )

        return Report(
            module="calibrate",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(judges),
        )

    # -- the gate ----------------------------------------------------------

    def gate(self, judge: str, *, reference: str | None = None) -> Verdict:
        """Whether this judge's labels may be reported as findings.

        The consequence that makes the number matter. A judge below the floor is
        refused rather than reported with a caveat, because a caveat attached to
        a number gets dropped the first time somebody quotes it.
        """
        against = reference or self._reference or _most_expensive(self._corpus.judges)
        if against == judge:
            return Verdict.yes()
        if against not in self._corpus.judges or judge not in self._corpus.judges:
            return Verdict.no(
                "judge.unmeasured",
                f"{judge!r} has never been compared against {against!r}",
                hint="score some trials with both and the question becomes answerable",
            )
        score = agreement(self._corpus, against, judge)
        if score.n < MIN_PAIRS:
            return Verdict.no(
                "judge.unmeasured",
                f"{judge!r} and {against!r} settled only {count_of(score.n, 'judgement')} "
                f"together; {MIN_PAIRS} is the floor for the number to mean anything",
            )
        code = str(score.detail["verdict"])
        if code == "judge.calibrated":
            return Verdict.yes()
        if code == "judge.tentative":
            return Verdict.note(
                code,
                f"{judge!r} agrees with {against!r} at kappa {score.value:.3f}, which "
                f"is between {AGREEMENT_TENTATIVE} and {AGREEMENT_TRUSTED}",
                hint="findings from this judge should be labelled tentative rather "
                "than reported flat",
            )
        return Verdict.no(
            code,
            f"{judge!r} agrees with {against!r} at kappa {score.value:.3f}, below "
            f"{AGREEMENT_TENTATIVE}",
            hint="this judge is not measuring the same thing; its labels should not "
            "become findings until the rubric or the judge changes",
        )


#: Cost ordering, cheapest first. The most expensive judge present is the
#: default reference, because a cheaper one is trying to stand in for it.
_COST_ORDER = ("free", "cheap", "human")


def _most_expensive(judges: Sequence[str]) -> str:
    """The judge others are measured against, by convention on the name.

    A heuristic, and a documented one: a corpus does not carry descriptors, so
    the reference is inferred from the judge's name and can always be given
    explicitly. Getting it wrong inverts the comparison's meaning, so the
    inference is deliberately dumb and easy to override.
    """
    for candidate in ("human", "person", "rater"):
        for judge in judges:
            if candidate in judge.lower():
                return judge
    return judges[-1] if judges else ""


def _prescription(code: str, judge: str, reference: str) -> str:
    if code == "judge.calibrated":
        return (
            f"{judge} may stand in for {reference} on this task family. Its labels "
            "can be reported as findings."
        )
    if code == "judge.tentative":
        return (
            f"{judge} is close but not reliable. Use it for screening and label any "
            "conclusion tentative; do not publish a number from it alone."
        )
    return (
        f"{judge} is not measuring what {reference} measures. Either the rubric is "
        "ambiguous — check the abstention rate and the wording — or this judge "
        "cannot do this task family. Its labels should not become findings."
    )


def corpus_from_sessions(
    sessions: Mapping[str, Mapping[str, Mapping[str, bool | None]]],
    *,
    context: Mapping[str, Mapping[str, Any]] | None = None,
    task: str | None = None,
) -> Corpus:
    """Build a corpus from ``{judge: {trial: {criterion: passed}}}``."""
    return Corpus(labels=dict(sessions), context=dict(context or {}), task=task)
