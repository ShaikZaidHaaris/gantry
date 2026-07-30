"""Did the curation actually help? The half that makes a plan more than an opinion.

Every published data-curation method reports its own wins. That is not dishonesty
— it is the only thing available when evaluation is expensive enough that you
run it once, at the end, on the configuration you were hoping for. It does mean
the field's claims are, structurally, unaudited.

This module is the audit. It takes a plan, insists on the conditions under which
testing it would mean anything, and then judges the retrain by a paired test on
scenes the plan's own signal never saw.

Two gates before anything runs
------------------------------
**Leakage.** The strongest curation signals read evaluation rollouts to decide
what to remove. Verifying such a plan on those same scenes guarantees a win and
measures nothing: the data was chosen to look good exactly there. So a plan
declares the seeds its signal consumed, and this refuses to score it on their
intersection. No published method guards this mechanically, and it is a set
intersection.

**Power.** A plan predicting a three-point improvement, verified with twenty
trials, cannot come back with anything but noise — and the noise will be read as
a verdict. Refusing before the retrain costs nothing; discovering it afterward
costs the retrain. The magnitude the plan predicted is exactly the effect size
this powers for, which is the other reason predictions must carry a number.

Selection
---------
A signal that proposes ten plans and reports the one that worked has rediscovered
the multiple-comparisons problem. The count of plans already tested from a signal
is an input here, and the threshold is corrected by it.

What comes out
--------------
A ``CurationOutcome``: both run references, the paired delta with its interval,
the exact test, and whether the prediction held. Written whether it held or not
— a refuted plan is a result, and a ledger that only records successes is a
brochure.
"""

from __future__ import annotations

from math import comb
from typing import Any, Mapping, Sequence

from gantry.contracts.curation import CurationOutcome, CurationPlan
from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, Verdict, mcnemar, proportion

VERSION = "0.1.0.dev0"

#: Cohort names this expects. Fixed, because the whole analysis is "did the
#: change help", and which side is which cannot be guessed from the data.
BASELINE, CURATED = "baseline", "curated"


def trials_needed(baseline: float, magnitude: float, *, alpha: float = 0.05,
                  power: float = 0.8, cap: int = 5000) -> int:
    """Paired trials needed to see a change of ``magnitude``, by exact search.

    Exact rather than the normal approximation because robot evaluations run at
    n where the approximation is wrong in the unsafe direction — it says twenty
    trials suffice when they do not, which is precisely the error this exists
    to prevent.

    Modelled on the paired test that will actually judge it: what matters is
    the rate of disagreeing trials, not the marginal rates.
    """
    target = min(max(baseline + magnitude, 0.0), 1.0)
    # Trials where the two arms disagree — the only ones McNemar counts. The
    # conservative assumption is that improvement is the only source of
    # disagreement, which understates n; discordance from noise is added back
    # by assuming an equal amount of it in the other direction.
    gain = abs(target - baseline)
    if gain <= 0:
        return cap
    discordant_rate = min(1.0, gain * 2)
    for n in range(4, cap + 1):
        expected_discordant = n * discordant_rate
        if expected_discordant < 4:
            continue
        k = int(round(expected_discordant))
        # Under the null a discordant trial favours either arm equally; the
        # exact binomial tail is what McNemar reports.
        wins = int(round(k * (gain / discordant_rate + 0.5)))
        tail = sum(comb(k, i) for i in range(wins, k + 1)) / (2 ** k)
        if 2 * tail <= alpha:
            return n
    return cap


def preflight(
    plan: CurationPlan,
    verification_seeds: Sequence[int],
    *,
    baseline_rate: float = 0.5,
    plans_already_tested: int = 0,
    alpha: float = 0.05,
) -> Verdict:
    """Is testing this plan going to mean anything? Asked before the retrain."""
    checks = [plan.validate()]

    leaked = sorted(set(plan.evidence_seeds) & set(verification_seeds))
    if leaked:
        checks.append(
            Verdict.no(
                "curation.leaky",
                f"{plan.signal!r} read {len(leaked)} of the {len(verification_seeds)} "
                f"scene(s) it would be verified on, e.g. seed {leaked[0]}",
                hint="hold out scenes the signal never saw; a plan tested where it "
                "was fitted always wins and the win means nothing",
            )
        )

    magnitude = plan.predicted.magnitude
    if magnitude > 0:
        needed = trials_needed(baseline_rate, magnitude, alpha=alpha)
        available = len(verification_seeds)
        if available and available < needed:
            checks.append(
                Verdict.no(
                    "curation.underpowered",
                    f"{plan.signal!r} predicts {plan.predicted}; separating that from "
                    f"a baseline of {baseline_rate:.0%} needs about {needed} paired "
                    f"trials and {available} are planned",
                    hint=f"run {needed}, or predict an effect this budget can see",
                )
            )

    if plans_already_tested >= 1:
        # Holm's correction on the k-th test from one signal: the threshold this
        # plan must beat, stated up front rather than discovered after.
        corrected = alpha / (plans_already_tested + 1)
        checks.append(
            Verdict.note(
                "curation.selection",
                f"{plan.signal!r} has had {plans_already_tested} plan(s) tested "
                f"already; this one must clear p < {corrected:.4f}, not {alpha}",
                hint="reporting the best of several tries without correcting is how "
                "a signal looks better than it is",
            )
        )
    return Verdict.all(checks)


def _paired(baseline: Sequence[Any], curated: Sequence[Any]) -> tuple[int, int, int, int]:
    """Wins, losses, and totals over the scenes both arms attempted.

    Keyed on the scene rather than on position, because two runs that skipped
    different trials line up wrongly otherwise and the pairing is the whole
    source of the test's power.
    """
    def by_scene(episodes: Sequence[Any]) -> Mapping[str, bool]:
        out = {}
        for episode in episodes:
            labels = getattr(episode, "labels", None)
            success = getattr(labels, "success", None)
            if success is None:
                continue
            key = str(getattr(getattr(episode, "meta", None), "id", "")) or str(len(out))
            out[key] = bool(success)
        return out

    left, right = by_scene(baseline), by_scene(curated)
    shared = sorted(set(left) & set(right))
    only_curated = sum(1 for k in shared if right[k] and not left[k])
    only_baseline = sum(1 for k in shared if left[k] and not right[k])
    return only_curated, only_baseline, len(shared), sum(1 for k in shared if right[k])


class CurationVerifier(FeedbackModule):
    """Judges one curation plan from a baseline run and a curated run."""

    def __init__(self, plan: CurationPlan, *, alpha: float = 0.05,
                 plans_already_tested: int = 0):
        self._plan = plan
        self._alpha = alpha
        self._tested = plans_already_tested

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="verify",
            version=VERSION,
            min_cohorts=2,
            # The comparison is over datasets — one curated, one not — so
            # everything else has to be held. Declared, and cross-checked
            # against the run's provenance like any other module's claim.
            holds=("policy", "evaluation", "task", "embodiment"),
            signal=self._plan.signal,
            rung=self._plan.rung,
        )

    def requirement(self) -> Requirement:
        # A bare Requirement() is not constructible — it needs a name and a
        # plane — and this went unnoticed because the tests call analyse()
        # directly while only run() consults it. Anything driven from a
        # manifest would have hit it.
        return requires_channels(
            "verify",
            "feedback",
            capabilities={"outcomes": True},
            description="whether a curation plan produced the effect it predicted",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        named = {cohort.name: cohort for cohort in cohorts}
        missing = [n for n in (BASELINE, CURATED) if n not in named]
        if missing:
            return Report(
                module="verify",
                notes=(
                    f"expected cohorts named {BASELINE!r} and {CURATED!r}; "
                    f"missing {missing}. Which side is which cannot be inferred.",
                ),
                cohorts=tuple(named),
            )

        base, cur = named[BASELINE], named[CURATED]
        wins, losses, shared, cur_ok = _paired(base.episodes, cur.episodes)
        base_ok = sum(
            1 for e in base.episodes if getattr(getattr(e, "labels", None), "success", None)
        )
        p = mcnemar(losses, wins) if shared else None
        before = proportion(base_ok, len(base.episodes)) if base.episodes else None
        after = proportion(cur_ok, shared) if shared else None
        delta = Measurement(
            value=round((after.value - before.value), 4) if before and after else 0.0,
            n=shared,
            method="paired difference in success rate",
        )

        threshold = self._alpha / (self._tested + 1)
        held = (
            p is not None
            and p < threshold
            and delta.value >= self._plan.predicted.magnitude
        )
        if self._plan.predicted.direction == "-":
            held = p is not None and p < threshold and -delta.value >= self._plan.predicted.magnitude

        finding = Finding(
            code="curation.verified" if held else "curation.refuted",
            summary=(
                f"{self._plan.signal!r} predicted {self._plan.predicted}; "
                f"observed {delta.value:+.3f} over {shared} shared scenes "
                f"(won {wins}, lost {losses}"
                + (f", p={p:.4f}" if p is not None else "")
                + f"). Threshold p<{threshold:.4f}."
            ),
            severity="strong" if held else "info",
            measurements={
                k: v for k, v in
                (("delta", delta), ("baseline", before), ("curated", after))
                if v is not None
            },
            evidence={
                "wins": wins, "losses": losses, "shared_scenes": shared,
                "p": p, "threshold": threshold,
                "plan": self._plan.summary(),
                "rung": self._plan.rung,
                "plans_already_tested": self._tested,
            },
            prescription=(
                f"Apply it: {self._plan.summary()}"
                if held
                else f"Do not apply it on this evidence. {self._plan.signal!r} did not "
                "produce the improvement it predicted."
            ),
            cohorts=(BASELINE, CURATED),
        )
        return Report(
            module="verify",
            findings=(finding,),
            measurements=finding.measurements,
            cohorts=(BASELINE, CURATED),
        )

    # -- the ledger entry --------------------------------------------------

    def outcome(
        self, cohorts: Sequence[Cohort], *, baseline_run: str, curated_run: str,
        cost: Mapping[str, Any] | None = None,
    ) -> CurationOutcome:
        """The same judgement, in the form the ledger accumulates."""
        report = self.analyse(list(cohorts))
        if not report.findings:
            return CurationOutcome(
                plan=self._plan, baseline_run=baseline_run, curated_run=curated_run,
                delta=Measurement(value=0.0, n=0, method="not run"),
                verdict=Verdict.no("curation.unjudged", "; ".join(report.notes)),
                cost=dict(cost or {}),
            )
        finding = report.findings[0]
        return CurationOutcome(
            plan=self._plan,
            baseline_run=baseline_run,
            curated_run=curated_run,
            delta=finding.measurements["delta"],
            p=finding.evidence.get("p"),
            verdict=(
                Verdict.yes()
                if finding.code == "curation.verified"
                else Verdict.note("curation.refuted", finding.summary)
            ),
            cost=dict(cost or {}),
        )
