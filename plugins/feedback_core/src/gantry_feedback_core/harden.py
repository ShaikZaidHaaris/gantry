"""Which findings are about the data, and which are about everything.

A finding from one dataset cannot tell you what to collect, because it cannot
distinguish two very different situations:

**Universal** — the same effect appears in every cohort, including the good
ones. It is a property of the task, the policy, or how the policy is run, and
collecting more data will not shift it. Acting on it wastes a collection
budget.

**Data-attributable** — the effect appears in some cohorts and not others. That
difference is the actionable part, and it is the only kind of finding that
should reach whoever decides what to record next.

Separating them structurally needs at least two cohorts, so this module refuses
below that rather than answering a narrower question in a confident voice.

The mechanism: run the same per-statistic test inside each cohort
independently, then classify by where it fires. Firing is judged on effect size
as well as significance, because a statistic sitting near the correction
threshold clears it in one cohort and misses in the next for no reason but
sampling — and reading that as "the actionable difference" is how a collection
budget gets spent on a coin flip.

When an effect fires in some cohorts but the others are merely unproven rather
than clearly quiet, the answer is that these cohorts cannot be told apart. Not
deciding is a real answer here, and a more decisive rule would only be more
confident, not more correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from gantry.contracts.feedback import Cohort, FeedbackModule, Finding, Report, feedback_descriptor
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement

from . import metrics, statistics as st
from .metrics import get_statistic, tabulate

VERSION = "0.1.0.dev0"

UNIVERSAL = "universal"
ATTRIBUTABLE = "data_attributable"
INCONSISTENT = "inconsistent"

#: Cliff's delta conventions. An effect must clear MATERIAL to count as firing,
#: and must be under NEGLIGIBLE in the other cohorts to count as absent.
MATERIAL, NEGLIGIBLE = 0.33, 0.147


@dataclass(frozen=True)
class PerCohort:
    """One statistic's verdict inside one cohort."""

    cohort: str
    delta: float
    q: float
    significant: bool


def _within(cohort: Cohort, alpha: float) -> dict[str, PerCohort]:
    episodes = [e for e in cohort.episodes if e.labels.success is not None]
    succeeded = [e for e in episodes if e.labels.success]
    failed = [e for e in episodes if not e.labels.success]
    if not succeeded or not failed:
        return {}
    good, _ = tabulate(succeeded)
    bad, _ = tabulate(failed)
    shared = sorted(set(good) & set(bad))
    pvalues = {name: st.mann_whitney(good[name], bad[name]) for name in shared}
    deltas = {name: st.cliffs_delta(good[name], bad[name]) for name in shared}
    return {
        corrected.name: PerCohort(
            cohort.name, deltas[corrected.name], corrected.q, corrected.significant
        )
        for corrected in st.benjamini_hochberg(pvalues, alpha)
    }


def classify(
    results: Sequence[PerCohort],
    total_cohorts: int,
    *,
    material: float = MATERIAL,
    negligible: float = NEGLIGIBLE,
) -> str:
    """Universal, data-attributable, or not decidable.

    Significance alone is too fragile to decide this. A statistic sitting near
    the correction threshold clears it in one cohort and misses in the next for
    no reason but sampling, and reading that as "the actionable difference"
    sends someone to collect data on a coin flip.

    So an effect must be *materially* large where it fires, and *negligibly*
    small where it does not. When it fires in some cohorts but is merely
    unproven in the others, the honest answer is that these cohorts cannot be
    told apart, not that a difference was found.
    """
    firing = [r for r in results if r.significant and abs(r.delta) >= material]
    if not firing:
        return INCONSISTENT
    if len({r.delta > 0 for r in firing}) > 1:
        return INCONSISTENT
    if len(firing) == total_cohorts:
        return UNIVERSAL
    quiet = [r for r in results if r not in firing]
    if len(firing) + len(quiet) < total_cohorts:
        return INCONSISTENT  # a cohort could not be tested at all
    if all(abs(r.delta) <= negligible for r in quiet):
        return ATTRIBUTABLE
    return INCONSISTENT


class Harden(FeedbackModule):
    """Cross-cohort classification of what is worth acting on."""

    def __init__(self, alpha: float = 0.05, aliases: dict[str, list[str]] | None = None):
        self.alpha = alpha
        self.aliases = dict(aliases or {})

    def descriptor(self) -> Descriptor:
        return feedback_descriptor("harden", VERSION, min_cohorts=2, prescribes=True)

    def requirement(self) -> Requirement:
        return metrics.requirement(
            "harden",
            self.aliases,
            capabilities={"outcomes": True},
            description="separates universal effects from data-attributable ones",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        per_cohort: dict[str, dict[str, PerCohort]] = {
            cohort.name: _within(cohort, self.alpha) for cohort in cohorts
        }
        usable = {name: table for name, table in per_cohort.items() if table}
        notes = [
            f"{name}: outcomes do not vary, so it cannot contribute"
            for name, table in per_cohort.items()
            if not table
        ]
        if len(usable) < 2:
            notes.append(
                "fewer than two cohorts could be tested; universal and data-attributable "
                "effects cannot be told apart"
            )
            return Report("harden", (), {}, tuple(notes), tuple(c.name for c in cohorts))

        statistics = sorted({name for table in usable.values() for name in table})
        findings, measurements = [], {}
        for name in statistics:
            results = [table[name] for table in usable.values() if name in table]
            verdict = classify(results, len(usable))
            if verdict == INCONSISTENT:
                continue
            statistic = get_statistic(name)
            firing = [r for r in results if r.significant]
            mean_delta = sum(r.delta for r in firing) / len(firing)
            measurements[name] = Measurement(
                mean_delta, n=len(firing), method="mean cliffs_delta across cohorts"
            )
            findings.append(
                Finding(
                    code=f"harden.{verdict}",
                    summary=_summarise(name, verdict, firing, len(usable)),
                    severity=_severity(verdict, statistic.prescribable),
                    measurements={name: measurements[name]},
                    evidence={
                        "classification": verdict,
                        "fires_in": [r.cohort for r in firing],
                        "cohorts_tested": list(usable),
                        "deltas": {r.cohort: r.delta for r in results},
                        "prescribable": statistic.prescribable,
                    },
                    prescription=_prescribe(verdict, name, statistic),
                    cohorts=tuple(usable),
                )
            )
        if not findings:
            notes.append("nothing fired consistently enough to classify")
        return Report("harden", tuple(findings), measurements, tuple(notes),
                      tuple(c.name for c in cohorts))


def _summarise(name: str, verdict: str, firing: Sequence[PerCohort], total: int) -> str:
    where = ", ".join(r.cohort for r in firing)
    if verdict == UNIVERSAL:
        return f"{name} separates outcomes in all {total} cohorts ({where})"
    return f"{name} separates outcomes in {len(firing)} of {total} cohorts ({where})"


def _severity(verdict: str, prescribable: bool) -> str:
    if not prescribable:
        return "info"
    return "strong" if verdict == ATTRIBUTABLE else "weak"


def _prescribe(verdict: str, name: str, statistic) -> str | None:
    if not statistic.prescribable:
        return None
    if verdict == UNIVERSAL:
        return (
            f"Do not collect for {name}. It behaves the same way in every cohort, including "
            f"the strongest, so it is a property of the task or of how the policy is run "
            f"rather than of the data."
        )
    return (
        f"{name} is the actionable difference between these cohorts. Weight collection "
        f"toward the cohorts where it does not fire."
    )
