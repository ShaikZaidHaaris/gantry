"""Judge a demonstration set before spending GPU time on it.

Three modes, in order of how much they claim:

``comparative`` (default)
    Rank cohorts against each other and report which statistics actually
    separate them. Needs no thresholds at all, which is why it is the default:
    it is the only mode that makes no claim about what "good" means in the
    abstract.

``reference``
    Fit thresholds from a cohort already known to train a working policy, then
    screen others against it. The thresholds are a property of that reference,
    and they travel with it.

``absolute``
    Only the fraction of demonstrations that succeed. It is the one statistic
    that survives a change of task, and shipping any other as a universal
    threshold would be inventing a constant.

That ordering is the correction to the obvious design. Absolute thresholds fit
on one task read as universal truths and are not: a path-efficiency floor is a
statement about a task with no obstacle in the way, and a step-count ceiling is
a statement about one horizon. Numbers fitted from a reference are reported as
fitted, never as facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from gantry.contracts.feedback import Cohort, FeedbackModule, Finding, Report, feedback_descriptor
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement

from . import metrics, statistics as st
from .metrics import HIGHER, LOWER, Statistic, known_statistics, outcomes_of, tabulate

VERSION = "0.1.0.dev0"
MODES = ("comparative", "reference", "absolute")


@dataclass(frozen=True)
class Threshold:
    """One fitted bound, remembering where it came from."""

    statistic: str
    low: float
    high: float
    fitted_from: str
    n: int

    def holds(self, value: float) -> bool:
        return self.low <= value <= self.high


class Screen(FeedbackModule):
    def __init__(
        self,
        mode: str = "comparative",
        reference: str | None = None,
        k: float = 3.0,
        aliases: dict[str, list[str]] | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if mode == "reference" and not reference:
            raise ValueError("reference mode needs the name of the reference cohort")
        self.mode = mode
        self.reference = reference
        self.k = k
        #: Which channel plays which role. The caller knows; the module cannot.
        self.aliases = dict(aliases or {})

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            "screen",
            VERSION,
            min_cohorts=2 if self.mode in {"comparative", "reference"} else 1,
            prescribes=True,
            mode=self.mode,
        )

    def requirement(self) -> Requirement:
        capabilities = {"outcomes": True} if self.mode == "absolute" else {}
        return metrics.requirement(
            "screen",
            self.aliases,
            capabilities=capabilities,
            description="statistics over a demonstration set",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        if self.mode == "absolute":
            return self._absolute(cohorts)
        if self.mode == "reference":
            return self._reference(cohorts)
        return self._comparative(cohorts)

    # -- absolute ----------------------------------------------------------

    def _absolute(self, cohorts: Sequence[Cohort]) -> Report:
        findings, measurements, notes = [], {}, [
            "absolute mode reports only the success fraction; every other "
            "threshold would be a constant fitted on some other task"
        ]
        for cohort in cohorts:
            outcomes, unlabelled = outcomes_of(cohort.episodes)
            if not outcomes:
                notes.append(f"{cohort.name}: no episode carries an outcome label")
                continue
            successes, n = int(sum(outcomes)), len(outcomes)
            rate = successes / n
            measurement = Measurement(
                rate, n=n, ci=st.wilson(successes, n), units="fraction", method="wilson"
            )
            measurements[f"{cohort.name}.success_rate"] = measurement
            if unlabelled:
                notes.append(f"{cohort.name}: {unlabelled} episode(s) had no outcome label")
            if measurement.ci[1] < 0.95:
                findings.append(
                    Finding(
                        code="screen.low_success",
                        summary=(
                            f"{cohort.name}: {rate:.0%} of demonstrations succeed "
                            f"({successes}/{n})"
                        ),
                        severity="strong" if rate < 0.5 else "weak",
                        measurements={"success_rate": measurement},
                        prescription=(
                            "Keep only the demonstrations that finish the task, or collect "
                            "a set that does. No training setting repairs a set that mostly fails."
                        ),
                        cohorts=(cohort.name,),
                    )
                )
        return Report("screen", tuple(findings), measurements, tuple(notes),
                      tuple(c.name for c in cohorts))

    # -- reference ---------------------------------------------------------

    def fit(self, cohort: Cohort) -> tuple[Threshold, ...]:
        """Derive bounds from a cohort known to be good."""
        columns, _ = tabulate(cohort.episodes)
        thresholds = []
        for name, values in columns.items():
            low, high = st.robust_bounds(values, self.k)
            thresholds.append(Threshold(name, low, high, cohort.name, len(values)))
        return tuple(thresholds)

    def _reference(self, cohorts: Sequence[Cohort]) -> Report:
        by_name = {cohort.name: cohort for cohort in cohorts}
        if self.reference not in by_name:
            return Report(
                "screen",
                notes=(f"reference cohort {self.reference!r} is not among {sorted(by_name)}",),
                cohorts=tuple(by_name),
            )
        thresholds = {t.statistic: t for t in self.fit(by_name[self.reference])}
        findings, measurements = [], {}
        notes = [
            f"thresholds fitted from {self.reference!r} (median ± {self.k}·MAD); "
            "they describe that cohort, not the task in general"
        ]
        for cohort in cohorts:
            if cohort.name == self.reference:
                continue
            columns, unavailable = tabulate(cohort.episodes)
            if unavailable:
                notes.append(f"{cohort.name}: not computable here: {', '.join(unavailable)}")
            for name, values in columns.items():
                threshold = thresholds.get(name)
                if threshold is None:
                    continue
                centre = float(np.median(values))
                measurement = Measurement(
                    centre, n=len(values), ci=st.bootstrap_ci(values, np.median),
                    method="median"
                )
                measurements[f"{cohort.name}.{name}"] = measurement
                if threshold.holds(centre):
                    continue
                statistic = next(s for s in known_statistics() if s.name == name)
                findings.append(
                    Finding(
                        code="screen.outside_reference",
                        summary=(
                            f"{cohort.name}: {name} is {centre:.4g}, outside the "
                            f"{threshold.low:.4g}–{threshold.high:.4g} band fitted from "
                            f"{threshold.fitted_from}"
                        ),
                        severity="weak",
                        measurements={name: measurement},
                        evidence={
                            "band": [threshold.low, threshold.high],
                            "fitted_from": threshold.fitted_from,
                            "prescribable": statistic.prescribable,
                        },
                        prescription=_prescribe(statistic, centre, threshold),
                        cohorts=(cohort.name, threshold.fitted_from),
                    )
                )
        return Report("screen", tuple(findings), measurements, tuple(notes),
                      tuple(c.name for c in cohorts))

    # -- comparative -------------------------------------------------------

    def _comparative(self, cohorts: Sequence[Cohort]) -> Report:
        tables = {c.name: tabulate(c.episodes)[0] for c in cohorts}
        shared = set.intersection(*(set(table) for table in tables.values())) if tables else set()
        findings, measurements = [], {}
        notes = ["comparative mode ranks cohorts against each other and asserts no thresholds"]

        pvalues: dict[str, float] = {}
        details: dict[str, dict] = {}
        for name in sorted(shared):
            groups = {cohort: tables[cohort][name] for cohort in tables}
            for cohort, values in groups.items():
                measurements[f"{cohort}.{name}"] = Measurement(
                    float(np.median(values)), n=len(values),
                    ci=st.bootstrap_ci(values, np.median), method="median"
                )
            statistic = next(s for s in known_statistics() if s.name == name)
            # Which end is "best" depends on the statistic, not on the sort.
            # Ranking every statistic ascending calls the jerkiest cohort the
            # best one on jerk, and the prescription then points at the data you
            # least want more of.
            ordered = sorted(
                groups.items(),
                key=lambda kv: float(np.median(kv[1])),
                reverse=statistic.direction == LOWER,
            )
            worst, best = ordered[0], ordered[-1]
            pvalues[name] = st.mann_whitney(best[1], worst[1])
            details[name] = {
                "best": best[0] if statistic.direction else None,
                "worst": worst[0] if statistic.direction else None,
                "delta": st.cliffs_delta(best[1], worst[1]),
                "ranking": [cohort for cohort, _ in ordered],
                "ordered_by": statistic.direction or "no better end; ordered by median",
            }

        for corrected in st.benjamini_hochberg(pvalues):
            if not corrected.significant:
                continue
            detail = details[corrected.name]
            statistic = next(s for s in known_statistics() if s.name == corrected.name)
            findings.append(
                Finding(
                    code="screen.separates",
                    summary=(
                        f"{corrected.name} separates the cohorts "
                        f"(q={corrected.q:.3g}, delta={detail['delta']:+.2f})"
                    ),
                    severity="strong" if abs(detail["delta"]) >= 0.474 else "weak",
                    evidence={**detail, "p": corrected.p, "q": corrected.q,
                              "prescribable": statistic.prescribable},
                    prescription=(
                        f"Collect more demonstrations resembling {detail['best']!r}, "
                        f"which is the better cohort on {corrected.name}."
                        if statistic.prescribable and detail["best"]
                        else None
                    ),
                    cohorts=tuple(detail["ranking"]),
                )
            )
        if not findings and shared:
            notes.append("no statistic separates these cohorts after FDR correction")
        return Report("screen", tuple(findings), measurements, tuple(notes),
                      tuple(c.name for c in cohorts))


def _prescribe(statistic: Statistic, value: float, threshold: Threshold) -> str | None:
    """Advice, but only from a statistic allowed to give it."""
    if not statistic.prescribable:
        return None
    if statistic.direction == HIGHER and value < threshold.low:
        return f"Collect demonstrations with higher {statistic.name}: {statistic.description}."
    if statistic.direction == LOWER and value > threshold.high:
        return f"Collect demonstrations with lower {statistic.name}: {statistic.description}."
    return None
