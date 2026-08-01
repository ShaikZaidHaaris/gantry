"""Which behaviours go with success, and which merely look like they do.

For every statistic, compare the episodes that succeeded against the ones that
did not: Cliff's delta for how big the gap is, Mann-Whitney for whether it
survives noise, Benjamini-Hochberg because a dozen statistics are being asked
at once and reporting whatever clears 0.05 manufactures about one finding from
nothing every time.

The quarantine is the important part. A statistic that is not duration-free, or
that is computed from the outcome, can be *reported* but can never become a
prescription. Both rules come from findings that looked strong and were not:

- A count-type statistic separates succeeded from failed beautifully when
  failed episodes are short, which they usually are. It is measuring length.
- A statistic derived from the outcome correlates perfectly with the outcome.
  "Episodes that reached more stages succeeded more" is arithmetic, not a
  lesson about data collection.

Both still appear in the report, marked, because they are useful context and
hiding them would just mean someone recomputes them by hand and believes them.
What they cannot do is spend anyone's collection budget.

Every finding also carries its rank correlation with episode length, so a
reader can see for themselves how much of it is duration.
"""

from __future__ import annotations

from typing import Sequence

from gantry.contracts.feedback import Cohort, FeedbackModule, Finding, Report, feedback_descriptor
from gantry.resolve import Requirement
from gantry.spine import Descriptor, Measurement, count_of

from . import metrics
from . import statistics as st
from .metrics import HIGHER, LOWER, get_statistic, known_statistics, tabulate

VERSION = "0.1.0.dev0"

#: Cliff's delta conventions: 0.147 small, 0.33 medium, 0.474 large.
LARGE, MEDIUM = 0.474, 0.33


class Attribution(FeedbackModule):
    def __init__(self, alpha: float = 0.05, aliases: dict[str, list[str]] | None = None):
        self.alpha = alpha
        self.aliases = dict(aliases or {})

    def descriptor(self) -> Descriptor:
        return feedback_descriptor("attribution", VERSION, min_cohorts=1, prescribes=True)

    def requirement(self) -> Requirement:
        return metrics.requirement(
            "attribution",
            self.aliases,
            capabilities={"outcomes": True},
            description="which behaviours separate success from failure",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings, measurements, notes = [], {}, []
        for cohort in cohorts:
            findings.extend(self._one(cohort, measurements, notes))
        return Report(
            "attribution",
            tuple(findings),
            measurements,
            tuple(notes),
            tuple(c.name for c in cohorts),
        )

    def _one(self, cohort: Cohort, measurements: dict, notes: list) -> list[Finding]:
        episodes = [e for e in cohort.episodes if e.labels.success is not None]
        if len(episodes) != len(cohort.episodes):
            notes.append(
                f"{cohort.name}: {count_of(len(cohort.episodes) - len(episodes), 'episode')} "
                "had no outcome and were excluded"
            )
        succeeded = [e for e in episodes if e.labels.success]
        failed = [e for e in episodes if not e.labels.success]
        if not succeeded or not failed:
            notes.append(
                f"{cohort.name}: every episode has the same outcome, so nothing can be "
                "attributed to behaviour"
            )
            return []

        good, _ = tabulate(succeeded)
        bad, _ = tabulate(failed)
        lengths = [float(len(e)) for e in episodes]

        shared = sorted(set(good) & set(bad))
        pvalues = {name: st.mann_whitney(good[name], bad[name]) for name in shared}
        deltas = {name: st.cliffs_delta(good[name], bad[name]) for name in shared}

        findings: list[Finding] = []
        for corrected in st.benjamini_hochberg(pvalues, self.alpha):
            if not corrected.significant:
                continue
            name = corrected.name
            statistic = get_statistic(name)
            delta = deltas[name]
            column, _ = tabulate(episodes, [statistic])
            with_duration = st.spearman(column.get(name, lengths), lengths)

            measurements[f"{cohort.name}.{name}"] = Measurement(
                delta,
                n=len(episodes),
                method="cliffs_delta",
                detail={"p": corrected.p, "q": corrected.q},
            )
            findings.append(
                Finding(
                    code="attribution.associated"
                    if statistic.prescribable
                    else "attribution.context",
                    summary=(
                        f"{cohort.name}: {name} separates success from failure "
                        f"(delta={delta:+.2f}, q={corrected.q:.3g})"
                        + ("" if statistic.prescribable else " — context only")
                    ),
                    severity=_severity(delta, statistic.prescribable),
                    measurements={name: measurements[f"{cohort.name}.{name}"]},
                    evidence={
                        "delta": delta,
                        "p": corrected.p,
                        "q": corrected.q,
                        "n_success": len(succeeded),
                        "n_failure": len(failed),
                        "rank_correlation_with_length": with_duration,
                        "prescribable": statistic.prescribable,
                        "duration_free": statistic.duration_free,
                        "outcome_derived": statistic.outcome_derived,
                    },
                    prescription=_prescribe(statistic, delta),
                    cohorts=(cohort.name,),
                )
            )
        if not findings:
            notes.append(f"{cohort.name}: no behaviour survived FDR correction")
        return findings


def _severity(delta: float, prescribable: bool) -> str:
    if not prescribable:
        return "info"
    magnitude = abs(delta)
    if magnitude >= LARGE:
        return "strong"
    return "weak" if magnitude >= MEDIUM else "info"


def _prescribe(statistic, delta: float) -> str | None:
    """Advice, and only from a statistic entitled to give it."""
    if not statistic.prescribable:
        return None
    if statistic.direction == HIGHER or (statistic.direction is None and delta > 0):
        way = "higher"
    elif statistic.direction == LOWER or (statistic.direction is None and delta < 0):
        way = "lower"
    else:
        return None
    return (
        f"Weight collection toward demonstrations with {way} {statistic.name} "
        f"({statistic.description})."
    )


def eligible_statistics() -> tuple[str, ...]:
    """Which statistics may produce advice, for anyone who wants to check."""
    return tuple(s.name for s in known_statistics() if s.prescribable)


def quarantined_statistics() -> tuple[str, ...]:
    return tuple(s.name for s in known_statistics() if not s.prescribable)
