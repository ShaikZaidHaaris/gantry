"""Plane 5: turning records into findings, and findings into prescriptions.

A feedback module consumes records and nothing else. It never imports a
simulator, a policy, or a connector, and it cannot tell whether an episode was
recorded by a person or produced by a policy — which is exactly why one module
can screen a dataset and diagnose an evaluation without being told which it
received.

Invariants
----------
1. ``requirement()`` states everything the module depends on, and a module
   refuses rather than degrades when a requirement is unmet. A funnel handed
   outcome-only data says so; it does not return an empty table for someone to
   read meaning into.
2. ``analyse()`` is pure with respect to its inputs: same cohorts, same report.
3. Every number in a report is a
   :class:`~gantry.spine.provenance.Measurement`, so it carries its own n and
   interval rather than arriving as a bare float.
4. A module declares ``min_cohorts``. Some questions are not askable of a
   single dataset, and asking anyway produces a confident answer to a
   different question.
5. Findings name their evidence. A prescription with no measurement behind it
   is an opinion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..resolve.requirement import Requirement
from ..spine import Descriptor, EpisodeRecord, Measurement, Provenance, Verdict

#: Version of this contract.
FEEDBACK_CONTRACT = "feedback@1.0"

#: How many cohorts the module needs before its question is meaningful.
CAP_MIN_COHORTS = "min_cohorts"
#: Whether the module emits actionable prescriptions, not just measurements.
CAP_PRESCRIBES = "prescribes"

#: How much weight a finding carries. Not a p-value: evidence strength as
#: judged by the module, so a reader can triage without re-deriving statistics.
SEVERITIES = ("info", "weak", "strong")


@dataclass(frozen=True)
class Cohort:
    """One named group of records to analyse.

    Named rather than positional because most feedback questions are
    comparative, and "which of these is better" needs the labels to survive
    into the answer.
    """

    name: str
    episodes: tuple[EpisodeRecord, ...]
    provenance: Provenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    @property
    def outcomes(self) -> tuple[bool | None, ...]:
        return tuple(episode.labels.success for episode in self.episodes)

    @property
    def has_outcomes(self) -> bool:
        return any(outcome is not None for outcome in self.outcomes)

    @property
    def has_stage_events(self) -> bool:
        return any(episode.labels.stage_events for episode in self.episodes)

    @property
    def stages(self) -> tuple[str, ...]:
        seen: list[str] = []
        for episode in self.episodes:
            for stage in episode.labels.stages:
                if stage not in seen:
                    seen.append(stage)
        return tuple(seen)


@dataclass(frozen=True)
class Finding:
    """One thing the module concluded, with the evidence behind it."""

    code: str
    summary: str
    severity: str = "info"
    measurements: Mapping[str, Measurement] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    prescription: str | None = None
    cohorts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}; expected one of {SEVERITIES}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "summary": self.summary,
            "severity": self.severity,
            "cohorts": list(self.cohorts),
            "measurements": {k: v.as_dict() for k, v in self.measurements.items()},
            "evidence": dict(self.evidence),
            "prescription": self.prescription,
        }

    def __str__(self) -> str:
        head = f"[{self.severity}] {self.summary}"
        return f"{head}\n    -> {self.prescription}" if self.prescription else head


@dataclass(frozen=True)
class Report:
    """Everything one module concluded about one set of cohorts."""

    module: str
    findings: tuple[Finding, ...] = ()
    measurements: Mapping[str, Measurement] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    cohorts: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def of_severity(self, *severities: str) -> tuple[Finding, ...]:
        wanted = set(severities)
        return tuple(f for f in self.findings if f.severity in wanted)

    def by_code(self, code: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.code == code)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(f.code for f in self.findings)

    @property
    def prescriptions(self) -> tuple[str, ...]:
        return tuple(f.prescription for f in self.findings if f.prescription)

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "cohorts": list(self.cohorts),
            "findings": [f.as_dict() for f in self.findings],
            "measurements": {k: v.as_dict() for k, v in self.measurements.items()},
            "notes": list(self.notes),
        }

    def explain(self) -> str:
        lines = [f"{self.module} ({', '.join(self.cohorts) or 'no cohorts'})"]
        for name, value in self.measurements.items():
            lines.append(f"  {name}: {value}")
        for finding in self.findings:
            lines.append(f"  {finding}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def feedback_descriptor(
    name: str,
    version: str,
    *,
    min_cohorts: int,
    prescribes: bool,
    isolation: str = "in-process",
    **metadata: Any,
) -> Descriptor:
    return Descriptor(
        plane="feedback",
        name=name,
        version=version,
        contract=FEEDBACK_CONTRACT,
        provides={CAP_MIN_COHORTS: min_cohorts, CAP_PRESCRIBES: prescribes},
        metadata=metadata,
    )


class FeedbackModule(ABC):
    """Analysis over records, with its needs declared up front."""

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Identity and capabilities, read before anything is asked of it."""

    @abstractmethod
    def requirement(self) -> Requirement:
        """What this module needs from whatever produced the records."""

    @abstractmethod
    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        """Do the analysis. Callers should use ``run``, which checks first."""

    # -- provided ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.descriptor().name

    def min_cohorts(self) -> int:
        return int(self.descriptor().provides.get(CAP_MIN_COHORTS, 1))

    def check_inputs(self, cohorts: Sequence[Cohort]) -> Verdict:
        """Refuse before analysing, rather than answering a different question."""
        checks = []
        needed = self.min_cohorts()
        if len(cohorts) < needed:
            checks.append(
                Verdict.no(
                    "feedback.too_few_cohorts",
                    f"{self.name} needs at least {needed} cohort(s), got {len(cohorts)}",
                    hint="this question is not askable of fewer",
                    module=self.name,
                    needed=needed,
                    given=len(cohorts),
                )
            )
        empty = [cohort.name for cohort in cohorts if not len(cohort)]
        if empty:
            checks.append(
                Verdict.no(
                    "feedback.empty_cohort",
                    f"{self.name}: cohort(s) {empty} contain no episodes",
                    module=self.name,
                )
            )
        names = [cohort.name for cohort in cohorts]
        repeated = {name for name in names if names.count(name) > 1}
        if repeated:
            checks.append(
                Verdict.no(
                    "feedback.duplicate_cohort",
                    f"{self.name}: cohort name(s) {sorted(repeated)} used more than once",
                    hint="results are reported by name; duplicates would be indistinguishable",
                )
            )
        return Verdict.all(checks)

    def bind(self, cohorts: Sequence[Cohort]) -> tuple[Cohort, ...]:
        """Present each cohort's channels under the names this module asked for.

        A module declares a requirement; this is what makes that declaration
        real. Channels are matched by name, by alias, and by meaning, and arrive
        renamed to the module's own roles — so ``analyse`` reads what it asked
        for and never goes looking.

        Done here rather than only in a runner because a module must be correct
        when it is called directly, which is how it is tested and how anyone
        uses it from a notebook. A binding that only happens inside one caller
        is a binding that silently does not happen everywhere else.

        Anything that will not bind is left as it is. A missing optional channel
        should cost its own statistics, not the whole analysis, and the module
        reports what it could not compute.
        """
        from ..resolve import adapt_all, bind

        requirement = self.requirement()
        if not requirement.channels:
            return tuple(cohorts)

        bound: list[Cohort] = []
        for cohort in cohorts:
            schema = cohort.episodes[0].schema if cohort.episodes else ()
            wiring, verdict = bind(requirement, schema)
            if wiring is None or not wiring.bindings:
                bound.append(cohort)
                continue
            bound.append(
                Cohort(
                    cohort.name,
                    adapt_all(cohort.episodes, wiring),
                    provenance=cohort.provenance,
                    metadata=cohort.metadata,
                )
            )
        return tuple(bound)

    def run(self, cohorts: Sequence[Cohort]) -> Report:
        """Check, bind, then analyse. The path callers should use."""
        self.check_inputs(cohorts).raise_if_refused(f"cannot run {self.name}")
        return self.analyse(self.bind(cohorts))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.descriptor().ref}>"
