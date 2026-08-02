"""Where a policy stops, and what fixing that one place would be worth.

An end-to-end success rate says a run failed. A funnel says where. Chaining
milestones into conditional rates -- reached, then engaged given reached, then
completed given engaged -- turns one uninformative number into a diagnosis, and
the counterfactual makes it actionable: if this one step were perfect and
nothing else changed, the end-to-end rate would move by *this much*.

That last quantity is what makes the difference between "grasping looks weak"
and "repairing grasping alone is worth 41 points", which are different
sentences to bring to whoever decides what to collect next.

Stage names come from the data. A funnel over approach/engage/transport and a
funnel over incise/suture/close are the same computation on different words,
and core has no opinion about which vocabulary is real.

Requires stage events, and says so rather than degrading. Handed outcome-only
records it refuses, because an empty funnel table looks like a finished
analysis and someone will read meaning into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gantry.contracts.feedback import Cohort, FeedbackModule, Finding, Report, feedback_descriptor
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, EpisodeRecord, Measurement

from . import statistics as st

VERSION = "0.1.0.dev0"


@dataclass(frozen=True)
class Step:
    """One rung: how many arrived, how many got through."""

    name: str
    arrived: int
    passed: int

    @property
    def measured(self) -> bool:
        """Did anything reach this rung at all?

        The distinction between "everybody failed here" and "nobody got here to
        try" is the whole difference between a rate of zero and no rate. A rung
        past a total blockage has no evidence either way, and treating its
        silence as failure is an assertion the data does not support.
        """
        return self.arrived > 0

    @property
    def rate(self) -> float:
        """Observed pass rate. Only meaningful when :attr:`measured`."""
        return self.passed / self.arrived if self.arrived else 0.0

    @property
    def chained(self) -> float:
        """The rate to use when multiplying rungs together.

        An unmeasured rung contributes 1.0 -- not because it is known to pass,
        but because the alternative asserts it fails. That makes every product
        involving it zero, which wipes out the uplift of the rung that is
        *actually* blocking and points the diagnosis at the wrong place. The
        report says out loud when this is happening.
        """
        return self.rate if self.measured else 1.0

    @property
    def ci(self) -> tuple[float, float]:
        return st.wilson(self.passed, self.arrived)

    def measurement(self) -> Measurement:
        return Measurement(
            self.rate if self.measured else float("nan"),
            n=self.arrived,
            ci=self.ci,
            units="fraction",
            method="wilson" if self.measured else "unmeasured: nothing reached this rung",
        )


#: Where a run records the milestones it was set up to look for. An evaluator
#: knows the whole vocabulary; the episodes only show the part that happened.
DECLARED_STAGES = "stages"


def stage_order(episodes: Sequence[EpisodeRecord]) -> tuple[str, ...]:
    """Order stages by when they typically happen, not by first sighting.

    Ordering by first appearance would let one unusual episode fix the shape of
    the whole funnel. Median step is stable against that.

    Inferring the vocabulary this way is a *fallback*, and a dangerous one: a
    stage no episode ever reached cannot appear here. See :func:`declared_stages`
    for why that matters.
    """
    steps: dict[str, list[int]] = {}
    for episode in episodes:
        for event in episode.labels.stage_events:
            steps.setdefault(event.name, []).append(event.step)
    return tuple(
        name for name, _ in sorted(steps.items(), key=lambda kv: sorted(kv[1])[len(kv[1]) // 2])
    )


def declared_stages(cohort: Cohort) -> tuple[str, ...] | None:
    """The milestones the run was set up to look for, if it recorded them.

    This exists because inferring the vocabulary from what happened is wrong in
    exactly the case that matters most. A policy that fails at the second
    milestone in *every* episode never produces a second milestone event, so an
    inferred funnel has one rung, reports it at 100%, and says the uplift from
    fixing it is zero -- a clean bill of health for a policy that never does
    anything. Reading the intended vocabulary instead turns that into the
    second rung sitting at 0%, which is the truth.
    """
    if cohort.provenance is None:
        return None
    declared = cohort.provenance.protocol.get(DECLARED_STAGES)
    if not declared:
        return None
    return tuple(str(stage) for stage in declared)


def build(
    episodes: Sequence[EpisodeRecord], stages: Sequence[str] | None = None
) -> tuple[Step, ...]:
    """Conditional pass rates along the chain."""
    ordered = tuple(stages) if stages else stage_order(episodes)
    steps: list[Step] = []
    surviving = list(episodes)
    for stage in ordered:
        arrived = len(surviving)
        through = [e for e in surviving if e.labels.reached(stage)]
        steps.append(Step(stage, arrived, len(through)))
        surviving = through
    return tuple(steps)


def end_to_end(steps: Sequence[Step]) -> float:
    """Chance of clearing the whole chain, over the rungs there is evidence for.

    When a rung further down was never reached, this is an upper bound rather
    than an estimate: it is what the chain would achieve if the unexercised part
    were free. :func:`fully_measured` says whether that caveat applies.
    """
    rate = 1.0
    for step in steps:
        rate *= step.chained
    return rate


def fully_measured(steps: Sequence[Step]) -> bool:
    """Was every rung actually exercised?"""
    return all(step.measured for step in steps)


def unmeasured(steps: Sequence[Step]) -> tuple[str, ...]:
    return tuple(step.name for step in steps if not step.measured)


def uplift(steps: Sequence[Step], target: str) -> float:
    """What perfecting one rung alone would add, holding the rest fixed."""
    baseline = end_to_end(steps)
    perfected = 1.0
    for step in steps:
        perfected *= 1.0 if step.name == target else step.chained
    return perfected - baseline


class Funnel(FeedbackModule):
    def __init__(self, stages: Sequence[str] | None = None):
        self.stages = tuple(stages) if stages else None

    def descriptor(self) -> Descriptor:
        return feedback_descriptor("funnel", VERSION, min_cohorts=1, prescribes=True)

    def requirement(self) -> Requirement:
        return requires_channels(
            "funnel",
            "feedback",
            capabilities={"stage_events": True},
            description="conditional stage rates and counterfactual uplift",
        )

    def stages_for(self, cohort: Cohort) -> tuple[tuple[str, ...], str]:
        """The vocabulary to build the funnel over, and where it came from.

        Configured beats declared beats inferred. Inference is last because it
        can only ever see milestones something reached, and the interesting case
        is the one nothing reached.
        """
        if self.stages:
            return self.stages, "configured"
        declared = declared_stages(cohort)
        if declared:
            return declared, "declared by the run"
        return stage_order(cohort.episodes), "inferred from what happened"

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings, measurements, notes = [], {}, []
        for cohort in cohorts:
            if not cohort.has_stage_events:
                notes.append(f"{cohort.name}: no stage events, so there is no funnel to build")
                continue
            stages, origin = self.stages_for(cohort)
            if origin.startswith("inferred"):
                notes.append(
                    f"{cohort.name}: milestone vocabulary was inferred from the episodes, "
                    "so a milestone nothing ever reached is invisible here"
                )
            steps = build(cohort.episodes, stages)
            if not steps:
                continue
            overall = end_to_end(steps)
            never_reached = unmeasured(steps)
            if never_reached:
                notes.append(
                    f"{cohort.name}: nothing reached {', '.join(never_reached)}, so those "
                    "rungs have no evidence either way; the end-to-end rate and the "
                    "uplift below are best cases that assume they are free"
                )
            measurements[f"{cohort.name}.end_to_end"] = Measurement(
                overall,
                n=len(cohort),
                units="fraction",
                method="chained conditional rates"
                + ("" if not never_reached else ", upper bound: some rungs unexercised"),
            )
            for step in steps:
                measurements[f"{cohort.name}.{step.name}"] = step.measurement()

            uplifts = {step.name: uplift(steps, step.name) for step in steps}
            weakest = max(uplifts, key=uplifts.get)
            gain = uplifts[weakest]
            step = next(s for s in steps if s.name == weakest)

            findings.append(
                Finding(
                    code="funnel.bottleneck",
                    summary=(
                        f"{cohort.name}: {weakest} is the weak link at "
                        f"{step.rate:.0%} ({step.passed}/{step.arrived}); "
                        f"perfecting it alone would add {gain:.1%} end to end"
                    ),
                    severity="strong" if gain >= 0.15 else "weak",
                    measurements={weakest: step.measurement()},
                    evidence={
                        "stages": [s.name for s in steps],
                        "rates": {s.name: s.rate for s in steps},
                        "uplift": uplifts,
                        "end_to_end": overall,
                    },
                    prescription=(
                        f"Before collecting data for {weakest}, check whether other cohorts "
                        f"fail there too. A weak rung that is weak everywhere is a property "
                        f"of how the policy is run, not of what it was taught."
                    ),
                    cohorts=(cohort.name,),
                )
            )
        return Report(
            "funnel", tuple(findings), measurements, tuple(notes), tuple(c.name for c in cohorts)
        )
