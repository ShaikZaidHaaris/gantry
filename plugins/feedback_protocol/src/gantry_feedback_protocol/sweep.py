"""Which execution setting was best, and how much it was worth.

Everything else in the feedback plane asks what the *data* is like. This asks
what the *protocol* is like, and it exists because that question is routinely
skipped and routinely worth more than the answer people go looking for.

In this project's own benchmark, changing how many actions of each predicted
chunk were executed before re-planning moved a fixed policy on fixed data by
fourteen points — no retraining, no collection, no new demonstrations. Nobody
had a module for it, so it was found by hand and could easily not have been.

How it works
------------
Cohorts are runs of the same policy on the same task at *different* protocols.
The module reads each cohort's protocol out of its provenance, checks that only
one setting varies, and ranks. Where the runs share seeds it compares them
paired, because the same scenes under two settings is a far tighter comparison
than two independent samples and the pairing is free.

What it refuses
---------------
Anything where the comparison would not mean what it looks like. Cohorts that
differ in policy or evaluator as well as protocol are not a protocol sweep, they
are a confound, and reporting a winner would attribute to the setting whatever
the other difference did. Cohorts carrying no protocol at all are not a sweep
either — that is a run that never recorded how it was executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.feedback import Cohort, FeedbackModule, Finding, Report, feedback_descriptor
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, Verdict

VERSION = "0.1.0.dev0"

#: Protocol keys that describe *how* a run was executed rather than what it ran.
#: These are the levers; anything else varying makes it not a sweep.
LEVERS = ("execute", "epochs", "seed_base", "max_trials", "horizon")

#: Planes that must be identical across the cohorts. A "protocol sweep" where
#: the policy also changed is measuring the policy.
HELD = ("policy", "evaluation")

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Arm:
    """One cohort, with the setting it was run at and how it scored."""

    name: str
    protocol: Mapping[str, Any]
    outcomes: tuple[bool, ...]
    by_scene: Mapping[str, bool]

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def rate(self) -> float:
        return sum(self.outcomes) / self.n if self.n else 0.0


def _scene_of(episode: Any) -> str:
    """The scene an episode attempted, so two arms can be paired on it."""
    annotations = episode.labels.annotations
    for key in ("scene", "scene_id"):
        if key in annotations:
            return str(annotations[key])
    return episode.meta.id.split("#", 1)[0]


def _arm(cohort: Cohort) -> Arm | None:
    outcomes = [e.labels.success for e in cohort.episodes if e.labels.success is not None]
    if not outcomes:
        return None
    protocol = dict(cohort.provenance.protocol) if cohort.provenance else {}
    by_scene = {
        _scene_of(e): bool(e.labels.success)
        for e in cohort.episodes
        if e.labels.success is not None
    }
    return Arm(cohort.name, protocol, tuple(bool(o) for o in outcomes), by_scene)


def varying(arms: Sequence[Arm]) -> tuple[str, ...]:
    """Protocol keys that differ across the arms."""
    keys = {key for arm in arms for key in arm.protocol}
    return tuple(
        sorted(
            key
            for key in keys
            if len({repr(arm.protocol.get(key)) for arm in arms}) > 1
        )
    )


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired(best: Arm, other: Arm) -> tuple[int, int, int, int] | None:
    """Counts over the scenes both arms attempted, or ``None`` if they share none."""
    shared = set(best.by_scene) & set(other.by_scene)
    if not shared:
        return None
    both = sum(best.by_scene[s] and other.by_scene[s] for s in shared)
    only_best = sum(best.by_scene[s] and not other.by_scene[s] for s in shared)
    only_other = sum(other.by_scene[s] and not best.by_scene[s] for s in shared)
    neither = len(shared) - both - only_best - only_other
    return both, only_best, only_other, neither


def mcnemar(only_best: int, only_other: int) -> float:
    """Exact two-sided p-value over the scenes where the arms disagreed."""
    from math import comb

    discordant = only_best + only_other
    if discordant == 0:
        return 1.0
    smaller = min(only_best, only_other)
    tail = sum(comb(discordant, k) for k in range(smaller + 1)) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


class ProtocolSweep(FeedbackModule):
    """Rank runs of one policy at different execution settings."""

    def __init__(self, levers: Sequence[str] = LEVERS):
        self.levers = tuple(levers)

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            "protocol", VERSION, min_cohorts=2, prescribes=True,
            # A sweep where the policy also changed is measuring the policy.
            # Declared rather than hand-checked: the base contract verifies it
            # from provenance for every module that declares it, so this one
            # stopped carrying its own copy of the logic.
            holds=HELD,
            levers=list(self.levers),
        )

    def requirement(self) -> Requirement:
        # Outcomes only. This module reads how a run was executed and whether it
        # worked, never what the trajectory looked like.
        return requires_channels(
            "protocol",
            "feedback",
            capabilities={"outcomes": True},
            description="which execution setting was best, and by how much",
        )

    def check_inputs(self, cohorts: Sequence[Cohort]) -> Verdict:
        checks = [super().check_inputs(cohorts)]
        arms = [arm for arm in (_arm(c) for c in cohorts) if arm is not None]
        if len(arms) < 2:
            checks.append(
                Verdict.no(
                    "protocol.no_outcomes",
                    "at least two cohorts must carry outcomes; a run that recorded no "
                    "result cannot be ranked",
                )
            )
            return Verdict.all(checks)
        if not any(arm.protocol for arm in arms):
            checks.append(
                Verdict.no(
                    "protocol.unrecorded",
                    "no cohort carries a protocol, so there is no setting to compare",
                    hint="the evaluator should record how it executed the run",
                )
            )
        return Verdict.all(checks)

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        arms = [arm for arm in (_arm(c) for c in cohorts) if arm is not None]
        swept = [key for key in varying(arms) if key in self.levers]
        notes: list[str] = []
        measurements: dict[str, Measurement] = {}

        for arm in arms:
            successes = sum(arm.outcomes)
            measurements[f"{arm.name}.success_rate"] = Measurement(
                arm.rate,
                n=arm.n,
                ci=wilson(successes, arm.n),
                units="fraction",
                method="wilson",
                detail={key: arm.protocol.get(key) for key in swept},
            )

        if not swept:
            notes.append(
                "no protocol lever varies across these cohorts, so there is nothing to "
                "sweep; they were run the same way"
            )
            return Report("protocol", (), measurements, tuple(notes),
                          tuple(c.name for c in cohorts))
        if len(swept) > 1:
            notes.append(
                f"{len(swept)} levers vary at once ({', '.join(swept)}), so a winner "
                "cannot be attributed to any one of them"
            )

        ranked = sorted(arms, key=lambda arm: arm.rate, reverse=True)
        best, worst = ranked[0], ranked[-1]
        findings = [self._verdict(best, worst, swept, len(swept) == 1)]
        notes.append(
            "ranked by success rate; every arm ran the same policy on the same task"
        )
        return Report("protocol", tuple(findings), measurements, tuple(notes),
                      tuple(c.name for c in cohorts))

    def _verdict(self, best: Arm, worst: Arm, swept: Sequence[str], single: bool) -> Finding:
        gain = best.rate - worst.rate
        setting = ", ".join(f"{key}={best.protocol.get(key)!r}" for key in swept)
        counts = paired(best, worst)
        evidence: dict[str, Any] = {
            "levers": list(swept),
            "best": best.name,
            "worst": worst.name,
            "best_setting": {key: best.protocol.get(key) for key in swept},
            "worst_setting": {key: worst.protocol.get(key) for key in swept},
            "gain": gain,
        }
        if counts:
            both, only_best, only_worst, neither = counts
            evidence.update(
                paired_scenes=both + only_best + only_worst + neither,
                won=only_best,
                lost=only_worst,
                p=mcnemar(only_best, only_worst),
            )
            comparison = (
                f"paired on {evidence['paired_scenes']} shared scenes, "
                f"won {only_best} lost {only_worst}, p={evidence['p']:.3g}"
            )
        else:
            comparison = "compared unpaired: the arms share no scene"
            evidence["paired_scenes"] = 0

        return Finding(
            code="protocol.best_setting",
            summary=(
                f"{best.name} is the best setting ({setting}) at {best.rate:.1%}, "
                f"{gain:+.1%} over {worst.name}; {comparison}"
            ),
            severity="strong" if gain >= 0.05 and single else "weak",
            measurements={best.name: Measurement(
                best.rate, n=best.n, ci=wilson(sum(best.outcomes), best.n),
                units="fraction", method="wilson",
            )},
            evidence=evidence,
            prescription=(
                f"Run at {setting}. It costs nothing: same policy, same data, "
                f"same task, {gain:+.1%} for changing how the run is executed."
                if single and gain > 0
                else None
            ),
            cohorts=(best.name, worst.name),
        )
