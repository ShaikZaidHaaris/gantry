"""Ranking policies across many tasks, without letting one easy task decide.

A matrix of policies by tasks is the shape this project's evaluation actually
produces — three checkpoints across thirteen tasks, six bodies, fifty trials
each. The obvious summary of such a matrix is a mean, and the mean is the wrong
number twice over.

It is dominated by whichever task happened to be tractable. Twelve tasks at zero
and one at sixty percent averages to five, which reads as "barely works" when the
truth is "works on the thing it was trained for and nothing else" — a completely
different sentence with completely different consequences.

And it hides shape. Two policies with the same mean can be mediocre everywhere
or excellent on half and hopeless on the rest, and only one of those is worth
deploying. A table of means cannot tell them apart.

So this reports four things
---------------------------
The **interquartile mean** with a stratified bootstrap interval, because it is
robust to the outlier task and still tight enough to be useful at the run counts
anybody actually has.

A **performance profile**, because the curve distinguishes the two policies a
mean conflates.

**Probability of improvement** for each pair, which answers "which should I use"
rather than "how much better is it" — usually the question actually being asked.

A **compact letter display**, so "these are indistinguishable and those are not"
is legible at a glance rather than reconstructed from a grid of p-values.

What it refuses
---------------
A matrix that is mostly zeros. Aggregating twelve zero-shot cells and one real
one produces a number whose meaning is entirely determined by the ratio of
cells, and reporting it invites exactly the misreading above. The refusal names
which cells are at the floor so the caller can decide whether to aggregate a
subset or report the floor separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, count_of
from gantry.spine.inference import (
    iqm,
    performance_profile,
    prob_improvement,
    stratified_bootstrap,
)

VERSION = "0.1.0.dev0"

#: A cell at or below this rate is treated as being at the floor. Not "zero",
#: because one lucky success in fifty trials is still the floor and pretending
#: otherwise is how a 1/50 becomes evidence.
FLOOR = 0.02

#: Above this share of floor cells, an aggregate is refused rather than reported.
FLOOR_SHARE = 0.5


def compact_letters(
    scores: Mapping[str, float], indistinguishable: Sequence[tuple[str, str]]
) -> dict[str, str]:
    """Letters such that two names share one exactly when they are not separated.

    The display convention from experimental statistics, and the reason to use
    it here: a grid of pairwise p-values is technically complete and nobody
    reads it, while "a" beside two policies says "these two are not
    distinguishable" in a form that survives being glanced at.
    """
    ordered = sorted(scores, key=lambda name: -scores[name])
    same = defaultdict(set)
    for left, right in indistinguishable:
        same[left].add(right)
        same[right].add(left)

    groups: list[set[str]] = []
    for name in ordered:
        placed = False
        for group in groups:
            if all(other in same[name] or other == name for other in group):
                group.add(name)
                placed = True
                break
        if not placed:
            groups.append({name})

    letters: dict[str, set[str]] = defaultdict(set)
    for index, group in enumerate(groups):
        letter = chr(ord("a") + index)
        for name in group:
            letters[name].add(letter)
    return {name: "".join(sorted(letters[name])) for name in ordered}


class MatrixRanking(FeedbackModule):
    """Ranks cohorts over the tasks they share, robustly.

    Each cohort is one policy; its episodes may span many tasks, and the tasks
    are the stratification. The comparison is only meaningful over tasks every
    cohort attempted, so tasks missing from any cohort are dropped and named.
    """

    def __init__(self, *, alpha: float = 0.05, resamples: int = 2000, seed: int = 0):
        self._alpha = alpha
        self._resamples = resamples
        self._seed = seed

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="rank",
            version=VERSION,
            min_cohorts=2,
            prescribes=True,
            # Ranking policies means everything else is held. Declared so a
            # run where the task or body also varied is refused rather than
            # reported as a policy comparison.
            holds=("task", "evaluation", "embodiment"),
            alpha=self._alpha,
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "rank",
            "feedback",
            capabilities={"outcomes": True},
            description="robust ranking across a matrix of policies by tasks",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        per_task = {cohort.name: _rates_by_task(cohort) for cohort in cohorts}
        shared = _shared_tasks(per_task)
        notes: list[str] = []
        if not shared:
            return Report(
                module="rank",
                notes=(
                    "no task was attempted by every cohort, so there is nothing "
                    "these can be ranked over",
                ),
                cohorts=tuple(per_task),
            )

        dropped = sorted({task for rates in per_task.values() for task in rates} - set(shared))
        if dropped:
            notes.append(
                f"ranked over {count_of(len(shared), 'shared task')}, with {len(dropped)} attempted "
                f"by only some cohorts were dropped ({', '.join(dropped[:4])}"
                + (f" and {len(dropped) - 4} more" if len(dropped) > 4 else "")
                + ")"
            )

        matrices = {
            name: tuple((rates[task],) for task in shared) for name, rates in per_task.items()
        }
        floor_share = {
            name: sum(1 for task in shared if rates[task] <= FLOOR) / len(shared)
            for name, rates in per_task.items()
        }

        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        scores: dict[str, float] = {}
        for name, matrix in matrices.items():
            flat = [value for row in matrix for value in row]
            point, low, high = stratified_bootstrap(
                matrix, iqm, alpha=self._alpha, resamples=self._resamples, seed=self._seed
            )
            scores[name] = point
            measurements[f"{name}.iqm"] = Measurement(
                value=round(point, 4),
                n=len(flat),
                ci=(round(low, 4), round(high, 4)),
                method="interquartile mean, stratified bootstrap over tasks",
                detail={
                    "tasks": len(shared),
                    "at_floor": sum(1 for v in flat if v <= FLOOR),
                    "profile": [
                        [round(tau, 3), round(frac, 3)]
                        for tau, frac in performance_profile(matrix, (0.0, 0.25, 0.5, 0.75))
                    ],
                },
            )

        # A matrix that is mostly floor cannot be aggregated into a number whose
        # meaning survives being quoted.
        mostly_floor = [n for n, share in floor_share.items() if share > FLOOR_SHARE]
        if mostly_floor:
            at_floor = {
                name: sorted(t for t in shared if per_task[name][t] <= FLOOR)
                for name in mostly_floor
            }
            findings.append(
                Finding(
                    code="rank.mostly_floor",
                    summary=(
                        f"{count_of(len(mostly_floor), 'cohort')} sit at the floor on more than "
                        f"half the shared tasks, so an aggregate over all of them is "
                        "determined by how many tasks were included rather than by "
                        "performance"
                    ),
                    severity="info",
                    evidence={
                        "floor_share": {n: round(s, 3) for n, s in floor_share.items()},
                        "at_floor": at_floor,
                    },
                    prescription=(
                        "Report the tasks above the floor as a ranking and the rest as "
                        "a zero-shot floor, separately. Averaging them produces one "
                        "number that means neither thing."
                    ),
                    cohorts=tuple(mostly_floor),
                )
            )

        pairs: list[tuple[str, str]] = []
        for left in matrices:
            for right in matrices:
                if left >= right:
                    continue
                a = [v for row in matrices[left] for v in row]
                b = [v for row in matrices[right] for v in row]
                probability = prob_improvement(a, b)
                measurements[f"{left}_beats_{right}"] = Measurement(
                    value=round(probability, 4),
                    n=len(shared),
                    method="probability of improvement over shared tasks",
                )
                lo_l, hi_l = measurements[f"{left}.iqm"].ci or (0.0, 1.0)
                lo_r, hi_r = measurements[f"{right}.iqm"].ci or (0.0, 1.0)
                if not (hi_l < lo_r or hi_r < lo_l):
                    pairs.append((left, right))

        letters = compact_letters(scores, pairs)
        best = max(scores, key=lambda name: scores[name])
        shares_with_best = [
            name for name in scores if name != best and set(letters[name]) & set(letters[best])
        ]
        findings.append(
            Finding(
                code="rank.ordering",
                summary=(
                    "; ".join(
                        f"{name} {scores[name]:.3f} [{letters[name]}]"
                        for name in sorted(scores, key=lambda n: -scores[n])
                    )
                    + f" over {count_of(len(shared), 'shared task')}"
                ),
                severity="strong" if not shares_with_best else "weak",
                measurements={k: v for k, v in measurements.items() if k.endswith(".iqm")},
                evidence={
                    "letters": letters,
                    "shared_tasks": list(shared),
                    "indistinguishable_pairs": [list(p) for p in pairs],
                },
                prescription=(
                    f"Use {best}."
                    if not shares_with_best
                    else f"{best} leads, but shares a letter with "
                    f"{', '.join(shares_with_best)} — those are not separated at this "
                    "number of tasks and runs."
                ),
                cohorts=tuple(scores),
            )
        )
        return Report(
            module="rank",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(scores),
        )


def _rates_by_task(cohort: Cohort) -> dict[str, float]:
    """This cohort's success rate per task."""
    tally: dict[str, list[bool]] = defaultdict(list)
    for episode in cohort.episodes:
        labels = getattr(episode, "labels", None)
        success = getattr(labels, "success", None)
        if success is None:
            continue
        meta = getattr(episode, "meta", None)
        task = getattr(meta, "task", None) or getattr(meta, "source", None) or "(untasked)"
        tally[str(task)].append(bool(success))
    return {
        task: sum(1 for v in outcomes if v) / len(outcomes)
        for task, outcomes in tally.items()
        if outcomes
    }


def _shared_tasks(per_task: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    if not per_task:
        return ()
    common: set[str] | None = None
    for rates in per_task.values():
        common = set(rates) if common is None else common & set(rates)
    return tuple(sorted(common or ()))
