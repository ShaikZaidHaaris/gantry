"""Does the data even contain the thing the benchmark is asking about?

The question this module exists for: somebody contributes a dataset of cooking
footage, it is evaluated on tabletop pick-and-place, the measured improvement is
zero, and a report goes out saying their data did not help. That report is
false. Their data was never given a chance to help, and nothing in the delta
distinguishes "this data is poor" from "this data is about something else".

Those two conclusions lead to opposite actions -- throw it away, or evaluate it on
something relevant -- so conflating them is not a rounding error, it is the single
most unfair thing a data-quality product can do. This runs before the delta is
interpreted, and where the overlap is low it says the number is about task match
rather than about quality.

The other direction matters too
-------------------------------
Coverage is asymmetric and both halves are worth reporting.

*Uncovered evaluation*: tasks the benchmark asks about that the data never
touches. Those cells are measuring transfer, not learning, and pooling them into
one mean makes an average that means neither.

*Unused data*: clips about things nothing in the benchmark evaluates. This is the
half people forget, and it is the more actionable one -- "31% of your footage is
about tasks we cannot see" is a statement about the benchmark's blind spot as
much as the contributor's, and it is the honest thing to tell somebody who paid
attention while filming.

How similarity is measured, and why the default is the dull one
---------------------------------------------------------------
By default: shared content words between instructions, with the verb overlap
reported separately because "pick up the mug" and "pick up the cube" are much
closer than "pick up the mug" and "open the drawer", and a bag of words alone
half-hides that.

It is crude, and it is the default because it is *inspectable*. A user told their
data does not match the benchmark is entitled to ask why, and "these words
appeared in your clips and these in the tasks" is an answer they can check. An
embedding model gives better numbers and no such answer.

``similarity`` is an argument, so a caller with a sentence encoder passes one in
and the method lands in the record. What is not negotiable is that the method
appears in the finding: two runs measured with different notions of similar are
not comparable, and the number alone does not say which was used.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, count_of

VERSION = "0.1.0.dev0"

#: Below this, an aggregate delta is reported as a task-match statement rather
#: than a data-quality one. Not tuned -- chosen as "less than a third of the
#: evaluated tasks have anything to do with this data", which is the point at
#: which the mean is dominated by cells that were never in scope.
COVERED = 0.34

#: Above this, coverage is not worth mentioning at all.
AMPLE = 0.75

#: Per-task, how similar an instruction has to be before the data counts as
#: touching that task.
MATCH = 0.34

#: Words carrying no task content. Deliberately short: a long stoplist starts
#: making judgements about which words matter, and the whole argument for the
#: default measure is that a user can check it.
STOPWORDS = frozenset(
    """a an the this that these those to of in on at from with and or for
    is are was were be been being it its his her their my your our
    into onto up down over under please then next now""".split()
)

#: Verbs common in manipulation instructions. Used to report verb overlap
#: separately, never to filter -- an instruction whose verb is not here is not
#: thereby less of an instruction.
VERBS = frozenset(
    """pick place put grasp grab lift move push pull open close press turn
    rotate slide insert stack pour wipe fold hang drop release reach hold
    carry transfer flip twist unscrew screw plug unplug serve cut stir""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> tuple[str, ...]:
    """Content words of an instruction, lowercased, in order, without stopwords."""
    return tuple(word for word in _WORD.findall(str(text).lower()) if word not in STOPWORDS)


def overlap(left: str, right: str) -> float:
    """Shared content words as a fraction of the smaller instruction.

    Over the *smaller* rather than the union, on purpose: "pick up the mug" and
    "pick up the red mug from the second shelf and put it in the sink" describe
    overlapping activity, and a Jaccard would score that pair low because one
    sentence is longer. Asymmetry is not a problem here -- the question is whether
    the shorter thing is contained in the longer one.
    """
    a, b = set(words(left)), set(words(right))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def verbs_of(text: str) -> frozenset[str]:
    return frozenset(word for word in words(text) if word in VERBS)


def instructions_of(cohort: Cohort) -> tuple[str, ...]:
    """Every instruction in a cohort, in order, including repeats.

    Repeats are kept because they are the weighting: a dataset with forty clips
    of one instruction and one of another is mostly about the first, and
    deduplicating would report it as evenly split between two things.
    """
    out: list[str] = []
    for episode in cohort.episodes:
        labels = getattr(episode, "labels", None)
        annotations = getattr(labels, "annotations", {}) or {}
        text = annotations.get("instruction") or getattr(
            getattr(episode, "meta", None), "task", None
        )
        if text:
            out.append(str(text))
    return tuple(out)


class Coverage(FeedbackModule):
    """Whether a dataset is about the tasks it is being evaluated on."""

    def __init__(
        self,
        *,
        evaluates: Sequence[str] = (),
        similarity: Callable[[str, str], float] = overlap,
        method: str = "shared content words, over the shorter instruction",
        match: float = MATCH,
        covered: float = COVERED,
    ):
        """``evaluates`` is what the benchmark asks about, as instructions.

        Supplied rather than discovered, because the evaluation suite is not in
        this module's hands and a coverage number computed against the wrong task
        list would be worse than none. An empty list produces a refusal rather
        than a coverage of zero -- "nothing to compare against" and "no overlap"
        are different sentences with different fixes.
        """
        self._evaluates = tuple(str(text) for text in evaluates if str(text).strip())
        self._similarity = similarity
        self._method = method
        self._match = float(match)
        self._covered = float(covered)

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="coverage",
            version=VERSION,
            # One dataset at a time. This is not a comparison -- it is a property
            # of one dataset against one benchmark.
            min_cohorts=1,
            prescribes=True,
            holds=(),
            method=self._method,
            evaluates=len(self._evaluates),
            match_threshold=self._match,
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "coverage",
            "feedback",
            description="instructions on the episodes, and the task list being evaluated",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        if not self._evaluates:
            return Report(
                module="coverage",
                findings=(
                    Finding(
                        code="coverage.no_task_list",
                        summary=(
                            "no evaluation task list was supplied, so whether this "
                            "data is about what is being measured cannot be assessed"
                        ),
                        severity="info",
                        prescription=(
                            "Pass the instructions the benchmark evaluates. Without "
                            "them a low delta cannot be told apart from a task "
                            "mismatch, and reporting one as the other is the most "
                            "unfair thing this pipeline can do."
                        ),
                    ),
                ),
                cohorts=tuple(cohort.name for cohort in cohorts),
            )

        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        notes: list[str] = []

        for cohort in cohorts:
            said = instructions_of(cohort)
            if not said:
                findings.append(
                    Finding(
                        code="coverage.no_instructions",
                        summary=(
                            f"{cohort.name} carries no instructions, so there is "
                            "nothing to compare against the task list"
                        ),
                        severity="strong",
                        prescription=(
                            "Every clip needs a sentence saying what was being done. "
                            "Without it a language-conditioned policy has nothing to "
                            "condition on and coverage cannot be measured at all."
                        ),
                        cohorts=(cohort.name,),
                    )
                )
                continue

            result = self._assess(said)
            measurements[f"{cohort.name}.coverage"] = Measurement(
                value=round(result["covered_fraction"], 4),
                n=len(self._evaluates),
                method=self._method,
                detail={
                    "clips": len(said),
                    "tasks_evaluated": len(self._evaluates),
                    "tasks_touched": len(result["touched"]),
                    "unused_clip_fraction": round(result["unused_fraction"], 4),
                },
            )
            measurements[f"{cohort.name}.verb_coverage"] = Measurement(
                value=round(result["verb_fraction"], 4),
                n=len(self._evaluates),
                method="fraction of evaluated tasks whose verb appears in the data",
            )
            findings.extend(self._findings_for(cohort, said, result))
            notes.append(
                f"{cohort.name}: {count_of(len(said), 'clip')} against "
                f"{count_of(len(self._evaluates), 'evaluated task')}"
            )

        return Report(
            module="coverage",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(cohort.name for cohort in cohorts),
        )

    # -- the assessment ----------------------------------------------------

    def _assess(self, said: Sequence[str]) -> dict[str, Any]:
        """Both directions of the overlap, plus what to say about each."""
        touched: list[str] = []
        untouched: list[str] = []
        best_for_task: dict[str, float] = {}
        for task in self._evaluates:
            best = max((self._similarity(task, text) for text in said), default=0.0)
            best_for_task[task] = round(float(best), 3)
            (touched if best >= self._match else untouched).append(task)

        used: list[str] = []
        unused: list[str] = []
        for text in said:
            best = max((self._similarity(text, task) for task in self._evaluates), default=0.0)
            (used if best >= self._match else unused).append(text)

        task_verbs = {verb for task in self._evaluates for verb in verbs_of(task)}
        said_verbs = {verb for text in said for verb in verbs_of(text)}
        verb_hits = sum(1 for task in self._evaluates if verbs_of(task) & said_verbs)
        return {
            "covered_fraction": len(touched) / len(self._evaluates),
            "touched": touched,
            "untouched": untouched,
            "unused_fraction": len(unused) / max(1, len(said)),
            "unused": unused,
            "per_task": best_for_task,
            "verb_fraction": verb_hits / len(self._evaluates),
            "missing_verbs": sorted(task_verbs - said_verbs),
            "extra_verbs": sorted(said_verbs - task_verbs),
        }

    def _findings_for(
        self, cohort: Cohort, said: Sequence[str], result: Mapping[str, Any]
    ) -> list[Finding]:
        findings: list[Finding] = []
        fraction = float(result["covered_fraction"])
        untouched = list(result["untouched"])
        common = [text for text, _ in Counter(result["unused"]).most_common(5)]

        if fraction < self._covered:
            findings.append(
                Finding(
                    code="coverage.mismatch",
                    summary=(
                        f"{cohort.name} touches {len(result['touched'])} of "
                        f"{count_of(len(self._evaluates), 'evaluated task')} "
                        f"({fraction:.0%}); a delta measured across all of them is "
                        "mostly a statement about task match rather than about this "
                        "data"
                    ),
                    severity="strong",
                    evidence={
                        "untouched": untouched[:10],
                        "per_task_best_match": dict(result["per_task"]),
                        "missing_verbs": result["missing_verbs"],
                        "method": self._method,
                    },
                    prescription=(
                        "Report the delta only over the tasks this data touches, and "
                        "say plainly that the rest were not in scope. A pooled number "
                        "here would read as 'this data did not help' when the truth "
                        f"is that {len(untouched)} of the tasks were never about it. "
                        "If a general claim is wanted, evaluate on a suite that "
                        "includes what this data actually shows."
                    ),
                    cohorts=(cohort.name,),
                )
            )
        elif fraction < AMPLE:
            findings.append(
                Finding(
                    code="coverage.partial",
                    summary=(
                        f"{cohort.name} touches {fraction:.0%} of the evaluated tasks; "
                        f"{len(untouched)} are outside what it shows"
                    ),
                    severity="weak",
                    evidence={"untouched": untouched[:10], "method": self._method},
                    prescription=(
                        "Split the table: the touched tasks measure what this data "
                        "taught, the rest measure transfer. Those are different claims "
                        "and one mean over both supports neither."
                    ),
                    cohorts=(cohort.name,),
                )
            )

        unused = float(result["unused_fraction"])
        if unused > 0.2:
            findings.append(
                Finding(
                    code="coverage.unused_data",
                    summary=(
                        f"{unused:.0%} of {cohort.name}'s clips are about things "
                        "nothing in this benchmark evaluates"
                    ),
                    severity="info",
                    evidence={"examples": common, "method": self._method},
                    prescription=(
                        "This is the benchmark's blind spot as much as the "
                        "contributor's, and it is worth saying so: that footage may be "
                        "the most valuable part of the upload and no number here can "
                        "see it. Either add tasks that cover it, or state that the "
                        "measured delta rests on the remaining "
                        f"{1 - unused:.0%} of the data."
                    ),
                    cohorts=(cohort.name,),
                )
            )

        if not findings:
            findings.append(
                Finding(
                    code="coverage.ample",
                    summary=(
                        f"{cohort.name} covers {fraction:.0%} of the evaluated tasks; "
                        "a delta over them is about this data rather than about scope"
                    ),
                    severity="info",
                    evidence={"method": self._method},
                    cohorts=(cohort.name,),
                )
            )
        return findings


def report_for(cohort: Cohort, evaluates: Sequence[str], **kwargs: Any) -> Report:
    """One dataset against one task list, in a line. For a GUI."""
    return Coverage(evaluates=evaluates, **kwargs).analyse([cohort])
