"""What to change about the filming, ordered by what it costs to ignore.

Generic advice is a blog post. Advice with the cost of ignoring it attached is
worth reading, and that is the only thing this module tries to be: not "keep your
hands in frame" but "your hands were out of frame for 22% of your footage, which
is roughly one clip in five discarded, and it is the largest single thing
separating this upload from the ones above it".

Where the numbers come from
---------------------------
Nowhere new. Every signal here was computed by something that already had to look
at the data — the connector counted scenes and durations, the hand estimator
reported how often it found a hand, the retargeter reported what fraction of the
reaching was inside the arm's workspace. This module reads those off the record
and turns them into sentences. It never opens a video, which is what keeps it
cheap enough to run on every upload and honest enough that a user can trace every
claim back to a measurement.

The honesty guard that matters
------------------------------
The cost attached to each fix is **correlational across datasets, not causal for
theirs**. Datasets scoring above them have hands off-frame 4% of the time; that
does not license "fix this and you gain five points". So the finding says what
the better datasets look like and stops, and :data:`CAUSAL_WORDS` exists as a
reminder of the phrasing this module is not allowed to use. It is the same rule
the rest of the feedback plane keeps — describe, do not promise — and it is
easier to break here than anywhere else, because a prescription *wants* to
promise something.

Ordering is the product
-----------------------
A list of eleven things wrong with somebody's footage is not usable. Three, in
order, with the biggest first, is. Severity is assigned from measured cost rather
than from how bad each defect sounds, so a small amount of a very expensive
problem outranks a large amount of a cheap one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement

VERSION = "0.1.0.dev0"

#: Phrasings this module must not use. Here as documentation rather than as a
#: filter — the point is that somebody editing a prescription reads them and
#: remembers why. The cost figures are correlational across datasets; they do not
#: license a causal promise about this one.
CAUSAL_WORDS = (
    "will gain",
    "you would score",
    "guarantees",
    "fix this and you",
)

#: How many fixes to lead with. More than this and the list stops being a list.
LEAD = 3


@dataclass(frozen=True)
class Check:
    """One measurable property of a capture, and what to say when it is off.

    A dataclass rather than a method per defect, because the set of things worth
    checking will grow and every one of them is the same shape: read a number,
    compare it to a threshold, say what it costs and what to do. A new check is
    an entry in :data:`CHECKS`.
    """

    #: The annotation or metadata key this reads.
    key: str
    code: str
    #: What counts as a problem. ``below`` means smaller is worse.
    threshold: float
    below: bool = True
    #: How much of the data this typically costs, as a fraction, when at the
    #: threshold. Used only to order the findings.
    weight: float = 1.0
    summary: str = ""
    fix: str = ""
    why: str = ""

    def fires(self, value: float) -> bool:
        return value < self.threshold if self.below else value > self.threshold

    def cost(self, value: float) -> float:
        """How far past the bar, as a fraction of the bar, times what it costs.

        A fraction rather than a raw distance because the thresholds are not on
        one scale: 0.9 for hand visibility and 0.05 for instruction variety are
        both "the bar", and a raw gap would make the check with the highest
        threshold always look worse. Missing the bar by a tenth of itself is the
        same amount of wrong either way.
        """
        distance = (self.threshold - value) if self.below else (value - self.threshold)
        scale = abs(float(self.threshold)) or 1.0
        return max(0.0, float(distance)) / scale * float(self.weight)


#: The checks themselves. Ordered here for readability only — findings come out
#: ordered by measured cost, not by position in this list.
CHECKS: tuple[Check, ...] = (
    Check(
        key="hands_visible",
        code="capture.hands_offscreen",
        threshold=0.9,
        below=True,
        weight=3.0,
        summary="hands were out of frame for {missing:.0%} of the footage",
        fix="Keep both hands inside the frame while working. Do not reach for "
        "anything you cannot see.",
        why="A frame with no hand in it cannot produce a hand pose, so those "
        "steps are dropped before anything is trained.",
    ),
    Check(
        key="in_reach",
        code="capture.out_of_workspace",
        threshold=0.8,
        below=True,
        weight=2.5,
        summary="{missing:.0%} of the reaching happens where the arm cannot go",
        fix="Stand closer to what you are working on, and keep the work within "
        "about an arm's length in front of you.",
        why="A trajectory the robot physically cannot follow teaches it nothing, "
        "whatever else is right about the clip.",
    ),
    Check(
        # A count, not a per-clip ratio. As a ratio, six clips in one kitchen
        # scored better than forty in one kitchen and passed — but one location
        # is one location, and the advice is identical either way. The ratio
        # conflated "few clips" with "many locations".
        key="scenes",
        code="capture.single_scene",
        threshold=3.0,
        below=True,
        weight=2.0,
        summary="all of it was filmed in {scenes} location(s) across {clips} clips",
        fix="Film the same tasks somewhere else — a different room, different "
        "lighting, different objects of the same kind.",
        why="A policy trained on one room learns that room. This is usually the "
        "largest single difference between the uploads that transfer and the "
        "ones that do not.",
    ),
    Check(
        key="usable_length",
        code="capture.clips_truncated",
        threshold=0.85,
        below=True,
        weight=1.5,
        summary="{missing:.0%} of clips are too short to contain a whole attempt",
        fix="Start recording before you reach and stop after you let go.",
        why="A clip cut mid-motion has no beginning or no end, and a policy "
        "cannot learn an action it never sees completed.",
    ),
    Check(
        key="motion_ok",
        code="capture.too_fast",
        threshold=0.8,
        below=True,
        weight=1.5,
        summary="{missing:.0%} of the motion is faster than the arm can follow",
        fix="Move deliberately, at about the pace you would use to teach someone.",
        why="Human hands move several times faster than the arm does; fast motion "
        "is both blurred and unreachable as a target.",
    ),
    Check(
        # Also a count, for the same reason. The failure worth catching is one
        # sentence reused for a whole upload, and that is a property of the set
        # rather than of its size.
        key="instructions",
        code="capture.one_instruction",
        threshold=2.0,
        below=True,
        weight=1.0,
        summary="{unique} distinct instruction(s) across {clips} clips",
        fix="Describe each clip by what was actually done in it, rather than "
        "reusing one sentence for the whole upload.",
        why="A language-conditioned policy trains on the sentence. One sentence "
        "everywhere means the language carries no information.",
    ),
    Check(
        key="stabilized",
        code="capture.stabilized",
        threshold=0.0,
        below=False,
        weight=1.0,
        summary="device stabilisation was detected on {affected} clip(s)",
        fix="Turn stabilisation off and upload the original file, without cropping or filters.",
        why="Stabilisation invents camera motion to cancel real motion, which "
        "destroys the head trajectory the hand positions are measured against.",
    ),
    Check(
        key="labelled",
        code="capture.unlabelled_outcomes",
        threshold=0.5,
        below=True,
        weight=0.5,
        summary="{missing:.0%} of clips do not say whether the attempt worked",
        fix="Mark each clip as a success or a failure. Failures are useful — "
        "unlabelled ones are not.",
        why="A dataset that is quietly all failures trains nothing, and there is "
        "no way to tell from the footage alone.",
    ),
)


def measured(cohort: Cohort, checks: Sequence[Check] = CHECKS) -> dict[str, float]:
    """The capture signals, averaged over the cohort's episodes.

    ``checks`` is passed rather than read from the module constant, because a
    caller who added their own check should get it read — the version that
    consulted the global reported every custom check as "nothing to read", which
    is exactly the wrong answer and a quiet one.

    Read off annotations that earlier stages wrote. A signal nobody measured is
    absent rather than zero — the difference between "your hands were never
    visible" and "nobody looked" is the whole distinction this project keeps
    everywhere else, and it would be especially cruel to get wrong here.
    """
    totals: dict[str, list[float]] = {}
    for episode in cohort.episodes:
        labels = getattr(episode, "labels", None)
        annotations = dict(getattr(labels, "annotations", {}) or {})
        meta = getattr(episode, "meta", None)
        annotations.update(dict(getattr(meta, "extra", {}) or {}))
        for check in checks:
            value = annotations.get(check.key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals.setdefault(check.key, []).append(float(value))
            elif isinstance(value, bool):
                totals.setdefault(check.key, []).append(1.0 if value else 0.0)
    return {key: sum(values) / len(values) for key, values in totals.items() if values}


def variety(cohort: Cohort) -> dict[str, float]:
    """Scene and instruction variety, which are properties of the set not the clip.

    Reported as counts rather than as per-clip ratios. A ratio looked tidier and
    was wrong: six clips in one kitchen scored better than forty in one kitchen,
    when the problem and the advice are identical. The ratios are kept alongside
    for anything that wants them.
    """
    scenes: set[str] = set()
    instructions: set[str] = set()
    for episode in cohort.episodes:
        labels = getattr(episode, "labels", None)
        annotations = dict(getattr(labels, "annotations", {}) or {})
        meta = getattr(episode, "meta", None)
        annotations.update(dict(getattr(meta, "extra", {}) or {}))
        if annotations.get("scene"):
            scenes.add(str(annotations["scene"]))
        if annotations.get("instruction"):
            instructions.add(str(annotations["instruction"]))
    clips = max(1, len(cohort.episodes))
    return {
        "scenes": float(len(scenes)),
        "instructions": float(len(instructions)),
        "scene_variety": len(scenes) / clips,
        "instruction_variety": len(instructions) / clips,
        "_clips": float(clips),
    }


class Capture(FeedbackModule):
    """Filming advice, from measurements something else already took."""

    def __init__(
        self,
        *,
        checks: Sequence[Check] = CHECKS,
        better: Mapping[str, float] | None = None,
        lead: int = LEAD,
        history: Callable[[str], float | None] | None = None,
    ):
        """``better`` is what the datasets scoring above this one look like.

        Optional, and the findings are useful without it — they simply say what
        is wrong rather than how far from the field it is. Supplied, every finding
        gains one comparative sentence, phrased as an observation about those
        datasets and never as a promise about this one.
        """
        self._checks = tuple(checks)
        self._better = dict(better or {})
        self._lead = int(lead)
        self._history = history

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="capture",
            version=VERSION,
            min_cohorts=1,
            prescribes=True,
            holds=(),
            checks=[check.code for check in self._checks],
            comparative=bool(self._better or self._history),
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "capture",
            "feedback",
            description="capture measurements written by the ingest and retargeting steps",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        notes: list[str] = []

        for cohort in cohorts:
            signals = {**measured(cohort, self._checks), **variety(cohort)}
            counts = {
                "clips": int(signals.get("_clips", len(cohort.episodes))),
                "scenes": int(signals.get("scenes", 0)),
                "unique": int(signals.get("instructions", 0)),
            }
            unmeasured = [check.code for check in self._checks if check.key not in signals]
            scored: list[tuple[float, Check, float]] = []
            for check in self._checks:
                if check.key not in signals:
                    continue
                value = float(signals[check.key])
                measurements[f"{cohort.name}.{check.key}"] = Measurement(
                    value=round(value, 4),
                    n=counts["clips"],
                    method="mean over the cohort's clips",
                    detail={"threshold": check.threshold, "fires": check.fires(value)},
                )
                if check.fires(value):
                    scored.append((check.cost(value), check, value))

            scored.sort(key=lambda item: -item[0])
            for rank, (_cost, check, value) in enumerate(scored):
                findings.append(self._finding(cohort, check, value, counts, rank))

            if unmeasured:
                notes.append(
                    f"{cohort.name}: {len(unmeasured)} check(s) had nothing to read "
                    f"({', '.join(unmeasured[:4])}"
                    + (f" and {len(unmeasured) - 4} more" if len(unmeasured) > 4 else "")
                    + ") — not measured is not the same as fine"
                )
            if not scored and not unmeasured:
                findings.append(
                    Finding(
                        code="capture.clean",
                        summary=f"nothing in {cohort.name}'s filming stands out as fixable",
                        severity="info",
                        cohorts=(cohort.name,),
                    )
                )

        return Report(
            module="capture",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(cohort.name for cohort in cohorts),
        )

    # -- one finding -------------------------------------------------------

    def _finding(
        self,
        cohort: Cohort,
        check: Check,
        value: float,
        counts: Mapping[str, int],
        rank: int,
    ) -> Finding:
        context = {
            "missing": max(0.0, check.threshold - value) if check.below else value,
            "affected": int(round(value * counts["clips"])),
            **counts,
        }
        summary = check.summary.format(**context)
        prescription = f"{check.fix} {check.why}"

        reference = self._reference(check.key)
        evidence: dict[str, Any] = {
            "measured": round(value, 4),
            "threshold": check.threshold,
            "clips": counts["clips"],
        }
        if reference is not None:
            evidence["datasets_above_this_one"] = round(float(reference), 4)
            # Phrased as an observation about those datasets. Not "fix this and
            # you gain N" — the association is across datasets and says nothing
            # causal about this one.
            prescription += (
                f" For reference, uploads that scored above this one average {reference:.0%} here."
            )

        return Finding(
            # Only the leading few are strong. A list of eleven equally urgent
            # things is not a list.
            code=check.code,
            summary=f"{cohort.name}: {summary}",
            severity="strong" if rank < self._lead else "weak",
            evidence=evidence,
            prescription=prescription,
            cohorts=(cohort.name,),
        )

    def _reference(self, key: str) -> float | None:
        if key in self._better:
            return float(self._better[key])
        if self._history is not None:
            return self._history(key)
        return None


def top_fixes(report: Report, limit: int = LEAD) -> list[str]:
    """The leading prescriptions as plain sentences, for a GUI.

    A list of eleven things wrong with somebody's footage is not usable; three,
    in order, is.
    """
    strong = [finding for finding in report.findings if finding.severity == "strong"]
    return [f"{finding.summary} — {finding.prescription}" for finding in strong[:limit]]
