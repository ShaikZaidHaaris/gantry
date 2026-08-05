"""Whether the pipeline read the footage well -- a different question from whether
the footage was good.

``feedback_capture`` tells a contributor what to change about their filming.
This tells *us* what to change about our extraction, and the separation is the
whole point: those two failures look identical in the final numbers and have
completely different owners.

A clip that yields nothing might have had no hands in it, or it might have had
hands the estimator could not find. The first is the contributor's to fix by
filming differently; the second is ours to fix by using a better model, and
telling a contributor to re-film because our detector was weak is both wrong and
the kind of wrong that loses a customer. Every finding here is addressed to the
operator of the pipeline.

The signals are all recorded already
------------------------------------
Nothing here computes anything from video. The estimator wrote how often it found
a hand and how well its poses reprojected; the retargeter wrote what fraction of
reaching was reachable; the action step wrote how much was held and how much was
dropped; the intrinsics carry whether they were calibrated or assumed. This reads
those off the record and says which of them is currently limiting the dataset.

Ranked by what is actually costing you
--------------------------------------
The useful output is not a list of everything imperfect -- it is *which one thing
to fix next*. So findings are ordered by how much data each defect is losing, and
the module is explicit that a defect losing 2% of frames is not worth engineering
time while one losing 40% is.

Assumptions are findings too
----------------------------
An intrinsics source of ``fov`` rather than ``calibrated`` is not an error and
does not lose a single frame -- it puts a systematic few-percent error on every
distance in the dataset, invisibly and forever. That is worth surfacing precisely
because nothing else will ever complain about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

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

#: Below this share of data lost, a defect is not worth engineering time. Stated
#: rather than implied, because the alternative is a report where eleven things
#: are wrong and none is ranked.
WORTH_FIXING = 0.05


@dataclass(frozen=True)
class Stage:
    """One step of the extraction, and how much it is currently losing."""

    key: str
    code: str
    stage: str
    #: Read as "what fraction survived". Lower is worse.
    summary: str
    #: What we would change. Addressed to whoever runs the pipeline.
    fix: str
    #: Why it is lost, so the fix is arguable rather than followed.
    why: str
    floor: float = 0.9


#: The extraction, in order. Each entry names a signal an earlier stage wrote.
STAGES: tuple[Stage, ...] = (
    Stage(
        key="hands_visible",
        code="extraction.detector_missing_hands",
        stage="detection",
        summary="the detector found a hand in {value:.0%} of frames",
        fix="Try a stronger 2D detector before anything else. RTMPose through "
        "rtmlib is Apache-2.0 and found hands in every frame of footage where "
        "MediaPipe managed two thirds.",
        why="A frame with no detection cannot produce a pose, so this is a hard "
        "ceiling on everything downstream, because no later stage can recover a frame "
        "the detector never saw.",
        floor=0.85,
    ),
    Stage(
        key="pose_solved",
        code="extraction.pose_not_solving",
        stage="metric pose",
        summary="a metric pose was recovered for {value:.0%} of detected hands",
        fix="Check the reprojection budget before assuming the poses are bad. A "
        "fixed pixel budget is a different standard at every distance; a budget "
        "relative to the hand's size in frame is the same standard everywhere.",
        why="Detection succeeding and pose failing means the 2D and the 3D model "
        "disagree. Usually a threshold demanding better agreement than two "
        "independent detectors can give each other.",
        floor=0.7,
    ),
    Stage(
        key="pose_plausible",
        code="extraction.poses_implausible",
        stage="metric pose",
        summary="{value:.0%} of recovered poses put the hand somewhere a hand can be",
        fix="Check the camera intrinsics first. Every distance scales with the "
        "focal length, so a wrong one moves the whole trajectory somewhere "
        "impossible while every internal check still passes.",
        why="A pose can reproject perfectly from an impossible place, at a large "
        "enough distance the projection barely changes, so only a statement "
        "about where hands physically are catches this.",
        floor=0.9,
    ),
    Stage(
        key="in_reach",
        code="extraction.outside_the_workspace",
        stage="retargeting",
        summary="{value:.0%} of the reaching lands where the arm can go",
        fix="This one is usually genuine rather than a pipeline fault, and it is "
        "the contributor's to fix by working closer in. Check the mount and the "
        "workspace ratio first, then pass it on as filming advice.",
        why="A trajectory the arm cannot follow teaches nothing, but unlike the "
        "stages above it is a fact about the footage rather than about the "
        "extraction.",
        floor=0.6,
    ),
    Stage(
        key="steps_kept",
        code="extraction.steps_dropped",
        stage="assembly",
        summary="{value:.0%} of steps survived into the training set",
        fix="If this is far below the pose-solve rate, the loss is in requiring "
        "every hand at once. Holding an idle arm at its last pose recovers "
        "one-handed working, which is most of real kitchen footage.",
        why="A bimanual step needs both hands at the same instant, and the "
        "intersection of two independent detection rates is much smaller than "
        "either.",
        floor=0.5,
    ),
)

#: Signals that are not a rate -- they describe an assumption rather than a loss.
ASSUMPTIONS = ("intrinsics_source", "scale", "trajectory_source", "estimator", "keypoints")


def signals(cohort: Cohort) -> dict[str, float]:
    """Every numeric extraction signal, averaged over the cohort's episodes."""
    totals: dict[str, list[float]] = {}
    for episode in cohort.episodes:
        for mapping in _readable(episode):
            for key, value in mapping.items():
                if isinstance(value, bool):
                    totals.setdefault(key, []).append(1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    totals.setdefault(key, []).append(float(value))
    out = {key: sum(v) / len(v) for key, v in totals.items() if v}

    # Two derived rates the earlier stages record only in pieces.
    if "steps_in" in totals and "steps_out" in totals:
        inn, keep = sum(totals["steps_in"]), sum(totals["steps_out"])
        out["steps_kept"] = keep / inn if inn else 0.0
    left = out.get("left_pose_plausible")
    right = out.get("right_pose_plausible")
    plausible = [v for v in (left, right) if v is not None]
    if plausible:
        out["pose_plausible"] = float(np.mean(plausible))
    return out


def stated(cohort: Cohort) -> dict[str, set[str]]:
    """Non-numeric declarations -- what was assumed rather than measured."""
    out: dict[str, set[str]] = {}
    for episode in cohort.episodes:
        for mapping in _readable(episode):
            for key, value in mapping.items():
                if key in ASSUMPTIONS and isinstance(value, str):
                    out.setdefault(key, set()).add(value)
    return out


def _readable(episode: Any) -> list[Mapping[str, Any]]:
    labels = getattr(episode, "labels", None)
    meta = getattr(episode, "meta", None)
    return [
        dict(getattr(labels, "annotations", {}) or {}),
        dict(getattr(meta, "extra", {}) or {}),
    ]


class Extraction(FeedbackModule):
    """What the pipeline is losing, and which stage to fix first."""

    def __init__(self, *, stages: Sequence[Stage] = STAGES, worth: float = WORTH_FIXING):
        self._stages = tuple(stages)
        self._worth = float(worth)

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="extraction",
            version=VERSION,
            min_cohorts=1,
            prescribes=True,
            holds=(),
            stages=[s.stage for s in self._stages],
            # Said explicitly because the distinction is the module's reason to
            # exist and a reader will otherwise assume this is capture advice.
            addressed_to="whoever operates the pipeline, not whoever filmed",
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "extraction",
            "feedback",
            description="signals written by the estimator, retargeter and assembly steps",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        notes: list[str] = []

        for cohort in cohorts:
            measured = signals(cohort)
            declarations = stated(cohort)
            unread = [s.code for s in self._stages if s.key not in measured]

            scored: list[tuple[float, Stage, float]] = []
            for stage in self._stages:
                if stage.key not in measured:
                    continue
                value = float(measured[stage.key])
                measurements[f"{cohort.name}.{stage.key}"] = Measurement(
                    value=round(value, 4),
                    n=len(cohort.episodes),
                    method=f"mean over episodes, {stage.stage} stage",
                    detail={"floor": stage.floor, "losing": round(max(0.0, 1.0 - value), 4)},
                )
                if value < stage.floor:
                    scored.append((1.0 - value, stage, value))

            scored.sort(key=lambda item: -item[0])
            for lost, stage, value in scored:
                findings.append(self._finding(cohort, stage, value, lost))

            findings.extend(self._assumptions(cohort, declarations))
            if unread:
                notes.append(
                    f"{cohort.name}: {count_of(len(unread), 'stage')} wrote no signal "
                    f"({', '.join(unread[:3])}). Not measured is not the same as fine"
                )
            if not scored:
                findings.append(
                    Finding(
                        code="extraction.healthy",
                        summary=(
                            f"every measured extraction stage for {cohort.name} is above "
                            "its floor; the limit on this dataset is the footage rather "
                            "than the pipeline"
                        ),
                        severity="info",
                        cohorts=(cohort.name,),
                    )
                )

        return Report(
            module="extraction",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(cohort.name for cohort in cohorts),
        )

    def _finding(self, cohort: Cohort, stage: Stage, value: float, lost: float) -> Finding:
        worth = lost >= self._worth
        return Finding(
            code=stage.code,
            summary=f"{cohort.name}: {stage.summary.format(value=value)}",
            # Ranked by data lost rather than by how bad it sounds. A stage
            # losing 2% is not worth engineering time; one losing 40% is the
            # only thing worth doing this week.
            severity="strong" if worth else "weak",
            evidence={
                "stage": stage.stage,
                "measured": round(value, 4),
                "floor": stage.floor,
                "losing": round(lost, 4),
                "worth_fixing": worth,
            },
            prescription=(
                f"{stage.fix} {stage.why}"
                if worth
                else f"Below the {self._worth:.0%} threshold where this is worth "
                f"engineering time. Noted rather than prescribed. {stage.why}"
            ),
            cohorts=(cohort.name,),
        )

    def _assumptions(self, cohort: Cohort, declarations: Mapping[str, set[str]]) -> list[Finding]:
        """Things that cost no frames and bias every number."""
        out: list[Finding] = []
        sources = declarations.get("intrinsics_source", set())
        if sources and sources != {"calibrated"}:
            out.append(
                Finding(
                    code="extraction.intrinsics_assumed",
                    severity="weak",
                    summary=(
                        f"{cohort.name}: the camera was never calibrated "
                        f"({', '.join(sorted(sources))}), so every distance carries the "
                        "same unmeasured error"
                    ),
                    evidence={"intrinsics_source": sorted(sources)},
                    prescription=(
                        "Calibrate once per camera model with a checkerboard and the "
                        "error goes away for every dataset that camera ever produces. "
                        "This costs no frames and biases every number, which is why "
                        "nothing else in the pipeline will ever complain about it."
                    ),
                    cohorts=(cohort.name,),
                )
            )
        scales = declarations.get("scale", set())
        if scales and scales != {"metric"}:
            others = sorted(scales - {"metric"})
            mixed = "metric" in scales
            out.append(
                Finding(
                    code="extraction.not_metric",
                    severity="strong",
                    summary=(
                        # A mixed cohort is the common case and the confusing
                        # one: some clips solved metrically and some fell back.
                        # Saying "positions are metric, normalized rather than
                        # metric" is what happens when a set is joined without
                        # noticing that one of its members is the good value.
                        f"{cohort.name}: some episodes fell back to "
                        f"{', '.join(others)} positions while others are metric"
                        if mixed
                        else f"{cohort.name}: positions are {', '.join(others)} rather than metric"
                    ),
                    evidence={"scale": sorted(scales)},
                    prescription=(
                        "Non-metric positions cannot be retargeted to a real arm at all, "
                        "so those episodes are carrying no usable trajectory. A mix means "
                        "the metric path failed on some clips rather than being off, "
                        "find which, and why, before treating the dataset as one thing."
                        if mixed
                        else "Non-metric positions cannot be retargeted to a real arm at "
                        "all. Either recover scale, which the hand's own measured size does "
                        ", or capture with a rig that reports metres."
                    ),
                    cohorts=(cohort.name,),
                )
            )
        return out


def worst_stage(report: Report) -> str | None:
    """The one thing to fix next, or ``None`` if nothing is worth fixing."""
    strong = [
        f for f in report.findings if f.severity == "strong" and f.code.startswith("extraction.")
    ]
    return strong[0].code if strong else None
