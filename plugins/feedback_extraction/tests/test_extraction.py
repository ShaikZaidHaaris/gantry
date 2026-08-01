"""Pipeline health, and the distinction it exists to keep."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_feedback_extraction import Extraction, Stage, signals, stated, worst_stage

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeLabels, episode_from_arrays

SPEC = ChannelSpec("x", "vector", (1,), "float32")


def episode(index=0, **annotations):
    return episode_from_arrays(
        {"x": np.zeros((2, 1), dtype="float32")},
        [SPEC],
        id=f"e{index}",
        source="ego",
        labels=EpisodeLabels(annotations=dict(annotations)),
    )


def cohort(clips, name="theirs"):
    return Cohort(name=name, episodes=tuple(episode(i, **c) for i, c in enumerate(clips)))


def healthy(**over):
    base = {
        "hands_visible": 0.95,
        "pose_solved": 0.92,
        "left_pose_plausible": 0.99,
        "right_pose_plausible": 0.98,
        "in_reach": 0.80,
        "steps_in": 100,
        "steps_out": 95,
        "intrinsics_source": "calibrated",
        "scale": "metric",
    }
    base.update(over)
    return base


def codes(report):
    return [f.code for f in report.findings]


# -- the distinction this module exists for ---------------------------------


def test_every_finding_is_addressed_to_the_pipeline_not_the_contributor():
    """A clip that yields nothing might have had no hands, or hands the estimator
    could not find. Telling a contributor to re-film because our detector was
    weak is wrong, and the kind of wrong that loses a customer."""
    made = Extraction()
    assert "not whoever filmed" in made.descriptor().metadata["addressed_to"]

    report = made.analyse([cohort([healthy(hands_visible=0.4)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.detector_missing_hands")
    assert "detector" in finding.prescription.lower()
    assert "re-film" not in finding.prescription.lower()


def test_the_one_stage_that_is_genuinely_the_contributors_says_so():
    """Reaching outside the workspace is a fact about the footage rather than
    about the extraction, and the module does not pretend otherwise."""
    report = Extraction().analyse([cohort([healthy(in_reach=0.3)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.outside_the_workspace")
    assert "contributor's to fix" in finding.prescription


# -- ranked by what it is actually costing ----------------------------------


def test_findings_are_ordered_by_how_much_data_each_defect_loses():
    """The useful output is not a list of everything imperfect, it is which one
    thing to fix next."""
    report = Extraction().analyse(
        [cohort([healthy(hands_visible=0.5, pose_solved=0.65, in_reach=0.55)] * 4)]
    )
    order = [c for c in codes(report) if c.startswith("extraction.")]
    assert order[0] == "extraction.detector_missing_hands"  # losing 50%
    assert worst_stage(report) == "extraction.detector_missing_hands"


def test_a_defect_losing_almost_nothing_is_noted_not_prescribed():
    """A stage losing 2% is not worth engineering time; one losing 40% is the
    only thing worth doing this week.

    Note what "losing" means: the absolute share of data gone, not the distance
    below the floor. Detection at 0.84 against a 0.85 floor only just fires, and
    is still losing 16% of every frame — which is worth fixing. The weak case
    needs a stage whose floor sits near one.
    """
    fussy = Stage(
        key="fussy",
        code="extraction.fussy",
        stage="polish",
        summary="{value:.0%} clean",
        fix="Tidy it.",
        why="Marginal.",
        floor=0.99,
    )
    report = Extraction(stages=(fussy,), worth=0.05).analyse([cohort([healthy(fussy=0.97)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.fussy")
    assert finding.severity == "weak"
    assert "Noted rather than prescribed" in finding.prescription
    assert finding.evidence["worth_fixing"] is False


def test_a_defect_losing_a_lot_is_strong_however_close_to_its_floor():
    report = Extraction(worth=0.05).analyse([cohort([healthy(hands_visible=0.84)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.detector_missing_hands")
    assert finding.severity == "strong"
    assert finding.evidence["losing"] == pytest.approx(0.16)


def test_a_healthy_pipeline_says_the_footage_is_the_limit():
    report = Extraction().analyse([cohort([healthy()] * 4)])
    assert codes(report) == ["extraction.healthy"]
    assert "footage rather than the pipeline" in report.findings[0].summary


# -- the specific stages -----------------------------------------------------


def test_detection_is_named_as_the_hard_ceiling():
    report = Extraction().analyse([cohort([healthy(hands_visible=0.6)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.detector_missing_hands")
    assert "hard ceiling" in finding.prescription
    assert "RTMPose" in finding.prescription


def test_pose_failing_while_detection_succeeds_points_at_the_threshold():
    """The real diagnosis from this pipeline: detection at 100% and solves at 10%
    was a budget demanding better agreement than two detectors can give."""
    report = Extraction().analyse([cohort([healthy(pose_solved=0.15)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.pose_not_solving")
    assert "reprojection budget" in finding.prescription
    assert "different standard at every distance" in finding.prescription


def test_implausible_poses_point_at_the_intrinsics():
    report = Extraction().analyse(
        [cohort([healthy(left_pose_plausible=0.3, right_pose_plausible=0.2)] * 4)]
    )
    finding = next(f for f in report.findings if f.code == "extraction.poses_implausible")
    assert "focal length" in finding.prescription
    assert "reproject perfectly from an impossible place" in finding.prescription


def test_a_low_step_yield_points_at_requiring_both_hands():
    """8% measured on real footage, because a bimanual step needs both hands at
    the same instant and the intersection of two rates is much smaller than
    either."""
    report = Extraction().analyse([cohort([healthy(steps_in=100, steps_out=8)] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.steps_dropped")
    assert "every hand at once" in finding.prescription
    assert "one-handed working" in finding.prescription


# -- assumptions cost no frames and bias everything -------------------------


def test_assumed_intrinsics_are_surfaced_although_they_lose_nothing():
    """Worth surfacing precisely because nothing else will ever complain."""
    report = Extraction().analyse([cohort([healthy(intrinsics_source="fov")] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.intrinsics_assumed")
    assert "never calibrated" in finding.summary
    assert "costs no frames and biases every number" in finding.prescription


def test_calibrated_intrinsics_produce_no_finding():
    report = Extraction().analyse([cohort([healthy(intrinsics_source="calibrated")] * 4)])
    assert "extraction.intrinsics_assumed" not in codes(report)


def test_non_metric_positions_are_a_strong_finding():
    report = Extraction().analyse([cohort([healthy(scale="normalized")] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.not_metric")
    assert finding.severity == "strong"
    assert "cannot be retargeted to a real arm" in finding.prescription


# -- reading the record ------------------------------------------------------


def test_signals_are_read_off_the_record_rather_than_recomputed():
    """Nothing here opens a video. That is what makes it cheap enough to run on
    every upload and traceable back to a measurement."""
    made = signals(cohort([healthy(hands_visible=0.9), healthy(hands_visible=0.7)]))
    assert made["hands_visible"] == pytest.approx(0.8)
    assert made["steps_kept"] == pytest.approx(0.95)
    assert made["pose_plausible"] == pytest.approx(0.985)


def test_declarations_are_collected_separately_from_rates():
    said = stated(cohort([healthy(), healthy(intrinsics_source="fov")]))
    assert said["intrinsics_source"] == {"calibrated", "fov"}
    assert said["scale"] == {"metric"}


def test_a_stage_that_wrote_no_signal_is_noted_rather_than_assumed_fine():
    report = Extraction().analyse([cohort([{"scale": "metric"}] * 3)])
    assert any("ot measured is not the same as fine" in n for n in report.notes)


def test_a_new_stage_is_an_entry_rather_than_a_code_change():
    extra = Stage(
        key="tracking_ok",
        code="extraction.tracking_lost",
        stage="tracking",
        summary="tracking held for {value:.0%} of frames",
        fix="Use a tracker with re-identification.",
        why="A lost track restarts the identity assignment.",
        floor=0.9,
    )
    report = Extraction(
        stages=(*[s for s in __import__("gantry_feedback_extraction").STAGES], extra)
    ).analyse([cohort([healthy(tracking_ok=0.3)] * 4)])
    assert "extraction.tracking_lost" in codes(report)


def test_it_analyses_one_cohort_and_prescribes():
    made = Extraction()
    assert made.descriptor().provides["min_cohorts"] == 1
    assert made.descriptor().provides["prescribes"] is True


def test_a_mixed_scale_cohort_reads_as_a_partial_failure_not_a_contradiction():
    """Some clips solving metrically and some falling back is the common case.
    Joining the set without noticing one member is the good value produced
    "positions are metric, normalized rather than metric", which reads as a bug
    in the report rather than a finding about the data."""
    report = Extraction().analyse(
        [cohort([healthy(scale="metric"), healthy(scale="normalized")] * 2)]
    )
    finding = next(f for f in report.findings if f.code == "extraction.not_metric")
    assert "some episodes fell back to normalized" in finding.summary
    assert "rather than being off" in finding.prescription


def test_a_wholly_non_metric_cohort_still_reads_plainly():
    report = Extraction().analyse([cohort([healthy(scale="normalized")] * 4)])
    finding = next(f for f in report.findings if f.code == "extraction.not_metric")
    assert finding.summary.endswith("positions are normalized rather than metric")
