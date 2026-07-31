"""The simulator's predicate, as one judge among several.

The interesting tests are not "does it read a boolean" — it does. They are about
the two things demoting it from definitional to comparable actually buys: that a
criterion it has no measurement for produces an abstention rather than a
failure, and that its disagreement with the rubric written beside it is
locatable.
"""

from __future__ import annotations

import pytest
from gantry_scorer_machine import MachinePredicate, ThresholdPredicate
from gantry_scorer_machine.machine import all_of, lifted, reached

from gantry.conformance import check_scorer
from gantry.contracts.scorer import FINAL_STATE, Evidence
from gantry.contracts.task import Criterion, Region, TaskDefinition, Thing
from gantry.spine import IncompatibleError

RUBRIC = (
    "The cube is clear of the table by at least 4 cm and held in the gripper for "
    "at least one second. A cube dropped immediately does not count."
)


def lift_task(*criteria: Criterion) -> TaskDefinition:
    return TaskDefinition(
        name="lift_cube",
        instruction="lift the cube",
        things=(Thing("cube", "cube_20mm", Region("table", (-0.03, 0.03), (-0.03, 0.03))),),
        success=criteria or (Criterion("lifted", {"object": "cube", "height": 0.04}, RUBRIC),),
    )


# -- the machine predicate ---------------------------------------------------


def test_it_reads_the_worlds_own_verdict():
    scorer = MachinePredicate()
    judgements = scorer.score(Evidence("seed_0", final_state={"success": True}), lift_task())
    assert judgements[0].passed is True
    assert scorer.overall(judgements) is True


def test_a_world_reporting_sub_goals_is_read_for_the_overall_answer_only():
    """A sub-goal is a milestone and belongs to the funnel, not to a label."""
    scorer = MachinePredicate()
    state = {"success": {"task": False, "grasp": True}}
    assert scorer.score(Evidence("s", final_state=state), lift_task())[0].passed is False


def test_a_world_that_reported_nothing_produces_an_abstention():
    scorer = MachinePredicate()
    judgements = scorer.score(Evidence("s", final_state={}), lift_task())
    assert judgements[0].passed is None
    assert scorer.overall(judgements) is None


def test_it_says_out_loud_that_it_cannot_tell_criteria_apart():
    """Two criteria, one boolean. Honest about it rather than pretending."""
    scorer = MachinePredicate()
    task = lift_task(
        Criterion("lifted", {"height": 0.04}, RUBRIC),
        Criterion("held", {"seconds": 1.0}, "The cube stays in the gripper for a second."),
    )
    judgements = scorer.score(Evidence("s", final_state={"success": True}), task)
    assert len(judgements) == 2
    assert all("does not distinguish" in j.rationale for j in judgements)


def test_it_refuses_evidence_it_cannot_read():
    """A video is not something a pose predicate can use, and accepting one
    would let it depend on evidence a real bench cannot supply."""
    scorer = MachinePredicate()
    with pytest.raises(IncompatibleError, match="scorer.evidence_missing"):
        scorer.score(Evidence("s", video="/tmp/nope.mp4"), lift_task())


def test_it_declares_itself_free_and_deterministic():
    scorer = MachinePredicate()
    assert scorer.cost == "free"
    assert scorer.deterministic is True
    assert scorer.needs == (FINAL_STATE,)


def test_it_passes_the_conformance_kit():
    verdict = check_scorer(
        MachinePredicate(), Evidence("s", final_state={"success": True}), lift_task()
    )
    assert verdict.ok, verdict.explain()


# -- per-criterion thresholds ------------------------------------------------


def test_a_threshold_scorer_judges_each_criterion_separately():
    scorer = ThresholdPredicate({"lifted": lifted})
    state = {"object_height_above_surface": 0.061}
    assert scorer.score(Evidence("s", final_state=state), lift_task())[0].passed is True


def test_below_the_threshold_is_a_failure_not_an_abstention():
    scorer = ThresholdPredicate({"lifted": lifted})
    state = {"object_height_above_surface": 0.01}
    assert scorer.score(Evidence("s", final_state=state), lift_task())[0].passed is False


def test_a_criterion_with_no_registered_check_abstains():
    """The distinction that keeps a gap in coverage from looking like evidence."""
    scorer = ThresholdPredicate({})
    judgements = scorer.score(Evidence("s", final_state={"anything": 1}), lift_task())
    assert judgements[0].passed is None
    assert "no check is registered" in judgements[0].rationale


def test_a_check_that_raises_abstains_rather_than_failing_the_trial():
    def explodes(args, state):
        raise KeyError("nope")

    scorer = ThresholdPredicate({"lifted": explodes})
    judgements = scorer.score(Evidence("s", final_state={"x": 1}), lift_task())
    assert judgements[0].passed is None
    assert "KeyError" in judgements[0].rationale


def test_a_missing_measurement_abstains():
    scorer = ThresholdPredicate({"lifted": lifted})
    assert scorer.score(Evidence("s", final_state={}), lift_task())[0].passed is None


def test_the_threshold_scorer_passes_the_conformance_kit():
    verdict = check_scorer(
        ThresholdPredicate({"lifted": lifted}),
        Evidence("s", final_state={"object_height_above_surface": 0.061}),
        lift_task(),
    )
    assert verdict.ok, verdict.explain()


# -- the gap between a predicate and the rubric beside it --------------------


def test_a_predicate_and_its_rubric_can_disagree_and_that_is_the_point():
    """The reason judging became an axis.

    The rubric says the cube must be *held*; this check tests only the height at
    the end of the episode. A policy that lifts and drops satisfies the
    predicate and fails the rubric — a disagreement that was invisible while the
    predicate was the definition of success, and is now a measurable difference
    between two named judges.
    """
    scorer = ThresholdPredicate({"lifted": lifted})
    task = lift_task()
    dropped_at_the_end = {"object_height_above_surface": 0.061, "in_gripper": False}
    assert scorer.score(Evidence("s", final_state=dropped_at_the_end), task)[0].passed is True
    assert "held in the gripper" in task.success[0].rubric


def test_a_stricter_check_can_be_registered_for_the_same_criterion():
    """Which is the fix once the disagreement is measured."""
    scorer = ThresholdPredicate({"lifted": all_of("high_enough", "in_gripper")})
    state = {"high_enough": True, "in_gripper": False}
    assert scorer.score(Evidence("s", final_state=state), lift_task())[0].passed is False


# -- helpers -----------------------------------------------------------------


def test_all_of_abstains_when_any_flag_is_absent():
    check = all_of("a", "b")
    assert check({}, {"a": True, "b": True}) is True
    assert check({}, {"a": True}) is None


def test_reached_reads_the_reported_stages():
    check = reached("grasp")
    assert check({}, {"stages": ("reach", "grasp")}) is True
    assert check({}, {"stages": ("reach",)}) is False
    assert check({}, {}) is None
