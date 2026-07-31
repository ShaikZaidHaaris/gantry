"""Whether a judge may be believed, and what happens when it may not.

The tests that matter are the ones about consequence. Computing kappa is
arithmetic; refusing a judge's findings because kappa was 0.4 is the thing that
makes cheap judgement safe to scale, and it is checked here.
"""

from __future__ import annotations

import pytest
from gantry_feedback_calibrate import Calibration, Corpus, agreement, hedges

from gantry.spine.inference import AGREEMENT_TENTATIVE, AGREEMENT_TRUSTED


def labels(outcomes, criterion="lifted"):
    return {f"t{i}": {criterion: v} for i, v in enumerate(outcomes)}


HUMAN = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]
AGREES = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]  # one disagreement
COIN = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]  # unrelated


def corpus(**judges):
    return Corpus(labels={name: labels(o) for name, o in judges.items()}, task="lift_cube")


# -- agreement ---------------------------------------------------------------


def test_a_judge_that_nearly_matches_is_calibrated():
    score = agreement(corpus(human=HUMAN, vlm=AGREES), "human", "vlm")
    assert score.value >= AGREEMENT_TRUSTED
    assert score.detail["verdict"] == "judge.calibrated"


def test_a_judge_that_is_guessing_is_not():
    score = agreement(corpus(human=HUMAN, vlm=COIN), "human", "vlm")
    assert score.value < 0
    assert score.detail["verdict"] == "judge.uncalibrated"


def test_raw_agreement_flatters_the_judge_least_worth_trusting():
    """On a task solved rarely, two judges who always say 'failed' agree 95% of
    the time and have established nothing."""
    skewed = [1] + [0] * 19
    lazy = [0] * 20
    score = agreement(corpus(human=skewed, lazy=lazy), "human", "lazy")
    assert score.detail["raw_agreement"] == pytest.approx(0.95)
    assert score.value <= 0.0
    assert score.detail["verdict"] == "judge.uncalibrated"


def test_abstentions_are_dropped_and_counted():
    """A pair where either judge could not tell carries no information about
    agreement, but the rate of them is itself a measurement."""
    with_abstain = [1, None, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    score = agreement(corpus(human=HUMAN, vlm=with_abstain), "human", "vlm")
    assert score.detail["dropped_abstentions"] == 1
    assert score.n == 11


# -- the gate ----------------------------------------------------------------


def test_a_calibrated_judge_passes_the_gate():
    gate = Calibration(corpus(human=HUMAN, vlm=AGREES)).gate("vlm")
    assert gate.ok


def test_an_uncalibrated_judge_is_refused_rather_than_caveated():
    """A caveat attached to a number gets dropped the first time it is quoted."""
    gate = Calibration(corpus(human=HUMAN, vlm=COIN)).gate("vlm")
    assert not gate.ok
    assert "judge.uncalibrated" in gate.codes()
    assert "should not become findings" in gate.because("judge.uncalibrated")[0].hint


def test_a_judge_in_the_middle_band_is_allowed_but_labelled():
    borderline = [1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    score = agreement(corpus(human=HUMAN, vlm=borderline), "human", "vlm")
    if AGREEMENT_TENTATIVE <= score.value < AGREEMENT_TRUSTED:
        gate = Calibration(corpus(human=HUMAN, vlm=borderline)).gate("vlm")
        assert gate.ok
        assert "judge.tentative" in gate.codes()


def test_too_few_judged_pairs_is_unmeasured_not_uncalibrated():
    """Kappa on four trials swings between -1 and 1 on one disagreement."""
    gate = Calibration(corpus(human=[1, 0, 1], vlm=[1, 0, 1])).gate("vlm")
    assert not gate.ok
    assert "judge.unmeasured" in gate.codes()


def test_a_judge_never_compared_is_unmeasured():
    gate = Calibration(corpus(human=HUMAN)).gate("some_model")
    assert not gate.ok
    assert "judge.unmeasured" in gate.codes()


def test_the_reference_judges_itself_trivially():
    assert Calibration(corpus(human=HUMAN, vlm=AGREES)).gate("human").ok


# -- the report --------------------------------------------------------------


def test_it_reports_one_finding_per_judge_against_the_reference():
    report = Calibration(corpus(human=HUMAN, good=AGREES, bad=COIN)).analyse([])
    codes = {f.code for f in report.findings}
    assert "judge.calibrated" in codes
    assert "judge.uncalibrated" in codes


def test_a_person_is_the_default_reference():
    """A cheaper judge is standing in for a person, not the reverse."""
    report = Calibration(corpus(human=HUMAN, vlm=AGREES)).analyse([])
    finding = next(f for f in report.findings if f.code.startswith("judge."))
    assert "against human" in finding.summary


def test_one_judge_alone_cannot_be_calibrated_and_says_so():
    report = Calibration(corpus(human=HUMAN)).analyse([])
    assert any("cannot be measured" in note for note in report.notes)


def test_frequent_abstention_is_surfaced_as_a_rubric_problem():
    unsure = [None] * 8 + [1, 1, 0, 0]
    report = Calibration(corpus(human=HUMAN, vlm=unsure)).analyse([])
    assert "judge.abstains_often" in {f.code for f in report.findings}


def test_three_judges_get_an_alpha_as_well_as_pairwise_kappas():
    report = Calibration(corpus(human=HUMAN, a=AGREES, b=AGREES)).analyse([])
    assert "alpha" in report.measurements


def test_the_module_holds_everything_but_the_scorer():
    holds = set(Calibration(corpus(human=HUMAN)).descriptor().provides["holds"])
    assert holds == {"policy", "task", "evaluation", "embodiment"}


# -- rubric wording ----------------------------------------------------------


def test_it_finds_the_hedge_in_our_own_door_rubric():
    """Measured on the real thing: open_door says 'roughly 17 degrees'."""
    assert hedges("The door has swung open by roughly 17 degrees or more.") == ("roughly",)


def test_a_decidable_rubric_has_no_hedges():
    assert hedges("The cube is clear of the table by at least 4 cm and held in the gripper.") == ()


def test_a_hedged_rubric_is_reported_with_what_to_do_about_it():
    report = Calibration(
        corpus(human=HUMAN, vlm=AGREES),
        task_rubrics={"opened": "The door is mostly open."},
    ).analyse([])
    finding = next(f for f in report.findings if f.code == "criterion.graded_rubric")
    assert "mostly" in finding.evidence["hedges"]
    assert "twenty points" in finding.prescription


# -- bias --------------------------------------------------------------------


def test_a_judge_whose_verdicts_track_episode_length_is_flagged():
    """Duration correlates with success on some tasks and against it on others,
    so the bias looks like signal in whichever direction the task runs."""
    verdicts = [1, 1, 1, 1, 0, 0, 0, 0]
    context = {f"t{i}": {"steps": 500 if v else 50} for i, v in enumerate(verdicts)}
    c = Corpus(labels={"human": labels(verdicts), "vlm": labels(verdicts)}, context=context)
    report = Calibration(c).analyse([])
    assert "judge.bias.length" in {f.code for f in report.findings}


def test_no_length_bias_is_reported_when_lengths_are_comparable():
    verdicts = [1, 1, 1, 1, 0, 0, 0, 0]
    context = {f"t{i}": {"steps": 300} for i in range(8)}
    c = Corpus(labels={"human": labels(verdicts), "vlm": labels(verdicts)}, context=context)
    report = Calibration(c).analyse([])
    assert "judge.bias.length" not in {f.code for f in report.findings}
