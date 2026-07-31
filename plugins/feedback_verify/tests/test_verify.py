"""Testing a curation plan, and refusing to test it badly.

The refusals here are the module. A verifier that always produces a number is
worth less than one that says "this test would not have meant anything", because
the first kind is how every curation method in the literature comes to report
its own win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from gantry_feedback_verify import CurationVerifier, preflight, trials_needed

from gantry.contracts.curation import (
    CurationAction,
    CurationPlan,
    Prediction,
)
from gantry.contracts.feedback import Cohort


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)
    stage_events: tuple = ()


@dataclass
class Meta:
    id: str
    uid: str = ""
    task: str | None = None


@dataclass
class Ep:
    meta: Meta
    labels: Labels


def episodes(outcomes, prefix="seed"):
    return tuple(Ep(Meta(f"{prefix}_{i}"), Labels(success=o)) for i, o in enumerate(outcomes))


def plan(magnitude=0.10, seeds=(), signal="labels", rung="screening"):
    return CurationPlan(
        actions=(CurationAction("drop", episodes=("mg/1",)),),
        signal=signal,
        rung=rung,
        predicted=Prediction(magnitude=magnitude, tasks=("lift_cube",)),
        evidence_seeds=tuple(seeds),
    )


# -- power -------------------------------------------------------------------


def test_bigger_effects_need_fewer_trials():
    assert trials_needed(0.35, 0.20) < trials_needed(0.35, 0.10) < trials_needed(0.35, 0.03)


def test_a_plan_the_budget_cannot_test_is_refused_before_the_retrain():
    verdict = preflight(plan(magnitude=0.03), list(range(20)), baseline_rate=0.35)
    assert not verdict.ok
    assert "curation.underpowered" in verdict.codes()
    # The refusal carries the number that would work, so it is actionable.
    reason = verdict.because("curation.underpowered")[0]
    assert "paired trials" in reason.message


def test_the_same_plan_with_enough_trials_passes():
    assert preflight(plan(magnitude=0.10), list(range(200)), baseline_rate=0.35).ok


# -- leakage: the guard nobody else has --------------------------------------


def test_a_plan_verified_on_the_scenes_that_produced_it_is_refused():
    # The signal read seeds 0-9 to decide what to drop. Verifying on 5-14
    # overlaps, and on the overlap the data was chosen to look good.
    verdict = preflight(
        plan(seeds=range(10), rung="influence"), list(range(5, 15)), baseline_rate=0.35
    )
    assert not verdict.ok
    assert "curation.leaky" in verdict.codes()


def test_held_out_scenes_are_not_leakage():
    verdict = preflight(
        plan(seeds=range(10), rung="influence"), list(range(100, 300)), baseline_rate=0.35
    )
    assert verdict.ok, verdict.explain()


def test_a_rollout_reading_plan_that_names_no_scenes_is_refused():
    # Without the list, leakage cannot be checked at all — so the plan is
    # refused for being unauditable rather than assumed clean.
    bare = CurationPlan(
        actions=(CurationAction("drop", episodes=("a",)),),
        signal="cupid",
        rung="influence",
        predicted=Prediction(magnitude=0.1),
    )
    verdict = preflight(bare, list(range(200)))
    assert not verdict.ok
    assert "plan.unnamed_evidence" in verdict.codes()


# -- selection ---------------------------------------------------------------


def test_the_tenth_plan_from_one_signal_faces_a_stricter_threshold():
    verdict = preflight(plan(), list(range(200)), baseline_rate=0.35, plans_already_tested=9)
    assert verdict.ok  # a note, not a refusal — it tightens rather than forbids
    reason = verdict.because("curation.selection")[0]
    assert "0.005" in reason.message


# -- judging -----------------------------------------------------------------


def make(baseline, curated, magnitude=0.10, tested=0):
    return CurationVerifier(plan(magnitude=magnitude), plans_already_tested=tested), [
        Cohort("baseline", episodes(baseline)),
        Cohort("curated", episodes(curated)),
    ]


def test_a_real_improvement_is_verified():
    # 20 shared scenes: baseline solves 4, curated solves 14, and every scene
    # the baseline solved the curated one also solved.
    base = [True] * 4 + [False] * 16
    cur = [True] * 14 + [False] * 6
    verifier, cohorts = make(base, cur)
    report = verifier.analyse(cohorts)
    finding = report.findings[0]
    assert finding.code == "curation.verified"
    assert finding.evidence["wins"] == 10 and finding.evidence["losses"] == 0
    assert report.measurements["delta"].value == pytest.approx(0.5)


def test_noise_is_refuted_not_reported_as_a_win():
    base = [True] * 7 + [False] * 13
    cur = [True] * 8 + [False] * 12  # one scene better; nothing to see
    verifier, cohorts = make(base, cur)
    assert verifier.analyse(cohorts).findings[0].code == "curation.refuted"


def test_a_move_in_the_right_direction_that_misses_the_prediction_is_refuted():
    # Significant, but the plan promised +30pp and delivered +25. The
    # prediction is the claim, and it did not hold.
    base = [True] * 2 + [False] * 18
    cur = [True] * 7 + [False] * 13
    verifier, cohorts = make(base, cur, magnitude=0.30)
    assert verifier.analyse(cohorts).findings[0].code == "curation.refuted"


def test_the_same_result_can_flip_once_selection_is_corrected():
    """Six wins, no losses, p=0.031. Real on its own; not the best of forty."""
    base = [True] * 4 + [False] * 16
    cur = [True] * 10 + [False] * 10
    _, cohorts = make(base, cur)
    plain = make(base, cur, magnitude=0.20, tested=0)[0].analyse(cohorts)
    corrected = make(base, cur, magnitude=0.20, tested=40)[0].analyse(cohorts)
    assert plain.findings[0].evidence["p"] == pytest.approx(0.03125)
    assert plain.findings[0].code == "curation.verified"
    assert corrected.findings[0].code == "curation.refuted"


def test_unnamed_cohorts_are_refused_rather_than_guessed():
    verifier = CurationVerifier(plan())
    report = verifier.analyse([Cohort("a", episodes([True])), Cohort("b", episodes([False]))])
    assert not report.findings
    assert "cannot be inferred" in report.notes[0]


def test_pairing_is_by_scene_not_by_position():
    # Two runs that attempted different scenes must line up on identity. If
    # this paired positionally the deltas would be nonsense and nothing would
    # say so.
    verifier = CurationVerifier(plan())
    base = (Ep(Meta("seed_1"), Labels(False)), Ep(Meta("seed_2"), Labels(True)))
    cur = (Ep(Meta("seed_2"), Labels(True)), Ep(Meta("seed_1"), Labels(True)))
    report = verifier.analyse([Cohort("baseline", base), Cohort("curated", cur)])
    assert report.findings[0].evidence["shared_scenes"] == 2
    assert report.findings[0].evidence["wins"] == 1


# -- the ledger entry --------------------------------------------------------


def test_a_refuted_plan_still_produces_an_outcome():
    """A ledger that only records successes is a brochure."""
    base = [True] * 7 + [False] * 13
    cur = [True] * 8 + [False] * 12
    verifier, cohorts = make(base, cur)
    outcome = verifier.outcome(
        cohorts,
        baseline_run="runs/aaa",
        curated_run="runs/bbb",
        cost={"gpu_minutes": 44},
    )
    assert not outcome.held
    assert outcome.as_dict()["cost"]["gpu_minutes"] == 44
    assert outcome.baseline_run == "runs/aaa"


def test_an_outcome_records_what_was_predicted_alongside_what_happened():
    base = [True] * 4 + [False] * 16
    cur = [True] * 14 + [False] * 6
    verifier, cohorts = make(base, cur)
    row = verifier.outcome(cohorts, baseline_run="a", curated_run="b").as_dict()
    assert row["held"] is True
    assert row["predicted"] == "success_rate +0.1 on lift_cube"
    assert row["rung"] == "screening"
