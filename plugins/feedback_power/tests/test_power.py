"""Refusing an experiment before it is run, sized from what has been run before.

The numbers are this project's own: a measured 56/66 percent baseline on Lift, a
twenty-trial habit, and one downstream comparison that came back at zero from a
budget four hundred times too small to say anything.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

import pytest
from gantry_feedback_power import Budget, PowerCheck, plan_for

from gantry.contracts.feedback import Cohort
from gantry.history import History
from gantry.spine import ComponentRef, Provenance, proportion


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)
    stage_events: tuple = ()


@dataclass
class Meta:
    id: str
    task: str | None = None
    source: str = "s"


@dataclass
class Ep:
    meta: Meta
    labels: Labels


@dataclass
class Run:
    provenance: Provenance
    episodes: tuple
    metrics: dict


def run(task, policy, outcomes):
    components = (
        ComponentRef("policy", policy, "1.0"),
        ComponentRef("scorer", "machine", "1.0"),
    )
    episodes = tuple(Ep(Meta(f"seed_{i}", task), Labels(o)) for i, o in enumerate(outcomes))
    return Run(
        Provenance(components=components),
        episodes,
        {"success_rate": proportion(sum(1 for o in outcomes if o), len(outcomes))},
    )


def measured_history():
    h = History(tempfile.mkdtemp())
    h.put(run("lift_cube", "ph_official", [True] * 28 + [False] * 22), keep_record=False)
    h.put(run("lift_cube", "mh_official", [True] * 33 + [False] * 17), keep_record=False)
    return h


def cohort(name, task, ok, n):
    return Cohort(name, tuple(Ep(Meta(f"seed_{i}", task), Labels(i < ok)) for i in range(n)))


# -- planning ----------------------------------------------------------------


def test_twenty_trials_cannot_see_a_ten_point_effect():
    """The habit this exists to break."""
    verdict = plan_for(
        Budget(trials=20, magnitude=0.10), history=measured_history(), task="lift_cube"
    )
    assert not verdict.ok
    assert "power.underpowered" in verdict.codes()


def test_a_sufficient_budget_proceeds():
    # 200, not the 60 this once said. Sixty was adequate only under a sizing
    # that ignored its own power argument; separating ten points at a 61%
    # baseline actually takes 168 paired trials.
    assert plan_for(
        Budget(trials=200, magnitude=0.10), history=measured_history(), task="lift_cube"
    ).ok


def test_the_refusal_carries_the_number_that_would_work():
    """A refusal nobody can act on is just an obstacle."""
    verdict = plan_for(
        Budget(trials=60, magnitude=0.05), history=measured_history(), task="lift_cube"
    )
    reason = verdict.because("power.underpowered")[0]
    assert reason.detail["needed"] > 60
    assert 0 < reason.detail["smallest_detectable"] < 1
    assert "run" in reason.hint


def test_a_hopeless_budget_says_so_rather_than_offering_a_hundred_points():
    """At twenty trials there is no effect size worth quoting, and the search
    used to answer 1.0 -- a reassuring sentence about a hopeless experiment."""
    verdict = plan_for(
        Budget(trials=20, magnitude=0.05), history=measured_history(), task="lift_cube"
    )
    reason = verdict.because("power.underpowered")[0]
    assert reason.detail["smallest_detectable"] is None
    assert "no smaller question" in reason.hint


def test_it_also_says_what_this_budget_could_see_instead():
    """The other actionable direction: keep the budget, weaken the claim."""
    verdict = plan_for(
        Budget(trials=60, magnitude=0.05), history=measured_history(), task="lift_cube"
    )
    detectable = verdict.because("power.underpowered")[0].detail["smallest_detectable"]
    assert plan_for(
        Budget(trials=60, magnitude=detectable + 0.01),
        history=measured_history(),
        task="lift_cube",
    ).ok


def test_the_baseline_comes_from_history_not_from_the_caller():
    """An invented rate produces an invented trial count, which is how an
    underpowered run gets approved by its own author."""
    verdict = plan_for(
        Budget(trials=20, magnitude=0.05), history=measured_history(), task="lift_cube"
    )
    baseline = verdict.because("power.underpowered")[0].detail["baseline"]
    assert baseline == pytest.approx(0.61, abs=0.01)  # (0.56 + 0.66) / 2


def test_an_untested_task_is_noted_and_sized_conservatively():
    verdict = plan_for(
        Budget(trials=50, magnitude=0.10), history=measured_history(), task="open_door"
    )
    assert "power.no_history" in verdict.codes()


def test_a_budget_planned_for_nothing_is_refused():
    verdict = plan_for(Budget(trials=100, magnitude=0.0), baseline=0.5)
    assert not verdict.ok
    assert "power.no_effect_named" in verdict.codes()


def test_a_budget_below_the_floor_is_not_an_experiment():
    verdict = plan_for(Budget(trials=2, magnitude=0.3), baseline=0.5)
    assert not verdict.ok
    assert "power.below_floor" in verdict.codes()


def test_a_very_large_budget_suggests_the_claim_was_hedged():
    verdict = plan_for(Budget(trials=2000, magnitude=0.30), baseline=0.5)
    assert verdict.ok
    assert "power.generous" in verdict.codes()


# -- selection ---------------------------------------------------------------


def test_repeated_attempts_tighten_the_threshold():
    once = Budget(trials=200, magnitude=0.10)
    tenth = Budget(trials=200, magnitude=0.10, attempts=9)
    assert tenth.corrected_alpha() == pytest.approx(once.corrected_alpha() / 10)


def test_the_tightened_threshold_is_stated_before_the_money_is_spent():
    verdict = plan_for(
        Budget(trials=200, magnitude=0.10, attempts=9),
        history=measured_history(),
        task="lift_cube",
    )
    assert "power.selection" in verdict.codes()
    assert "0.0050" in verdict.because("power.selection")[0].message


def test_correction_can_turn_an_adequate_budget_inadequate():
    """Which is the point: the tenth look is not the same claim as the first."""
    first = plan_for(Budget(trials=200, magnitude=0.10), baseline=0.6)
    fiftieth = plan_for(Budget(trials=200, magnitude=0.10, attempts=49), baseline=0.6)
    assert first.ok
    assert not fiftieth.ok


# -- retrospective -----------------------------------------------------------


def test_it_says_what_a_finished_run_could_ever_have_seen():
    """The Can result: 0/20 is not a finding about the policy."""
    report = PowerCheck(magnitude=0.10).analyse([cohort("can", "pick_place_can", 0, 20)])
    finding = report.findings[0]
    assert finding.code == "power.underpowered"
    assert finding.evidence["needed"] > 20
    assert "before reading this as a comparison" in finding.prescription


def test_an_adequate_run_produces_no_complaint():
    # 56 paired trials separate thirty points at this rate, so fifty-six is the
    # bar and fifty is not it.
    report = PowerCheck(magnitude=0.30).analyse([cohort("ph", "lift_cube", 34, 60)])
    assert [f.code for f in report.findings] == []
    assert report.measurements["ph.success_rate"].value == pytest.approx(34 / 60)


def test_a_cohort_with_no_outcomes_is_reported_rather_than_skipped():
    unscored = Cohort("nothing", (Ep(Meta("s", "t"), Labels(None)),))
    report = PowerCheck().analyse([unscored])
    assert report.findings[0].code == "power.unscored"


def test_it_holds_nothing_because_a_sample_size_claim_is_true_regardless():
    assert list(PowerCheck().descriptor().provides["holds"]) == []
