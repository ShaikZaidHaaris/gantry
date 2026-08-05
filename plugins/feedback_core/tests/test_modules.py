"""Acceptance: the fixtures are the answer key.

A detector is judged on two things, and the second is the one that is usually
skipped: what it finds, and what it leaves alone. Every planted defect must be
reported, and the clean and decoy suites must produce nothing.
"""

from __future__ import annotations

import pytest
from gantry_feedback_core import (
    ATTRIBUTABLE,
    INCONSISTENT,
    UNIVERSAL,
    Attribution,
    Funnel,
    Harden,
    Screen,
    build,
    classify,
    end_to_end,
    quarantined_statistics,
    stage_order,
    uplift,
)
from gantry_feedback_core.harden import PerCohort

from gantry.contracts.feedback import Cohort
from gantry.fixtures import make_clean, make_defective, make_duration_confound
from gantry.spine import IncompatibleError


def cohort(name: str, suite) -> Cohort:
    return Cohort(name, suite.episodes)


def mixed(
    defect: str, n: int = 40, fraction: float = 0.5, seed: int = 0, name: str | None = None
) -> Cohort:
    return cohort(name or defect, make_defective(defect, n=n, fraction=fraction, seed=seed))


# ==========================================================================
# screen
# ==========================================================================


def test_comparative_mode_asserts_no_thresholds():
    report = Screen("comparative").run(
        [cohort("clean", make_clean(n=30, seed=1)), mixed("path_detour", n=30, fraction=1.0)]
    )
    assert "asserts no thresholds" in report.notes[0]


def test_comparative_mode_finds_the_statistic_that_separates():
    report = Screen("comparative").run(
        [
            cohort("clean", make_clean(n=30, seed=1)),
            cohort("detoured", make_defective("path_detour", n=30, fraction=1.0, seed=1)),
        ]
    )
    separating = {f.evidence["ranking"][-1]: f for f in report.by_code("screen.separates")}
    assert "clean" in separating
    codes = [f.summary.split()[0] for f in report.by_code("screen.separates")]
    assert "path_efficiency" in codes


def test_comparative_mode_stays_quiet_on_two_clean_cohorts():
    report = Screen("comparative").run(
        [cohort("a", make_clean(n=25, seed=1)), cohort("b", make_clean(n=25, seed=2))]
    )
    assert report.by_code("screen.separates") == ()
    assert "no statistic separates" in report.notes[-1]


def test_reference_mode_says_its_thresholds_are_fitted_not_universal():
    report = Screen("reference", reference="good").run(
        [cohort("good", make_clean(n=30, seed=1)), mixed("actuation_jerk", n=30, fraction=1.0)]
    )
    assert "they describe that cohort, not the task in general" in report.notes[0]
    assert any("fitted from good" in f.summary for f in report.findings)


def test_reference_mode_flags_a_cohort_outside_the_band():
    report = Screen("reference", reference="good").run(
        [
            cohort("good", make_clean(n=30, seed=1)),
            cohort("jerky", make_defective("actuation_jerk", n=30, fraction=1.0, seed=1)),
        ]
    )
    assert any(
        f.code == "screen.outside_reference" and "actuation_jerk" in f.summary
        for f in report.findings
    )


def test_reference_mode_does_not_flag_the_reference_against_itself():
    report = Screen("reference", reference="good").run(
        [cohort("good", make_clean(n=30, seed=1)), cohort("also_good", make_clean(n=30, seed=9))]
    )
    assert all("good:" not in f.summary for f in report.findings)


def test_an_unknown_reference_is_reported_not_crashed():
    report = Screen("reference", reference="ghost").run(
        [cohort("a", make_clean(n=5)), cohort("b", make_clean(n=5, seed=2))]
    )
    assert "is not among" in report.notes[0]


def test_absolute_mode_reports_only_the_success_fraction():
    report = Screen("absolute").run([mixed("never_completes", n=40, fraction=0.5)])
    assert "every other" in report.notes[0]
    assert set(report.measurements) == {"never_completes.success_rate"}


def test_absolute_mode_catches_a_set_that_mostly_fails():
    report = Screen("absolute").run(
        [cohort("broken", make_defective("never_completes", n=40, fraction=0.9))]
    )
    finding = report.by_code("screen.low_success")[0]
    assert finding.severity == "strong"
    assert "No training setting repairs" in finding.prescription


def test_absolute_mode_passes_a_fully_successful_set():
    report = Screen("absolute").run([cohort("clean", make_clean(n=40))])
    assert report.by_code("screen.low_success") == ()


def test_screen_refuses_a_mode_it_does_not_have():
    with pytest.raises(ValueError, match="unknown mode"):
        Screen("vibes")


def test_comparative_screen_refuses_a_single_cohort():
    with pytest.raises(IncompatibleError, match="at least 2"):
        Screen("comparative").run([cohort("only", make_clean(n=10))])


# ==========================================================================
# funnel
# ==========================================================================


def test_stage_order_comes_from_the_data_not_from_core():
    suite = make_clean(n=10, stages=("incise", "grip", "suture", "close"))
    assert stage_order(suite.episodes) == ("incise", "grip", "suture", "close")


def test_stage_order_is_robust_to_one_odd_episode():
    """Ordering by first sighting would let a single episode reshape the funnel."""
    suite = make_clean(n=12, seed=4)
    assert stage_order(suite.episodes) == ("approach", "engage", "transport", "release")


def test_a_clean_funnel_passes_everything_through():
    steps = build(make_clean(n=20).episodes)
    assert [s.rate for s in steps] == [1.0, 1.0, 1.0, 1.0]
    assert end_to_end(steps) == 1.0


def test_truncated_episodes_show_up_as_a_broken_rung():
    steps = build(make_defective("never_completes", n=40, fraction=0.5).episodes)
    rates = {s.name: s.rate for s in steps}
    # truncation cuts at the engage step, so approach survives and engage is
    # where half the episodes stop
    assert rates["approach"] == 1.0
    assert rates["engage"] == pytest.approx(0.5, abs=0.05)
    assert rates["transport"] == 1.0  # of those that engaged, all continued


def test_uplift_measures_what_fixing_one_rung_alone_is_worth():
    steps = build(make_defective("never_completes", n=40, fraction=0.5).episodes)
    baseline = end_to_end(steps)
    gain = uplift(steps, "engage")
    engage = next(s for s in steps if s.name == "engage")
    assert gain > 0
    # perfecting one rung divides the chain by that rung's rate
    assert baseline + gain == pytest.approx(baseline / engage.rate, abs=1e-9)


def test_uplift_is_zero_for_a_rung_that_is_already_perfect():
    steps = build(make_defective("never_completes", n=40, fraction=0.5).episodes)
    assert uplift(steps, "approach") == pytest.approx(0.0, abs=1e-9)


def test_the_funnel_names_the_weak_link_and_what_it_is_worth():
    report = Funnel().run([mixed("never_completes", n=40, fraction=0.5)])
    finding = report.by_code("funnel.bottleneck")[0]
    assert "engage is the weak link" in finding.summary
    assert "would add" in finding.summary
    assert finding.evidence["uplift"]["engage"] > 0
    assert finding.evidence["uplift"]["approach"] == pytest.approx(0.0, abs=1e-9)


def test_the_funnel_prescription_warns_against_collecting_for_a_universal_weakness():
    report = Funnel().run([mixed("never_completes", n=40, fraction=0.5)])
    prescription = report.by_code("funnel.bottleneck")[0].prescription
    assert "check whether other cohorts fail there too" in prescription


def test_the_funnel_refuses_records_with_no_stage_events():
    stripped = [
        e.with_labels(type(e.labels)(success=e.labels.success)) for e in make_clean(n=10).episodes
    ]
    report = Funnel().run([Cohort("outcome_only", tuple(stripped))])
    assert report.findings == ()
    assert "no stage events, so there is no funnel to build" in report.notes[0]


def test_the_funnel_works_on_a_vocabulary_it_has_never_seen():
    suite = make_clean(n=20, stages=("incise", "grip", "suture", "close"))
    report = Funnel().run([cohort("surgery", suite)])
    assert report.measurements["surgery.incise"].value == 1.0


# ==========================================================================
# attribution -- including what it must NOT say
# ==========================================================================


def test_attribution_finds_a_behaviour_that_tracks_failure():
    report = Attribution().run([mixed("never_completes", n=60, fraction=0.5)])
    assert report.findings, report.explain()


def test_attribution_refuses_when_every_outcome_is_the_same():
    report = Attribution().run([cohort("all_good", make_clean(n=30))])
    assert report.findings == ()
    assert "every episode has the same outcome" in report.notes[0]


def test_duration_and_counts_can_never_become_advice():
    """The quarantine, checked on the case that produced it.

    Truncated episodes are short *because* they failed, so length separates the
    outcomes perfectly. Reporting that as a cause would tell someone to collect
    longer demonstrations, which is advice to avoid failing by not failing.
    """
    report = Attribution().run([mixed("never_completes", n=60, fraction=0.5)])
    for finding in report.findings:
        statistic = finding.summary.split(": ")[1].split()[0]
        if statistic in quarantined_statistics():
            assert finding.prescription is None, statistic
            assert finding.code == "attribution.context"
            assert finding.severity == "info"


def test_every_prescription_comes_from_an_eligible_statistic():
    report = Attribution().run([mixed("never_completes", n=60, fraction=0.5)])
    for finding in report.findings:
        if finding.prescription is not None:
            assert finding.evidence["prescribable"] is True
            assert finding.evidence["duration_free"] is True
            assert finding.evidence["outcome_derived"] is False


def test_every_finding_shows_how_much_of_it_is_duration():
    report = Attribution().run([mixed("never_completes", n=60, fraction=0.5)])
    for finding in report.findings:
        assert "rank_correlation_with_length" in finding.evidence


def test_attribution_stays_quiet_on_the_duration_decoy():
    """The suite that must produce nothing.

    Per-step behaviour is identical across these episodes and only length
    varies. Anything reported here would be a measurement of duration wearing
    the language of quality.
    """
    suite = make_duration_confound(n=40)
    report = Attribution().run([cohort("decoy", suite)])
    assert report.prescriptions == ()


# ==========================================================================
# harden
# ==========================================================================


def test_harden_refuses_a_single_cohort():
    with pytest.raises(IncompatibleError, match="at least 2"):
        Harden().run([mixed("never_completes", n=30)])


def test_an_effect_present_everywhere_is_universal_and_is_not_actionable():
    report = Harden().run(
        [
            mixed("never_completes", n=60, fraction=0.5, seed=1, name="run_a"),
            mixed("never_completes", n=60, fraction=0.5, seed=2, name="run_b"),
        ]
    )
    universal = report.by_code(f"harden.{UNIVERSAL}")
    assert universal, report.explain()
    for finding in universal:
        if finding.prescription:
            assert "Do not collect" in finding.prescription


def test_when_two_cohorts_fail_the_same_way_everything_reads_universal():
    """The honest result, and the reason this module exists.

    Both cohorts fail by stopping early, and stopping early shifts nearly every
    statistic at once. Handed either cohort alone, attribution would produce a
    page of confident advice. Seen together, none of it is about the data.
    """
    report = Harden().run(
        [
            mixed("never_completes", n=60, fraction=0.5, seed=1, name="run_a"),
            mixed("never_completes", n=60, fraction=0.5, seed=2, name="run_b"),
        ]
    )
    assert report.by_code(f"harden.{ATTRIBUTABLE}") == ()
    assert len(report.by_code(f"harden.{UNIVERSAL}")) >= 4


def test_classify_calls_an_effect_attributable_when_it_fires_in_some_cohorts():
    """The classifier itself, stated exactly.

    Building this end to end would need two cohorts that fail for genuinely
    different reasons, which the synthetic fixtures do not yet produce. The
    rule is small enough to specify directly, and specifying it here is better
    than an integration test that happens to arrange the right accident.
    """
    fires_in_one = [
        PerCohort("a", delta=0.8, q=0.001, significant=True),
        PerCohort("b", delta=0.05, q=0.400, significant=False),
    ]
    assert classify(fires_in_one, total_cohorts=2) == ATTRIBUTABLE


def test_an_effect_that_merely_straddles_the_threshold_is_not_actionable():
    """Significant in one cohort and nearly so in the other is sampling, not a
    difference, and acting on it would spend a collection budget on noise."""
    borderline = [
        PerCohort("a", delta=0.52, q=0.049, significant=True),
        PerCohort("b", delta=0.48, q=0.051, significant=False),
    ]
    assert classify(borderline, total_cohorts=2) == INCONSISTENT


def test_a_significant_but_tiny_effect_does_not_count_as_firing():
    trivial = [
        PerCohort("a", delta=0.05, q=0.001, significant=True),
        PerCohort("b", delta=0.04, q=0.900, significant=False),
    ]
    assert classify(trivial, total_cohorts=2) == INCONSISTENT


def test_classify_calls_an_effect_universal_when_it_fires_everywhere():
    everywhere = [
        PerCohort("a", delta=0.8, q=0.001, significant=True),
        PerCohort("b", delta=0.7, q=0.002, significant=True),
    ]
    assert classify(everywhere, total_cohorts=2) == UNIVERSAL


def test_an_effect_pointing_opposite_ways_is_not_a_finding():
    """Significant in both cohorts but in opposite directions is noise wearing
    a p-value, and calling it universal would be worse than saying nothing."""
    contradictory = [
        PerCohort("a", delta=0.8, q=0.001, significant=True),
        PerCohort("b", delta=-0.8, q=0.001, significant=True),
    ]
    assert classify(contradictory, total_cohorts=2) == INCONSISTENT


def test_classify_says_nothing_when_nothing_fires():
    quiet = [PerCohort("a", 0.0, 0.9, False), PerCohort("b", 0.0, 0.9, False)]
    assert classify(quiet, total_cohorts=2) == INCONSISTENT


def test_a_cohort_with_no_outcome_variation_is_excluded_and_said_so():
    report = Harden().run(
        [
            cohort("constant", make_clean(n=30)),
            mixed("never_completes", n=40, fraction=0.5),
        ]
    )
    assert any("outcomes do not vary" in note for note in report.notes)
    assert any("cannot be told apart" in note for note in report.notes)


# ==========================================================================
# reports
# ==========================================================================


def test_a_report_is_json_able():
    import json

    report = Funnel().run([mixed("never_completes", n=20)])
    assert json.loads(json.dumps(report.as_dict()))["module"] == "funnel"


def test_every_number_in_a_report_carries_its_own_n():
    report = Funnel().run([mixed("never_completes", n=40, fraction=0.5)])
    assert all(m.n is not None for m in report.measurements.values())


def test_a_duplicate_cohort_name_is_refused():
    with pytest.raises(IncompatibleError, match="more than once"):
        Screen("comparative").run(
            [cohort("same", make_clean(n=5)), cohort("same", make_clean(n=5, seed=2))]
        )


def test_an_empty_cohort_is_refused():
    with pytest.raises(IncompatibleError, match="no episodes"):
        Funnel().run([Cohort("nothing", ())])


# ==========================================================================
# the milestone-vocabulary trap
# ==========================================================================


def _run_that_always_stops_at_stage_two():
    """A run where no episode ever reaches the second milestone."""
    import numpy as np

    from gantry.spine import EpisodeLabels, Provenance, StageEvent, episode_from_arrays

    episodes = tuple(
        episode_from_arrays(
            {"position": np.zeros((6, 3), dtype="float32")},
            (make_clean(n=1).episodes[0].channel("position"),),
            id=f"e{i}",
            source="run",
            labels=EpisodeLabels(success=False, stage_events=(StageEvent("approach", 1),)),
        )
        for i in range(20)
    )
    provenance = Provenance(protocol={"stages": ["approach", "engage", "transport", "release"]})
    return episodes, provenance


def test_a_milestone_nothing_reached_is_invisible_to_inference():
    """The trap, shown directly.

    Every episode stops at the first milestone, so the later ones never appear
    as events. Inferring the vocabulary from what happened gives a one-rung
    funnel at 100% and an uplift of zero -- a clean bill of health for a policy
    that never does anything.
    """
    episodes, _ = _run_that_always_stops_at_stage_two()
    report = Funnel().run([Cohort("inferred", episodes)])
    finding = report.by_code("funnel.bottleneck")[0]
    assert "approach is the weak link at 100%" in finding.summary
    assert any("was inferred" in note for note in report.notes)


def test_the_declared_vocabulary_tells_the_truth_instead():
    """Same episodes, but the run recorded what it was looking for."""
    episodes, provenance = _run_that_always_stops_at_stage_two()
    report = Funnel().run([Cohort("declared", episodes, provenance=provenance)])
    finding = report.by_code("funnel.bottleneck")[0]
    assert "engage is the weak link at 0%" in finding.summary
    assert report.measurements["declared.end_to_end"].value == 0.0
    assert not any("was inferred" in note for note in report.notes)


def test_configured_stages_beat_a_declaration():
    episodes, provenance = _run_that_always_stops_at_stage_two()
    cohort = Cohort("c", episodes, provenance=provenance)
    assert Funnel(stages=("approach", "engage")).stages_for(cohort) == (
        ("approach", "engage"),
        "configured",
    )


def test_a_declaration_beats_inference():
    episodes, provenance = _run_that_always_stops_at_stage_two()
    stages, origin = Funnel().stages_for(Cohort("c", episodes, provenance=provenance))
    assert origin == "declared by the run"
    assert len(stages) == 4
