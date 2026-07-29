"""Ranking policies, and refusing to rank anything else."""

from __future__ import annotations

import pytest
from gantry_feedback_compare import PolicyComparison, paired_counts, scene_of

from gantry.conformance import check_feedback
from gantry.contracts.feedback import Cohort
from gantry.spine import ComponentRef, EpisodeLabels, IncompatibleError, Provenance
from gantry.spine.episode import episode_from_labels

RIG = ComponentRef("evaluation", "robosuite", "1.0")


def arm(name, outcomes, *, policy="p", evaluation=RIG, offset=0, trained_on=None):
    episodes = tuple(
        episode_from_labels(
            id=f"scene-{i + offset:02d}", source=name,
            labels=EpisodeLabels(success=won),
        )
        for i, won in enumerate(outcomes)
    )
    protocol = {"execute": 8}
    if trained_on:
        protocol["trained_on"] = trained_on
    return Cohort(
        name, episodes,
        provenance=Provenance(
            components=(ComponentRef("policy", policy, "1.0"), evaluation),
            protocol=protocol,
        ),
    )


@pytest.fixture
def arms():
    """Ten shared scenes; ph wins six that mg loses and loses none.

    Six, not five, on purpose: five disagreements all in one direction gives
    p=0.0625 and does not clear the bar. The exact test is doing real work at
    this sample size, which is the reason for using it.
    """
    ph = arm("ph", [True] * 8 + [False] * 2, policy="ft-ph", trained_on="lift/ph")
    mg = arm("mg", [True, True] + [False] * 8, policy="ft-mg", trained_on="lift/mg")
    return [ph, mg]


def test_conforms(arms):
    verdict = check_feedback(PolicyComparison(), arms, strict=True)
    assert verdict.ok, verdict.explain()


def test_it_holds_the_world_and_not_the_policy():
    """Every other comparative module holds the policy. This is the inverse."""
    module = PolicyComparison()
    assert "policy" not in module.holds()
    assert "evaluation" in module.holds()


def test_it_names_the_winner_with_a_paired_test(arms):
    (finding,) = PolicyComparison().run(arms).findings
    assert finding.code == "compare.best_policy"
    assert finding.evidence["best"] == "ph"
    assert finding.evidence["won"] == 6 and finding.evidence["lost"] == 0
    assert finding.evidence["p"] < 0.05
    assert finding.severity == "strong"
    assert "Use ph" in finding.prescription


def test_every_arm_gets_a_rate_with_an_interval(arms):
    report = PolicyComparison().run(arms)
    rate = report.measurements["ph.success_rate"]
    assert rate.value == pytest.approx(0.8)
    assert rate.n == 10 and rate.ci[0] < 0.8 < rate.ci[1]


def test_the_training_set_is_context_not_a_conclusion(arms):
    report = PolicyComparison().run(arms)
    (finding,) = report.findings
    assert finding.evidence["best_trained_on"] == "lift/ph"
    assert "cannot separate the training data from the training seed" in " ".join(report.notes)
    assert "data is better" not in (finding.prescription or "")


def test_a_small_difference_is_reported_without_a_prescription():
    close = [arm("a", [True] * 5 + [False] * 5, policy="a"),
             arm("b", [True] * 4 + [False] * 6, policy="b")]
    (finding,) = PolicyComparison().run(close).findings
    assert finding.severity == "weak"
    assert finding.prescription is None


def test_arms_sharing_no_scene_are_ranked_unpaired():
    left = arm("a", [True, True, False], policy="a")
    right = arm("b", [False, False, False], policy="b", offset=50)
    (finding,) = PolicyComparison().run([left, right]).findings
    assert finding.evidence["paired_scenes"] == 0
    assert "share no scene" in finding.summary
    assert finding.prescription is None, "unpaired is not convincing on its own"


def test_all_zero_ranks_nothing_and_says_why():
    """The failure mode this project cares most about not faking."""
    flat = [arm("a", [False] * 8, policy="a"), arm("b", [False] * 8, policy="b")]
    report = PolicyComparison().run(flat)
    assert not report.findings
    assert any("no resolving power" in note for note in report.notes)


def test_a_different_world_is_refused(arms):
    other = arm("mg", [True] * 10, policy="ft-mg",
                evaluation=ComponentRef("evaluation", "libero", "1.0"))
    verdict = PolicyComparison().check_inputs([arms[0], other])
    assert "feedback.incomparable" in verdict.codes()
    with pytest.raises(IncompatibleError, match="differ on"):
        PolicyComparison().run([arms[0], other])


def test_comparing_one_policy_against_itself_is_noted():
    same = [arm("a", [True, False], policy="same"), arm("b", [True, True], policy="same")]
    verdict = PolicyComparison().check_inputs(same)
    assert "compare.same_policy" in verdict.codes()
    assert verdict.ok, "a note, not a refusal — it is a legitimate variance check"


def test_cohorts_without_outcomes_are_refused():
    empty = [
        Cohort(n, (episode_from_labels(id="s", source=n, labels=EpisodeLabels()),),
               provenance=Provenance(components=(ComponentRef("policy", n, "1"), RIG)))
        for n in ("a", "b")
    ]
    verdict = PolicyComparison().check_inputs(empty)
    assert "compare.no_outcomes" in verdict.codes()


def test_pairing_counts_only_shared_scenes():
    left = arm("a", [True, False, True], policy="a")
    right = arm("b", [True, True], policy="b")
    from gantry_feedback_compare import arm_of

    both, only_l, only_r, neither = paired_counts(arm_of(left), arm_of(right))
    assert both + only_l + only_r + neither == 2


def test_the_scene_is_read_from_an_annotation_when_there_is_one():
    episode = episode_from_labels(
        id="run-9", source="x",
        labels=EpisodeLabels(success=True, annotations={"initial_state": 4}),
    )
    assert scene_of(episode) == "4"
