"""Ranking execution settings, and refusing to rank anything else."""

from __future__ import annotations

import pytest
from gantry_feedback_protocol import ProtocolSweep, mcnemar, paired, varying, wilson

from gantry.conformance import check_feedback
from gantry.contracts.feedback import Cohort
from gantry.spine import ComponentRef, EpisodeLabels, IncompatibleError, Provenance
from gantry.spine.episode import episode_from_labels

POLICY = ComponentRef("policy", "fixed", "1.0")
EVALUATOR = ComponentRef("evaluation", "rig", "1.0")


def arm(
    name: str,
    outcomes: list[bool],
    *,
    execute: int | None = None,
    policy: ComponentRef = POLICY,
    evaluator: ComponentRef = EVALUATOR,
    offset: int = 0,
    protocol: dict | None = None,
) -> Cohort:
    """One cohort of trials, scene ids shared across arms unless offset."""
    episodes = tuple(
        episode_from_labels(
            id=f"scene-{index + offset:02d}",
            source=name,
            labels=EpisodeLabels(success=outcome),
        )
        for index, outcome in enumerate(outcomes)
    )
    if protocol is None:
        protocol = {"epochs": 1, "seed_base": 0}
        if execute is not None:
            protocol["execute"] = execute
    return Cohort(
        name,
        episodes,
        provenance=Provenance(components=(policy, evaluator), protocol=protocol),
    )


@pytest.fixture
def arms():
    """Eight scenes; the longer chunk wins three of them and loses one."""
    short = arm("execute-1", [True, False, False, True, False, True, False, False], execute=1)
    long = arm("execute-8", [True, True, True, True, True, False, False, True], execute=8)
    return [short, long]


# -- the contract ----------------------------------------------------------


def test_conforms(arms):
    verdict = check_feedback(ProtocolSweep(), arms, strict=True)
    assert verdict.ok, verdict.explain()


def test_it_declares_what_it_needs(arms):
    requirement = ProtocolSweep().requirement()
    assert requirement.plane == "feedback"
    assert requirement.capabilities == {"outcomes": True}
    assert not requirement.channels, "this module reads outcomes, never trajectories"


# -- ranking ---------------------------------------------------------------


def test_it_names_the_best_setting(arms):
    report = ProtocolSweep().run(arms)
    (finding,) = report.findings
    assert finding.code == "protocol.best_setting"
    assert finding.evidence["best"] == "execute-8"
    assert finding.evidence["best_setting"] == {"execute": 8}
    assert finding.evidence["gain"] == pytest.approx(0.375)


def test_the_gain_is_worth_prescribing(arms):
    (finding,) = ProtocolSweep().run(arms).findings
    assert finding.severity == "strong"
    assert "execute=8" in finding.prescription


def test_every_arm_gets_a_rate_with_an_interval(arms):
    report = ProtocolSweep().run(arms)
    rate = report.measurements["execute-8.success_rate"]
    assert rate.value == pytest.approx(0.75)
    assert rate.n == 8
    assert rate.ci[0] < rate.value < rate.ci[1]
    assert rate.detail == {"execute": 8}


def test_only_the_lever_that_moved_is_reported(arms):
    """epochs and seed_base are identical, so neither is part of the answer."""
    (finding,) = ProtocolSweep().run(arms).findings
    assert finding.evidence["levers"] == ["execute"]


# -- pairing ---------------------------------------------------------------


def test_arms_sharing_scenes_are_compared_paired(arms):
    (finding,) = ProtocolSweep().run(arms).findings
    assert finding.evidence["paired_scenes"] == 8
    assert (finding.evidence["won"], finding.evidence["lost"]) == (4, 1)
    assert "p" in finding.evidence


def test_arms_sharing_no_scene_are_compared_unpaired():
    """Two runs over different scenes are still rankable, just not paired."""
    left = arm("execute-1", [True, False, False, False], execute=1)
    right = arm("execute-8", [True, True, True, False], execute=8, offset=100)
    (finding,) = ProtocolSweep().run([left, right]).findings
    assert finding.evidence["paired_scenes"] == 0
    assert "share no scene" in finding.summary


def test_pairing_counts_only_the_scenes_both_arms_attempted():
    left = arm("a", [True, False, True], execute=1)
    right = arm("b", [True, True], execute=8)
    counts = paired(_arm_of(right), _arm_of(left))
    both, only_right, only_left, neither = counts
    assert both + only_right + only_left + neither == 2


def _arm_of(cohort: Cohort):
    from gantry_feedback_protocol.sweep import _arm

    return _arm(cohort)


def test_mcnemar_reads_only_the_disagreements():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 5) == 1.0
    assert mcnemar(10, 0) < 0.01
    assert mcnemar(4, 1) > mcnemar(9, 1)


def test_wilson_stays_inside_the_unit_interval():
    for successes, n in [(0, 5), (5, 5), (1, 100), (3, 7)]:
        low, high = wilson(successes, n)
        assert 0.0 <= low <= successes / n <= high <= 1.0


def test_wilson_narrows_as_evidence_accumulates():
    small = wilson(5, 10)
    large = wilson(50, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


# -- refusals --------------------------------------------------------------


def test_a_sweep_that_also_changed_the_policy_is_refused(arms):
    confounded = [
        arms[0],
        arm(
            "execute-8",
            [True] * 8,
            execute=8,
            policy=ComponentRef("policy", "different", "2.0"),
        ),
    ]
    verdict = ProtocolSweep().check_inputs(confounded)
    # The refusal now comes from the contract rather than this module: it
    # declares which planes it holds, and the base class checks provenance.
    assert "feedback.incomparable" in verdict.codes()
    assert ProtocolSweep().holds() == ("policy", "evaluation")
    with pytest.raises(IncompatibleError, match="differ on"):
        ProtocolSweep().run(confounded)


def test_a_run_that_never_recorded_its_protocol_is_refused():
    cohorts = [
        arm("a", [True, False], protocol={}),
        arm("b", [True, True], protocol={}),
    ]
    verdict = ProtocolSweep().check_inputs(cohorts)
    assert "protocol.unrecorded" in verdict.codes()


def test_cohorts_without_outcomes_are_refused():
    cohorts = [
        Cohort(
            name,
            (episode_from_labels(id="s0", source=name, labels=EpisodeLabels()),),
            provenance=Provenance(components=(POLICY, EVALUATOR), protocol={"execute": 1}),
        )
        for name in ("a", "b")
    ]
    verdict = ProtocolSweep().check_inputs(cohorts)
    assert "protocol.no_outcomes" in verdict.codes()


def test_one_cohort_is_not_a_sweep(arms):
    with pytest.raises(IncompatibleError, match="at least 2"):
        ProtocolSweep().run(arms[:1])


# -- honest silence --------------------------------------------------------


def test_two_arms_run_the_same_way_produce_no_winner():
    same = [arm("a", [True, False, True], execute=4), arm("b", [False, False, True], execute=4)]
    report = ProtocolSweep().run(same)
    assert not report.findings
    assert any("nothing to sweep" in note for note in report.notes)
    assert report.measurements["a.success_rate"].value == pytest.approx(2 / 3)


def test_two_levers_moving_at_once_is_reported_and_not_prescribed():
    left = arm("a", [False, False], protocol={"execute": 1, "epochs": 1})
    right = arm("b", [True, True], protocol={"execute": 8, "epochs": 3})
    report = ProtocolSweep().run([left, right])
    (finding,) = report.findings
    assert finding.severity == "weak"
    assert finding.prescription is None
    assert any("cannot be attributed" in note for note in report.notes)


def test_a_lever_nobody_declared_is_not_swept():
    """Only keys the module was told are levers count as a setting."""
    left = arm("a", [True], protocol={"temperature": 0.1})
    right = arm("b", [False], protocol={"temperature": 0.9})
    assert varying([_arm_of(left), _arm_of(right)]) == ("temperature",)
    assert not ProtocolSweep().run([left, right]).findings
    assert ProtocolSweep(levers=("temperature",)).run([left, right]).findings


def test_a_scene_annotation_wins_over_the_episode_id():
    """Two runs that numbered their episodes differently still pair up."""

    def annotated(name: str, ids: list[str], scenes: list[str], outcomes: list[bool]) -> Cohort:
        return Cohort(
            name,
            tuple(
                episode_from_labels(
                    id=episode_id,
                    source=name,
                    labels=EpisodeLabels(success=outcome, annotations={"scene": scene}),
                )
                for episode_id, scene, outcome in zip(ids, scenes, outcomes)
            ),
            provenance=Provenance(
                components=(POLICY, EVALUATOR),
                protocol={"execute": 1 if name == "a" else 8},
            ),
        )

    left = annotated("a", ["0", "1"], ["kitchen", "desk"], [False, False])
    right = annotated("b", ["run-7", "run-8"], ["kitchen", "desk"], [True, True])
    (finding,) = ProtocolSweep().run([left, right]).findings
    assert finding.evidence["paired_scenes"] == 2
    assert finding.evidence["won"] == 2
