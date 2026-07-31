"""The control arm, and the claim it is the only thing that licenses."""

from __future__ import annotations

import numpy as np
from gantry_feedback_control import Control, errors_of, outcomes_of

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeLabels, episode_from_arrays

SPEC = ChannelSpec("x", "vector", (1,), "float32")


def arm(name, wins=0, total=0, errors=None):
    episodes = []
    for index in range(total):
        episodes.append(
            episode_from_arrays(
                {"x": np.zeros((2, 1), dtype="float32")},
                [SPEC],
                id=f"{name}-{index}",
                source=name,
                labels=EpisodeLabels(success=index < wins),
            )
        )
    for index, err in enumerate(errors or []):
        episodes.append(
            episode_from_arrays(
                {"x": np.zeros((2, 1), dtype="float32")},
                [SPEC],
                id=f"{name}-e{index}",
                source=name,
                labels=EpisodeLabels(annotations={"action_error": float(err)}),
            )
        )
    return Cohort(name=name, episodes=tuple(episodes))


def codes(report):
    return [f.code for f in report.findings]


# -- the reason the module exists -------------------------------------------


def test_without_a_control_the_module_refuses_and_says_why():
    """Fine-tuning a large pretrained model on anything moves it. A
    baseline-versus-treatment table shows a difference for every contributor,
    including the ones whose data is worthless."""
    report = Control().analyse([arm("ego", 30, 50), arm("base", 20, 50)])
    assert codes(report) == ["control.no_control"]
    finding = report.findings[0]
    assert finding.severity == "strong"
    assert "including the ones whose data is worthless" in finding.prescription
    assert "detached from the frames" in finding.prescription


def test_beating_the_control_is_the_claim_worth_making():
    report = Control().analyse(
        [
            arm("ego", 40, 60),
            arm("shuffled", 12, 60),
            arm("base", 10, 60),
        ]
    )
    assert "control.data_carried_information" in codes(report)
    finding = next(f for f in report.findings if f.code == "control.data_carried_information")
    assert "correspondence between the images and the actions" in finding.summary
    # and the claim is bounded
    assert "narrower than it sounds" in finding.prescription
    assert "different experiment" in finding.prescription


def test_not_beating_the_control_is_reported_as_such_not_rounded_past():
    """The point of a control is to be capable of embarrassing you."""
    report = Control().analyse(
        [
            arm("ego", 31, 60),
            arm("shuffled", 29, 60),
            arm("base", 12, 60),
        ]
    )
    assert "control.data_carried_information" not in codes(report)
    finding = next(f for f in report.findings if f.code == "control.not_separated")
    assert "has not been shown to carry information" in finding.summary
    assert "noise with a decimal point on it" in finding.prescription


def test_a_control_that_wins_stops_the_report_rather_than_appearing_in_it():
    """A control outperforming the real thing is not a data-quality result — it
    means labels are misaligned or the split leaked."""
    report = Control().analyse(
        [
            arm("ego", 10, 60),
            arm("shuffled", 40, 60),
            arm("base", 9, 60),
        ]
    )
    finding = next(f for f in report.findings if f.code == "control.control_wins")
    assert finding.severity == "strong"
    assert "not a finding about the data" in finding.summary
    assert "Stop and check the pipeline" in finding.prescription


def test_both_arms_moving_off_base_is_reported_as_what_fine_tuning_does():
    report = Control().analyse(
        [
            arm("ego", 40, 60),
            arm("shuffled", 35, 60),
            arm("base", 5, 60),
        ]
    )
    finding = next(f for f in report.findings if f.code == "control.fine_tuning_effect")
    assert "regardless of what it is shown" in finding.summary
    assert finding.evidence["both_moved_off_base"] is True
    assert "would have been reported as a data result" in finding.prescription


# -- the offline fallback, honest about being one ---------------------------


def test_offline_prediction_error_is_used_when_there_are_no_outcomes():
    report = Control().analyse(
        [
            arm("ego", errors=[0.10, 0.11, 0.09, 0.10, 0.11]),
            arm("shuffled", errors=[0.30, 0.31, 0.29, 0.30, 0.31]),
        ]
    )
    finding = report.findings[0]
    assert finding.code == "control.data_carried_information"
    assert "offline" in finding.summary
    assert "measures fit, not capability" in finding.prescription
    assert "rung on the ladder" in finding.prescription


def test_offline_abstains_when_the_arms_overlap():
    report = Control().analyse(
        [
            arm("ego", errors=[0.20, 0.25, 0.15, 0.30, 0.10]),
            arm("shuffled", errors=[0.21, 0.24, 0.16, 0.29, 0.11]),
        ]
    )
    assert report.findings[0].code == "control.not_separated"
    assert "not separated" in report.findings[0].summary


def test_nothing_measured_at_all_says_the_loss_curve_is_not_evidence():
    report = Control().analyse([arm("ego"), arm("shuffled")])
    finding = report.findings[0]
    assert finding.code == "control.nothing_measured"
    assert "true of the shuffled control as well" in finding.prescription


# -- the plumbing ------------------------------------------------------------


def test_the_arms_must_differ_only_in_their_data():
    holds = Control().descriptor().provides["holds"]
    for plane in ("policy", "evaluation", "task", "embodiment"):
        assert plane in holds


def test_abstentions_are_dropped_rather_than_counted_as_losses():
    episodes = [
        episode_from_arrays(
            {"x": np.zeros((2, 1), dtype="float32")},
            [SPEC],
            id=f"e{i}",
            source="ego",
            labels=EpisodeLabels(success=s),
        )
        for i, s in enumerate([True, False, None, None])
    ]
    wins, total = outcomes_of(Cohort(name="ego", episodes=tuple(episodes)))
    assert (wins, total) == (1, 2)


def test_arm_names_tolerate_the_prefixes_a_real_run_adds():
    report = Control().analyse(
        [
            arm("ego_v2", 40, 60),
            arm("shuffled_v2", 12, 60),
        ]
    )
    assert "control.data_carried_information" in codes(report)


def test_errors_are_read_from_annotations():
    values = errors_of(arm("ego", errors=[0.1, 0.2]))
    assert list(values) == [0.1, 0.2]


def test_it_needs_two_arms():
    assert Control().descriptor().provides["min_cohorts"] == 2
