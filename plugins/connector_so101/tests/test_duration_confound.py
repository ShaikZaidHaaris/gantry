"""The suite where nothing is wrong and everything varies in length.

``docs/writing-a-plugin.md``: ``make_duration_confound()`` "is a suite where
nothing is wrong and everything varies in length, and a module that reports a
finding on it is measuring duration. Getting that suite to stay silent is harder
and more valuable than getting the defective suites to speak."

These tests run it end to end — fixture suite, rendered as an SO-101 recording
on disk, read back through this connector, bound by the resolver, screened by
the feedback plane — and check three separate things:

1. the connector adds no artefact: cohorts that share the length distribution
   are reported as indistinguishable, with no finding at all;
2. the connector's *description* is what stops the joint vector being read as a
   Cartesian position. Six percentages of calibrated travel are not a point in
   space, and the statistics that assume one are refused a channel rather than
   handed the nearest thing of the right width;
3. what does separate a duration-ordered split, and why each one is a property
   of an absolute joint-command channel rather than of the data. That last set
   is pinned, so a change in the feedback plane's statistics shows up here as a
   test failure and not as a new finding on a clean corpus.
"""

from __future__ import annotations

import pytest
from gantry_connector_so101 import FOLLOWER_REF, SO101Connector
from gantry_connector_so101.fixtures import write_corpus
from gantry_feedback_core.metrics import get_statistic, tabulate
from gantry_feedback_core.screen import Screen

from gantry.contracts.feedback import Cohort
from gantry.fixtures import make_duration_confound
from gantry.resolve import adapt_all, bind


@pytest.fixture
def screened(tmp_path):
    """The confound suite, on disk as SO-101, read back and bound for the screen."""
    suite = make_duration_confound(n=16)
    root = write_corpus(suite.episodes, tmp_path / "confound")
    connector = SO101Connector(root, embodiment=FOLLOWER_REF)
    episodes = tuple(connector.episodes())
    screen = Screen()
    wiring, verdict = bind(screen.requirement(), episodes[0].schema)
    assert verdict.ok, verdict.explain()
    return screen, wiring, adapt_all(episodes, wiring)


def test_the_corpus_really_does_vary_in_length(screened):
    """Otherwise the rest of this file proves nothing."""
    _, _, episodes = screened
    lengths = [len(episode) for episode in episodes]
    assert max(lengths) >= 2 * min(lengths)


def test_the_confound_stays_silent_through_this_connector(screened):
    """Two cohorts that share the length distribution: nothing separates them.

    This is the test that the connector is not the thing producing a finding.
    Every episode goes through the same rendering, the same parquet, the same
    lazy window reads and the same channel description, so if any of that
    introduced a per-episode artefact it would land here as a difference
    between two halves that differ in nothing else.
    """
    screen, _, episodes = screened
    report = screen.analyse(
        [Cohort("odd", tuple(episodes[0::2])), Cohort("even", tuple(episodes[1::2]))]
    )
    assert report.findings == (), [finding.summary for finding in report.findings]
    assert "no statistic separates these cohorts after FDR correction" in report.notes


def test_joint_percent_does_not_bind_to_the_position_role(screened):
    """The description is the guard, and it is doing the work here.

    ``observation.state`` is six numbers wide and float32, which is exactly what
    the screen's ``position`` role asks for. The only reason it does not bind is
    that this connector declares what those numbers *mean* —
    ``so101.joint_position_normalized``, dimensionless percent of calibrated
    travel — and the role is not that. Path efficiency over percentages of
    travel would be a number: net over total in a space where the gripper's jaw
    and the shoulder's rotation are the same unit, and where the SO-101 URDF
    carries no linear gripper joint so no Cartesian path exists to be efficient
    in. It is unavailable instead, which is the correct outcome.
    """
    _, wiring, episodes = screened
    assert "position" in wiring.skipped
    assert "actuation" in wiring.skipped  # the gripper is dimension 5, not a channel
    assert episodes[0].channel_names == ("command",)

    _, unavailable = tabulate(episodes)
    assert "path_efficiency" in unavailable
    assert "direction_change_rate" in unavailable


#: What a duration-ordered split of a corpus with nothing wrong in it does
#: separate, and why. Every one of these is a property of the *channel*, not of
#: the demonstrations, and none of them is a defect this corpus has.
EXPECTED_ON_A_DURATION_SPLIT = {
    "episode_length": (
        "the duration itself. Registered in the feedback plane as a canary and "
        "barred from advice (duration_free=False)"
    ),
    "actuation_jerk": (
        "mean |step-to-step change| over an ABSOLUTE joint command. Every episode "
        "contains exactly one grasp and one release, so the gripper's fixed total "
        "travel is divided by a variable number of steps and the mean falls as 1/T. "
        "This is a real property of the SO-101 action channel, not of the fixture: "
        "a longer demonstration of the same task still closes the gripper once"
    ),
    "command_magnitude": (
        "mean ||action|| over an ABSOLUTE joint command, which is the mean distance "
        "of the pose from the origin of the joint coordinate system rather than a "
        "behaviour. It is declared duration_free=True, which holds for a delta "
        "command and does not hold for an absolute one"
    ),
}


def test_a_duration_ordered_split_only_moves_what_duration_moves(screened):
    """The hard direction: cohorts that differ in nothing but length.

    Findings appear, and every one of them is accounted for above. The
    assertion is on the exact set, so a new statistic in the feedback plane —
    or a fix to one of these two — surfaces here rather than quietly becoming a
    finding about a corpus with nothing wrong with it.
    """
    screen, _, episodes = screened
    half = len(episodes) // 2
    report = screen.analyse(
        [Cohort("short", tuple(episodes[:half])), Cohort("long", tuple(episodes[half:]))]
    )
    separated = {
        finding.summary.split(" separates")[0] for finding in report.findings
    }
    assert separated == set(EXPECTED_ON_A_DURATION_SPLIT), (
        f"unexpected: {sorted(separated - set(EXPECTED_ON_A_DURATION_SPLIT))}, "
        f"missing: {sorted(set(EXPECTED_ON_A_DURATION_SPLIT) - separated)}"
    )
    assert not any(
        finding.evidence.get("prescribable") is False and finding.prescription
        for finding in report.findings
    )


def test_the_two_that_fire_are_declared_duration_free_and_are_not(screened):
    """Named here so the gap is a written claim rather than an omission.

    ``actuation_jerk`` and ``command_magnitude`` are registered
    ``duration_free=True`` and may therefore become advice. On an absolute
    joint-command channel with a discrete gripper they are not duration-free, so
    on this corpus the screen offers to collect more demonstrations resembling
    the longer cohort — which is advice to record longer episodes and nothing
    else. Fixing that belongs to the feedback plane (a per-step normalisation,
    or splitting the gripper dimension out through a declared retargeter); this
    connector's part is to describe the channel well enough that the problem is
    visible, which is what makes this test possible to write.
    """
    for name in ("actuation_jerk", "command_magnitude"):
        assert get_statistic(name).duration_free is True
