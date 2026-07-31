"""Cutting a corpus in two, and the four ways that quietly answers a different question.

None of these raise on their own. A leaked participant, an unmatched size, a
truncated trajectory and an unnamed criterion all produce two cohorts that look
exactly like two cohorts, and the ranking that comes out of them looks exactly
like a ranking.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_curate_split import Split, group_of, match_frames, moving_fraction

from gantry.errors import ConfigError
from gantry.spine import ArraySource, ChannelSpec, EpisodeLabels, EpisodeMeta, EpisodeRecord

WIDTH = 14


def episode(name, steps=100, left_moving=1.0, right_moving=1.0, participant=None):
    """One clip, with each arm moving for a given fraction of it."""
    values = np.zeros((steps, WIDTH))
    for index, fraction in enumerate((left_moving, right_moving)):
        live = int(steps * fraction)
        # a moving arm walks; a frozen one holds its last value, which is what
        # the connector does when the tracker loses a hand
        values[:live, index * 7] = np.arange(live) * 0.01
        values[live:, index * 7] = values[max(0, live - 1), index * 7]
    return EpisodeRecord(
        meta=EpisodeMeta(
            id=name,
            source="ego",
            task="t",
            extra={"participant": participant} if participant else {},
        ),
        schema=(ChannelSpec("action", "vector", (WIDTH,), "float32", semantics="actuation"),),
        source=ArraySource({"action": values.astype("float32")}),
        labels=EpisodeLabels(success=None),
    )


def splitter(**kwargs):
    return Split(
        measure=moving_fraction(),
        threshold=0.5,
        names=("two_handed", "one_handed"),
        criterion="fraction of frames with both hands moving",
        **kwargs,
    )


# -- the measure ----------------------------------------------------------------


def test_a_frozen_arm_reads_as_not_moving():
    """The tracker losing a hand is invisible to every shape and dtype check:
    the connector holds the last value, so the block is well formed and
    constant."""
    measure = moving_fraction()
    assert measure(episode("a", left_moving=1.0, right_moving=1.0)) > 0.95
    assert measure(episode("b", left_moving=0.01, right_moving=1.0)) < 0.05


def test_a_clip_is_only_two_handed_if_both_hands_are():
    """The smaller of the two, not the mean. One good arm and one absent one
    averages to something that reads as adequate."""
    measure = moving_fraction()
    assert measure(episode("c", left_moving=1.0, right_moving=0.2)) == pytest.approx(0.2, abs=0.02)


# -- the four quiet failures ----------------------------------------------------


def test_a_participant_in_both_halves_is_refused():
    """Two halves that share a person are one contributor cut in half, and any
    difference between them is pulled toward zero by however much they
    overlap."""
    episodes = [
        episode("a", left_moving=1.0, participant="P01"),
        episode("b", left_moving=0.0, participant="P01"),
    ]
    with pytest.raises(ConfigError, match="one contributor cut in half"):
        splitter().cut(episodes)


def test_sizes_are_matched_by_dropping_whole_episodes():
    """Never by trimming: truncating a trajectory to hit a frame count teaches a
    policy that reaching stops halfway."""
    episodes = [episode(f"hi{i}", steps=100, participant=f"H{i}") for i in range(6)]
    episodes += [
        episode(f"lo{i}", steps=100, left_moving=0.0, participant=f"L{i}") for i in range(2)
    ]
    parts = match_frames(splitter().cut(episodes))
    assert parts["two_handed"].frames == parts["one_handed"].frames == 200
    assert all(len(e) == 100 for p in parts.values() for e in p.episodes)
    assert len(parts["two_handed"].dropped) == 4


def test_what_was_dropped_to_match_is_named():
    episodes = [episode(f"hi{i}", steps=50, participant=f"H{i}") for i in range(4)]
    episodes += [episode("lo0", steps=50, left_moving=0.0, participant="L0")]
    parts = match_frames(splitter().cut(episodes))
    assert len(parts["two_handed"].dropped) == 3
    assert parts["two_handed"].summary()["dropped_to_match"]


def test_parts_that_cannot_be_matched_are_refused_rather_than_compared():
    """A comparison between halves of visibly different size is a comparison of
    sizes. Whole episodes are the unit, so a part of a few long clips cannot
    always be trimmed finely."""
    episodes = [episode("hi0", steps=1000, participant="H0")]
    episodes += [episode("lo0", steps=10, left_moving=0.0, participant="L0")]
    with pytest.raises(ConfigError, match="could not match these parts"):
        match_frames(splitter().cut(episodes))


def test_a_split_has_to_say_what_it_split_on():
    """'cohort A' and 'cohort B' are not a criterion anybody can disagree with."""
    with pytest.raises(ConfigError, match="split on"):
        Split(measure=moving_fraction(), threshold=0.5, criterion="  ")


# -- what it records ------------------------------------------------------------


def test_each_episodes_measured_value_is_kept_not_just_its_side():
    parts = splitter().cut(
        [episode("a", participant="A"), episode("b", left_moving=0.0, participant="B")]
    )
    assert len(parts["two_handed"].measured) == 1
    assert parts["two_handed"].summary()["measured_median"] > 0.9
    assert parts["one_handed"].summary()["measured_range"][1] < 0.1


def test_the_grouping_key_falls_back_to_the_episode_rather_than_to_nothing():
    """Each episode its own group is the safe reading; a corpus that says
    nothing about who filmed it should not be assumed to be all one person, nor
    all different ones."""
    assert group_of(episode("clip7")) == "clip7"
    assert group_of(episode("clip7", participant="P03")) == "P03"


def test_an_empty_part_is_still_a_part():
    """A threshold nothing clears is a fact about the corpus, and reporting it
    as an empty cohort says so. Dropping it silently would leave a one-sided
    comparison looking like a two-sided one."""
    parts = splitter().cut([episode("a", participant="A")])
    assert len(parts["one_handed"]) == 0
    assert parts["one_handed"].frames == 0
