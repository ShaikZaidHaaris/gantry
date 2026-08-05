"""Coverage, and the unfairness it exists to prevent."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_feedback_coverage import Coverage, instructions_of, overlap, report_for, words

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeLabels, episode_from_arrays

SPEC = ChannelSpec("x", "vector", (1,), "float32")

KITCHEN = [
    "pick up the mug and put it in the sink",
    "open the fridge door",
    "pour water into the glass",
    "wipe the counter with a cloth",
]

TABLETOP = [
    "pick up the cube",
    "stack the red block on the green block",
    "push the block to the target",
    "open the drawer",
    "close the drawer",
    "insert the peg into the hole",
]


def cohort(instructions, name="theirs"):
    episodes = []
    for index, text in enumerate(instructions):
        episodes.append(
            episode_from_arrays(
                {"x": np.zeros((2, 1), dtype="float32")},
                [SPEC],
                id=f"{name}-{index}",
                source=name,
                labels=EpisodeLabels(annotations={"instruction": text}),
            )
        )
    return Cohort(name=name, episodes=tuple(episodes))


def codes(report):
    return [finding.code for finding in report.findings]


# -- the unfairness ----------------------------------------------------------


def test_cooking_footage_evaluated_on_tabletop_reads_as_a_task_mismatch():
    """The report this module exists to prevent: a contributor's kitchen data
    measured against pick-and-place, zero improvement, and a verdict that their
    data did not help. Their data was never given a chance to help, and the two
    conclusions lead to opposite actions."""
    report = report_for(cohort(KITCHEN), TABLETOP)

    assert "coverage.mismatch" in codes(report)
    finding = next(f for f in report.findings if f.code == "coverage.mismatch")
    assert finding.severity == "strong"
    assert "task match rather than about this data" in finding.summary
    assert "never about it" in finding.prescription


def test_matching_data_is_reported_as_ample_and_says_the_delta_is_interpretable():
    report = report_for(cohort(TABLETOP), TABLETOP)
    assert codes(report) == ["coverage.ample"]
    assert "about this data rather than about scope" in report.findings[0].summary


def test_partial_coverage_asks_for_the_table_to_be_split():
    """The touched tasks measure what the data taught; the rest measure transfer.
    Those are different claims and one mean over both supports neither."""
    half = TABLETOP[:3] + KITCHEN[:2]
    report = report_for(cohort(half), TABLETOP)
    assert "coverage.partial" in codes(report)
    finding = next(f for f in report.findings if f.code == "coverage.partial")
    assert finding.severity == "weak"
    assert "measure transfer" in finding.prescription


# -- the direction people forget --------------------------------------------


def test_data_about_things_nothing_evaluates_is_reported_as_the_benchmarks_blind_spot():
    """The more actionable half, and the honest thing to tell somebody who paid
    attention while filming."""
    mixed = TABLETOP[:2] + KITCHEN
    report = report_for(cohort(mixed), TABLETOP)

    assert "coverage.unused_data" in codes(report)
    finding = next(f for f in report.findings if f.code == "coverage.unused_data")
    assert "nothing in this benchmark evaluates" in finding.summary
    assert "benchmark's blind spot" in finding.prescription
    assert finding.evidence["examples"]


def test_the_unused_fraction_is_measured():
    mixed = TABLETOP[:2] + ["pour the water", "stir the soup", "fold the towel"]
    report = report_for(cohort(mixed), TABLETOP)
    detail = report.measurements["theirs.coverage"].detail
    assert detail["clips"] == 5
    assert detail["unused_clip_fraction"] == pytest.approx(0.6)


def test_the_default_measure_is_generous_with_a_shared_verb_and_says_so():
    """A known limitation, pinned rather than hidden. "pick up the mug and put it
    in the sink" and "pick up the cube" share only `pick`, and over the shorter
    instruction that is half its content -- enough to count as a match.

    That is the cost of a measure a user can check by eye. It errs toward
    claiming coverage, which is the safer direction here: over-claiming coverage
    understates a mismatch finding, while under-claiming it would manufacture
    mismatches for data that is genuinely relevant. Anyone who needs the
    distinction passes a sentence encoder.
    """
    assert overlap("pick up the mug and put it in the sink", "pick up the cube") == 0.5
    assert overlap("stir the soup", "pick up the cube") == 0.0


# -- the refusals ------------------------------------------------------------


def test_no_task_list_is_a_refusal_rather_than_a_coverage_of_zero():
    """'Nothing to compare against' and 'no overlap' are different sentences with
    different fixes."""
    report = Coverage(evaluates=()).analyse([cohort(KITCHEN)])
    assert codes(report) == ["coverage.no_task_list"]
    assert "most unfair thing" in report.findings[0].prescription
    assert not report.measurements


def test_a_cohort_with_no_instructions_is_named():
    episodes = (
        episode_from_arrays(
            {"x": np.zeros((2, 1), dtype="float32")},
            [SPEC],
            id="bare",
            source="theirs",
            labels=EpisodeLabels(),
        ),
    )
    report = Coverage(evaluates=TABLETOP).analyse([Cohort(name="theirs", episodes=episodes)])
    assert codes(report) == ["coverage.no_instructions"]
    assert report.findings[0].severity == "strong"


# -- the measure itself, which a user is entitled to check ------------------


def test_the_method_appears_in_every_finding_and_measurement():
    """Two runs measured with different notions of similar are not comparable,
    and the number alone does not say which was used."""
    report = report_for(cohort(KITCHEN), TABLETOP)
    assert report.measurements["theirs.coverage"].method
    for finding in report.findings:
        assert "method" in finding.evidence


def test_similarity_is_over_the_shorter_instruction_not_the_union():
    """'pick up the mug' and 'pick up the red mug from the second shelf' describe
    overlapping activity, and a Jaccard would score that pair low purely because
    one sentence is longer."""
    short, long = "pick up the mug", "pick up the red mug from the second shelf"
    assert overlap(short, long) == 1.0
    assert overlap(long, short) == 1.0
    assert overlap("pick up the mug", "open the drawer") == 0.0


def test_stopwords_are_dropped_and_content_words_kept_in_order():
    assert words("Pick up the RED block from the table") == (
        "pick",
        "red",
        "block",
        "table",
    )


def test_repeated_instructions_are_the_weighting():
    """A dataset with forty clips of one thing and one of another is mostly about
    the first; deduplicating would report it as evenly split."""
    said = instructions_of(cohort(["pick up the cube"] * 4 + ["open the drawer"]))
    assert len(said) == 5
    assert said.count("pick up the cube") == 4


def test_an_encoder_can_be_swapped_in_and_lands_in_the_record():
    """The default is the dull one because it is inspectable. Somebody with a
    sentence encoder passes one in, and which was used has to be visible."""

    def everything_matches(left, right):
        return 1.0

    report = Coverage(
        evaluates=TABLETOP,
        similarity=everything_matches,
        method="cosine over a sentence encoder",
    ).analyse([cohort(KITCHEN)])

    assert codes(report) == ["coverage.ample"]
    assert report.measurements["theirs.coverage"].value == 1.0
    assert "sentence encoder" in report.measurements["theirs.coverage"].method


def test_verb_coverage_is_reported_separately():
    """'pick up the mug' and 'pick up the cube' are much closer than 'pick up the
    mug' and 'open the drawer', and a bag of words alone half-hides that."""
    report = report_for(cohort(["pick up the mug"]), ["pick up the cube"])
    assert report.measurements["theirs.verb_coverage"].value == 1.0

    report = report_for(cohort(["pour the water"]), ["open the drawer"])
    assert report.measurements["theirs.verb_coverage"].value == 0.0


def test_the_missing_verbs_are_named_in_the_evidence():
    report = report_for(cohort(["pour the water into the glass"]), TABLETOP)
    finding = next(f for f in report.findings if f.code == "coverage.mismatch")
    assert "stack" in finding.evidence["missing_verbs"]


def test_the_per_task_best_match_is_shown_so_a_user_can_check_it():
    report = report_for(cohort(KITCHEN), TABLETOP)
    per_task = next(f for f in report.findings if f.code == "coverage.mismatch").evidence[
        "per_task_best_match"
    ]
    assert set(per_task) == set(TABLETOP)
    assert all(0.0 <= score <= 1.0 for score in per_task.values())


# -- the plumbing ------------------------------------------------------------


def test_it_analyses_one_cohort_because_this_is_not_a_comparison():
    made = Coverage(evaluates=TABLETOP)
    assert made.descriptor().provides["min_cohorts"] == 1
    assert made.descriptor().provides["prescribes"] is True


def test_several_cohorts_are_each_assessed_separately():
    report = Coverage(evaluates=TABLETOP).analyse(
        [cohort(TABLETOP, "matching"), cohort(KITCHEN, "mismatched")]
    )
    assert "matching.coverage" in report.measurements
    assert "mismatched.coverage" in report.measurements
    assert report.measurements["matching.coverage"].value > (
        report.measurements["mismatched.coverage"].value
    )


def test_the_instruction_falls_back_to_the_episode_task():
    episodes = (
        episode_from_arrays(
            {"x": np.zeros((2, 1), dtype="float32")},
            [SPEC],
            id="e0",
            source="theirs",
            task="pick up the cube",
            labels=EpisodeLabels(),
        ),
    )
    said = instructions_of(Cohort(name="theirs", episodes=episodes))
    assert said == ("pick up the cube",)
