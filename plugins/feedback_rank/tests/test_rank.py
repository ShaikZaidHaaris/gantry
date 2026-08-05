"""Ranking a matrix of policies by tasks, and refusing to aggregate a floor.

The fixtures are this project's measured Lift results and the shape of its actual
thirteen-task matrix, because the misreading this module prevents -- averaging one
real task with twelve zero-shot ones -- is one this project was about to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from gantry_feedback_rank import MatrixRanking, compact_letters

from gantry.contracts.feedback import Cohort


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


def cohort(name, per_task):
    episodes = []
    for task, (ok, n) in per_task.items():
        episodes += [Ep(Meta(f"{task}_{i}", task), Labels(i < ok)) for i in range(n)]
    return Cohort(name, tuple(episodes))


def measured():
    return [
        cohort("ph_official", {"lift_cube": (28, 50), "lift_cube_wide": (9, 50)}),
        cohort("mh_official", {"lift_cube": (33, 50), "lift_cube_wide": (29, 50)}),
        cohort("mg_official", {"lift_cube": (0, 50), "lift_cube_wide": (1, 50)}),
    ]


# -- ordering ----------------------------------------------------------------


def test_it_ranks_the_measured_results_the_way_the_evidence_does():
    report = MatrixRanking(resamples=400).analyse(measured())
    ordering = next(f for f in report.findings if f.code == "rank.ordering")
    assert ordering.summary.index("mh_official") < ordering.summary.index("ph_official")
    assert ordering.summary.index("ph_official") < ordering.summary.index("mg_official")


def test_separated_cohorts_get_different_letters():
    report = MatrixRanking(resamples=400).analyse(measured())
    letters = next(f for f in report.findings if f.code == "rank.ordering").evidence["letters"]
    assert letters["mh_official"] != letters["ph_official"] != letters["mg_official"]


def test_it_names_the_winner_when_the_winner_is_separated():
    report = MatrixRanking(resamples=400).analyse(measured())
    ordering = next(f for f in report.findings if f.code == "rank.ordering")
    assert ordering.prescription == "Use mh_official."
    assert ordering.severity == "strong"


def test_indistinguishable_cohorts_share_a_letter_and_the_claim_weakens():
    twins = [
        cohort("a", {"t1": (25, 50), "t2": (25, 50)}),
        cohort("b", {"t1": (26, 50), "t2": (24, 50)}),
    ]
    report = MatrixRanking(resamples=400).analyse(twins)
    ordering = next(f for f in report.findings if f.code == "rank.ordering")
    letters = ordering.evidence["letters"]
    assert set(letters["a"]) & set(letters["b"])
    assert ordering.severity == "weak"
    assert "not separated" in ordering.prescription


# -- the aggregate is robust -------------------------------------------------


def test_the_interquartile_mean_is_reported_with_an_interval():
    report = MatrixRanking(resamples=400).analyse(measured())
    measurement = report.measurements["mh_official.iqm"]
    assert measurement.ci is not None
    assert measurement.ci[0] <= measurement.value <= measurement.ci[1]
    assert "interquartile" in measurement.method


def test_a_profile_travels_with_the_aggregate():
    """So two cohorts a mean would conflate can still be told apart."""
    report = MatrixRanking(resamples=200).analyse(measured())
    profile = report.measurements["mh_official.iqm"].detail["profile"]
    assert len(profile) == 4
    assert all(0.0 <= fraction <= 1.0 for _, fraction in profile)


def test_probability_of_improvement_is_reported_per_pair():
    """Keys are alphabetical, so the same pair always lands in the same place
    rather than depending on the order cohorts were passed in."""
    report = MatrixRanking(resamples=200).analyse(measured())
    # mg before mh alphabetically, so the key reads mg_beats_mh -- and mg loses.
    assert report.measurements["mg_official_beats_mh_official"].value == pytest.approx(0.0)
    assert report.measurements["mg_official_beats_ph_official"].value == pytest.approx(0.0)


# -- what it refuses ---------------------------------------------------------


def test_a_matrix_that_is_mostly_floor_is_not_aggregated_silently():
    """The misreading this project was one step from making.

    One trained task and twelve zero-shot ones: the aggregate is decided by how
    many tasks were included, not by performance.
    """

    def thirteen(name, real):
        per = {"lift_cube": (int(real * 50), 50)}
        per.update({f"zero_shot_{i}": (0, 50) for i in range(12)})
        return cohort(name, per)

    report = MatrixRanking(resamples=200).analyse([thirteen("ph", 0.56), thirteen("mh", 0.66)])
    floor = next(f for f in report.findings if f.code == "rank.mostly_floor")
    assert "how many tasks were included" in floor.summary
    assert "separately" in floor.prescription
    assert floor.evidence["floor_share"]["ph"] > 0.5


def test_the_floor_finding_names_which_cells_are_at_it():
    report = MatrixRanking(resamples=200).analyse(measured())
    floor = next(f for f in report.findings if f.code == "rank.mostly_floor")
    assert floor.evidence["at_floor"]["mg_official"] == ["lift_cube", "lift_cube_wide"]


def test_one_lucky_success_is_still_the_floor():
    """1/50 is not evidence, and treating it as above the floor is how it becomes so."""
    report = MatrixRanking(resamples=200).analyse(measured())
    assert "mg_official" in report.findings[0].evidence["floor_share"]
    assert report.findings[0].evidence["floor_share"]["mg_official"] == 1.0


def test_cohorts_sharing_no_task_cannot_be_ranked():
    apart = [cohort("a", {"t1": (10, 20)}), cohort("b", {"t2": (10, 20)})]
    report = MatrixRanking().analyse(apart)
    assert not report.findings
    assert "nothing these can be ranked over" in report.notes[0]


def test_tasks_only_some_cohorts_attempted_are_dropped_and_named():
    uneven = [
        cohort("a", {"shared": (10, 20), "only_a": (5, 20)}),
        cohort("b", {"shared": (12, 20)}),
    ]
    report = MatrixRanking(resamples=200).analyse(uneven)
    assert any("only_a" in note for note in report.notes)


# -- the letter display ------------------------------------------------------


def test_letters_are_shared_exactly_when_a_pair_is_indistinguishable():
    letters = compact_letters({"a": 0.6, "b": 0.59, "c": 0.1}, [("a", "b")])
    assert set(letters["a"]) & set(letters["b"])
    assert not set(letters["c"]) & set(letters["a"])


def test_a_fully_separated_set_gets_one_letter_each():
    letters = compact_letters({"a": 0.9, "b": 0.5, "c": 0.1}, [])
    assert len({letters["a"], letters["b"], letters["c"]}) == 3


def test_the_module_holds_everything_except_the_policy():
    assert set(MatrixRanking().descriptor().provides["holds"]) == {
        "task",
        "evaluation",
        "embodiment",
    }
