"""Dropping the demonstrations that failed, and not dropping the ones nobody scored."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from gantry_curate_labels import FailedDemonstrations

from gantry.conformance import check_curator
from gantry.errors import GantryError


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)


@dataclass
class Meta:
    uid: str


@dataclass
class Ep:
    meta: Meta
    labels: Labels


def dataset(succeed: int, fail: int, unlabelled: int = 0, source="mg"):
    out = [Ep(Meta(f"{source}/{i}"), Labels(True)) for i in range(succeed)]
    out += [Ep(Meta(f"{source}/{succeed + i}"), Labels(False)) for i in range(fail)]
    out += [Ep(Meta(f"{source}/{succeed + fail + i}"), Labels(None)) for i in range(unlabelled)]
    return out


def test_it_drops_exactly_the_failures():
    plan = FailedDemonstrations().plan(dataset(succeed=16, fail=84))
    assert len(plan.drops) == 84
    assert all(uid.startswith("mg/") for uid in plan.drops)
    assert plan.rung == "screening"


def test_it_reports_the_rate_that_justified_the_drop():
    """The number that makes the case, in the plan rather than in a log."""
    plan = FailedDemonstrations().plan(dataset(succeed=16, fail=84))
    assert plan.metadata["note"] == "16/100 labelled demonstrations succeed"
    detail = plan.actions[0].detail
    assert detail["failed"] == 84
    assert detail["success_rate_before"] == pytest.approx(0.16)


def test_unlabelled_is_not_failed():
    # The distinction between "16% good" and "16% *labelled* good". Treating
    # the second as the first silently deletes data nobody judged.
    plan = FailedDemonstrations().plan(dataset(succeed=10, fail=10, unlabelled=50))
    assert len(plan.drops) == 10
    assert plan.metadata["unlabelled"] == 50


def test_dropping_the_unlabelled_is_possible_but_has_to_be_asked_for():
    plan = FailedDemonstrations(drop_unlabelled=True).plan(
        dataset(succeed=10, fail=10, unlabelled=50)
    )
    assert len(plan.drops) == 60
    assert plan.actions[0].detail["unlabelled_dropped"] == 50


def test_a_clean_dataset_produces_a_plan_that_does_nothing_rather_than_no_plan():
    """Having no objection is an answer, and has to be recordable."""
    plan = FailedDemonstrations().plan(dataset(succeed=50, fail=0))
    assert plan.drops == ()
    assert len(plan.keeps) == 50
    assert plan.validate().ok
    assert plan.predicted.magnitude == 0.0


def test_the_prediction_scales_with_how_much_failure_there_is():
    dirty = FailedDemonstrations().plan(dataset(succeed=16, fail=84))
    clean = FailedDemonstrations().plan(dataset(succeed=95, fail=5))
    assert dirty.predicted.magnitude > clean.predicted.magnitude


def test_it_refuses_an_empty_dataset():
    with pytest.raises(GantryError, match="no episodes"):
        FailedDemonstrations().plan([])


def test_it_needs_no_rollouts():
    """The cheapest rung works before anything has been trained."""
    assert FailedDemonstrations().needs_runs is False
    assert FailedDemonstrations().plan(dataset(succeed=1, fail=1), runs=[]).drops


def test_it_passes_the_conformance_kit():
    episodes = dataset(succeed=16, fail=84)
    verdict = check_curator(FailedDemonstrations(), episodes)
    assert verdict.ok, verdict.explain()
