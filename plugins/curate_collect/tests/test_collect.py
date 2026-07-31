"""Turning failed scenes into an order for what to film next."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from gantry_curate_collect import FromFailures, cluster_note, failure_scenes

from gantry.conformance import check_curator
from gantry.errors import GantryError


@dataclass
class Stage:
    name: str
    step: int = 0


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)
    stage_events: tuple = ()


@dataclass
class Meta:
    uid: str
    id: str = ""
    task: str | None = None


@dataclass
class Ep:
    meta: Meta
    labels: Labels


@dataclass
class Run:
    episodes: tuple


def run(outcomes, task="lift_cube", stages=None, starts=None):
    eps = []
    for i, ok in enumerate(outcomes):
        annotations = {"initial_state": 1000 + i}
        if starts is not None:
            annotations["object_start"] = starts[i]
        events = ()
        if stages is not None and stages[i]:
            events = (Stage(stages[i]),)
        eps.append(Ep(Meta(f"r/{i}", f"seed_{1000 + i}", task), Labels(ok, annotations, events)))
    return Run(tuple(eps))


def dataset(n=5):
    return [Ep(Meta(f"d/{i}"), Labels(True)) for i in range(n)]


# -- reading the failures ----------------------------------------------------


def test_it_finds_the_seeds_that_failed():
    seeds, tasks, _ = failure_scenes([run([True, False, False, True])])
    assert seeds == [1001, 1002]
    assert tasks == ["lift_cube", "lift_cube"]


def test_it_counts_where_the_attempts_died():
    r = run([False, False, False], stages=["grasp", "grasp", "reach"])
    _, _, stages = failure_scenes([r])
    assert stages["grasp"] == 2 and stages["reach"] == 1


# -- the order ---------------------------------------------------------------


def test_the_order_names_the_scenes_that_beat_the_policy():
    plan = FromFailures().plan(dataset(), [run([True, False, False, False])])
    (order,) = plan.orders
    assert order.task == "lift_cube"
    assert order.seeds == (1001, 1002, 1003)
    assert order.n == 3


def test_the_order_names_the_stage_when_a_funnel_knows_it():
    """'More lifting data' and 'more grasping data' commission different work."""
    r = run([False, False, False], stages=["grasp", "grasp", "reach"])
    (order,) = FromFailures().plan(dataset(), [r]).orders
    assert order.stage == "grasp"
    assert "grasp" in str(order)


def test_the_order_says_where_in_the_workspace_the_failures_were():
    starts = [[0.07, 0.0], [0.08, 0.0], [0.06, 0.0], [0.075, 0.0]]
    r = run([False] * 4, starts=starts)
    (order,) = FromFailures().plan(dataset(), [r]).orders
    assert "toward the" in order.note and "x" in order.note


def test_an_order_is_capped_because_it_is_somebody_s_week():
    r = run([False] * 300)
    (order,) = FromFailures(cap=50).plan(dataset(), [r]).orders
    assert order.n == 50
    assert plan_detail(FromFailures(cap=50), r)["capped"] is True


def plan_detail(curator, r):
    return curator.plan(dataset(), [r]).actions[0].detail


def test_nothing_failed_means_nothing_ordered():
    """Manufacturing work to look useful is the failure mode here."""
    plan = FromFailures().plan(dataset(), [run([True, True, True])])
    assert plan.orders == ()
    assert plan.predicted.magnitude == 0.0
    assert "nothing failed" in plan.metadata["note"]


# -- what it declares about itself -------------------------------------------


def test_it_refuses_without_rollouts():
    # This rung cannot be reached from a dataset alone, and says so rather
    # than inventing an order.
    assert FromFailures().needs_runs is True
    with pytest.raises(GantryError, match="rollouts"):
        FromFailures().plan(dataset(), [])


def test_the_scenes_it_read_are_recorded_so_verification_can_hold_them_out():
    """Otherwise the demos get filmed at the layouts they are then tested on."""
    plan = FromFailures().plan(dataset(), [run([False, False])])
    assert plan.evidence_seeds == (1000, 1001)


def test_it_passes_the_conformance_kit():
    verdict = check_curator(FromFailures(), dataset(), [run([True, False, False])])
    assert verdict.ok, verdict.explain()


def test_cluster_note_stays_quiet_when_there_is_nothing_to_say():
    assert cluster_note([run([False], starts=[[0.0, 0.0]])]) == ""
