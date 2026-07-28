"""A closed loop with no dependencies: it produces milestones, so it can be diagnosed."""

from __future__ import annotations

import numpy as np
import pytest
from gantry.conformance import check_evaluator, check_policy
from gantry.contracts.evaluator import Protocol
from gantry.contracts.feedback import Cohort
from gantry.contracts.policy import Observation
from gantry.spine import IncompatibleError

from gantry_evaluator_waypoint import GreedyPolicy, WaypointWorld

TIGHT_ENGAGE = [0.08, 0.03, 0.08, 0.08]


def world(**kw):
    return WaypointWorld(**kw)


# -- conformance ------------------------------------------------------------


def test_the_world_conforms():
    w = world()
    verdict = check_evaluator(w, GreedyPolicy(skill=1.0), w.task_for(scenes=6), strict=True)
    assert verdict.ok, verdict.explain()


def test_the_policy_conforms():
    observations = [
        Observation(i, {"position": np.zeros(3, "float32"), "target": np.ones(3, "float32")})
        for i in range(3)
    ]
    for policy in (GreedyPolicy(skill=1.0), GreedyPolicy(skill=0.7, chunk=8)):
        verdict = check_policy(policy, observations, strict=True)
        assert verdict.ok, verdict.explain()


def test_it_declares_what_it_can_report():
    provides = world().descriptor().provides
    assert provides["stage_events"] and provides["outcomes"]
    assert provides["closed_loop"] and provides["seedable"]


# -- the world behaves ------------------------------------------------------


def test_a_perfect_policy_clears_everything():
    w = world()
    run = w.evaluate(GreedyPolicy(skill=1.0), w.task_for(scenes=20))
    assert run.metrics["success_rate"].value == 1.0
    assert all(e.labels.success for e in run.episodes)


def test_milestones_are_recorded_in_order():
    w = world()
    episode = w.evaluate(GreedyPolicy(skill=1.0), w.task_for(scenes=4)).episodes[0]
    assert episode.labels.stages == ("approach", "engage", "transport", "release")
    steps = [episode.labels.step_of(s) for s in episode.labels.stages]
    assert steps == sorted(steps)


def test_a_real_trajectory_comes_back():
    w = world()
    episode = w.evaluate(GreedyPolicy(skill=1.0), w.task_for(scenes=2)).episodes[0]
    assert episode.channel_names == ("position", "target", "command")
    assert episode.validate(deep=True).ok
    assert len(episode) > 4


def test_lower_aim_accuracy_reaches_fewer_milestones():
    w = world(tolerance=TIGHT_ENGAGE)
    task = w.task_for(scenes=30)
    reached = [
        w.evaluate(GreedyPolicy(skill=s), task).metrics["stages_reached"].value
        for s in (1.0, 0.85, 0.6)
    ]
    assert reached == sorted(reached, reverse=True)


def test_the_same_seed_gives_the_same_run():
    w = world()
    task = w.task_for(scenes=8)
    a = w.evaluate(GreedyPolicy(skill=0.8), task, Protocol(seed_base=5))
    b = w.evaluate(GreedyPolicy(skill=0.8), task, Protocol(seed_base=5))
    assert [e.labels.success for e in a.episodes] == [e.labels.success for e in b.episodes]
    assert a.digest == b.digest


def test_a_different_seed_gives_a_different_run():
    w = world()
    task = w.task_for(scenes=8)
    a = w.evaluate(GreedyPolicy(skill=0.8), task, Protocol(seed_base=1))
    b = w.evaluate(GreedyPolicy(skill=0.8), task, Protocol(seed_base=2))
    assert not np.array_equal(a.episodes[0].array("position"), b.episodes[0].array("position"))


def test_the_stage_vocabulary_is_a_parameter():
    w = world(stages=("incise", "suture", "close"))
    episode = w.evaluate(GreedyPolicy(skill=1.0), w.task_for(scenes=2)).episodes[0]
    assert episode.labels.stages == ("incise", "suture", "close")


# -- the loop this was built to close --------------------------------------


def test_the_funnel_localises_the_stage_that_is_actually_tight():
    """The whole point: Gantry now produces the milestones it diagnoses.

    Only 'engage' has a tight tolerance, so an imprecise policy must fail there
    and nowhere else. A funnel that named a different stage would be wrong in a
    way no fixture could have shown.
    """
    from gantry_feedback_core import Funnel

    w = world(tolerance=TIGHT_ENGAGE)
    run = w.evaluate(GreedyPolicy(skill=0.75), w.task_for(scenes=60))
    report = Funnel().run([Cohort("waypoints", run.episodes)])
    finding = report.by_code("funnel.bottleneck")[0]
    assert "engage is the weak link" in finding.summary
    assert report.measurements["waypoints.approach"].value > 0.9
    assert report.measurements["waypoints.engage"].value < 0.2


def test_a_run_survives_a_round_trip_to_disk(tmp_path):
    from gantry.store import read_run, same_run, write_run

    w = world()
    run = w.evaluate(GreedyPolicy(skill=0.9), w.task_for(scenes=5))
    assert same_run(read_run(write_run(run, tmp_path / "run")), run)


# -- protocol ---------------------------------------------------------------


def test_executing_a_whole_chunk_open_loop_differs_from_replanning():
    w = world()
    task = w.task_for(scenes=20)
    policy = GreedyPolicy(skill=0.9, chunk=8)
    every_step = w.evaluate(policy, task, Protocol(execute=1))
    whole_chunk = w.evaluate(policy, task, Protocol(execute=8))
    assert every_step.digest != whole_chunk.digest


def test_epochs_give_one_episode_per_attempt():
    w = world()
    assert len(w.evaluate(GreedyPolicy(), w.task_for(scenes=5), Protocol(epochs=3))) == 15


# -- refusals ---------------------------------------------------------------


def test_an_impossible_skill_is_refused():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        GreedyPolicy(skill=1.5)


def test_mismatched_tolerances_are_refused():
    with pytest.raises(ValueError, match="tolerance"):
        WaypointWorld(stages=("a", "b"), tolerance=[0.1, 0.1, 0.1])


def test_a_task_with_no_scenes_is_refused():
    from gantry.contracts.evaluator import TaskSpec

    with pytest.raises(IncompatibleError, match="no scenes"):
        world().evaluate(GreedyPolicy(), TaskSpec("empty", (), horizon=10))


def test_a_policy_that_raises_fails_one_trial_not_the_run():
    class Broken(GreedyPolicy):
        def act(self, observation):
            if observation.step > 2:
                raise RuntimeError("inference died")
            return super().act(observation)

    w = world()
    run = w.evaluate(Broken(skill=1.0), w.task_for(scenes=5))
    assert len(run) == 5
    assert all(e.labels.success is None for e in run.episodes)
    assert any("trial(s) failed" in n for n in run.provenance.notes)
