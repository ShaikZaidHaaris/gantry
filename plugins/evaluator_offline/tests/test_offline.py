"""The offline evaluator: what it measures, and what it refuses to pretend."""

from __future__ import annotations

import pytest
from gantry_evaluator_offline import OfflineEvaluator
from gantry_policy_basic import ConstantPolicy, NoisyReplayPolicy, ReplayPolicy

from gantry.conformance import check_evaluator, evaluator_checks
from gantry.contracts.evaluator import Protocol, Scene, TaskSpec
from gantry.errors import ComponentError
from gantry.fixtures import make_clean, make_defective
from gantry.resolve import Registry, requires_channels, resolve
from gantry.spine import IncompatibleError

SUITE = make_clean(n=8, seed=5)
ACTION = SUITE.episodes[0].channel("action")


def evaluator(suite=SUITE):
    return OfflineEvaluator(suite.episodes, "action")


# -- the contract ----------------------------------------------------------


def test_conforms():
    ev = evaluator()
    verdict = check_evaluator(ev, ReplayPolicy(ACTION), ev.task_for(), strict=True)
    assert verdict.ok, verdict.explain()


def test_conforms_across_epochs():
    ev = evaluator()
    verdict = check_evaluator(ev, ReplayPolicy(ACTION), ev.task_for(), Protocol(epochs=3))
    assert verdict.ok, verdict.explain()


def test_the_kit_is_discoverable():
    assert "capability_stage_events" in dict(evaluator_checks())


# -- the scale is real -----------------------------------------------------


def test_replay_is_exactly_the_ceiling():
    """A harness that does not put perfect replay at zero is broken."""
    ev = evaluator()
    assert ev.evaluate(ReplayPolicy(ACTION), ev.task_for()).metrics["action_mse"].value == 0.0


def test_error_recovers_the_noise_variance():
    """MSE of replay-plus-noise should land on sigma squared."""
    ev = evaluator()
    for sigma in (0.05, 0.2):
        mse = (
            ev.evaluate(NoisyReplayPolicy(ACTION, sigma=sigma), ev.task_for())
            .metrics["action_mse"]
            .value
        )
        assert mse == pytest.approx(sigma**2, rel=0.15)


def test_more_noise_scores_worse_monotonically():
    ev = evaluator()
    scores = [
        ev.evaluate(NoisyReplayPolicy(ACTION, sigma=s), ev.task_for()).metrics["action_mse"].value
        for s in (0.0, 0.05, 0.1, 0.2, 0.4)
    ]
    assert scores == sorted(scores)


def test_a_constant_policy_is_the_floor():
    ev = evaluator()
    floor = ev.evaluate(ConstantPolicy(ACTION), ev.task_for()).metrics["action_mse"].value
    ceiling = ev.evaluate(ReplayPolicy(ACTION), ev.task_for()).metrics["action_mse"].value
    assert floor > ceiling


def test_every_metric_carries_an_interval():
    run = evaluator().evaluate(NoisyReplayPolicy(ACTION, sigma=0.1), evaluator().task_for())
    for measurement in run.metrics.values():
        assert measurement.n == 8 and measurement.ci is not None


# -- what it refuses to claim ----------------------------------------------


def test_it_declares_no_outcomes_and_no_milestones():
    """There is no world here, so there is nothing to succeed in."""
    provides = evaluator().descriptor().provides
    assert provides["outcomes"] is False
    assert provides["stage_events"] is False
    assert provides["closed_loop"] is False


def test_the_record_carries_no_success_labels():
    run = evaluator().evaluate(ReplayPolicy(ACTION), evaluator().task_for())
    assert all(episode.labels.success is None for episode in run.episodes)


def test_the_provenance_names_the_limitation():
    run = evaluator().evaluate(ReplayPolicy(ACTION), evaluator().task_for())
    assert any("compounding error is not measured" in note for note in run.provenance.notes)


def test_a_funnel_is_refused_against_this_evaluator():
    registry = Registry()
    registry.register("evaluation", "offline", lambda **_: evaluator())
    resolution = resolve(
        registry,
        components={"evaluation": {"name": "offline"}},
        consumers=[
            requires_channels("funnel", "feedback", capabilities={"stage_events": True}),
            requires_channels("error_rate", "feedback", capabilities={"closed_loop": False}),
        ],
        provider_plane="evaluation",
        provided_channels=[],
    )
    assert not resolution.ok
    assert "funnel needs stage_events" in resolution.explain()
    assert resolution.alternatives == ("error_rate",)


# -- protocol is a lever, not a setting ------------------------------------


def test_executing_fewer_actions_per_chunk_changes_the_answer():
    """The lever that moved a real benchmark by fourteen points."""
    ev = evaluator()
    policy = NoisyReplayPolicy(ACTION, sigma=0.1, chunk=8)
    long_chunk = ev.evaluate(policy, ev.task_for(), Protocol(execute=8))
    short_chunk = ev.evaluate(policy, ev.task_for(), Protocol(execute=1))
    assert long_chunk.metrics["action_mse"].value != short_chunk.metrics["action_mse"].value
    assert long_chunk.digest != short_chunk.digest


def test_the_protocol_is_recorded():
    run = evaluator().evaluate(ReplayPolicy(ACTION), evaluator().task_for(), Protocol(execute=4))
    assert run.provenance.protocol["execute"] == 4


def test_epochs_produce_one_episode_per_attempt():
    ev = evaluator()
    assert len(ev.evaluate(ReplayPolicy(ACTION), ev.task_for(), Protocol(epochs=3))) == 24


def test_max_trials_caps_the_work():
    ev = evaluator()
    assert len(ev.evaluate(ReplayPolicy(ACTION), ev.task_for(), Protocol(max_trials=3))) == 3


# -- refusals --------------------------------------------------------------


def test_a_task_with_no_scenes_is_refused():
    with pytest.raises(IncompatibleError, match="no scenes"):
        evaluator().evaluate(ReplayPolicy(ACTION), TaskSpec("empty", (), horizon=10))


def test_zero_epochs_is_refused():
    ev = evaluator()
    with pytest.raises(IncompatibleError, match="runs nothing"):
        ev.evaluate(ReplayPolicy(ACTION), ev.task_for(), Protocol(epochs=0))


def test_a_scene_that_is_not_held_out_is_refused_by_name():
    ev = evaluator()
    task = TaskSpec("wrong", (Scene(id="not-a-real-episode"),), horizon=10)
    with pytest.raises(ComponentError, match="not among the held-out episodes"):
        ev.evaluate(ReplayPolicy(ACTION), task)


def test_an_evaluator_with_nothing_held_out_is_unbound_not_broken():
    """Constructing without data is legitimate; running without it is not.

    A manifest builds every plane independently and cannot hand this one a
    dataset at that point, so an unbound evaluator has to be a describable
    object. It refuses at the point it would produce a number.
    """
    from gantry.contracts.evaluator import Protocol
    from gantry.spine import IncompatibleError

    bare = OfflineEvaluator(action_name="action")
    assert bare.bound is False
    assert bare.descriptor().provides["needs_dataset"] is True
    assert bare.requires().channels[0].shape == (None,), "width is unknown until bound"

    with pytest.raises(IncompatibleError, match="not been bound"):
        bare.evaluate(ReplayPolicy(ACTION), bare.task_for(), Protocol())

    with pytest.raises(ValueError, match="at least one held-out episode"):
        bare.bind([])


def test_truncated_episodes_still_score():
    suite = make_defective("never_completes", n=8, fraction=0.5, seed=2)
    ev = OfflineEvaluator(suite.episodes, "action")
    assert ev.evaluate(ReplayPolicy(ACTION), ev.task_for()).metrics["action_mse"].value == 0.0
