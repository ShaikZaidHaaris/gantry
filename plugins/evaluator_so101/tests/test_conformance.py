"""The SO-101 rig held to the same kits as everything else, with no rig present.

Every test here runs on a laptop with no driver board, no USB device and no
serial library installed. That is not a convenience. An evaluator whose
correctness can only be checked while standing next to the hardware is an
evaluator whose correctness is checked once, by whoever was standing there — and
this one is the instrument the whole SO-101 campaign is measured with.

Two kits run, because this package occupies two planes: the arms are the machine
and the operator station is the evaluator. A descriptor nobody checks is where a
plugin can claim anything, and the claims are what the resolver plans from.

Everything is built through :func:`_rig`, so the seam with the other four
modules is named in one place.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from gantry_evaluator_so101 import (
    ACTION,
    GRIPPER,
    JOINTS,
    MOTOR_NAMES,
    QUANTISATION,
    RANGE,
    REQUESTED,
    STATE,
    PoseSource,
    Sample,
    ScriptedConsole,
    SerialTransport,
    SO101Evaluator,
    VirtualClock,
    action_spec,
    gripper_stall,
    mock_pair,
    open_serial,
    so101_embodiment,
    state_spec,
)
from gantry_policy_basic import ConstantPolicy

from gantry.conformance import check_embodiment, check_evaluator
from gantry.contracts.embodiment import CAP_SEEDABLE, CAP_SIMULATED
from gantry.contracts.evaluator import Protocol, TaskSpec
from gantry.errors import ConfigError
from gantry.fixtures import make_duration_confound
from gantry.spine import IncompatibleError

#: Mid-range in the 0..100 convention: a pose inside every joint's calibrated
#: travel, so a constant policy is a legal thing to run rather than a request to
#: drive six servos into their end stops.
MIDDLE = 50.0

#: Trial lengths here are short on purpose. A conformance pass runs the whole
#: task several times, and at 30 Hz a realistic 600-step horizon would make this
#: file something nobody runs before pushing.
SCENES, HORIZON = 4, 12


def _rig(*, outcomes=(True, False, True), **kwargs) -> SO101Evaluator:
    """One construction site for the whole file.

    ``VirtualClock`` is what makes a 30 Hz loop testable: the run jumps to each
    deadline instead of sleeping to it, and the record says ``paced=False`` so
    nothing downstream mistakes the timings in it for measurements.

    The *same* clock goes to the mock bus. The simulated servos integrate their
    motion against whatever clock they are given, so leaving them on wall time
    while the loop jumps between deadlines makes the arm move by however long the
    interpreter happened to take: two identical runs then disagree, and every
    number measured off the mock is host scheduling noise wearing a unit. One
    clock, both layers, and the mock is reproducible.

    ``ScriptedConsole(repeat=True)`` is the rehearsed operator. Repeating is
    right here and would be wrong at the rig: a conformance pass runs the same
    task an unknown number of times, and running out of answers mid-kit would
    fail the plugin for the test harness's arithmetic rather than its own.
    """
    kwargs.setdefault("scenes", SCENES)
    kwargs.setdefault("horizon", HORIZON)
    clock = VirtualClock()
    return SO101Evaluator(
        mock_pair(bus_clock=clock),
        console=ScriptedConsole(outcomes=outcomes, repeat=True),
        clock=clock,
        **kwargs,
    )


def _clock_watcher() -> PoseSource:
    """A stand-in for the perception rig that does not exist yet.

    It calls its milestones off the step counter, which is exactly what a real
    pose source must not do. It is here to prove the plumbing carries events and
    that the capability flips with it — not to stand in for perception.
    """
    names = ("reached", "grasped", "lifted")

    def milestones(sample: Sample) -> tuple[str, ...]:
        return tuple(name for name, at in zip(names, (1, 3, 5)) if sample.step >= at)

    return PoseSource(milestones=milestones, names=names, description="a clock, not a camera")


def _policy(value: float = MIDDLE, spec=None) -> ConstantPolicy:
    return ConstantPolicy(spec or action_spec(), value)


# -- the two kits -----------------------------------------------------------


def test_the_evaluator_conforms():
    rig = _rig()
    verdict = check_evaluator(rig, _policy(), rig.task_for("so101"), Protocol(), strict=True)
    assert verdict.ok, verdict.explain()


def test_the_embodiment_conforms():
    verdict = check_embodiment(so101_embodiment(), strict=True)
    assert verdict.ok, verdict.explain()


def test_the_evaluator_conforms_across_epochs():
    """Several attempts at one scene is the normal shape of a hardware run.

    Each attempt is its own episode. Collapsing epochs would discard the
    trial-to-trial variation that is the only reason to run a rig more than once.
    """
    rig = _rig()
    verdict = check_evaluator(rig, _policy(), rig.task_for("so101"), Protocol(epochs=3))
    assert verdict.ok, verdict.explain()


# -- the three declarations -------------------------------------------------


def test_it_does_not_claim_to_be_seedable():
    """The declaration that decides whether a paired analysis may be planned.

    Nothing here is reproducible from a number: the object is placed by a
    person, the servos are warm or cold, the table has been nudged. True would
    let the resolver plan CORPUS-PROTOCOL.md's paired comparison against
    hardware, where it is not valid, and the output would look like any other.
    """
    provides = _rig().descriptor().provides
    assert provides[CAP_SEEDABLE] is False
    assert provides["closed_loop"] is True
    assert provides["outcomes"] is True


def test_the_machine_claims_neither_seeding_nor_simulation():
    capabilities = so101_embodiment().capabilities
    assert CAP_SEEDABLE not in capabilities
    assert CAP_SIMULATED not in capabilities


def test_a_seed_names_a_start_pose_and_does_not_promise_to_reproduce_it():
    """Scenes carry seeds, which is not the same as being seedable.

    The seed indexes a start pose in the pre-registered list. A hand then places
    the object, so the episode says in words that the placement was neither
    verified nor reproducible — otherwise the mere presence of a seed reads, to
    the next person, as a reproducible trial.
    """
    rig = _rig(scenes=1, horizon=4)
    task = rig.task_for("so101")
    assert task.scenes[0].seed is not None

    episode = rig.evaluate(_policy(), task, Protocol()).episodes[0]
    placement = str(episode.labels.annotations["placement"]).lower()
    assert "by hand" in placement and "not reproducible" in placement


def test_without_a_pose_source_there_are_no_stage_events_and_none_are_claimed():
    """The empty-funnel failure, refused in both directions.

    Claiming stage events with no perception yields a funnel over nothing that
    prints as a finished analysis. Emitting them while declaring False is as bad
    the other way: the resolver would refuse a diagnosis this run could feed.
    The kit checks both, so this asserts the pair, and the record carries the
    reason in words for whoever reads it later.
    """
    rig = _rig()
    assert rig.descriptor().provides["stage_events"] is False
    record = rig.evaluate(_policy(), rig.task_for("so101"), Protocol())
    assert not record.has_stage_events
    assert record.stages == ()
    assert any("no pose source" in note for note in record.provenance.notes)


def test_a_pose_source_is_the_only_thing_that_turns_the_funnel_on():
    rig = _rig(pose_source=_clock_watcher())
    assert rig.descriptor().provides["stage_events"] is True
    verdict = check_evaluator(rig, _policy(), rig.task_for("so101"), Protocol(), strict=True)
    assert verdict.ok, verdict.explain()
    record = rig.evaluate(_policy(), rig.task_for("so101"), Protocol())
    assert set(record.stages) == {"reached", "grasped", "lifted"}


def test_the_gripper_stall_proxy_is_an_annotation_and_never_a_stage_event():
    """One rung is not a funnel.

    These servos report position and nothing else, so contact is inferable from
    a gripper that stopped following its command — and that is a GRASP candidate
    with no lift and no place behind it. Reporting it as a stage event would let
    the resolver plan a funnel this rig cannot fill, so the proxy is configured,
    read, and recorded beside the episode instead.
    """
    rig = _rig(contact=gripper_stall(closes_toward=RANGE[0]))
    assert rig.descriptor().provides["stage_events"] is False
    record = rig.evaluate(_policy(), rig.task_for("so101"), Protocol())
    assert not record.has_stage_events
    assert all("contact_step" in e.labels.annotations for e in record.episodes)


# -- safety -----------------------------------------------------------------


def test_the_clamp_is_mandatory_and_cannot_be_switched_off():
    """LeRobot's ``max_relative_target`` defaults to None, which is no clamp.

    Inheriting that default means one bad action chunk drives the follower into
    the table at full servo speed. Refusing None is the difference between a
    safety property and a suggestion.
    """
    with pytest.raises(ConfigError, match="clamp"):
        _rig(max_relative_target=None)
    with pytest.raises(ConfigError):
        _rig(max_relative_target=0.0)


def test_a_policy_asking_for_a_leap_gets_the_clamp_not_the_leap():
    """A wild command costs a slow trial, never a collision.

    The follower sits mid-travel and is asked for the top of the range on the
    first step. Three things have to be true afterwards: no single command moved
    it further than the clamp allows, the record kept what the policy actually
    asked for beside what the arm was told, and the provenance declares the
    clamp as a loss — because a run where the clamp bound on most steps measured
    the clamp, not the policy.

    The tolerance on the first of those is one encoder tick, and it is a physical
    fact rather than slack. The clamp is computed in the 0..100 convention and
    executed in the integer register the servo actually has, so the value that
    reaches the goal register is the clamped target rounded to the nearest tick —
    up to half a tick past the limit, which on this arm is a few hundredths of a
    unit. Asserting exact equality would be asserting that the servos have
    infinite resolution, and the only way to pass it would be to record the value
    the arm was asked for instead of the one it was given.
    """
    limit = 5.0
    rig = _rig(scenes=1, horizon=8, max_relative_target=limit)
    episode = rig.evaluate(_policy(RANGE[1]), rig.task_for("so101"), Protocol()).episodes[0]

    reach = np.abs(episode.array(ACTION) - episode.array(STATE))
    assert reach.max() <= limit + QUANTISATION, f"a command reached {reach.max()} past the arm"
    assert episode.array(REQUESTED).max() == pytest.approx(RANGE[1]), (
        "what the policy asked for is not recoverable from the record"
    )
    assert episode.labels.annotations["clamped_steps"] > 0

    record = rig.evaluate(_policy(RANGE[1]), rig.task_for("so101"), Protocol())
    assert record.provenance.lossy
    assert any("limited" in loss for loss in record.provenance.losses)


def test_an_action_of_the_wrong_width_costs_the_trial_and_is_never_padded():
    """Six joints, and the sixth is the gripper. There is nothing to pad with.

    A five-wide action is a policy trained on a different machine. Inventing the
    sixth value turns a wrist command into a gripper command, and the run looks
    like every other run. The trial is recorded as the failure it was, with
    nothing in it.
    """
    narrow = action_spec()
    narrow = type(narrow)(**{**narrow.__dict__, "shape": (5,), "dim_labels": JOINTS[:5]})
    rig = _rig(scenes=2, horizon=4)
    record = rig.evaluate(_policy(spec=narrow), rig.task_for("so101"), Protocol())

    assert len(record) == 2, "a failed trial stays in the denominator"
    for episode in record.episodes:
        assert episode.channel_names == ()
        assert episode.labels.success is None
        assert "padded" in episode.labels.annotations["error"]
    assert record.provenance.notes


def test_a_task_with_no_scenes_is_refused():
    rig = _rig()
    with pytest.raises(IncompatibleError):
        rig.evaluate(_policy(), TaskSpec("empty", scenes=(), horizon=10), Protocol())


# -- the calibration trap ---------------------------------------------------


def test_the_joint_order_is_lerobots_and_the_gripper_is_the_sixth():
    """Matching by width alone is matching by coincidence.

    Six float32s in the wrong order is a legal action vector on this arm and a
    different pose entirely. The labels are what make that a refusal rather than
    a motion.
    """
    assert JOINTS == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    assert JOINTS[GRIPPER] == "gripper.pos"
    assert action_spec().dim_labels == JOINTS
    assert state_spec().dim_labels == JOINTS


def test_the_calibration_is_declared_load_bearing_on_every_channel():
    """LeRobot #3758: a ~17 degree offset cost ~6 cm at the gripper and was
    invisible during training — the policy learns the biased mapping and the
    loss curve looks healthy. Nothing structural catches it, because a
    mis-calibrated channel has the right name, width, units and meaning.

    So the calibration variant rides on the channel as a discriminator. Two
    corpora recorded under different calibrations then refuse to bind instead of
    averaging into one silently wrong mapping.
    """
    for spec in (action_spec(), state_spec()):
        assert "calibration_variant" in spec.discriminators
        assert spec.metadata["calibration_variant"]
        assert spec.rate_hz == 30.0


def test_the_calibration_of_the_arms_that_ran_is_stamped_into_provenance():
    """Which is what makes a recalibration break comparability by itself.

    The hash rides on the embodiment component, so two runs across a
    recalibration are not comparable while holding the embodiment fixed and
    nobody has to remember that.
    """
    rig = _rig(scenes=1, horizon=4)
    record = rig.evaluate(_policy(), rig.task_for("so101"), Protocol())
    machine = record.provenance.component("embodiment")
    assert machine is not None and machine.artifact_digest
    assert record.provenance.protocol["calibration_variant"]


# -- the suite that has to stay silent --------------------------------------


def test_duration_alone_moves_none_of_the_numbers():
    """``make_duration_confound`` is the suite where nothing is wrong and
    everything varies in length. The plugin doc singles it out because getting
    silence is harder, and worth more, than getting the defective suites to
    speak.

    It applies here for a reason specific to hardware: trial length varies by an
    order of magnitude for reasons that have nothing to do with the policy — a
    slow approach, a long reset, an operator watching one more second before
    calling it. Any quality computed as a raw count over a trial scores a
    patient policy as a worse one.

    So the same rig and the same scripted operator run twice, at the short and
    the long lengths of the confound suite, and every metric must agree.
    Anything honestly *about* time is exempt by its declared units: reporting a
    duration is fine, reporting a quality that is secretly a duration is not.
    """
    lengths = sorted(len(episode) for episode in make_duration_confound(n=12, seed=3).episodes)
    short, long = lengths[0], lengths[-1]
    assert long > 2 * short, "the fixture is meant to spread durations"

    outcomes = (True, False, True, True, False, True)
    brief = _rig(outcomes=outcomes, scenes=6, horizon=short)
    drawn = _rig(outcomes=outcomes, scenes=6, horizon=long)
    quick = brief.evaluate(_policy(), brief.task_for("confound", scenes=6, horizon=short), Protocol())
    slow = drawn.evaluate(_policy(), drawn.task_for("confound", scenes=6, horizon=long), Protocol())

    assert set(quick.metrics) == set(slow.metrics)
    #: Units that make a metric a statement about time rather than about the
    #: policy. Nothing else gets an exemption.
    timed = {"s", "ms", "min", "step", "count"}
    for name, measured in quick.metrics.items():
        other = slow.metrics[name]
        assert measured.n and other.n, f"{name} reports no denominator"
        if measured.units in timed:
            continue
        assert measured.value == pytest.approx(other.value), (
            f"{name} moved from {measured.value} to {other.value} on episode length alone"
        )


# -- no hardware, no serial stack, no excuse --------------------------------


def test_the_serial_stack_is_optional_and_refuses_by_name_when_absent():
    """A plugin that cannot be imported cannot explain itself.

    So a missing extra is a refusal at the point the port would be opened,
    naming what to install, rather than an ImportError at package import that
    takes the mock path down with it. Everything else in this file is the proof
    that the mock path is the whole plugin.
    """
    if importlib.util.find_spec("serial") is not None:
        pytest.skip("pyserial is installed, so the refusal path cannot be reached")
    with pytest.raises(ConfigError, match="pyserial"):
        open_serial("COM3")


def test_a_real_transport_refuses_a_fiction_and_will_not_guess_a_port():
    """Two refusals that only matter when hardware is attached.

    An unmeasured calibration makes every normalised value a fiction — that is
    #3758 again — and a port this code went looking for on its own is somebody
    else's device being written to. Both are checked here because both are
    unreachable from the mock path, which is where the rest of this file lives.
    """
    from gantry_evaluator_so101.transport import Calibration, SafetyLimits

    invented = Calibration.fixture(MOTOR_NAMES)
    limits = SafetyLimits.normalized(5.0)
    with pytest.raises(ConfigError, match="measured"):
        SerialTransport(invented, max_relative_target=limits, port="COM3")
    with pytest.raises(ConfigError, match="port"):
        SerialTransport(
            invented, max_relative_target=limits, allow_unmeasured_calibration=True
        )


def test_the_whole_run_happens_against_the_mock_bus():
    """The claim this file exists to make, asserted rather than implied."""
    rig = _rig(scenes=2, horizon=6)
    record = rig.evaluate(_policy(), rig.task_for("so101"), Protocol())
    assert len(record) == 2
    assert record.has_outcomes
    assert record.provenance.component("evaluation") is not None
    assert record.provenance.protocol["horizon"] == 6
    assert record.provenance.protocol["reset"].startswith("human")
