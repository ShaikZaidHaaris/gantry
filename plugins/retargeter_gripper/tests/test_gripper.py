"""Reading one gripper in another's terms.

The calibrations here are the ones measured in robosuite 1.4.1, not invented
for the test, so a change in behaviour shows up against real hardware numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_retargeter_gripper import (
    GripperAperture,
    GripperCalibration,
    calibration_from,
    gripper_block,
    state_spec_for,
)

from gantry.conformance import check_retargeter
from gantry.spine import ChannelSpec

PANDA = GripperCalibration(
    "PandaGripper", (0.000499, -0.0005), (0.039884, -0.039851), "robosuite-1.4.1"
)
RETHINK = GripperCalibration(
    "RethinkGripper", (-0.011837, 0.011837), (0.011398, -0.011393), "robosuite-1.4.1"
)
ROBOTIQ85 = GripperCalibration(
    "Robotiq85Gripper",
    (0.501677, -0.253756, 0.268663, 0.502147, -0.2521, 0.268371),
    (-0.027371, -0.274956, -0.20152, -0.02737, -0.274954, -0.201521),
    "robosuite-1.4.1",
)

POSE = ("x", "y", "z", "roll", "pitch", "yaw")


def state(calibration: GripperCalibration) -> ChannelSpec:
    return ChannelSpec(
        "observation.state",
        "vector",
        (6 + calibration.joints,),
        "float32",
        semantics="pose",
        dim_labels=POSE + tuple(f"{calibration.name}.{i}" for i in range(calibration.joints)),
    )


def steps(calibration: GripperCalibration, *fractions: float) -> np.ndarray:
    hand = calibration.reading(np.array(fractions))
    pose = np.tile(np.arange(6, dtype=float), (len(fractions), 1))
    return np.concatenate([pose, hand], axis=1)


# -- the quantity that actually crosses ----------------------------------------


def test_each_gripper_reads_its_own_stops_as_shut_and_wide():
    for calibration in (PANDA, RETHINK, ROBOTIQ85):
        assert calibration.fraction(np.array([calibration.closed])) == pytest.approx(0.0)
        assert calibration.fraction(np.array([calibration.open])) == pytest.approx(1.0)


def test_a_reading_beyond_the_measured_travel_is_clamped_not_extrapolated():
    # A calibration narrower than reality would otherwise drive the target
    # gripper past a stop it physically has.
    beyond = np.array(PANDA.open) * 3
    assert PANDA.fraction(np.array([beyond])) == pytest.approx(1.0)
    assert PANDA.fraction(np.array([-beyond])) == pytest.approx(0.0)


def test_six_coupled_joints_reduce_to_one_number_and_come_back_on_the_line():
    half = ROBOTIQ85.reading(np.array([0.5]))
    assert ROBOTIQ85.fraction(half) == pytest.approx(0.5)
    # Rebuilt on the line between the stops, so it is a pose the hand can hold.
    assert half.shape == (1, 6)


# -- the case a width check cannot catch ---------------------------------------


def test_two_grippers_of_equal_width_are_still_not_interchangeable():
    # The whole reason this package exists. Both hands report two numbers, so
    # no width check objects to swapping them — and yet, measured in robosuite,
    # a Rethink hand read on Panda's scale spans only 0 to 0.28. Its entire
    # travel, shut to wide open, looks like a hand that never opens past a
    # quarter. A policy trained on Panda would never see it let go.
    assert len(PANDA.closed) == len(RETHINK.closed)
    wide_open_on_pandas_scale = PANDA.fraction(np.array([RETHINK.open]))[0]
    assert wide_open_on_pandas_scale == pytest.approx(0.277, abs=0.01)
    # Through the retargeter, each hand is read against its own travel, so a
    # wide-open Rethink is wide open.
    assert RETHINK.fraction(np.array([RETHINK.open]))[0] == pytest.approx(1.0)
    assert RETHINK.fraction(np.array([RETHINK.closed]))[0] == pytest.approx(0.0)


def test_converting_a_shut_hand_gives_the_targets_own_shut_reading():
    out = GripperAperture(RETHINK, PANDA).apply(
        steps(RETHINK, 0.0), state(RETHINK), state(PANDA)
    )
    assert out[0, 6:] == pytest.approx(np.array(PANDA.closed), abs=1e-6)


def test_twelve_wide_becomes_eight_wide_with_the_pose_untouched():
    source, target = state(ROBOTIQ85), state(PANDA)
    values = steps(ROBOTIQ85, 0.0, 0.5, 1.0)
    out = GripperAperture(ROBOTIQ85, PANDA).apply(values, source, target)
    assert values.shape == (3, 12) and out.shape == (3, 8)
    assert out[:, :6] == pytest.approx(values[:, :6])  # pose crosses unchanged
    assert out[:, 6:] == pytest.approx(PANDA.reading(np.array([0.0, 0.5, 1.0])), abs=1e-6)


def test_the_same_gripper_both_ways_is_a_round_trip():
    source, target = state(ROBOTIQ85), state(ROBOTIQ85)
    values = steps(ROBOTIQ85, 0.2, 0.7)
    out = GripperAperture(ROBOTIQ85, ROBOTIQ85).apply(values, source, target)
    assert out == pytest.approx(values, abs=1e-6)


# -- what it refuses -----------------------------------------------------------


def test_it_refuses_a_channel_that_does_not_say_which_gripper_it_is():
    bare = ChannelSpec("observation.state", "vector", (8,), "float32")
    assert gripper_block(bare) is None
    v = GripperAperture(PANDA, PANDA).accepts(bare, state(PANDA))
    assert not v.ok and v.reasons[0].code == "gripper.unlabelled"


def test_it_refuses_a_gripper_it_was_not_built_for():
    v = GripperAperture(PANDA, PANDA).accepts(state(ROBOTIQ85), state(PANDA))
    assert not v.ok
    assert "gripper.wrong_calibration" in [r.code for r in v.reasons]


def test_a_calibration_that_never_moved_is_refused():
    stuck = GripperCalibration("Stuck", (0.0, 0.0), (0.0, 0.0), "test")
    v = stuck.validate()
    assert not v.ok and v.reasons[0].code == "gripper.no_travel"
    with pytest.raises(Exception):
        GripperAperture(stuck, PANDA)


def test_an_unattributed_calibration_is_noted_not_refused():
    typed = GripperCalibration("Typed", (0.0, 0.0), (1.0, -1.0))
    v = typed.validate()
    assert v.ok
    assert [r.code for r in v.reasons] == ["gripper.unattributed_calibration"]


def test_an_embodiment_with_no_calibration_refuses_rather_than_defaulting():
    class Bare:
        name = "unknown_arm"
        metadata = {}

    with pytest.raises(ValueError, match="no gripper calibration"):
        calibration_from(Bare())


# -- honesty about the cost ----------------------------------------------------


def test_it_declares_what_does_not_cross():
    losses = GripperAperture(ROBOTIQ85, PANDA).losses(state(ROBOTIQ85), state(PANDA))
    assert losses, "a 12 -> 8 map that declares no loss would be refused by check()"
    joined = " ".join(losses).lower()
    assert "force" in joined and "contact" in joined


def test_a_width_change_with_declared_loss_passes_the_contract_check():
    assert GripperAperture(ROBOTIQ85, PANDA).check(state(ROBOTIQ85), state(PANDA)).ok


def test_the_target_spec_relabels_the_hand_and_keeps_the_pose():
    spec = state_spec_for(state(ROBOTIQ85), PANDA)
    assert spec.shape == (8,)
    assert spec.dim_labels[:6] == POSE
    assert spec.dim_labels[6:] == ("PandaGripper.0", "PandaGripper.1")


def test_it_passes_the_conformance_kit():
    verdict = check_retargeter(
        GripperAperture(ROBOTIQ85, PANDA), state(ROBOTIQ85), state(PANDA)
    )
    assert verdict.ok, verdict.explain()
