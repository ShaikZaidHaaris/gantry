"""The single arm's three retargeters, watched refusing a number that is not a pose.

The travel check in these transforms is written as ``(array < low) | (array >
high)``, and that expression is blind to exactly one value. NaN compares False
against both bounds, so the test catches ``+/-inf`` and lets NaN through
untouched — ``LeaderToFollower().apply()`` would hand a NaN back as a follower
command and declare it in range.

There is an arm layer downstream that refuses non-finite values before anything
reaches the bus, so this was never a path to a moving servo. It is still the
wrong place to catch it, for two reasons:

* the refusal there names the arm, not the transform. "The follower has a
  non-finite value on ['gripper.pos']" sends an operator to the rig; the NaN was
  produced by a conversion that ran on a laptop, and the message that says so is
  the one that gets it fixed.
* two of these three conversions have no arm behind them at all. Normalized to
  joint angles exists to compare a real trajectory against a simulated one and to
  feed forward kinematics — nothing in that path is ever commanded, so a NaN that
  enters it is refused nowhere and lands in whatever the comparison produced.

``bimanual._in_travel`` already checks this, and the wording is asserted to match
below: two modules describing the same rig must not disagree about whether a
number is a position.

Everything here runs on a laptop with no arms and no serial library.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_so101.bimanual import BimanualToSingle, so101_bimanual
from gantry_evaluator_so101.embodiment import (
    ACTION,
    DOF,
    JOINT_NAMES,
    STATE,
    JointAnglesToNormalized,
    LeaderToFollower,
    NormalizedToJointAngles,
    so101_embodiment,
    so101_leader,
)

from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

#: A calibrated travel, in degrees, for the two conversions that need one. These
#: are plausible numbers and nobody's arm: the whole argument of ``_Calibrated``
#: is that this table is read off the machine in front of you and can never be a
#: default, so a test that reached for a built-in one would be exercising the
#: defect the class exists to prevent.
TRAVEL_DEG = {
    "shoulder_pan.pos": (-110.0, 110.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-90.0, 90.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-160.0, 160.0),
    "gripper.pos": (0.0, 40.0),
}

#: Mid-travel in ``RANGE_0_100``: a pose inside every joint's range, so a
#: refusal below is about the one value that was spoiled and not about the rest
#: of the array.
MIDDLE = 50.0


def _leader() -> ChannelSpec:
    return so101_leader().state[0]


def _follower(name: str = ACTION) -> ChannelSpec:
    machine = so101_embodiment()
    return machine.action[0] if name == ACTION else machine.state[0]


def _angles() -> ChannelSpec:
    """The degree channel the calibrated conversions read from and write to."""
    return ChannelSpec(
        name="observation.joint_angles",
        kind="vector",
        shape=(DOF,),
        dtype="float32",
        units="deg",
        frame="so101_follower_joints",
        dim_labels=JOINT_NAMES,
    )


def _mid(steps: int = 3, value: float = MIDDLE) -> np.ndarray:
    return np.full((steps, DOF), value, dtype=float)


# -- the pairs are legal, so a refusal below is about the values -------------


def test_the_three_pairs_used_here_are_ones_the_retargeters_accept():
    """Otherwise every refusal below could be a refusal of the channels.

    ``apply`` does not call ``accepts``, so a test could feed these transforms a
    pair the resolver would never have planned and still watch the value check
    fire. Asserting the pairs bind first is what makes the rest of this file a
    statement about NaN.
    """
    assert LeaderToFollower().accepts(_leader(), _follower(ACTION)).ok
    assert NormalizedToJointAngles(TRAVEL_DEG).accepts(_follower(STATE), _angles()).ok
    assert JointAnglesToNormalized(TRAVEL_DEG).accepts(_angles(), _follower(ACTION)).ok


# -- the value that the range test cannot see -------------------------------


def test_a_nan_leader_reading_is_not_a_follower_command():
    """The defect, at the transform that moves an arm.

    Numerically ``LeaderToFollower`` is an identity, which is why this matters:
    whatever the leader read is what the follower is commanded, and a NaN that
    survives the travel check survives the whole transform. The refusal has to
    name the joint and the transform, because ``nan`` on its own tells an
    operator nothing about where it came from.
    """
    spoiled = _mid()
    spoiled[1, DOF - 1] = np.nan
    with pytest.raises(ConfigError, match="not a finite position") as raised:
        LeaderToFollower().apply(spoiled, _leader(), _follower(ACTION))
    message = str(raised.value)
    assert "gripper.pos=nan" in message, message
    assert "so101-leader-to-follower" in message, message
    assert "step 1" in message, message


def test_a_nan_is_refused_on_the_way_to_joint_angles():
    """The direction with no arm behind it, and therefore no second chance.

    ``SO101Arm`` refuses non-finite values before the bus, but this conversion
    feeds a kinematics or sim-vs-real comparison rather than a servo. A NaN that
    passes here is refused nowhere: it becomes ``nan`` degrees, and every spatial
    number computed from it is silently absent rather than wrong.
    """
    spoiled = _mid()
    spoiled[0, 2] = np.nan
    with pytest.raises(ConfigError, match="not a finite position") as raised:
        NormalizedToJointAngles(TRAVEL_DEG).apply(spoiled, _follower(STATE), _angles())
    assert "elbow_flex.pos=nan" in str(raised.value)
    assert "so101-normalized-to-joint-angles" in str(raised.value)


def test_a_nan_angle_is_refused_on_the_way_back_to_a_command():
    """The same helper, reached through the other conversion.

    This one checks its *output* against the target's declared range, because
    the source is in degrees and the target is the range the arm is actually
    driven over. A NaN in degrees is still a NaN after the affine map, so the
    check catches it there — the value quoted is the normalized one and the
    channel named is the target, which is the number and the channel that would
    have been commanded.
    """
    spoiled = np.zeros((2, DOF), dtype=float)
    spoiled[0, 0] = np.nan
    with pytest.raises(ConfigError, match="not a finite position") as raised:
        JointAnglesToNormalized(TRAVEL_DEG).apply(spoiled, _angles(), _follower(ACTION))
    assert "shoulder_pan.pos=nan" in str(raised.value)
    assert "so101-joint-angles-to-normalized" in str(raised.value)


def test_infinity_was_already_refused_and_stays_refused():
    """The half of the check that worked, kept working.

    ``inf`` fails the range test on its own, so it was refused before this fix
    and must not become a pass now that a separate branch handles it. It now
    arrives at the non-finite message rather than the out-of-travel one, which
    is the more accurate of the two: infinity is not a position past the end of
    travel, it is not a position.
    """
    for bad in (np.inf, -np.inf):
        spoiled = _mid()
        spoiled[2, 0] = bad
        with pytest.raises(ConfigError, match="not a finite position"):
            LeaderToFollower().apply(spoiled, _leader(), _follower(ACTION))


def test_the_travel_refusal_is_still_live_and_still_says_it_is_not_clamping():
    """The finite check must not have swallowed the check it was added in front of.

    An out-of-travel number is finite, so it has to fall through to the range
    test and come back with the clamping sentence — that sentence is the one
    that tells a reader the value was not quietly pulled to the nearest limit.
    """
    spoiled = _mid()
    spoiled[0, 3] = 140.0
    with pytest.raises(ConfigError, match="outside the declared travel") as raised:
        LeaderToFollower().apply(spoiled, _leader(), _follower(ACTION))
    assert "refusing rather than clamping" in str(raised.value)
    assert "wrist_flex.pos=140" in str(raised.value)


def test_a_clean_trajectory_still_goes_through_all_three():
    """The baseline. Without it, a helper that refused everything would pass.

    The identity is also asserted to be a copy: a follower command aliasing the
    leader's buffer turns any later in-place edit into a retroactive change to
    what was commanded.
    """
    clean = _mid()
    commanded = LeaderToFollower().apply(clean, _leader(), _follower(ACTION))
    assert np.allclose(commanded, MIDDLE)
    assert not np.shares_memory(commanded, clean)

    degrees = NormalizedToJointAngles(TRAVEL_DEG).apply(clean, _follower(STATE), _angles())
    assert np.all(np.isfinite(degrees))
    # Mid-range in, mid-travel out, joint by joint.
    for column, label in enumerate(JOINT_NAMES):
        low, high = TRAVEL_DEG[label]
        assert degrees[:, column] == pytest.approx((low + high) / 2.0, abs=1e-4)

    back = JointAnglesToNormalized(TRAVEL_DEG).apply(degrees, _angles(), _follower(ACTION))
    assert back == pytest.approx(clean, abs=1e-3)


# -- the two modules describing this rig must not disagree ------------------


def test_the_single_arm_and_the_bimanual_module_refuse_a_nan_the_same_way():
    """Same rig, same value, same sentence.

    ``bimanual._in_travel`` is a deliberate near-copy of ``embodiment._in_range``
    — kept separate only so it can say ``left_gripper.pos`` where this one says
    ``gripper.pos``. Two copies drift, and a drift here means one module treats a
    NaN as a position and the other does not. Asserting the shared wording is the
    cheapest thing that notices.
    """
    single = _mid()
    single[0, 0] = np.nan
    with pytest.raises(ConfigError) as from_single:
        LeaderToFollower().apply(single, _leader(), _follower(ACTION))

    machine = so101_bimanual()
    both = np.full((3, machine.action[0].width), MIDDLE, dtype=float)
    both[0, 0] = np.nan
    with pytest.raises(ConfigError) as from_bimanual:
        BimanualToSingle("left").apply(both, machine.action[0], _follower(ACTION))

    shared = "which is not a finite position; refusing rather than passing it on"
    assert shared in str(from_single.value)
    assert shared in str(from_bimanual.value)
    # And each still names its own side of the bench.
    assert "gripper.pos=nan" not in str(from_single.value), "column 0 is the shoulder"
    assert "shoulder_pan.pos=nan" in str(from_single.value)
    assert "left_shoulder_pan.pos=nan" in str(from_bimanual.value)
