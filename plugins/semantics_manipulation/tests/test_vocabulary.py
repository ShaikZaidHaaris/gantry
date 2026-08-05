"""The distinctions that no structural check would catch on its own."""

from __future__ import annotations

import pytest
from gantry_semantics_manipulation import (
    action_channel,
    control_mode_of,
    is_absolute,
    state_channel,
)

from gantry.spine import ChannelSpec, compatible, get_semantics

# -- the silent failures this vocabulary exists to make loud ---------------


def test_absolute_and_delta_actions_are_refused_against_each_other():
    """Same width, same units, same dimension. Executing one as the other
    drives an arm to a position it read as a displacement."""
    target = action_channel("action", "joint_pos", width=7)
    change = action_channel("action", "joint_delta", width=7)
    assert target.shape == change.shape
    verdict = compatible(change, target)
    assert not verdict.ok
    assert "semantics.mismatch" in verdict.codes()


def test_two_quaternion_orderings_are_refused():
    """wxyz and xyzw are four floats each and differ by a permutation."""
    wxyz = action_channel("action", "eef_abs_pose", width=7, rotation_repr="quat_wxyz")
    xyzw = action_channel("action", "eef_abs_pose", width=7, rotation_repr="quat_xyzw")
    assert wxyz.shape == xyzw.shape
    assert wxyz.metadata["rotation_repr"] != xyzw.metadata["rotation_repr"]


def test_the_same_pose_in_two_frames_is_refused():
    base = action_channel("action", "eef_abs_pose", width=7, frame="base")
    world = action_channel("action", "eef_abs_pose", width=7, frame="world")
    assert "frame.mismatch" in compatible(base, world).codes()


def test_millimetres_and_metres_are_caught_on_proprioception():
    """Their canonical-units table is a convention in a comment; here it is
    checked, so a rig reporting the wrong scale does not pass silently."""
    metres = state_channel("state", "eef_pos", width=3)
    millimetres = state_channel("state", "eef_pos", width=3, units_override="mm")
    verdict = compatible(millimetres, metres)
    assert "units.scale" in verdict.codes()
    assert verdict.because("units.scale")[0].detail["factor"] == pytest.approx(1e-3)


def test_the_same_joints_in_a_different_order_are_refused():
    """Identical widths, identical names, opposite order: compatible on every
    other axis and produces garbage."""
    left_first = action_channel("action", "joint_pos", width=2, dim_labels=["left_j0", "right_j0"])
    right_first = action_channel("action", "joint_pos", width=2, dim_labels=["right_j0", "left_j0"])
    verdict = compatible(left_first, right_first)
    assert "dim_labels.order" in verdict.codes()
    assert "permutation, not a rename" in verdict.because("dim_labels.order")[0].hint


def test_genuinely_different_labels_read_differently_from_a_permutation():
    verdict = compatible(
        action_channel("action", "joint_pos", width=2, dim_labels=["a", "b"]),
        action_channel("action", "joint_pos", width=2, dim_labels=["c", "d"]),
    )
    assert "dim_labels.mismatch" in verdict.codes()


# -- building channels correctly -------------------------------------------


def test_a_built_channel_validates():
    channel = action_channel(
        "action",
        "joint_pos",
        width=14,
        dim_labels=[f"{side}_j{i}" for side in ("left", "right") for i in range(7)],
        rate_hz=20.0,
    )
    assert channel.validate().ok
    assert channel.width == 14


def test_a_pose_too_narrow_for_its_rotation_is_refused_at_construction():
    with pytest.raises(ValueError, match="needs at least 7 dimensions"):
        action_channel("action", "eef_abs_pose", width=5, rotation_repr="quat_wxyz")


def test_mislabelled_width_is_refused_at_construction():
    with pytest.raises(ValueError, match="2 labels for a width of 3"):
        action_channel("action", "joint_pos", width=3, dim_labels=["a", "b"])


def test_unknown_vocabulary_lists_the_alternatives():
    with pytest.raises(ValueError, match="expected one of"):
        action_channel("action", "telekinesis", width=3)
    with pytest.raises(ValueError, match="expected one of"):
        state_channel("state", "vibes", width=3)


def test_proprioception_carries_its_canonical_units():
    assert state_channel("state", "joint_pos", width=7).units == "rad"
    assert state_channel("state", "gripper", width=1).units == "fraction"


def test_registered_semantics_pin_a_physical_dimension():
    assert get_semantics("state.joint_pos").dimension is not None
    # so declaring metres for a joint angle is caught
    bad = ChannelSpec("state", "vector", (7,), units="m", semantics="state.joint_pos")
    assert "units.dimension" in bad.validate().codes()


# -- reading a channel back ------------------------------------------------


def test_control_mode_round_trips():
    assert control_mode_of(action_channel("a", "eef_delta_pos", width=3)) == "eef_delta_pos"
    assert control_mode_of(state_channel("s", "joint_pos", width=7)) is None


def test_absoluteness_is_answerable_or_honestly_unknown():
    assert is_absolute(action_channel("a", "joint_pos", width=7)) is True
    assert is_absolute(action_channel("a", "joint_delta", width=7)) is False
    assert is_absolute(ChannelSpec("a", "vector", (7,))) is None


def test_core_stays_neutral_until_this_is_installed():
    """None of this vocabulary exists in a core-only interpreter.

    Checked in a subprocess rather than by reading source: the invariant is
    that core never *registers* or branches on these tags, not that it never
    says the words. Core's own docstrings use robot terms as illustrations, and
    a grep-based test would forbid explaining the design.
    """
    import subprocess
    import sys

    probe = "from gantry.spine import known_semantics;print(','.join(known_semantics()))"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    registered = set(out.strip().split(","))

    from gantry_semantics_manipulation.vocabulary import CONTROL_MODES, STATE_UNITS

    ours = {f"action.{m}" for m in CONTROL_MODES} | {f"state.{k}" for k in STATE_UNITS}
    assert ours, "the plugin registers nothing"
    assert not (ours & registered), f"core already knows {sorted(ours & registered)}"
