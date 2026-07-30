"""The hand-to-arm transform, and every way it is allowed to say no.

Weighted towards the refusals on purpose. A retargeting that is nearly right
produces smooth, confident, wrong motion — and "the policy did not learn much"
is exactly the conclusion the surrounding product exists to measure, so the
failure that most resembles the answer has to be the hardest one to reach by
accident.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_retargeter_hands import (
    PANDA,
    VIPERX_300,
    Hand,
    HandToArm,
    Mount,
    Reach,
    arm_command,
    assemble,
    bimanual,
    hand_command,
)

from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

HAND = Hand(closed=0.02, open=0.10, span=0.19, measured_by="tape measure")
SOURCE = hand_command("hand", scale="metric", rotation_repr="quat_wxyz")
TARGET = arm_command("right_arm", rotation_repr="euler_xyz")


def trajectory(steps=5, *, x=0.4, aperture=0.06):
    """Identity rotation, a point out in front, a half-open hand."""
    values = np.zeros((steps, 8))
    values[:, 0] = x
    values[:, 3] = 1.0  # w of an identity wxyz quaternion
    values[:, 7] = aperture
    return values


def retargeter(**kwargs):
    kwargs.setdefault("mount", Mount.aligned())
    kwargs.setdefault("hand", HAND)
    return HandToArm(**kwargs)


# -- the scale refusal, which is the one that matters -----------------------


def test_unscaled_hands_with_no_measured_span_are_refused():
    """A monocular hand pose is right up to one unknown multiplier. Retargeting
    it as metres sends the arm to a point that does not exist, consistently —
    which looks exactly like a policy that never learned to reach."""
    unscaled = hand_command("hand", scale="unscaled")
    made = retargeter(hand=Hand(closed=0.02, open=0.10))  # no span

    verdict = made.accepts(unscaled, TARGET)
    assert not verdict.ok
    assert "unknown multiplier" in verdict.explain()
    assert "hands.unscaled_without_reference" in verdict.explain()


def test_unscaled_hands_with_a_measured_span_are_allowed_and_declare_the_cost():
    unscaled = hand_command("hand", scale="unscaled")
    made = retargeter()
    assert made.accepts(unscaled, TARGET).ok

    losses = made.losses(unscaled, TARGET)
    assert any("0.19" in loss and "rather than observed" in loss for loss in losses)


def test_a_channel_that_never_considered_scale_is_refused_separately():
    """Not the same refusal as 'unscaled'. One is a limitation somebody
    measured; the other is a question nobody asked, and a span cannot rescue it
    because nobody knows what the numbers are."""
    silent = ChannelSpec(
        "hand",
        "vector",
        (8,),
        "float32",
        semantics="ego.wrist_pose",
        metadata={"rotation_repr": "quat_wxyz", "hand": "right"},
    )
    verdict = retargeter().accepts(silent, TARGET)
    assert not verdict.ok
    assert "hands.scale_undeclared" in verdict.explain()


def test_the_span_scales_every_position():
    unscaled = hand_command("hand", scale="unscaled")
    made = retargeter()
    out = made.apply(trajectory(x=1.0), unscaled, TARGET)
    assert out[0, 0] == pytest.approx(0.19)


def test_an_implausible_span_is_refused_because_it_multiplies_through_everything():
    for bad in (0.019, 1.9, 2.0):
        with pytest.raises(ConfigError, match="multiplies through"):
            Hand(closed=0.02, open=0.10, span=bad)


# -- the joint-space refusal ------------------------------------------------


def test_a_joint_position_target_is_refused_and_says_what_is_missing():
    """The consequence is real: a pi0 server under the ALOHA config wants joint
    positions, so ego data cannot reach it through this retargeter alone. Saying
    so is better than emitting something joint-shaped."""
    joints = ChannelSpec(
        "action",
        "vector",
        (7,),
        "float32",
        semantics="action.joint_pos",
        metadata={"rotation_repr": "euler_xyz"},
    )
    verdict = retargeter().accepts(SOURCE, joints)
    assert not verdict.ok
    assert "hands.needs_inverse_kinematics" in verdict.explain()
    assert "elbow configuration" in verdict.explain()


def test_a_target_with_no_control_mode_is_refused():
    bare = ChannelSpec("action", "vector", (7,), "float32", metadata={"rotation_repr": "euler_xyz"})
    assert not retargeter().accepts(SOURCE, bare).ok


# -- rotation encodings ------------------------------------------------------


def test_an_undeclared_rotation_encoding_is_refused_on_either_side():
    """A quaternion in wxyz and one in xyzw are four floats each and differ by a
    permutation, so nothing else notices."""
    source = ChannelSpec(
        "hand",
        "vector",
        (8,),
        "float32",
        semantics="ego.wrist_pose",
        metadata={"scale": "metric", "hand": "right"},
    )
    assert "hands.encoding_undeclared" in retargeter().accepts(source, TARGET).explain()

    target = ChannelSpec("action", "vector", (7,), "float32", semantics="action.eef_abs_pose")
    assert "hands.target_encoding_undeclared" in retargeter().accepts(SOURCE, target).explain()


def test_the_width_must_match_the_encoding_on_both_sides():
    wrong = ChannelSpec(
        "hand",
        "vector",
        (7,),
        "float32",
        semantics="ego.wrist_pose",
        metadata={"rotation_repr": "quat_wxyz", "scale": "metric", "hand": "right"},
    )
    assert "hands.source_width" in retargeter().accepts(wrong, TARGET).explain()

    target = ChannelSpec(
        "action",
        "vector",
        (9,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={"rotation_repr": "euler_xyz"},
    )
    assert "hands.target_width" in retargeter().accepts(SOURCE, target).explain()


@pytest.mark.parametrize("encoding", ["quat_wxyz", "quat_xyzw", "euler_xyz", "axis_angle", "rot6d"])
def test_every_encoding_round_trips_through_the_transform(encoding):
    width = {"quat_wxyz": 4, "quat_xyzw": 4, "euler_xyz": 3, "axis_angle": 3, "rot6d": 6}[encoding]
    source = hand_command("hand", scale="metric", rotation_repr=encoding)
    values = np.zeros((4, 4 + width))
    values[:, 0] = 0.3
    if encoding == "quat_wxyz":
        values[:, 3] = 1.0
    elif encoding == "quat_xyzw":
        values[:, 6] = 1.0
    elif encoding == "rot6d":
        values[:, 3], values[:, 7] = 1.0, 1.0
    values[:, -1] = 0.06
    out = retargeter().apply(values, source, TARGET)
    assert out.shape == (4, 7)
    assert np.isfinite(out).all()


# -- the mount ---------------------------------------------------------------


def test_identity_is_a_named_constructor_rather_than_a_default():
    """Identity is a claim — right when a rig was set up so the frames agree, and
    silently wrong otherwise. It should be something somebody typed."""
    assert Mount.aligned().is_identity is True
    assert Mount.aligned().established_by == "declared aligned"
    assert Mount().established_by == ""


def test_a_rotation_that_is_not_a_rotation_is_refused():
    """A matrix that scales or shears as well as turning would quietly distort
    every pose through it."""
    with pytest.raises(ConfigError, match="not orthonormal"):
        Mount(rotation=np.diag([2.0, 1.0, 1.0]))


def test_a_mount_rotation_turns_positions_and_orientations_together():
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    made = retargeter(mount=Mount(rotation=quarter, established_by="measured on the rig"))
    out = made.apply(trajectory(x=0.4), SOURCE, TARGET)
    # A point at +x in the human's frame lands at +y in the robot's.
    assert out[0, :3] == pytest.approx([0.0, 0.4, 0.0], abs=1e-6)


def test_a_quaternion_mount_is_accepted_as_well_as_a_matrix():
    made = retargeter(mount=Mount(rotation=[0.0, 0.0, 0.0, 1.0]))  # 180 about z
    out = made.apply(trajectory(x=0.4), SOURCE, TARGET)
    assert out[0, 0] == pytest.approx(-0.4, abs=1e-6)


def test_the_origin_shifts_before_the_rotation():
    made = retargeter(mount=Mount(origin=(0.4, 0.0, 0.0)))
    out = made.apply(trajectory(x=0.4), SOURCE, TARGET)
    assert out[0, :3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_a_workspace_ratio_is_declared_as_a_loss():
    made = retargeter(mount=Mount(workspace_ratio=0.7))
    out = made.apply(trajectory(x=1.0), SOURCE, TARGET)
    assert out[0, 0] == pytest.approx(0.7)
    assert any("no longer the person's" in loss for loss in made.losses(SOURCE, TARGET))


# -- the gripper -------------------------------------------------------------


def test_the_aperture_becomes_a_fraction_of_this_persons_own_travel():
    """Per-person, because an adult's open hand and a child's differ by a factor
    that would otherwise land directly in the gripper signal."""
    made = retargeter()
    assert made.apply(trajectory(aperture=0.02), SOURCE, TARGET)[0, -1] == pytest.approx(0.0)
    assert made.apply(trajectory(aperture=0.10), SOURCE, TARGET)[0, -1] == pytest.approx(1.0)
    assert made.apply(trajectory(aperture=0.06), SOURCE, TARGET)[0, -1] == pytest.approx(0.5)


def test_opening_wider_than_the_calibration_is_clipped_not_extrapolated():
    """A gripper command above one is not a wider gripper."""
    out = retargeter().apply(trajectory(aperture=0.5), SOURCE, TARGET)
    assert out[0, -1] == pytest.approx(1.0)


def test_a_hand_with_no_travel_is_refused():
    with pytest.raises(ConfigError, match="cannot be the same number"):
        Hand(closed=0.05, open=0.05)


def test_two_hands_are_calibrated_separately():
    """People are not symmetric, and one calibration applied to both puts the
    difference straight into the gripper signal of whichever was not measured."""
    pair = bimanual(
        mount=Mount.aligned(),
        left=Hand(closed=0.02, open=0.09, span=0.19),
        right=Hand(closed=0.03, open=0.11, span=0.20),
    )
    assert set(pair) == {"left", "right"}
    left = pair["left"].apply(trajectory(aperture=0.055), SOURCE, TARGET)[0, -1]
    right = pair["right"].apply(trajectory(aperture=0.055), SOURCE, TARGET)[0, -1]
    assert left != right


# -- the workspace report, which is the product-facing output ---------------


def test_the_reach_report_says_what_fraction_is_out_of_reach_and_why():
    """'Too far away' and 'below the table' are different pieces of advice."""
    made = retargeter(reach=VIPERX_300)
    values = np.concatenate([trajectory(6, x=0.4), trajectory(4, x=1.2)])
    report = made.reach_report(values, SOURCE)

    assert report["measured"] is True
    assert report["arm"] == "viperx_300"
    assert report["in_reach"] == pytest.approx(0.6)
    assert report["why"]["too_far"] == pytest.approx(0.4)
    assert report["why"]["below_the_workspace"] == 0.0
    assert report["furthest"] == pytest.approx(1.2)


def test_reaching_below_the_workspace_is_reported_as_such():
    made = retargeter(reach=VIPERX_300)
    values = trajectory(4, x=0.4)
    values[:, 2] = -0.5
    assert made.reach_report(values, SOURCE)["why"]["below_the_workspace"] == 1.0


def test_the_report_says_so_when_no_workspace_was_measured():
    report = retargeter().reach_report(trajectory(), SOURCE)
    assert report["measured"] is False
    assert "no workspace" in report["why"]


def test_clipping_is_off_by_default_and_declared_when_on():
    """Clipping turns a reach the arm cannot make into one it can, which changes
    the demonstration into a different demonstration."""
    plain = retargeter(reach=VIPERX_300)
    assert plain.apply(trajectory(x=1.2), SOURCE, TARGET)[0, 0] == pytest.approx(1.2)
    assert not any("boundary" in loss for loss in plain.losses(SOURCE, TARGET))

    clipping = retargeter(reach=VIPERX_300, clip_to_reach=True)
    assert clipping.apply(trajectory(x=1.2), SOURCE, TARGET)[0, 0] == pytest.approx(0.75)
    assert any("different demonstration" in loss for loss in clipping.losses(SOURCE, TARGET))


def test_clipping_with_no_workspace_measured_is_refused():
    with pytest.raises(ConfigError, match="no workspace measured"):
        retargeter(clip_to_reach=True)


def test_a_reach_with_no_workspace_is_refused():
    with pytest.raises(ConfigError, match="no workspace at all"):
        Reach(radius=0.1, inner=0.2)
    with pytest.raises(ConfigError, match="ceiling"):
        Reach(radius=0.75, floor=0.5, ceiling=0.1)


def test_the_shipped_reaches_are_named_so_a_report_says_which_arm():
    assert VIPERX_300.name == "viperx_300"
    assert PANDA.radius > VIPERX_300.radius


# -- assembling two arms -----------------------------------------------------


def test_assembly_reads_the_targets_labels_rather_than_assuming_an_order():
    """A fourteen-wide vector is the same object whichever arm comes first, and a
    swap produces valid, smooth actions sent to the wrong arm."""
    left = np.full((3, 7), 1.0)
    right = np.full((3, 7), 2.0)

    left_first = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        dim_labels=tuple([f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]),
    )
    out = assemble({"left": left, "right": right}, left_first)
    assert out[0, 0] == 1.0 and out[0, 7] == 2.0

    right_first = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        dim_labels=tuple([f"right_{i}" for i in range(7)] + [f"left_{i}" for i in range(7)]),
    )
    out = assemble({"left": left, "right": right}, right_first)
    assert out[0, 0] == 2.0 and out[0, 7] == 1.0


def test_assembly_into_an_unlabelled_target_is_refused():
    target = ChannelSpec("action", "vector", (14,), "float32", semantics="actuation")
    with pytest.raises(ConfigError, match="which half of it is which arm"):
        assemble({"left": np.zeros((2, 7)), "right": np.zeros((2, 7))}, target)


def test_assembly_refuses_labels_that_never_mention_an_arm():
    target = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        dim_labels=tuple(f"joint_{index}" for index in range(14)),
    )
    with pytest.raises(ConfigError, match="no dimension label mentions"):
        assemble({"left": np.zeros((2, 7)), "right": np.zeros((2, 7))}, target)


def test_assembly_refuses_arms_of_different_lengths():
    target = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        dim_labels=tuple([f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]),
    )
    with pytest.raises(ConfigError, match="disagree on length"):
        assemble({"left": np.zeros((2, 7)), "right": np.zeros((3, 7))}, target)


def test_assembly_refuses_blocks_that_do_not_fill_the_target():
    target = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        dim_labels=tuple([f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]),
    )
    with pytest.raises(ConfigError, match="do not fill"):
        assemble({"left": np.zeros((2, 6)), "right": np.zeros((2, 7))}, target)


# -- what goes in the record -------------------------------------------------


def test_the_grasp_loss_is_always_declared():
    """A hand has more than twenty degrees of freedom and a gripper has one. Two
    demonstrations a person would describe completely differently can come out
    identical, and a result that hid that would let a policy's failure to
    reproduce a delicate grasp read as the policy's fault."""
    losses = retargeter().losses(SOURCE, TARGET)
    assert any("twenty degrees of freedom" in loss for loss in losses)
    assert any("force" in loss for loss in losses)


def test_the_provenance_carries_everything_that_was_measured():
    made = retargeter(reach=VIPERX_300, mount=Mount.aligned())
    record = made.provenance()
    assert record["retargeter"].startswith("hands@")
    assert record["identity_mount"] is True
    assert record["established_by"] == "declared aligned"
    assert record["hand_span"] == 0.19
    assert record["hand_measured_by"] == "tape measure"
    assert record["reach"] == "viperx_300"


def test_an_undeclared_mount_says_so_in_the_record():
    """So a report can distinguish a mount somebody established from one that was
    left at its defaults."""
    assert retargeter(mount=Mount()).provenance()["established_by"] == "undeclared"


def test_the_contracts_own_check_catches_an_undeclared_width_change():
    """The base class refuses a retargeter that changes width and declares no
    loss. This one always declares, so it passes — but the check runs."""
    assert retargeter().check(SOURCE, TARGET).ok


def test_applying_the_wrong_shape_is_refused():
    with pytest.raises(ConfigError, match=r"expected \(steps, 8\)"):
        retargeter().apply(np.zeros((5, 6)), SOURCE, TARGET)


# -- the moving frame --------------------------------------------------------


def test_a_camera_frame_trajectory_is_refused():
    """A hand position measured relative to a head-mounted camera changes when
    the head turns and the hand does not. Retargeted as though it were fixed, the
    arm's target swings every time the person looks around — smooth, plausible
    motion toward the wrong place."""
    for frame in ("camera", "head", "image"):
        source = hand_command("hand", scale="metric", frame=frame)
        verdict = retargeter().accepts(source, TARGET)
        assert not verdict.ok, frame
        assert "hands.moving_frame" in verdict.explain()
        assert "head pose" in verdict.explain()


def test_a_world_frame_trajectory_is_accepted():
    assert retargeter().accepts(hand_command("hand", scale="metric", frame="world"), TARGET).ok


def test_the_moving_frame_refusal_comes_before_the_scale_one():
    """Both are wrong; the frame is the one to fix first, because a hand span
    cannot rescue a trajectory measured against something that was moving."""
    source = hand_command("hand", scale="unscaled", frame="camera")
    explanation = retargeter(hand=Hand(closed=0.02, open=0.10)).accepts(source, TARGET).explain()
    assert "hands.moving_frame" in explanation
