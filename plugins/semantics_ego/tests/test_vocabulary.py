"""The ego vocabulary, and the two refusals it exists for."""

from __future__ import annotations

import pytest
from gantry_semantics_ego import (
    KEYPOINTS,
    SCALES,
    aperture_channel,
    describe,
    ego_camera,
    gaze_channel,
    hand_channel,
    head_pose_channel,
    is_metric,
    scale_of,
    sequence_of,
    wrist_channel,
)

from gantry.spine import compatible, known_semantics


def test_the_ego_meanings_are_registered():
    known = known_semantics()
    for name in ("ego.rgb", "ego.hand_keypoints", "ego.wrist_pose", "ego.aperture"):
        assert name in known


# -- the scale question ------------------------------------------------------


def test_an_unscaled_hand_cannot_be_bound_where_metres_are_wanted():
    """The failure this discriminator exists for. Both are float32 (21, 3). One
    is metres; the other is metres times an unknown constant. Retarget the second
    as the first and the arm reaches for a point that does not exist — which
    looks exactly like a policy that has not learned to reach."""
    estimated = hand_channel("hands", scale="unscaled")
    wanted = hand_channel("hands", scale="metric")

    verdict = compatible(estimated, wanted)
    assert not verdict.ok
    assert "scale" in verdict.explain()

    assert compatible(hand_channel("hands", scale="metric"), wanted).ok


def test_a_channel_that_never_considered_scale_is_not_the_same_as_one_that_says_unscaled():
    """`None` should worry a caller more than `unscaled` does. One is a
    limitation somebody measured; the other is a question nobody asked."""
    from gantry.spine import ChannelSpec

    silent = ChannelSpec("hands", "tensor", (21, 3), "float32", semantics="ego.hand_keypoints")
    assert scale_of(silent) is None
    assert is_metric(silent) is None

    assert scale_of(hand_channel("hands", scale="unscaled")) == "unscaled"
    assert is_metric(hand_channel("hands", scale="unscaled")) is False
    assert is_metric(hand_channel("hands", scale="metric")) is True


def test_metric_hands_carry_metres_and_unscaled_ones_carry_no_units():
    """A unit on an unscaled channel would be a lie with a dimension attached."""
    assert hand_channel("h", scale="metric").units is not None
    assert hand_channel("h", scale="unscaled").units is None
    assert hand_channel("h", scale="normalized").units is None


def test_normalized_is_kept_distinct_from_unscaled():
    """Different recoveries: a normalised hand needs a bone length, an unscaled
    one needs a measurement. Collapsing them loses which."""
    assert set(SCALES) == {"metric", "unscaled", "normalized"}
    assert not compatible(
        hand_channel("h", scale="normalized"), hand_channel("h", scale="unscaled")
    ).ok


# -- the keypoint convention -------------------------------------------------


def test_mano_and_mediapipe_are_the_same_width_and_do_not_fit_each_other():
    """Twenty-one joints each, different orderings, different roots. Feed one to
    a retargeter written for the other and every finger attaches to the wrong
    knuckle — shapes agree, dtypes agree, nothing complains."""
    assert KEYPOINTS["mano"] == KEYPOINTS["mediapipe"] == 21

    mano = hand_channel("hands", keypoints="mano")
    mediapipe = hand_channel("hands", keypoints="mediapipe")
    assert mano.shape == mediapipe.shape

    verdict = compatible(mediapipe, mano)
    assert not verdict.ok
    assert "keypoints" in verdict.explain()


def test_the_joint_count_follows_from_the_convention():
    """Not a separate argument, because a caller who could get them out of step
    is a caller who will."""
    assert hand_channel("h", keypoints="mano").shape == (21, 3)
    assert hand_channel("h", keypoints="arkit").shape == (26, 3)
    assert hand_channel("h", keypoints="wrist_only").shape == (3,)


def test_an_unknown_convention_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="wrong knuckle"):
        hand_channel("h", keypoints="whatever")


# -- handedness --------------------------------------------------------------


def test_left_and_right_hands_do_not_fit_each_other():
    """Mirror images. An estimator that reports them by array slot rather than by
    identity swaps them whenever the person crosses their arms."""
    assert not compatible(hand_channel("h", hand="left"), hand_channel("h", hand="right")).ok
    assert compatible(hand_channel("h", hand="right"), hand_channel("h", hand="right")).ok


def test_an_unknown_hand_is_refused():
    with pytest.raises(ValueError, match="unknown hand"):
        hand_channel("h", hand="middle")


# -- the other channels ------------------------------------------------------


def test_a_wrist_pose_borrows_the_manipulation_rotation_vocabulary():
    """Deliberately the same spelling as the robot side. A wrist pose and an
    end-effector pose are the same kind of object, and the retargeter cannot
    relate them if the two planes disagree about what a quaternion is called."""
    quat = wrist_channel("wrist", rotation_repr="quat_wxyz")
    assert quat.shape == (7,)
    assert wrist_channel("wrist", rotation_repr="euler_xyz").shape == (6,)
    assert wrist_channel("wrist", rotation_repr="rotmat").shape == (12,)

    assert not compatible(
        wrist_channel("wrist", rotation_repr="quat_xyzw"),
        wrist_channel("wrist", rotation_repr="quat_wxyz"),
    ).ok


def test_the_aperture_carries_its_scale_because_the_gripper_mapping_needs_it():
    """'Eight centimetres open' maps to a gripper. '0.31 of an unknown unit' does
    not."""
    assert aperture_channel("grip", scale="metric").units is not None
    assert not compatible(
        aperture_channel("grip", scale="unscaled"),
        aperture_channel("grip", scale="metric"),
    ).ok


def test_the_ego_camera_says_it_moves():
    """A fixed-camera assumption downstream is wrong in a way that only shows up
    as poor performance."""
    camera = ego_camera(height=224, width=224, rate_hz=30.0)
    assert camera.shape == (224, 224, 3)
    assert camera.frame == "camera"
    assert camera.metadata["moving"] is True


def test_a_head_pose_lives_in_the_world_frame():
    pose = head_pose_channel(rate_hz=30.0)
    assert pose.frame == "world"
    assert pose.shape == (7,)


def test_gaze_is_a_direction():
    assert gaze_channel().shape == (3,)
    assert gaze_channel().dim_labels == ("x", "y", "z")


def test_every_builder_makes_an_internally_valid_spec():
    for spec in (
        hand_channel("h"),
        wrist_channel("w"),
        aperture_channel("a"),
        ego_camera(height=8, width=8),
        head_pose_channel(),
        gaze_channel(),
    ):
        assert spec.validate().ok, spec.validate().explain()


# -- the report-facing helpers ----------------------------------------------


def test_describe_says_the_thing_that_limits_what_the_data_can_be_used_for():
    """'hand_keypoints (21, 3) float32' tells a user nothing."""
    text = describe(hand_channel("hands", hand="right", keypoints="mano", scale="unscaled"))
    assert "right hand" in text
    assert "MANO" in text
    assert "unknown multiplier" in text

    assert "calibrated" in describe(hand_channel("h", scale="metric"))
    assert "wrist only" in describe(hand_channel("h", keypoints="wrist_only"))


def test_a_schema_can_be_mixed_and_the_grouping_shows_it():
    """A rig that tracked itself and an estimator that filled in the hands is a
    real and common situation, and it is two rows rather than one verdict."""
    from gantry.spine import ChannelSpec

    schema = [
        ego_camera(height=8, width=8),
        head_pose_channel(scale="metric"),
        hand_channel("left_hand", hand="left", scale="unscaled"),
        hand_channel("right_hand", hand="right", scale="unscaled"),
        ChannelSpec("action", "vector", (7,), "float32", semantics="actuation"),
    ]
    grouped = sequence_of(schema)
    assert grouped["metric"] == ["head_pose"]
    assert sorted(grouped["unscaled"]) == ["left_hand", "right_hand"]
    # The camera declares no scale, and the non-ego channel is not counted at all.
    assert grouped["undeclared"] == ["ego_rgb"]
    assert "action" not in {name for names in grouped.values() for name in names}
