"""The ego chain's output, accepted by a dual-arm evaluator.

This is the join that did not exist. ``retargeter_hands`` refuses to produce
joint positions — inverse kinematics needs link lengths, joint limits and a
choice of elbow configuration, none of which a retargeter has — and every
bimanual config installed until now wanted joint positions. So the ego pipeline
produced commands nothing could execute, and the report correctly refused to say
whether the data helped.

RoboTwin accepts end-effector poses. What this file checks is that the two
halves genuinely meet: that what the retargeter emits is the shape and the space
the evaluator reads, and that the mismatch which used to block it is still
caught when it is real.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_robotwin import RoboTwinEvaluator, for_ego, labels_for, width_of
from gantry_retargeter_hands import (
    VIPERX_300,
    Hand,
    HandToArm,
    Mount,
    arm_command,
    hand_command,
)

from gantry.errors import ConfigError
from gantry.spine import compatible


def retargeter():
    return HandToArm(
        mount=Mount.aligned(),
        hand=Hand(closed=0.02, open=0.10, span=0.19, measured_by="tape"),
        reach=VIPERX_300,
    )


def test_the_retargeter_still_refuses_joint_space_and_names_what_is_missing():
    """The refusal that made this necessary. It has not been weakened — the
    evaluator met the retargeter, not the other way round."""
    joints = RoboTwinEvaluator("dual_bottles_pick_easy", action_type="qpos")
    source = hand_command("right_wrist", scale="metric", frame="world")
    target = ChannelSpec = joints.action()

    # A qpos target is joint space by another name; the retargeter's own guard
    # is on the control mode it is handed.
    verdict = retargeter().accepts(
        source,
        arm_command("arm", rotation_repr="euler_xyz", control_mode="joint_pos")
        if False
        else _joint_target(),
    )
    assert not verdict.ok
    assert "hands.needs_inverse_kinematics" in verdict.explain()
    assert "elbow configuration" in verdict.explain()


def _joint_target():
    from gantry.spine import ChannelSpec

    return ChannelSpec(
        "action", "vector", (7,), "float32", semantics="action.joint_pos",
        metadata={"rotation_repr": "euler_xyz"},
    )


def test_end_effector_mode_is_the_space_the_retargeter_emits():
    made = for_ego(factory=lambda task, **_: None)
    assert made.descriptor().metadata["accepts_end_effector"] is True

    source = hand_command("right_wrist", scale="metric", frame="world")
    pose_target = arm_command("right_arm", rotation_repr="quat_wxyz")
    verdict = retargeter().accepts(source, pose_target)
    assert verdict.ok, verdict.explain()


def test_the_widths_line_up_per_arm():
    """Eight numbers an arm on both sides — position, wxyz quaternion, gripper —
    so two arms concatenated is exactly what the evaluator reads."""
    per_arm = arm_command("arm", rotation_repr="quat_wxyz").width
    assert per_arm == 8
    assert width_of("ee") == per_arm * 2


def test_the_arm_order_is_written_down_on_both_sides():
    """A sixteen-wide float32 is the same object whichever arm comes first, and
    a swap produces valid commands sent to the wrong arm."""
    labels = labels_for("ee")
    assert labels[:8] == tuple(f"left_{p}" for p in
                               ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper"))
    assert labels[8:] == tuple(f"right_{p}" for p in
                               ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper"))


def test_a_joint_space_policy_cannot_be_run_against_an_end_effector_evaluator():
    """The mismatch this discriminator exists for, still caught."""
    joints = RoboTwinEvaluator("t", action_type="qpos").action()
    poses = RoboTwinEvaluator("t", action_type="ee").action()
    assert not compatible(joints, poses).ok


def test_a_real_retargeted_trajectory_is_the_right_shape_for_the_evaluator():
    steps = 12
    values = np.zeros((steps, 8))
    values[:, 0] = 0.4          # 40 cm in front
    values[:, 3] = 1.0          # identity wxyz
    values[:, 7] = 0.06         # aperture

    source = hand_command("wrist", scale="metric", frame="world", rotation_repr="quat_wxyz")
    target = arm_command("arm", rotation_repr="quat_wxyz")
    per_arm = retargeter().apply(values, source, target)
    assert per_arm.shape == (steps, 8)

    bimanual = np.concatenate([per_arm, per_arm], axis=1)
    assert bimanual.shape[1] == width_of("ee")
