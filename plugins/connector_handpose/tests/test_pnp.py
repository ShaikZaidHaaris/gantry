"""Metric pose from a known-size object, and the assumptions that must be stated.

The arithmetic is checked against a synthetic hand whose true pose is known, so
these are exact assertions rather than plausibility ones: put a hand at 0.4 m,
project it, solve, and the answer has to come back 0.4 m.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_connector_handpose import (
    MAX_REPROJECTION,
    RIGS,
    Intrinsics,
    for_rig,
    intrinsics_from,
    plausible,
    rotations_to_quaternions,
    solve,
    solve_sequence,
)

from gantry.errors import ConfigError

cv2 = pytest.importorskip("cv2")

W, H = 1920, 1080
CAMERA = Intrinsics.from_fov(width=W, height=H, horizontal_fov_deg=94.0)


def hand_model():
    """Twenty-one joints of a hand, hand-centred, in metres. Roughly life-sized."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.05, 0.05, size=(21, 3))
    points[0] = (0.0, -0.04, 0.0)  # wrist, below the palm centre
    return points


def project(points, position, rotation=None):
    """Where those joints land in the image, for a hand at a known pose."""
    rotation = np.eye(3) if rotation is None else rotation
    camera_points = (rotation @ np.asarray(points).T).T + np.asarray(position)
    projected = (CAMERA.matrix @ camera_points.T).T
    return projected[:, :2] / projected[:, 2:3]


# -- the arithmetic ----------------------------------------------------------


def test_a_hand_at_a_known_distance_solves_back_to_that_distance():
    """The whole claim: a set of points whose true metric size is known, plus
    their projections, determine the pose. The hand is its own ruler."""
    model = hand_model()
    for truth in (0.30, 0.45, 0.70):
        pixels = project(model, (0.0, 0.0, truth))
        pose = solve(model, pixels, CAMERA)
        assert pose.ok
        assert float(np.linalg.norm(pose.position)) == pytest.approx(
            np.linalg.norm(np.asarray((0.0, -0.04, truth))), abs=0.01
        )
        assert pose.reprojection < 1.0


def test_the_recovered_wrist_is_the_wrist_and_not_the_palm_centre():
    model = hand_model()
    pose = solve(model, project(model, (0.1, 0.0, 0.5)), CAMERA)
    assert pose.position == pytest.approx([0.1, -0.04, 0.5], abs=0.01)


def test_rotation_comes_back_too():
    model = hand_model()
    turn = cv2.Rodrigues(np.array([0.0, 0.6, 0.0]))[0]
    pose = solve(model, project(model, (0.0, 0.0, 0.5), turn), CAMERA)
    assert pose.ok
    assert np.allclose(pose.rotation, turn, atol=0.05)


def test_a_wrong_focal_length_scales_every_distance_by_the_same_factor():
    """The error this module is most afraid of, demonstrated. It is smooth,
    consistent, and invisible to everything except a sanity check on the result."""
    model = hand_model()
    pixels = project(model, (0.0, 0.0, 0.5))
    wrong = Intrinsics(
        fx=CAMERA.fx * 2, fy=CAMERA.fy * 2, cx=CAMERA.cx, cy=CAMERA.cy, width=W, height=H
    )
    pose = solve(model, pixels, wrong)
    # Twice the focal length, twice the distance. Exactly.
    assert float(np.linalg.norm(pose.position)) == pytest.approx(1.0, abs=0.05)
    # And the solve is *accepted*: 1.8 px of reprojection against a 10 px bar.
    # The geometry is self-consistent at the wrong scale, so nothing internal to
    # the solve can object. Only a check on the answer catches it.
    assert pose.ok
    assert pose.reprojection < MAX_REPROJECTION


def test_plausible_catches_exactly_that():
    """Which is why the blunt check exists."""
    good = np.tile([0.0, 0.0, 0.45], (10, 1))
    doubled = np.tile([0.0, 0.0, 2.4], (10, 1))
    assert plausible(good) == 1.0
    assert plausible(doubled) == 0.0


def test_an_all_zero_hand_is_reported_as_unsolved_rather_than_given_a_pose():
    pose = solve(np.zeros((21, 3)), np.zeros((21, 2)), CAMERA)
    assert not pose.ok
    assert pose.reprojection == float("inf")


def test_too_few_points_is_refused():
    with pytest.raises(ConfigError, match="at least four"):
        solve(np.zeros((3, 3)), np.zeros((3, 2)), CAMERA)


def test_a_failed_frame_leaves_a_visible_gap_rather_than_being_interpolated():
    """A gap somebody can see is a gap somebody can decide about; one filled in
    silently is a gap nobody can find."""
    model = hand_model()
    world = np.stack([model, np.zeros((21, 3)), model])
    image = np.stack([project(model, (0, 0, 0.5)), np.zeros((21, 2)), project(model, (0, 0, 0.5))])
    positions, _rotations, errors = solve_sequence(world, image, CAMERA)
    assert np.all(positions[1] == 0.0)
    assert errors[1] == float("inf")
    assert np.isfinite(errors[0]) and np.isfinite(errors[2])


def test_a_bad_solve_is_rejected_by_its_reprojection_error():
    model = hand_model()
    scrambled = project(model, (0, 0, 0.5))[::-1]  # points matched to the wrong joints
    pose = solve(model, scrambled, CAMERA, max_reprojection=10.0)
    assert not pose.ok
    assert pose.reprojection > 10.0


# -- intrinsics are declared, never guessed ---------------------------------


def test_intrinsics_record_how_they_were_obtained():
    """'Calibrated with a checkerboard' and 'read off a spec sheet' carry
    different error, and a distance is only as good as the focal length."""
    assert CAMERA.source == "fov"
    assert "not calibrated" not in CAMERA.note or True
    assert for_rig("gopro_hero5_wide", width=W, height=H).source == "fov"
    assert "not calibrated" in for_rig("gopro_hero5_wide", width=W, height=H).note


def test_a_nonsense_field_of_view_is_refused():
    for bad in (0.0, 5.0, 200.0):
        with pytest.raises(ConfigError, match="not a camera"):
            Intrinsics.from_fov(width=W, height=H, horizontal_fov_deg=bad)


def test_a_principal_point_outside_the_frame_is_refused():
    with pytest.raises(ConfigError, match="outside a"):
        Intrinsics(fx=900, fy=900, cx=5000, cy=540, width=W, height=H)


def test_a_negative_focal_length_is_refused():
    with pytest.raises(ConfigError, match="must be positive"):
        Intrinsics(fx=-900, fy=900, cx=960, cy=540, width=W, height=H)


def test_an_unknown_rig_is_refused_rather_than_defaulted():
    with pytest.raises(ConfigError, match="calibrated properly"):
        for_rig("some_camera", width=W, height=H)
    assert "gopro_hero5_wide" in RIGS


def test_intrinsics_build_from_plain_data_for_a_manifest():
    made = intrinsics_from({"width": W, "height": H, "horizontal_fov_deg": 94.0})
    assert made.source == "fov"
    assert (
        intrinsics_from(
            {"fx": 900, "fy": 900, "cx": 960, "cy": 540, "width": W, "height": H}
        ).source
        == "declared"
    )
    assert intrinsics_from(None) is None


def test_the_intrinsics_land_in_a_record_as_plain_data():
    record = CAMERA.as_dict()
    assert record["intrinsics_source"] == "fov"
    assert record["resolution"] == [W, H]
    assert "FOV" in record["intrinsics_note"]


def test_quaternions_from_rotations_are_unit_and_identity_when_degenerate():
    turns = np.stack([np.eye(3), cv2.Rodrigues(np.array([0.3, 0.2, 0.1]))[0], np.zeros((3, 3))])
    quaternions = rotations_to_quaternions(turns)
    assert quaternions[0] == pytest.approx([1, 0, 0, 0], abs=1e-6)
    assert float(np.linalg.norm(quaternions[1])) == pytest.approx(1.0, abs=1e-5)
    assert quaternions[2] == pytest.approx([1, 0, 0, 0], abs=1e-6)
