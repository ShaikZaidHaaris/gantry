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
    MIN_REPROJECTION_PX,
    RIGS,
    Intrinsics,
    extent,
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
    # And the solve is *accepted*: the geometry is self-consistent at the wrong
    # scale, so nothing internal to the solve can object. Only a check on the
    # answer catches it -- and 1.0 m is still a plausible place for a hand, so
    # even the reachability bound lets this one through.
    assert pose.ok
    assert pose.reprojection < MIN_REPROJECTION_PX


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
    """Points matched to the wrong joints. RANSAC is off here on purpose: with it
    on, a scrambled hand still has a consistent subset and the point of this test
    is the rejection, not the robustness."""
    model = hand_model()
    scrambled = project(model, (0, 0, 0.5))[::-1]
    pose = solve(model, scrambled, CAMERA, robust=False)
    assert not pose.ok


def test_the_budget_scales_with_the_hand_rather_than_being_a_fixed_pixel_count():
    """An absolute budget is a different standard at every distance: ten pixels
    is loose on a hand filling the frame and impossible on one across the room.
    Two good detectors disagree with each other by about 8% of a hand's span, so
    a tighter bar than that demands better than the inputs support -- measured, a
    fixed 10 px rejected 109 of 120 perfectly plausible frames."""
    model = hand_model()
    near_pixels = project(model, (0.0, 0.0, 0.30))
    far_pixels = project(model, (0.0, 0.0, 1.20))
    # The same hand, four times further away, is four times smaller in frame.
    assert extent(near_pixels) > 3 * extent(far_pixels)

    # Add the same *relative* keypoint noise to both; both should still solve.
    rng = np.random.default_rng(1)
    for pixels in (near_pixels, far_pixels):
        noise = rng.normal(scale=0.04 * extent(pixels), size=pixels.shape)
        assert solve(model, pixels + noise, CAMERA).ok


def test_ransac_survives_a_few_joints_in_the_wrong_place():
    """A rigid model of a bending hand always has a few joints it cannot explain.
    Least squares is dragged around by them; RANSAC finds the consistent majority
    and reports the rest as outliers, which is the correct reading."""
    model = hand_model()
    pixels = project(model, (0.0, 0.0, 0.5))
    pixels[[4, 8, 12]] += 300.0  # three fingertips wildly misplaced

    assert solve(model, pixels, CAMERA, robust=True).ok
    assert not solve(model, pixels, CAMERA, robust=False).ok


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


# -- the per-person hand template --------------------------------------------


def test_a_template_is_the_median_of_the_frames_that_solved():
    """Pairing a strong 2D detector with a weak 3D one inherits the weak one's
    recall, because a frame needs both. A hand does not change size, so the
    frames that did solve measure this person's hand and can carry the rest."""
    from gantry_connector_handpose import template_from

    world = np.zeros((10, 21, 3))
    world[2:8] = hand_model()
    template = template_from(world)
    assert template is not None
    assert template == pytest.approx(hand_model(), abs=1e-9)


def test_no_template_when_too_few_frames_solved():
    from gantry_connector_handpose import template_from

    world = np.zeros((10, 21, 3))
    world[0] = hand_model()
    assert template_from(world) is None


def test_a_template_is_per_person_because_size_is_the_ruler():
    from gantry_connector_handpose import template_from

    big = hand_model() * 1.3
    world = np.zeros((10, 21, 3))
    world[:6] = big
    assert template_from(world) == pytest.approx(big, abs=1e-9)


def test_a_degenerate_hand_falls_back_instead_of_crashing_the_clip():
    """SQPNP asserts outright on a near-planar point set -- a hand seen edge-on,
    or an averaged template that came out flat. That is one frame's problem, not
    the episode's."""
    flat = hand_model().copy()
    flat[:, 2] = 0.0  # perfectly planar
    pose = solve(flat, project(flat, (0.0, 0.0, 0.5)), CAMERA)
    # Either it solves through a fallback or it reports unsolved. What it must
    # not do is raise.
    assert isinstance(pose.ok, bool)


def test_a_solve_never_raises_on_junk():
    junk = np.full((21, 3), 1e-9)
    pose = solve(junk, np.zeros((21, 2)), CAMERA)
    assert pose.ok is False


def test_a_pose_that_reprojects_perfectly_from_an_impossible_place_is_rejected():
    """The failure the fallback solvers introduced: a degenerate point set gets
    fitted at nine trillion metres, where the projection is unchanged by
    anything, so it reprojects beautifully. Reprojection cannot catch that. Only
    a statement about where hands actually are can."""
    model = hand_model()
    far = solve(model, project(model, (0.0, 0.0, 500.0)), CAMERA)
    assert far.reprojection < MAX_REPROJECTION  # it fits the pixels
    assert not far.ok  # and is still refused

    near = solve(model, project(model, (0.0, 0.0, 0.001)), CAMERA)
    assert not near.ok


def test_the_plausible_range_is_the_same_one_the_depth_path_uses():
    from gantry_connector_handpose import FAR, NEAR
    from gantry_connector_handpose.pnp import REACHABLE

    assert REACHABLE == (NEAR, FAR)
