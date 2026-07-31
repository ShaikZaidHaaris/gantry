"""Rotation conversions, checked by round trip and against known values."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_adapters_rotation import KNOWN, ROTATION, WIDTHS, convert, from_matrix, to_matrix
from gantry_adapters_rotation.rotation import ARMS_KEY, KEY, OFFSET_KEY, _offsets
from gantry_semantics_manipulation import action_channel

from gantry.conformance import check_adapter
from gantry.resolve import AdapterRegistry, bind_channel, requires_channels
from gantry.spine import ChannelSpec, compatible

NEED = requires_channels("consumer", "policy")


def pose(encoding, *, gripper=False, name="action"):
    width = 3 + WIDTHS[encoding] + (1 if gripper else 0)
    return action_channel(
        name,
        "eef_abs_pose",
        width=width,
        rotation_repr=encoding,
        gripper="continuous" if gripper else "none",
    )


def random_rotations(n=32, seed=0):
    """Uniform random rotations, as canonical wxyz quaternions."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.where((q[:, 0] < 0)[:, None], -q, q)


# -- the maths -------------------------------------------------------------


@pytest.mark.parametrize("encoding", sorted(KNOWN))
def test_every_encoding_round_trips_through_a_matrix(encoding):
    start = from_matrix(to_matrix(random_rotations(), "quat_wxyz"), encoding)
    back = from_matrix(to_matrix(start, encoding), encoding)
    assert np.allclose(to_matrix(start, encoding), to_matrix(back, encoding), atol=1e-8)


@pytest.mark.parametrize("encoding", sorted(KNOWN))
def test_a_matrix_survives_a_trip_through_every_encoding(encoding):
    matrices = to_matrix(random_rotations(), "quat_wxyz")
    assert np.allclose(to_matrix(from_matrix(matrices, encoding), encoding), matrices, atol=1e-8)


def test_identity_stays_identity():
    for encoding in sorted(KNOWN):
        matrices = np.tile(np.eye(3), (4, 1, 1))
        assert np.allclose(
            to_matrix(from_matrix(matrices, encoding), encoding), matrices, atol=1e-9
        )


def test_a_quarter_turn_about_z_is_what_it_should_be():
    quarter = from_matrix(
        np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]), "euler_xyz"
    )
    assert np.allclose(quarter, [[0.0, 0.0, np.pi / 2]], atol=1e-9)


def test_wxyz_and_xyzw_are_the_same_rotation_written_differently():
    wxyz = random_rotations()
    xyzw = from_matrix(to_matrix(wxyz, "quat_wxyz"), "quat_xyzw")
    assert np.allclose(xyzw, wxyz[:, [1, 2, 3, 0]], atol=1e-9)


def test_the_antipode_is_resolved_rather_than_left_to_flip():
    """q and -q are the same rotation and not the same numbers."""
    q = random_rotations(8)
    flipped = -q
    assert np.allclose(to_matrix(q, "quat_wxyz"), to_matrix(flipped, "quat_wxyz"), atol=1e-9)
    assert np.allclose(from_matrix(to_matrix(flipped, "quat_wxyz"), "quat_wxyz"), q, atol=1e-9)


def test_a_half_turn_does_not_divide_by_zero():
    """The branch a naive matrix-to-quaternion formula gets wrong."""
    half = np.array([[[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]])
    q = from_matrix(half, "quat_wxyz")
    assert np.all(np.isfinite(q))
    assert np.allclose(to_matrix(q, "quat_wxyz"), half, atol=1e-8)


# -- as a channel transform ------------------------------------------------


def test_position_and_gripper_pass_through_untouched():
    source, target = pose("quat_wxyz", gripper=True), pose("axis_angle", gripper=True)
    values = np.concatenate(
        [
            np.arange(4 * 3).reshape(4, 3) * 1.0,
            random_rotations(4),
            np.array([[0.2], [0.4], [0.6], [0.8]]),
        ],
        axis=1,
    )
    out = convert(values, source, target)
    assert out.shape == (4, 3 + 3 + 1)
    assert np.allclose(out[:, :3], values[:, :3])
    assert np.allclose(out[:, -1], values[:, -1])


def test_the_rotation_itself_is_preserved():
    source, target = pose("quat_wxyz"), pose("euler_xyz")
    values = np.concatenate([np.zeros((16, 3)), random_rotations(16)], axis=1)
    out = convert(values, source, target)
    assert np.allclose(
        to_matrix(out[:, 3:], "euler_xyz"), to_matrix(values[:, 3:], "quat_wxyz"), atol=1e-8
    )


# -- through the resolver --------------------------------------------------


def test_two_encodings_are_refused_without_the_adapter():
    verdict = compatible(pose("quat_wxyz"), pose("quat_xyzw"))
    assert "metadata.mismatch" in verdict.codes()


def test_the_adapter_closes_that_refusal():
    binding, verdict = bind_channel(
        pose("quat_xyzw"), [pose("quat_wxyz")], NEED, AdapterRegistry([ROTATION])
    )
    assert verdict.ok, verdict.explain()
    assert str(binding.chain) == f"rotation@{ROTATION.version}"
    assert not binding.chain.lossy


def test_it_closes_a_width_change_too():
    """quat is four numbers and axis-angle is three, so the pose width changes."""
    binding, verdict = bind_channel(
        pose("axis_angle"), [pose("quat_wxyz")], NEED, AdapterRegistry([ROTATION])
    )
    assert verdict.ok, verdict.explain()
    assert binding.want.width == 6 and binding.provided.width == 7


def test_the_data_really_comes_through_re_encoded():
    from gantry.resolve import adapt_episode, bind
    from gantry.spine import episode_from_arrays

    source, target = pose("quat_wxyz"), pose("axis_angle")
    values = np.concatenate([np.zeros((5, 3)), random_rotations(5)], axis=1).astype("float32")
    episode = episode_from_arrays({"action": values}, (source,), id="e", source="t")
    wiring, verdict = bind(
        requires_channels("c", "policy", target), episode.schema, AdapterRegistry([ROTATION])
    )
    assert verdict.ok
    out = adapt_episode(episode, wiring).array("action")
    assert out.shape == (5, 6)
    assert np.allclose(
        to_matrix(out[:, 3:], "axis_angle"), to_matrix(values[:, 3:], "quat_wxyz"), atol=1e-6
    )


# -- conformance and refusals ---------------------------------------------


def test_it_conforms():
    verdict = check_adapter(ROTATION, pose("quat_wxyz"), pose("quat_xyzw"))
    assert verdict.ok, verdict.explain()


def test_an_undeclared_encoding_is_refused():
    bare = ChannelSpec("action", "vector", (7,), "float32")
    assert "adapter.rotation_undeclared" in ROTATION.applies(bare, pose("quat_wxyz")).codes()


def test_an_unknown_encoding_is_refused_with_the_list():
    odd = ChannelSpec("action", "vector", (7,), "float32", metadata={"rotation_repr": "runes"})
    verdict = ROTATION.applies(odd, pose("quat_wxyz"))
    assert "adapter.rotation_unknown" in verdict.codes()
    assert "quat_wxyz" in verdict.reasons[0].message


def test_a_mismatched_layout_is_refused():
    """Only the rotation block is converted, so everything else must line up."""
    verdict = ROTATION.applies(pose("quat_wxyz", gripper=True), pose("quat_xyzw"))
    assert "adapter.rotation_layout" in verdict.codes()


def test_a_bare_rotation_is_the_one_unambiguous_layout():
    bare = ChannelSpec(
        "q",
        "vector",
        (4,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_wxyz"},
    )
    other = ChannelSpec(
        "q",
        "vector",
        (4,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_xyzw"},
    )
    assert ROTATION.applies(bare, other).ok
    values = random_rotations(4)
    assert np.allclose(convert(values, bare, other), values[:, [1, 2, 3, 0]], atol=1e-9)


def test_an_unknowable_offset_is_refused_with_the_fix():
    """A width alone does not say how the other numbers sit around the rotation."""
    padded = ChannelSpec(
        "x",
        "vector",
        (9,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_wxyz"},
    )
    target = ChannelSpec(
        "x",
        "vector",
        (8,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "axis_angle"},
    )
    verdict = ROTATION.applies(padded, target)
    assert "adapter.rotation_offset" in verdict.codes()
    assert "rotation_offset" in verdict.reasons[0].hint


def test_a_declared_offset_makes_any_layout_workable():
    padded = ChannelSpec(
        "x",
        "vector",
        (9,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_wxyz", "rotation_offset": 5},
    )
    target = ChannelSpec(
        "x",
        "vector",
        (8,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "axis_angle", "rotation_offset": 5},
    )
    assert ROTATION.applies(padded, target).ok
    values = np.concatenate([np.arange(4 * 5).reshape(4, 5) * 1.0, random_rotations(4)], axis=1)
    out = convert(values, padded, target)
    assert out.shape == (4, 8)
    assert np.allclose(out[:, :5], values[:, :5])


# -- one rotation per arm ------------------------------------------------------
#
# A two-armed command holds two rotation blocks and they do not sit at the same
# offset on both sides: replacing three Euler numbers with four quaternion ones
# shifts everything after the first block. Converting only the first would leave
# the second arm's three numbers to be read as the start of a quaternion — a
# well-formed vector, a smooth trajectory, and the wrong arm pointing somewhere
# arbitrary. Nothing downstream can detect that, which is why it is checked here.


def bimanual(encoding, *, name="action", arms=None):
    """[xyz rot grip] twice, laid end to end."""
    per_arm = 3 + WIDTHS[encoding] + 1
    metadata = {KEY: encoding, OFFSET_KEY: (3, per_arm + 3)}
    if arms is not None:
        metadata[ARMS_KEY] = arms
    return ChannelSpec(
        name,
        "vector",
        (per_arm * 2,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata=metadata,
    )


def test_both_arms_are_converted_not_just_the_first():
    source, target = bimanual("euler_xyz"), bimanual("quat_wxyz")
    assert source.width == 14 and target.width == 16

    values = np.zeros((5, 14))
    values[:, 0:3] = [0.1, 0.2, 0.3]  # left position
    values[:, 3:6] = [0.0, 0.0, np.pi / 2]  # left rotation: a quarter turn in z
    values[:, 6] = 0.04  # left gripper
    values[:, 7:10] = [0.5, 0.6, 0.7]  # right position
    values[:, 10:13] = [np.pi / 2, 0.0, 0.0]  # right rotation: a quarter turn in x
    values[:, 13] = 0.08  # right gripper

    out = convert(values, source, target)
    assert out.shape == (5, 16)

    # positions and grippers land where the target says, untouched
    assert np.allclose(out[:, 0:3], [0.1, 0.2, 0.3])
    assert np.allclose(out[:, 7], 0.04)
    assert np.allclose(out[:, 8:11], [0.5, 0.6, 0.7])
    assert np.allclose(out[:, 15], 0.08)

    # and both rotations really were converted, to different quaternions
    half = np.sqrt(0.5)
    assert np.allclose(out[:, 3:7], [half, 0, 0, half])  # z quarter turn
    assert np.allclose(out[:, 11:15], [half, half, 0, 0])  # x quarter turn


def test_the_round_trip_holds_for_both_arms():
    source, target = bimanual("euler_xyz"), bimanual("quat_wxyz")
    rng = np.random.default_rng(0)
    values = rng.uniform(-1.0, 1.0, size=(20, 14))
    back = convert(convert(values, source, target), target, source)
    assert np.allclose(back, values, atol=1e-9)


def test_a_single_block_channel_cannot_be_converted_into_a_two_block_one():
    single = ChannelSpec(
        "one",
        "vector",
        (7,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "euler_xyz"},
    )
    verdict = ROTATION.guard(single, bimanual("quat_wxyz"))
    assert not verdict.ok
    assert "adapter.rotation_blocks" in verdict.explain()


def test_a_declared_arm_count_that_disagrees_with_the_blocks_is_refused():
    """Two arms and one rotation is a layout claim that contradicts itself."""
    wrong = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "euler_xyz", OFFSET_KEY: (3,), ARMS_KEY: 2},
    )
    verdict = ROTATION.guard(wrong, bimanual("quat_wxyz"))
    assert not verdict.ok
    assert "adapter.rotation_blocks" in verdict.explain()


def test_a_matching_arm_count_passes():
    verdict = ROTATION.guard(bimanual("euler_xyz", arms=2), bimanual("quat_wxyz", arms=2))
    assert verdict.ok, verdict.explain()


def test_arms_are_never_inferred_only_checked():
    """Width 14 is as consistent with one rotation and a long tail as with two."""
    undeclared = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "euler_xyz", ARMS_KEY: 2},
    )
    assert _offsets(undeclared, "euler_xyz") == (3,)  # the conventional single block
    verdict = ROTATION.guard(undeclared, bimanual("quat_wxyz"))
    assert not verdict.ok  # and it is refused rather than half-converted


def test_grippers_between_the_arms_must_line_up():
    """Same encodings and same block count, but one side puts its grippers at
    the end instead of after each arm. The rotations would convert; the numbers
    around them would silently change meaning."""
    trailing = ChannelSpec(
        "trailing",
        "vector",
        (16,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "quat_wxyz", OFFSET_KEY: (3, 10)},
    )
    verdict = ROTATION.guard(bimanual("euler_xyz"), trailing)
    assert not verdict.ok
    assert "adapter.rotation_layout" in verdict.explain()
    assert "must line up" in verdict.explain()


def test_overlapping_blocks_are_refused_rather_than_producing_a_wider_vector():
    broken = ChannelSpec(
        "broken",
        "vector",
        (14,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "euler_xyz", OFFSET_KEY: (3, 4)},
    )
    verdict = ROTATION.guard(broken, bimanual("quat_wxyz"))
    assert not verdict.ok
    assert "overlap" in verdict.explain()


def test_a_block_running_off_the_end_is_refused():
    over = ChannelSpec(
        "over",
        "vector",
        (14,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "euler_xyz", OFFSET_KEY: (3, 12)},
    )
    assert not ROTATION.guard(over, bimanual("quat_wxyz")).ok


def test_the_single_arm_case_still_works_exactly_as_before():
    source = ChannelSpec(
        "one",
        "vector",
        (7,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "euler_xyz"},
    )
    target = ChannelSpec(
        "one",
        "vector",
        (8,),
        "float32",
        semantics="action.eef_abs_pose",
        metadata={KEY: "quat_wxyz"},
    )
    assert ROTATION.guard(source, target).ok
    values = np.zeros((3, 7))
    values[:, 6] = 0.05
    out = convert(values, source, target)
    assert out.shape == (3, 8) and np.allclose(out[:, 7], 0.05)
