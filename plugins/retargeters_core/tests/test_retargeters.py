"""Structural reductions, and the refusals that keep them honest."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_retargeters_core import DropDimensions, PoseToPosition

from gantry.resolve import RETARGETER, RetargeterRegistry, bind_channel, requires_channels
from gantry.spine import ChannelSpec

NEED = requires_channels("consumer", "policy")


def joints(width, labels, name="action"):
    return ChannelSpec(
        name,
        "vector",
        (width,),
        "float32",
        units="rad",
        frame="base",
        semantics="joint_position",
        dim_labels=tuple(labels),
    )


SEVEN = joints(7, [f"j{i}" for i in range(7)])
SIX = joints(6, [f"j{i}" for i in range(6)])


# -- dropping dimensions ----------------------------------------------------


def test_it_drops_the_named_dimension():
    retargeter = DropDimensions(["j6"])
    assert retargeter.check(SEVEN, SIX).ok
    values = np.arange(14, dtype=float).reshape(2, 7)
    out = retargeter.apply(values, SEVEN, SIX)
    assert out.shape == (2, 6)
    assert np.array_equal(out, values[:, :6])


def test_the_loss_names_what_went():
    losses = DropDimensions(["j6"]).losses(SEVEN, SIX)
    assert "dropped dimension(s) j6" in losses[0]


def test_an_unlabelled_source_is_refused():
    bare = ChannelSpec("action", "vector", (7,), "float32")
    verdict = DropDimensions(["j6"]).check(bare, SIX)
    assert "retarget.unlabelled" in verdict.codes()


def test_dropping_a_dimension_that_is_not_there_is_refused():
    verdict = DropDimensions(["wrist"]).check(SEVEN, SIX)
    assert "retarget.no_such_dimension" in verdict.codes()
    assert "j0" in verdict.reasons[0].hint


def test_dropping_the_wrong_count_is_refused():
    verdict = DropDimensions(["j5", "j6"]).check(SEVEN, SIX)
    assert "retarget.width" in verdict.codes()


def test_keeping_the_wrong_dimensions_is_refused():
    """Same width, different joints. A width check alone would let this through."""
    other = joints(6, ["j1", "j2", "j3", "j4", "j5", "j6"])
    verdict = DropDimensions(["j0"]).check(SEVEN, other)
    assert verdict.ok
    verdict = DropDimensions(["j6"]).check(SEVEN, other)
    assert "retarget.labels" in verdict.codes()


def test_naming_nothing_to_drop_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        DropDimensions([])


# -- pose to position -------------------------------------------------------


def pose(width, encoding):
    return ChannelSpec(
        "eef",
        "vector",
        (width,),
        "float32",
        units="m",
        frame="base",
        semantics="pose",
        metadata={"rotation_repr": encoding},
    )


POSITION = ChannelSpec("eef", "vector", (3,), "float32", units="m", frame="base", semantics="pose")


def test_it_keeps_the_leading_three():
    retargeter = PoseToPosition()
    source = pose(7, "quat_wxyz")
    assert retargeter.check(source, POSITION).ok
    values = np.arange(14, dtype=float).reshape(2, 7)
    assert np.array_equal(retargeter.apply(values, source, POSITION), values[:, :3])


def test_the_loss_names_the_encoding_that_went():
    losses = PoseToPosition().losses(pose(7, "quat_wxyz"), POSITION)
    assert "quat_wxyz" in losses[0]


@pytest.mark.parametrize("encoding,width", [("quat_wxyz", 7), ("rot6d", 9), ("axis_angle", 6)])
def test_every_encoding_width_is_accepted(encoding, width):
    assert PoseToPosition().check(pose(width, encoding), POSITION).ok


def test_a_width_that_does_not_match_the_encoding_is_refused():
    verdict = PoseToPosition().check(pose(6, "quat_wxyz"), POSITION)
    assert "retarget.pose_width" in verdict.codes()


def test_a_target_that_is_not_a_position_is_refused():
    verdict = PoseToPosition().check(pose(7, "quat_wxyz"), joints(7, [f"j{i}" for i in range(7)]))
    assert "retarget.not_a_position" in verdict.codes()


# -- through the resolver ---------------------------------------------------


def test_the_resolver_finds_a_retargeter_for_a_width_gap():
    """A width gap used to be a flat refusal; now it is one if nothing handles it."""
    registry = RetargeterRegistry([DropDimensions(["j6"])])
    binding, verdict = bind_channel(SIX, [SEVEN], NEED, None, registry)
    assert verdict.ok, verdict.explain()
    assert binding.chain.kinds == (RETARGETER,)
    assert binding.chain.lossy


def test_the_retargeters_loss_reaches_the_binding():
    registry = RetargeterRegistry([DropDimensions(["j6"])])
    binding, _ = bind_channel(SIX, [SEVEN], NEED, None, registry)
    assert "dropped dimension(s) j6" in binding.chain.losses[0]


def test_with_none_installed_the_refusal_says_so():
    binding, verdict = bind_channel(SIX, [SEVEN], NEED, None, RetargeterRegistry())
    assert binding is None
    assert "retarget.none_installed" in verdict.codes()


def test_with_one_that_declines_the_refusal_counts_them():
    registry = RetargeterRegistry([DropDimensions(["wrist"])])
    binding, verdict = bind_channel(SIX, [SEVEN], NEED, None, registry)
    assert binding is None
    assert "retarget.no_match" in verdict.codes()
    assert "1 retargeter(s) declined" in verdict.because("retarget.no_match")[0].hint


def test_a_modality_gap_is_never_retargeted():
    """Nothing turns an image into a vector, and no plugin may claim otherwise."""
    image = ChannelSpec("view", "image", (8, 8, 3), "uint8")
    registry = RetargeterRegistry([DropDimensions(["j6"])])
    binding, verdict = bind_channel(SIX, [image], NEED, None, registry)
    assert binding is None
    assert "resolve.no_candidate" in verdict.codes()


def test_the_data_really_comes_through_reduced():
    from gantry.resolve import adapt_episode, bind
    from gantry.spine import episode_from_arrays

    episode = episode_from_arrays(
        {"action": np.arange(21, dtype="float32").reshape(3, 7)},
        (SEVEN,),
        id="e0",
        source="test",
    )
    requirement = requires_channels("consumer", "policy", SIX)
    wiring, verdict = bind(
        requirement, episode.schema, None, RetargeterRegistry([DropDimensions(["j6"])])
    )
    assert verdict.ok
    reduced = adapt_episode(episode, wiring)
    assert reduced.array("action").shape == (3, 6)
    assert np.array_equal(reduced.array("action"), episode.array("action")[:, :6])
