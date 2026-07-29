"""Adapters: two that cost nothing, one that costs something and says so."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_adapters_core import (
    PERMUTE,
    RESAMPLE,
    UNIT_CONVERT,
    default_registry,
    permutation_between,
    resample_to,
)

from gantry.resolve import bind_channel, requires_channels
from gantry.spine import ChannelSpec


def spec(**over):
    base = dict(name="position", kind="vector", shape=(3,), dtype="float32",
                units="m", frame="world", rate_hz=20.0, semantics="position")
    return ChannelSpec(**{**base, **over})


NEED = requires_channels("consumer", "feedback")


# -- units -----------------------------------------------------------------


def test_conversion_is_exact_and_declares_no_loss():
    values = np.array([[1000.0, 2000.0, 3000.0]])
    out = UNIT_CONVERT.transform(values, spec(units="mm"), spec(units="m"))
    assert np.allclose(out, [[1.0, 2.0, 3.0]])
    assert UNIT_CONVERT.losses(spec(units="mm"), spec(units="m")) == ()


def test_conversion_round_trips():
    values = np.random.default_rng(0).normal(size=(20, 3))
    once = UNIT_CONVERT.transform(values, spec(units="m"), spec(units="mm"))
    back = UNIT_CONVERT.transform(once, spec(units="mm"), spec(units="m"))
    assert np.allclose(back, values)


def test_it_declines_undeclared_units():
    assert "adapter.units_undeclared" in UNIT_CONVERT.applies(spec(units=None), spec()).codes()


def test_it_declines_different_quantities():
    verdict = UNIT_CONVERT.applies(spec(units="m", semantics=None), spec(units="s", semantics=None))
    assert not verdict.ok


# -- permutation -----------------------------------------------------------


def test_a_permutation_reorders_columns():
    source = spec(dim_labels=("a", "b", "c"))
    target = spec(dim_labels=("c", "a", "b"))
    assert permutation_between(source, target) == (2, 0, 1)
    out = PERMUTE.transform(np.array([[1.0, 2.0, 3.0]]), source, target)
    assert np.allclose(out, [[3.0, 1.0, 2.0]])


def test_a_permutation_is_lossless():
    assert PERMUTE.losses(spec(dim_labels=("a", "b", "c")), spec(dim_labels=("c", "b", "a"))) == ()


def test_different_label_sets_are_not_a_permutation():
    verdict = PERMUTE.applies(spec(dim_labels=("a", "b", "c")), spec(dim_labels=("x", "y", "z")))
    assert "adapter.not_a_permutation" in verdict.codes()
    assert "retargeter" in verdict.reasons[0].hint


# -- resampling ------------------------------------------------------------


def test_downsampling_says_what_it_threw_away():
    losses = RESAMPLE.losses(spec(rate_hz=30.0), spec(rate_hz=20.0))
    assert losses and "67% of the timeline" in losses[0]
    assert "detail faster than 20 Hz is gone" in losses[0]


def test_upsampling_says_the_new_samples_are_invented():
    losses = RESAMPLE.losses(spec(rate_hz=10.0), spec(rate_hz=20.0))
    assert "invented, not measured" in losses[0]


def test_resampling_preserves_the_endpoints():
    values = np.linspace(0, 1, 31).reshape(31, 1)
    out = resample_to(values, spec(shape=(1,), rate_hz=30.0), spec(shape=(1,), rate_hz=20.0))
    assert out[0, 0] == pytest.approx(0.0) and out[-1, 0] == pytest.approx(1.0)


def test_resampling_lands_on_the_right_length():
    values = np.zeros((31, 3))
    out = resample_to(values, spec(rate_hz=30.0), spec(rate_hz=20.0))
    assert len(out) == 21  # one second at 20 Hz, endpoints included


def test_an_equal_rate_is_a_no_op():
    values = np.random.default_rng(0).normal(size=(10, 3))
    assert np.array_equal(resample_to(values, spec(), spec()), values)


def test_it_declines_a_modality_it_cannot_interpolate():
    image = ChannelSpec("view", "image", (8, 8, 3), "uint8", rate_hz=30.0)
    target = ChannelSpec("view", "image", (8, 8, 3), "uint8", rate_hz=20.0)
    verdict = RESAMPLE.applies(image, target)
    assert "adapter.rate_kind" in verdict.codes()
    assert "purpose-built" in verdict.reasons[0].hint


# -- through the resolver --------------------------------------------------


def test_the_resolver_finds_and_plans_them():
    registry = default_registry()
    assert registry.codes() == ("dim_labels.order", "rate.mismatch", "units.scale")

    binding, verdict = bind_channel(
        spec(units="m", rate_hz=20.0), [spec(units="mm", rate_hz=30.0)], NEED, registry
    )
    assert verdict.ok
    assert len(binding.chain) == 2


def test_the_loss_of_the_lossy_one_reaches_the_binding():
    binding, _ = bind_channel(
        spec(rate_hz=20.0), [spec(rate_hz=30.0)], NEED, default_registry()
    )
    assert binding.chain.lossy
    assert "detail faster than 20 Hz is gone" in binding.chain.losses[0]


def test_a_width_change_is_still_refused_with_every_adapter_installed():
    """No adapter closes a modality or width gap, and none pretends to."""
    binding, verdict = bind_channel(
        spec(shape=(7,)), [spec(shape=(3,))], NEED, default_registry()
    )
    assert binding is None
    assert "resolve.channel_incompatible" in verdict.codes()
