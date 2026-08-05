from __future__ import annotations

import numpy as np

from gantry.spine import ChannelSpec, compatible, register_semantics, units


def spec(**overrides) -> ChannelSpec:
    base = dict(name="state", kind="vector", shape=(7,), dtype="float32")
    return ChannelSpec(**{**base, **overrides})


# -- validation ------------------------------------------------------------


def test_valid_spec_passes():
    assert spec(units="m", semantics="position").validate().ok


def test_unknown_kind_is_refused_with_the_list():
    verdict = spec(kind="hologram").validate()
    assert not verdict.ok
    assert "kind.unknown" in verdict.codes()
    assert "vector" in verdict.reasons[0].hint


def test_rank_must_match_kind():
    verdict = spec(kind="scalar", shape=(7,)).validate()
    assert "kind.rank" in verdict.codes()


def test_units_must_match_semantic_dimension():
    verdict = spec(semantics="position", units="s").validate()
    assert "units.dimension" in verdict.codes()


def test_unregistered_semantics_is_a_note_not_a_failure():
    verdict = spec(semantics="tentacle_curl").validate()
    assert verdict.ok
    assert "semantics.unregistered" in verdict.codes()


def test_plugin_can_register_its_own_semantics_and_get_checking_for_free():
    register_semantics("tentacle_curl", "curl of a soft actuator", units.ANGLE)
    assert spec(semantics="tentacle_curl", units="rad").validate().ok
    # once registered, its dimension is enforced like any built-in tag
    verdict = spec(semantics="tentacle_curl", units="m").validate()
    assert "units.dimension" in verdict.codes()


# -- data agreement --------------------------------------------------------


def test_accepts_matching_array():
    assert spec().accepts(np.zeros((10, 7), dtype="float32")).ok


def test_rejects_wrong_width():
    verdict = spec().accepts(np.zeros((10, 6), dtype="float32"))
    assert "data.shape" in verdict.codes()


def test_rejects_wrong_kind_of_dtype():
    verdict = spec().accepts(np.zeros((10, 7), dtype="int32").astype("O"))
    assert "data.dtype" in verdict.codes()


def test_wildcard_width_accepts_anything():
    assert (
        ChannelSpec("img", "image", (None, None, 3), "uint8")
        .accepts(np.zeros((4, 64, 48, 3), dtype="uint8"))
        .ok
    )


# -- compatibility ---------------------------------------------------------


def test_identical_specs_are_compatible():
    assert compatible(spec(), spec()).ok


def test_wildcard_consumer_accepts_concrete_provider():
    assert compatible(spec(shape=(7,)), spec(shape=(None,))).ok


def test_mismatches_are_itemised_not_collapsed():
    provider = spec(units="mm", frame="base", rate_hz=30.0)
    consumer = spec(units="m", frame="world", rate_hz=20.0)
    verdict = compatible(provider, consumer)
    assert not verdict.ok
    assert set(verdict.codes()) >= {"units.scale", "frame.mismatch", "rate.mismatch"}


def test_scale_mismatch_reports_the_factor():
    verdict = compatible(spec(units="mm"), spec(units="m"))
    reason = verdict.because("units.scale")[0]
    assert reason.detail["factor"] == 1e-3


def test_different_quantities_say_no_scaling_helps():
    verdict = compatible(spec(units="m"), spec(units="s"))
    assert "units.dimension" in verdict.codes()
    assert "no scaling" in verdict.because("units.dimension")[0].hint


def test_one_sided_declaration_is_a_note():
    verdict = compatible(spec(frame="base"), spec(frame=None))
    assert verdict.ok
    assert "frame.undeclared" in verdict.codes()
