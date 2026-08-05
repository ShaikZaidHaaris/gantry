from __future__ import annotations

import pytest

from gantry.resolve import Adapter, AdapterRegistry, Chain, bind, bind_channel, requires_channels
from gantry.spine import ChannelSpec, Verdict, units


def provided(**overrides) -> ChannelSpec:
    base = dict(
        name="position",
        kind="vector",
        shape=(3,),
        dtype="float32",
        units="m",
        frame="world",
        rate_hz=20.0,
        semantics="position",
    )
    return ChannelSpec(**{**base, **overrides})


def want(**overrides) -> ChannelSpec:
    return provided(**overrides)


def need(*channels, **kwargs):
    return requires_channels("consumer", "feedback", *channels, **kwargs)


# Stand-ins for the real adapters. They carry a transform because the registry
# refuses one that cannot be applied -- an adapter that closes a gap on paper and
# not in the data is exactly what that guard exists to stop.
def _identity(values, provider, consumer):
    return values


UNIT_CONVERTER = Adapter(
    name="unit-convert",
    version="1.0",
    closes=("units.scale",),
    cost=lambda p, c: (),  # exact rescaling loses nothing
    transform=_identity,
)

RESAMPLER = Adapter(
    name="resample",
    version="1.0",
    closes=("rate.mismatch",),
    cost=lambda p, c: (f"resampled {p.rate_hz} Hz -> {c.rate_hz} Hz",),
    transform=_identity,
    preserves_length=False,
)


# -- direct matches --------------------------------------------------------


def test_identical_channels_bind_directly():
    binding, verdict = bind_channel(want(), [provided()], need(), AdapterRegistry())
    assert verdict.ok
    assert binding.direct and binding.matched_by == "name"


def test_a_wildcard_want_accepts_a_concrete_channel():
    binding, _ = bind_channel(want(shape=(None,)), [provided()], need(), AdapterRegistry())
    assert binding is not None and binding.direct


def test_an_alias_binds_a_differently_named_channel():
    requirement = need(aliases={"position": ("eef_xyz",)})
    binding, verdict = bind_channel(
        want(), [provided(name="eef_xyz")], requirement, AdapterRegistry()
    )
    assert verdict.ok
    assert binding.matched_by == "alias"
    assert str(binding) == "eef_xyz -> position"


def test_meaning_binds_when_names_differ_and_semantics_agree():
    binding, verdict = bind_channel(want(), [provided(name="tool_tip")], need(), AdapterRegistry())
    assert verdict.ok
    assert binding.matched_by == "semantics"


def test_name_beats_meaning_when_both_are_available():
    candidates = [provided(name="tool_tip"), provided(name="position")]
    binding, _ = bind_channel(want(), candidates, need(), AdapterRegistry())
    assert binding.provided.name == "position"


# -- refusals --------------------------------------------------------------


def test_nothing_matching_is_refused_with_what_was_available():
    binding, verdict = bind_channel(
        want(name="torque", semantics=None), [provided()], need(), AdapterRegistry()
    )
    assert binding is None
    assert "resolve.no_candidate" in verdict.codes()
    assert "available: position" in verdict.reasons[0].hint


def test_shape_alone_never_matches():
    """Two three-wide vectors are not the same thing just because they fit."""
    wrist = provided(name="wrist", semantics=None)
    binding, verdict = bind_channel(
        want(name="base", semantics=None), [wrist], need(), AdapterRegistry()
    )
    assert binding is None
    assert "resolve.no_candidate" in verdict.codes()


def test_width_mismatch_cannot_be_adapted_away():
    everything = AdapterRegistry([UNIT_CONVERTER, RESAMPLER])
    binding, verdict = bind_channel(want(shape=(7,)), [provided(shape=(3,))], need(), everything)
    assert binding is None
    assert "resolve.channel_incompatible" in verdict.codes()
    assert "no adapter can change" in verdict.because("resolve.channel_incompatible")[0].hint


def test_a_gap_with_no_adapter_names_the_code_to_install():
    binding, verdict = bind_channel(
        want(rate_hz=20.0), [provided(rate_hz=30.0)], need(), AdapterRegistry()
    )
    assert binding is None
    reason = verdict.because("resolve.no_adapter")[0]
    assert reason.detail["unclosed"] == ["rate.mismatch"]
    assert "'rate.mismatch'" in reason.hint


# -- adapters --------------------------------------------------------------


def test_an_adapter_closes_a_gap_and_is_planned_in():
    registry = AdapterRegistry([UNIT_CONVERTER])
    binding, verdict = bind_channel(want(units="m"), [provided(units="mm")], need(), registry)
    assert verdict.ok
    assert not binding.direct
    assert str(binding.chain) == "unit-convert@1.0"


def test_a_lossy_adapter_states_its_loss():
    registry = AdapterRegistry([RESAMPLER])
    binding, _ = bind_channel(want(rate_hz=20.0), [provided(rate_hz=30.0)], need(), registry)
    assert binding.chain.lossy
    assert binding.chain.losses == ("resampled 30.0 Hz -> 20.0 Hz",)


def test_several_gaps_need_several_adapters():
    registry = AdapterRegistry([UNIT_CONVERTER, RESAMPLER])
    binding, verdict = bind_channel(
        want(units="m", rate_hz=20.0),
        [provided(units="mm", rate_hz=30.0)],
        need(),
        registry,
    )
    assert verdict.ok
    assert len(binding.chain) == 2


def test_a_partly_closed_gap_is_no_better_than_an_open_one():
    """Half a bridge is not a bridge: adapting only the units still delivers
    data at the wrong rate, which would be silently wrong rather than refused."""
    registry = AdapterRegistry([UNIT_CONVERTER])
    binding, verdict = bind_channel(
        want(units="m", rate_hz=20.0),
        [provided(units="mm", rate_hz=30.0)],
        need(),
        registry,
    )
    assert binding is None
    assert verdict.because("resolve.no_adapter")[0].detail["unclosed"] == ["rate.mismatch"]


def test_an_adapter_that_declines_this_pair_is_not_used():
    picky = Adapter(
        "picky",
        "1.0",
        closes=("units.scale",),
        guard=lambda p, c: Verdict.no("nope", "only converts to metres"),
        transform=_identity,
    )
    binding, verdict = bind_channel(
        want(units="cm"), [provided(units="mm")], need(), AdapterRegistry([picky])
    )
    assert binding is None
    assert "resolve.no_adapter" in verdict.codes()


def test_different_quantities_are_never_adaptable_even_with_a_converter():
    registry = AdapterRegistry([UNIT_CONVERTER])
    binding, verdict = bind_channel(
        want(units="s", semantics=None),
        [provided(units="m", semantics=None)],
        need(),
        registry,
    )
    assert binding is None
    assert "units.dimension" in verdict.codes()


# -- whole consumers -------------------------------------------------------


def test_binding_every_channel_of_a_consumer():
    requirement = need(
        want(),
        want(
            name="engagement",
            kind="scalar",
            shape=(),
            units="fraction",
            semantics="actuation",
            frame=None,
        ),
    )
    wiring, verdict = bind(
        requirement,
        [
            provided(),
            provided(
                name="engagement",
                kind="scalar",
                shape=(),
                units="fraction",
                semantics="actuation",
                frame=None,
            ),
        ],
    )
    assert verdict.ok
    assert len(wiring.bindings) == 2


def test_an_optional_channel_may_simply_be_absent():
    requirement = need(
        want(),
        want(
            name="view",
            kind="image",
            shape=(None, None, 3),
            dtype="uint8",
            units=None,
            frame=None,
            semantics=None,
            optional=True,
        ),
    )
    wiring, verdict = bind(requirement, [provided()])
    assert verdict.ok
    assert wiring.skipped == ("view",)


def test_a_required_channel_may_not():
    requirement = need(
        want(),
        want(
            name="view",
            kind="image",
            shape=(None, None, 3),
            dtype="uint8",
            units=None,
            frame=None,
            semantics=None,
        ),
    )
    wiring, verdict = bind(requirement, [provided()])
    assert wiring is None
    assert "resolve.no_candidate" in verdict.codes()


def test_one_sided_declarations_pass_but_are_reported():
    binding, verdict = bind_channel(
        want(frame=None), [provided(frame="world")], need(), AdapterRegistry()
    )
    assert verdict.ok
    assert "frame.undeclared" in verdict.codes()


# -- adapter registry ------------------------------------------------------


def test_registry_indexes_by_code():
    registry = AdapterRegistry([UNIT_CONVERTER, RESAMPLER])
    assert registry.codes() == ("rate.mismatch", "units.scale")
    assert len(registry) == 2
    assert registry.for_code("units.scale") == (UNIT_CONVERTER,)


def test_an_adapter_that_cannot_be_applied_is_refused_at_registration():
    paper_only = Adapter("paper", "1.0", closes=("units.scale",))
    with pytest.raises(ValueError, match="on paper and not in the data"):
        AdapterRegistry([paper_only])


def test_an_empty_chain_is_falsy_and_reads_as_direct():
    assert not Chain()
    assert str(Chain()) == "direct"


def test_unit_conversion_factor_is_available_to_an_adapter():
    """The spine already computed the factor; an adapter need not re-derive it."""
    assert units.conversion_factor("mm", "m") == pytest.approx(1e-3)
