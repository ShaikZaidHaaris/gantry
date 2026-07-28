from __future__ import annotations

import pytest

from gantry.resolve import (
    AdapterRegistry,
    DuplicateRegistration,
    Registry,
    check_capabilities,
    requires_channels,
    resolve,
)
from gantry.spine import ChannelSpec, Descriptor, IncompatibleError

CONNECTOR = "connector@1.0"


class FakeConnector:
    """A stand-in provider. The resolver only ever reads its descriptor."""

    def __init__(self, *, stage_events=True, outcomes=True, version="1.0", contract=CONNECTOR):
        self._descriptor = Descriptor(
            plane="dataset",
            name="fake",
            version=version,
            contract=contract,
            provides={
                "lazy": True,
                "stage_events": stage_events,
                "outcomes": outcomes,
                "media": False,
            },
        )

    def descriptor(self):
        return self._descriptor


POSITION = ChannelSpec(
    "position", "vector", (3,), "float32", units="m", frame="world", semantics="position"
)


def registry(**kwargs) -> Registry:
    reg = Registry()
    reg.register("dataset", "fake", lambda **config: FakeConnector(**{**kwargs, **config}))
    return reg


def screen():
    return requires_channels("screen", "feedback", POSITION, description="dataset statistics")


def funnel():
    return requires_channels(
        "funnel",
        "feedback",
        POSITION,
        capabilities={"stage_events": True},
        description="conditional stage rates",
    )


def attribution():
    return requires_channels(
        "attribution", "feedback", POSITION, capabilities={"outcomes": True}
    )


# -- registry --------------------------------------------------------------


def test_register_and_look_up():
    reg = registry()
    assert reg.has("dataset", "fake")
    assert reg.names("dataset") == ("fake",)
    assert reg.describe_all() == {"dataset": ["fake"]}


def test_unknown_component_lists_what_is_installed():
    with pytest.raises(KeyError, match="installed on the dataset plane: fake"):
        registry().get("dataset", "nope")


def test_unknown_plane_on_an_empty_registry_says_so():
    with pytest.raises(KeyError, match="nothing is installed on the policy plane"):
        Registry().get("policy", "anything")


def test_double_registration_is_refused_unless_deliberate():
    reg = registry()
    with pytest.raises(DuplicateRegistration, match="replace=True"):
        reg.register("dataset", "fake", lambda: None)
    reg.register("dataset", "fake", lambda: None, replace=True)


def test_unknown_plane_is_rejected():
    with pytest.raises(ValueError, match="unknown plane"):
        Registry().register("vibes", "x", lambda: None)


def test_discovery_finds_an_installed_plugin():
    """The CSV plugin declares an entry point, so it appears without editing core."""
    reg = Registry()
    found = reg.discover()
    assert "dataset:csv" in found
    assert reg.get("dataset", "csv").origin == "entry-point:gantry.connectors"


def test_discovery_does_not_import_the_plugin_until_it_is_built():
    reg = Registry()
    reg.discover()
    registration = reg.get("dataset", "csv")
    assert "lazy" in repr(registration.factory)


# -- capabilities ----------------------------------------------------------


def test_a_capability_that_is_present_passes():
    assert check_capabilities(funnel(), FakeConnector().descriptor()).ok


def test_a_missing_capability_names_the_consumer_and_the_provider():
    verdict = check_capabilities(funnel(), FakeConnector(stage_events=False).descriptor())
    assert not verdict.ok
    reason = verdict.because("resolve.capability")[0]
    assert reason.detail["capabilities"] == ["stage_events"]
    assert "dataset:fake@1.0" in reason.message


# -- whole runs ------------------------------------------------------------


def test_a_satisfiable_run_produces_a_plan():
    resolution = resolve(
        registry(),
        components={"dataset": {"name": "fake"}},
        consumers=[screen(), funnel()],
        provided_channels=[POSITION],
        protocol={"n": 100},
    )
    assert resolution.ok
    plan = resolution.require()
    assert plan.component("dataset").ref == "dataset:fake@1.0"
    assert [w.consumer for w in plan.wirings] == ["screen", "funnel"]


def test_the_plan_determines_provenance_before_anything_runs():
    plan = resolve(
        registry(),
        components={"dataset": {"name": "fake", "config": {"version": "2.0"}}},
        protocol={"n": 100, "chunk": 8},
    ).require()
    provenance = plan.provenance()
    assert provenance.component("dataset").ref == "dataset:fake@2.0"
    assert provenance.protocol == {"n": 100, "chunk": 8}
    assert provenance.digest  # identity exists before a single episode is read


def test_a_missing_component_is_refused_by_name():
    resolution = resolve(registry(), components={"dataset": {"name": "ghost"}})
    assert not resolution.ok
    assert "resolve.not_installed" in resolution.verdict.codes()


def test_an_entry_without_a_name_is_refused():
    resolution = resolve(registry(), components={"dataset": {}})
    assert "resolve.unnamed" in resolution.verdict.codes()


def test_a_component_that_cannot_describe_itself_is_refused_not_crashed():
    reg = Registry()

    def explode(**_):
        raise RuntimeError("no descriptor for you")

    reg.register("dataset", "broken", explode)
    resolution = resolve(reg, components={"dataset": {"name": "broken"}})
    assert "resolve.describe_failed" in resolution.verdict.codes()
    assert "no descriptor for you" in resolution.explain()


def test_a_component_declaring_the_wrong_plane_is_caught():
    reg = Registry()
    reg.register(
        "dataset",
        "confused",
        lambda **_: type(
            "C", (), {"descriptor": lambda self: Descriptor("feedback", "c", "1.0", "x@1.0")}
        )(),
    )
    resolution = resolve(reg, components={"dataset": {"name": "confused"}})
    assert "resolve.plane_mismatch" in resolution.verdict.codes()


def test_an_incompatible_contract_major_is_refused():
    resolution = resolve(
        registry(contract="connector@2.0"), components={"dataset": {"name": "fake"}}
    )
    assert "contract.major" in resolution.verdict.codes()


def _stub(plane: str, name: str, contract: str):
    return lambda **_: type(
        "S", (), {"descriptor": lambda self: Descriptor(plane, name, "1.0", contract)}
    )()


def test_a_plane_with_no_published_contract_says_so_rather_than_pretending():
    """Adapters have no published contract yet, and the resolver admits it
    rather than reporting a check it did not perform."""
    reg = Registry()
    reg.register("adapter", "some-adapter", _stub("adapter", "some-adapter", "adapter@1.0"))
    resolution = resolve(reg, components={"adapter": {"name": "some-adapter"}})
    assert resolution.ok
    assert "resolve.contract_unpublished" in resolution.verdict.codes()


@pytest.mark.parametrize(
    "plane,contract",
    [
        ("policy", "policy@1.0"),
        ("evaluation", "evaluator@1.1"),
        ("embodiment", "embodiment@1.0"),
        ("feedback", "feedback@1.0"),
    ],
)
def test_every_published_plane_is_version_checked(plane, contract):
    reg = Registry()
    reg.register(plane, "good", _stub(plane, "good", contract))
    reg.register(plane, "stale", _stub(plane, "stale", contract.split("@")[0] + "@0.9"))

    assert resolve(reg, components={plane: {"name": "good"}}).ok
    stale = resolve(reg, components={plane: {"name": "stale"}})
    assert "contract.major" in stale.verdict.codes()


# -- the refusal that names alternatives -----------------------------------


def test_an_unsatisfiable_consumer_names_the_ones_that_would_run():
    """The flagship refusal: say what is impossible *and* what is not."""
    resolution = resolve(
        registry(stage_events=False),
        components={"dataset": {"name": "fake"}},
        consumers=[screen(), funnel(), attribution()],
        provided_channels=[POSITION],
    )
    assert not resolution.ok
    assert resolution.alternatives == ("screen", "attribution")

    text = resolution.explain()
    assert "funnel needs stage_events" in text
    assert "runnable as requested: screen, attribution" in text


def test_requiring_a_refused_plan_raises_with_the_whole_explanation():
    resolution = resolve(
        registry(stage_events=False),
        components={"dataset": {"name": "fake"}},
        consumers=[funnel()],
        provided_channels=[POSITION],
    )
    with pytest.raises(IncompatibleError, match="cannot plan this run"):
        resolution.require()


def test_consumers_without_a_provider_are_refused():
    resolution = resolve(Registry(), components={}, consumers=[screen()])
    assert "resolve.no_provider" in resolution.verdict.codes()


# -- losses travel ---------------------------------------------------------


def test_a_lossy_adapter_reaches_the_plan_and_then_provenance():
    from gantry.resolve import Adapter

    resampler = Adapter(
        "resample",
        "1.0",
        closes=("rate.mismatch",),
        cost=lambda p, c: ("30 Hz -> 20 Hz",),
        transform=lambda values, provider, consumer: values,
        preserves_length=False,
    )
    resolution = resolve(
        registry(),
        components={"dataset": {"name": "fake"}},
        consumers=[
            requires_channels("screen", "feedback", ChannelSpec(
                "position", "vector", (3,), "float32", units="m", frame="world",
                semantics="position", rate_hz=20.0,
            ))
        ],
        provided_channels=[
            ChannelSpec("position", "vector", (3,), "float32", units="m", frame="world",
                        semantics="position", rate_hz=30.0)
        ],
        adapters=AdapterRegistry([resampler]),
    )
    plan = resolution.require()
    assert plan.lossy
    assert plan.provenance().losses == ("30 Hz -> 20 Hz",)
    assert "resample is lossy: 30 Hz -> 20 Hz" in plan.explain()


def test_a_direct_plan_is_not_lossy():
    plan = resolve(
        registry(),
        components={"dataset": {"name": "fake"}},
        consumers=[screen()],
        provided_channels=[POSITION],
    ).require()
    assert not plan.lossy
    assert plan.provenance().losses == ()
