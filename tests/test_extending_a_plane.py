"""Can somebody add a whole new plane without editing core?

This is the claim that separates "swappable parts" from "extensible framework",
and it is easy to believe without checking. Everything below registers a plane
Gantry has never heard of and then uses it exactly as if it were built in — the
registry accepts it, descriptors validate against it, provenance records it,
manifests carry it, the resolver version-checks it, and the CLI lists it.

If any of that needed a core edit, the modularity claim would be about the five
planes that happen to exist rather than about planes.
"""

from __future__ import annotations

import pytest

from gantry.manifest import Manifest
from gantry.resolve import Registry
from gantry.runner import check_manifest
from gantry.spine import (
    CORE_PLANES,
    ComponentRef,
    Descriptor,
    Plane,
    contract_for,
    entry_point_groups,
    get_plane,
    known_planes,
    register_plane,
)

CURATION = Plane(
    name="curation",
    description="chooses which episodes are worth keeping",
    contract="curator@1.0",
    entry_point_group="gantry.curators",
)


@pytest.fixture(autouse=True)
def plane():
    register_plane(CURATION)
    return CURATION


class Curator:
    """A component on a plane core has never heard of."""

    def descriptor(self) -> Descriptor:
        return Descriptor(
            plane="curation",
            name="keep-the-best",
            version="1.0",
            contract="curator@1.0",
            provides={"ranks": True},
        )


# -- registration -----------------------------------------------------------


def test_the_plane_registers_and_is_not_a_core_plane():
    assert "curation" in known_planes()
    assert "curation" not in CORE_PLANES


def test_registering_it_twice_is_fine_and_redefining_it_is_not():
    register_plane(CURATION)
    with pytest.raises(ValueError, match="cannot be redefined"):
        register_plane(Plane("curation", "something else entirely"))


def test_its_entry_point_group_becomes_discoverable():
    assert entry_point_groups()["gantry.curators"] == "curation"


def test_its_contract_is_looked_up_like_any_other():
    assert contract_for("curation") == "curator@1.0"


def test_it_is_fully_described():
    described = get_plane("curation")
    assert described.description and not described.many


# -- it works everywhere a plane works --------------------------------------


def test_a_descriptor_on_the_new_plane_validates():
    assert Curator().descriptor().validate().ok


def test_provenance_accepts_it():
    ref = ComponentRef("curation", "keep-the-best", "1.0")
    assert ref.ref == "curation:keep-the-best@1.0"


def test_an_unregistered_plane_is_still_refused():
    with pytest.raises(ValueError, match="unknown plane"):
        ComponentRef("astrology", "x", "1.0")


def test_the_registry_accepts_components_on_it():
    registry = Registry()
    registry.register("curation", "keep-the-best", lambda **_: Curator())
    assert registry.names("curation") == ("keep-the-best",)
    assert registry.describe_all()["curation"] == ["keep-the-best"]


def test_a_manifest_carries_it_with_no_manifest_change():
    manifest = Manifest.from_dict(
        {
            "name": "curated",
            "cohorts": {"a": "csv"},
            "curation": {"name": "keep-the-best", "config": {"top": 100}},
            "feedback": ["screen"],
        }
    )
    assert manifest.components["curation"].name == "keep-the-best"
    assert manifest.components["curation"].config == {"top": 100}


def test_it_round_trips_through_the_manifest_file():
    original = Manifest.from_dict(
        {"name": "curated", "cohorts": {"a": "csv"}, "curation": "keep-the-best"}
    )
    import json

    assert Manifest.from_dict(json.loads(original.to_json())) == original


def test_a_missing_component_on_the_new_plane_is_reported_like_any_other():
    manifest = Manifest.from_dict(
        {"name": "curated", "cohorts": {"a": "csv"}, "curation": "not-installed"}
    )
    verdict = check_manifest(manifest, Registry())
    assert "manifest.not_installed" in verdict.codes()
    assert any(
        r.detail.get("plane") == "curation"
        for r in verdict.because("manifest.not_installed")
    )


def test_the_resolver_version_checks_it():
    from gantry.resolve import resolve

    registry = Registry()
    registry.register("curation", "stale", lambda **_: _StaleCurator())
    resolution = resolve(registry, components={"curation": {"name": "stale"}})
    assert "contract.major" in resolution.verdict.codes()


class _StaleCurator:
    def descriptor(self) -> Descriptor:
        return Descriptor("curation", "stale", "1.0", "curator@0.9")


def test_a_typo_in_a_manifest_key_is_refused_rather_than_ignored():
    """Silently dropping an unrecognised key is how a whole plane goes missing."""
    from gantry.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown key"):
        Manifest.from_dict({"name": "x", "cohorts": {"a": "csv"}, "curaton": "typo"})
