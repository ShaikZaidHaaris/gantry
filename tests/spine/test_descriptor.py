from __future__ import annotations

import pytest

from gantry.spine import ContractVersion, Descriptor


def make(**overrides) -> Descriptor:
    base = dict(
        plane="dataset",
        name="example",
        version="1.2.0",
        contract="connector@1.0",
        provides={"stage_events": False, "lazy": True},
    )
    return Descriptor(**{**base, **overrides})


def test_round_trips_through_json():
    original = make()
    assert Descriptor.from_json(original.to_json()) == original


def test_ref_is_stable_and_readable():
    assert make().ref == "dataset:example@1.2.0"


def test_unknown_plane_is_rejected():
    with pytest.raises(ValueError, match="unknown plane"):
        make(plane="vibes")


def test_unknown_isolation_is_rejected():
    with pytest.raises(ValueError, match="unknown isolation"):
        make(isolation="hope")


def test_missing_fields_name_themselves():
    with pytest.raises(ValueError, match="missing required field"):
        Descriptor.from_dict({"plane": "dataset", "name": "x"})


# -- capability matching ---------------------------------------------------


def test_missing_capability_lists_what_is_available():
    verdict = make().provides_all({"stage_events": None, "teleport": None})
    assert not verdict.ok
    assert verdict.because("capability.missing")[0].detail["capability"] == "teleport"
    assert "lazy" in verdict.because("capability.missing")[0].hint


def test_capability_value_must_match_when_specified():
    assert make().provides_all({"lazy": True}).ok
    assert "capability.value" in make().provides_all({"stage_events": True}).codes()


def test_presence_only_check_ignores_the_value():
    assert make().provides_all({"stage_events": None}).ok


# -- contract semver -------------------------------------------------------


def test_same_major_and_high_enough_minor_satisfies():
    assert ContractVersion.parse("policy@2.3").satisfies(ContractVersion.parse("policy@2.1")).ok


def test_older_minor_is_refused_with_a_reason():
    verdict = ContractVersion.parse("policy@2.0").satisfies(ContractVersion.parse("policy@2.4"))
    assert "contract.minor" in verdict.codes()


def test_major_bump_is_a_break():
    verdict = ContractVersion.parse("policy@1.9").satisfies(ContractVersion.parse("policy@2.0"))
    assert "contract.major" in verdict.codes()
    assert "breaking change" in verdict.reasons[0].hint


def test_different_contracts_never_satisfy_each_other():
    verdict = ContractVersion.parse("policy@1.0").satisfies(ContractVersion.parse("connector@1.0"))
    assert "contract.name" in verdict.codes()


def test_malformed_contract_is_rejected_at_construction():
    with pytest.raises(ValueError, match="malformed contract version"):
        make(contract="just-a-name")


# -- provenance stamping ---------------------------------------------------


def test_component_ref_carries_config_and_artifact_digests():
    ref = make().component_ref(config={"path": "/data"}, artifact_digest="deadbeef")
    assert ref.plane == "dataset"
    assert ref.artifact_digest == "deadbeef"
    assert ref.config_digest and len(ref.config_digest) == 16
    assert ref.ref.endswith("+deadbeef")


def test_config_digest_changes_with_config():
    a = make().component_ref(config={"path": "/a"})
    b = make().component_ref(config={"path": "/b"})
    assert a.config_digest != b.config_digest
