"""Describing a machine, and refusing to invent what nobody wrote down."""

from __future__ import annotations

import json

import pytest
from gantry_embodiment_declared import from_file, from_mapping, from_schema, to_file

from gantry.contracts.embodiment import EmbodimentDescriptor
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec, IncompatibleError

SPEC = {
    "name": "seven-axis",
    "version": "1.0",
    "control_hz": 20,
    "state": [
        {
            "name": "joint_pos",
            "kind": "vector",
            "shape": [7],
            "units": "rad",
            "semantics": "joint_position",
        }
    ],
    "action": [
        {
            "name": "action",
            "kind": "vector",
            "shape": [7],
            "units": "rad",
            "dim_labels": ["j1", "j2", "j3", "j4", "j5", "j6", "grip"],
        }
    ],
    "kinematics": "arm.urdf",
    "notes": "gripper: 1 open, -1 closed",
    "capabilities": ["resettable", "simulated"],
}


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(SPEC))
    return path


# -- from a file ------------------------------------------------------------


def test_a_file_describes_a_machine(spec_file):
    arm = from_file(spec_file)
    assert isinstance(arm, EmbodimentDescriptor)
    assert arm.ref == "embodiment:seven-axis@1.0"
    assert arm.control_hz == 20
    assert arm.kinematics == "arm.urdf"
    assert "resettable" in arm.capabilities
    assert arm.channel("action").dim_labels[-1] == "grip"
    assert arm.validate().ok


def test_the_descriptor_reaches_the_registry_shape(spec_file):
    descriptor = from_file(spec_file).descriptor()
    assert descriptor.plane == "embodiment"
    assert descriptor.provides["action_width"] == 7
    assert descriptor.provides["control_hz"] == 20


def test_it_records_where_the_description_came_from(spec_file):
    assert str(spec_file) in from_file(spec_file).metadata["described_by"]


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ConfigError, match="no embodiment description"):
        from_file(tmp_path / "nope.json")


def test_malformed_json_says_so(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        from_file(path)


def test_a_description_without_a_name_is_refused():
    with pytest.raises(ConfigError, match="needs a 'name'"):
        from_mapping({"version": "1.0"})


def test_a_channel_without_a_name_is_refused():
    with pytest.raises(ConfigError, match="needs a 'name'"):
        from_mapping({"name": "x", "version": "1", "action": [{"shape": [3]}]})


def test_an_incoherent_machine_is_refused():
    with pytest.raises(IncompatibleError):
        from_mapping(
            {
                "name": "x",
                "version": "1",
                "control_hz": -5,
                "action": [{"name": "a", "kind": "vector", "shape": [3]}],
            }
        )


# -- from a dataset's schema ------------------------------------------------


SCHEMA = (
    ChannelSpec(
        "observation.state",
        "vector",
        (8,),
        "float32",
        rate_hz=20.0,
        dim_labels=("x", "y", "z", "roll", "pitch", "yaw", "g.0", "g.1"),
    ),
    ChannelSpec(
        "action",
        "vector",
        (7,),
        "float32",
        rate_hz=20.0,
        dim_labels=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
    ),
)


def test_a_schema_gives_a_first_draft():
    arm = from_schema(SCHEMA, name="lift-arm")
    assert arm.channel("action").shape == (7,)
    assert arm.channel("observation.state").dim_labels[0] == "x"
    assert arm.control_hz == 20.0, "one declared rate, so it is the control rate"
    assert arm.validate().ok


def test_a_derived_machine_says_it_has_no_limits():
    """A recording shows what was done, not what is allowed."""
    arm = from_schema(SCHEMA, name="lift-arm")
    assert arm.metadata["derived_from_recording"] is True
    assert "not established" in arm.metadata["limits"]


def test_deriving_without_an_action_channel_is_refused():
    with pytest.raises(ConfigError, match="no action channel"):
        from_schema(SCHEMA[:1], name="stateless")


def test_conflicting_rates_leave_the_control_rate_unstated():
    mixed = (
        SCHEMA[1],
        ChannelSpec("slow", "vector", (2,), "float32", rate_hz=5.0),
    )
    assert from_schema(mixed, name="mixed").control_hz is None


def test_a_derived_description_can_be_written_down_and_read_back(tmp_path):
    """The intended path from cheap to real: derive, edit, then rely on it."""
    derived = from_schema(SCHEMA, name="lift-arm")
    path = to_file(derived, tmp_path / "lift.json")
    again = from_file(path)
    assert again.name == derived.name
    assert again.channel("action").dim_labels == derived.channel("action").dim_labels
    assert again.control_hz == derived.control_hz


# -- the check this plane exists for ---------------------------------------


def test_a_policy_that_cannot_command_the_machine_is_refused():
    """The whole point of the plane: catch it before the first step."""
    from gantry.contracts.policy import policy_descriptor
    from gantry.resolve import AdapterRegistry, RetargeterRegistry
    from gantry.runner import check_policy_against_embodiment

    arm = from_schema(SCHEMA, name="lift-arm")

    class Policy:
        name = "wrong-width"

        def descriptor(self):
            return policy_descriptor("wrong-width", "1", chunk=1, deterministic=True)

        def action_spec(self):
            return ChannelSpec("action", "vector", (3,), "float32")

    verdict = check_policy_against_embodiment(
        Policy(), arm, AdapterRegistry(), RetargeterRegistry()
    )
    assert not verdict.ok
    assert "runner.policy_embodiment" in verdict.codes()


def test_a_policy_that_fits_is_accepted():
    from gantry.contracts.policy import policy_descriptor
    from gantry.resolve import AdapterRegistry, RetargeterRegistry
    from gantry.runner import check_policy_against_embodiment

    arm = from_schema(SCHEMA, name="lift-arm")

    class Policy:
        name = "fits"

        def descriptor(self):
            return policy_descriptor("fits", "1", chunk=1, deterministic=True)

        def action_spec(self):
            return SCHEMA[1]

    verdict = check_policy_against_embodiment(
        Policy(), arm, AdapterRegistry(), RetargeterRegistry()
    )
    assert verdict.ok, verdict.explain()
