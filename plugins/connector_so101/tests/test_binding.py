"""The width guard, and what the binding buys that a plain LeRobot read does not.

A recorded dataset is float arrays. Everything asserted here is a claim the
LeRobot reader is right not to make and this one is only allowed to make because
it checks it first.
"""

from __future__ import annotations

import pytest
from gantry_connector_lerobot import LeRobotConnector
from gantry_connector_so101 import BIMANUAL_REF, FOLLOWER_REF, SO101Connector
from gantry_connector_so101.fixtures import write_corpus

from gantry.errors import ConfigError
from gantry.fixtures import make_clean


@pytest.fixture
def single(tmp_path):
    return write_corpus(make_clean(n=3).episodes, tmp_path / "single")


@pytest.fixture
def bimanual(tmp_path):
    return write_corpus(make_clean(n=3).episodes, tmp_path / "bimanual", embodiment=BIMANUAL_REF)


# -- the guard, both directions --------------------------------------------


def test_a_bimanual_corpus_opened_as_one_arm_is_refused(bimanual):
    with pytest.raises(ConfigError, match="12 wide but embodiment:so101_follower@1.0"):
        SO101Connector(bimanual, embodiment=FOLLOWER_REF)


def test_a_single_arm_corpus_opened_as_bimanual_is_refused(single):
    with pytest.raises(ConfigError, match="6 wide but embodiment:so101_bimanual@1.0"):
        SO101Connector(single, embodiment=BIMANUAL_REF)


def test_the_refusal_names_the_embodiment_the_width_actually_is(bimanual):
    with pytest.raises(ConfigError, match="A 12-wide recording is embodiment:so101_bimanual@1.0"):
        SO101Connector(bimanual, embodiment=FOLLOWER_REF)


def test_the_guard_fires_on_the_declared_width_before_any_data_is_read(tmp_path):
    """The metadata lies and the parquet is fine; the description is what binds."""
    root = write_corpus(
        make_clean(n=3).episodes, tmp_path / "mislabelled", widths={"action": 12}
    )
    # The plain reader is content: 12 is a width like any other.
    assert LeRobotConnector(root).schema("episode_000000")[0] is not None
    with pytest.raises(ConfigError, match="12 wide"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_a_permuted_joint_order_is_refused_as_a_permutation(tmp_path):
    """Same names, same width, same units — and every joint driven to another's target."""
    from gantry_evaluator_so101.embodiment import JOINT_NAMES

    swapped = (JOINT_NAMES[1], JOINT_NAMES[0], *JOINT_NAMES[2:])
    root = write_corpus(make_clean(n=3).episodes, tmp_path / "swapped", dim_labels=swapped)
    with pytest.raises(ConfigError, match="a permutation, not a rename"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_foreign_dimension_names_are_refused_as_a_different_machine(tmp_path):
    root = write_corpus(
        make_clean(n=3).episodes,
        tmp_path / "foreign",
        dim_labels=("j0", "j1", "j2", "j3", "j4", "j5"),
    )
    with pytest.raises(ConfigError, match="not recorded on the declared machine"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_an_unknown_embodiment_ref_is_refused_by_name(single):
    with pytest.raises(ConfigError, match="unknown embodiment ref"):
        SO101Connector(single, embodiment="embodiment:franka@1.0")


def test_the_ref_is_required_and_not_inferred_from_robot_type(single):
    """LeRobot's robot_type is a free string; binding to it would be binding to a typo."""
    with pytest.raises(TypeError):
        SO101Connector(single)  # type: ignore[call-arg]


# -- what the binding then declares ----------------------------------------


def test_the_joint_channels_stop_being_matchable_by_width_alone(single):
    """The plain reader declines to invent this, and is right to. Here it is declared."""
    plain = {spec.name: spec for spec in LeRobotConnector(single).schema("episode_000000")}
    assert plain["action"].units is None and plain["action"].semantics is None

    bound = {
        spec.name: spec
        for spec in SO101Connector(single, embodiment=FOLLOWER_REF).schema("episode_000000")
    }
    action = bound["action"]
    assert action.units == "percent"
    assert action.semantics == "so101.action.joint_position_normalized"
    assert action.frame == "so101_follower_joints"
    assert action.metadata["normalization"] == "RANGE_0_100"
    assert action.metadata["calibration_variant"] == "so101_new_calib"
    assert "calibration_variant" in action.discriminators


def test_state_and_action_carry_different_meaning_tags(single):
    bound = {
        spec.name: spec
        for spec in SO101Connector(single, embodiment=FOLLOWER_REF).schema("episode_000000")
    }
    assert bound["observation.state"].semantics != bound["action"].semantics


def test_the_declared_ref_lands_on_every_record(single):
    connector = SO101Connector(single, embodiment=FOLLOWER_REF)
    episode = connector.open("episode_000000")
    assert episode.meta.embodiment == FOLLOWER_REF
    assert episode.meta.extra["embodiment_ref"] == FOLLOWER_REF


def test_a_mounting_geometry_is_refused_on_a_one_armed_ref(single):
    from gantry_evaluator_so101.bimanual import MountingGeometry

    with pytest.raises(ConfigError, match="two-arm bench"):
        SO101Connector(
            single, embodiment=FOLLOWER_REF, mounting=MountingGeometry.nominal()
        )


def test_the_bimanual_binding_carries_the_mounting_discriminator(bimanual):
    action = {
        spec.name: spec
        for spec in SO101Connector(bimanual, embodiment=BIMANUAL_REF).schema("episode_000000")
    }["action"]
    assert "mounting_id" in action.discriminators
    assert action.metadata["mounting"]["measured"] is False


def test_timestamp_is_described_and_other_channels_are_left_alone(single):
    schema = {
        spec.name: spec
        for spec in SO101Connector(single, embodiment=FOLLOWER_REF).schema("episode_000000")
    }
    assert schema["timestamp"].units == "s"
    assert schema["timestamp"].semantics == "time"
