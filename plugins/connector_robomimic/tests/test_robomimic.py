"""Read a RoboMimic file, against a real one where there is one."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from gantry_connector_robomimic import CONVENTIONS, RoboMimicConnector

from gantry.conformance import check_connector
from gantry.errors import ConfigError

REAL = Path("/tmp/gantry-real/robomimic_data/lift/ph/low_dim.hdf5")
real_only = pytest.mark.skipif(not REAL.exists(), reason="no real RoboMimic file present")


def build(path: Path, *, demos: int = 4, steps: int = 10, succeed: int = 3) -> Path:
    """Write a file in the layout robomimic actually ships."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = '{"env_name": "Lift", "env_kwargs": {"robots": "Panda"}}'
        data.attrs["total"] = demos * steps
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["num_samples"] = steps
            demo.create_dataset("actions", data=rng.normal(size=(steps, 7)))
            dones = np.zeros(steps, dtype=np.int64)
            if index < succeed:
                dones[-1] = 1
            demo.create_dataset("dones", data=dones)
            demo.create_dataset("rewards", data=dones.astype(float))
            demo.create_dataset("states", data=rng.normal(size=(steps, 32)))
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=rng.normal(size=(steps, 3)))
            obs.create_dataset("robot0_eef_quat", data=rng.normal(size=(steps, 4)))
            obs.create_dataset("robot0_joint_pos", data=rng.normal(size=(steps, 7)))
            obs.create_dataset("robot0_gripper_qpos", data=rng.normal(size=(steps, 2)))
    return path


@pytest.fixture
def dataset(tmp_path):
    return build(tmp_path / "low_dim.hdf5")


# -- the contract ----------------------------------------------------------


def test_conforms(dataset):
    verdict = check_connector(RoboMimicConnector(dataset), strict=True)
    assert verdict.ok, verdict.explain()


def test_it_reports_the_outcomes_the_format_carries(dataset):
    connector = RoboMimicConnector(dataset)
    assert connector.descriptor().provides["outcomes"] is True
    outcomes = [connector.open(i).labels.success for i in connector.episode_ids()]
    assert outcomes == [True, True, True, False]


def test_it_declares_no_milestones(dataset):
    assert RoboMimicConnector(dataset).descriptor().provides["stage_events"] is False


def test_demos_are_ordered_numerically(tmp_path):
    """HDF5 iterates lexicographically, which puts demo_100 next to demo_10.

    A "first fifty episodes" that silently means a different fifty is the kind
    of thing that changes a benchmark without anyone noticing.
    """
    connector = RoboMimicConnector(build(tmp_path / "many.hdf5", demos=12))
    assert connector.episode_ids()[:3] == ("demo_0", "demo_1", "demo_2")
    assert connector.episode_ids()[-1] == "demo_11"


# -- the format's own vocabulary -------------------------------------------


def test_robosuite_conventions_are_carried_through(dataset):
    schema = {s.name: s for s in RoboMimicConnector(dataset).schema("demo_0")}
    assert schema["robot0_eef_pos"].units == "m"
    assert schema["robot0_eef_pos"].semantics == "state.eef_pos"
    assert schema["robot0_joint_pos"].units == "rad"


def test_the_quaternion_convention_is_declared_and_load_bearing(dataset):
    """robosuite stores scalar-last, and that difference is invisible in a shape."""
    schema = {s.name: s for s in RoboMimicConnector(dataset).schema("demo_0")}
    quat = schema["robot0_eef_quat"]
    assert quat.metadata["rotation_repr"] == "quat_xyzw"
    assert "rotation_repr" in quat.discriminators


def test_a_scalar_first_consumer_is_refused_without_a_converter(dataset):
    from gantry.spine import ChannelSpec, compatible

    provided = {s.name: s for s in RoboMimicConnector(dataset).schema("demo_0")}["robot0_eef_quat"]
    wanted = ChannelSpec(
        "robot0_eef_quat", "vector", (4,), provided.dtype, units="1", frame="world",
        semantics="state.eef_quat", discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_wxyz"},
    )
    assert "metadata.mismatch" in compatible(provided, wanted).codes()


def test_the_rotation_adapter_closes_it(dataset):
    from gantry_adapters_rotation import ROTATION

    from gantry.resolve import AdapterRegistry, bind_channel, requires_channels
    from gantry.spine import ChannelSpec


    provided = {s.name: s for s in RoboMimicConnector(dataset).schema("demo_0")}["robot0_eef_quat"]
    wanted = ChannelSpec(
        "robot0_eef_quat", "vector", (4,), provided.dtype, units="1", frame="world",
        semantics="state.eef_quat", discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_wxyz"},
    )
    # A bare quaternion is entirely rotation, which is the one unambiguous
    # layout, so the converter takes it.
    assert ROTATION.applies(provided, wanted).ok
    binding, verdict = bind_channel(
        wanted, [provided], requires_channels("c", "policy"), AdapterRegistry([ROTATION])
    )
    assert verdict.ok, verdict.explain()
    assert str(binding.chain).startswith("rotation@")


def test_an_unrecognised_key_is_left_undescribed(dataset):
    """Only the documented vocabulary is claimed; nothing else is guessed at."""
    assert "states" not in CONVENTIONS
    schema = {s.name: s for s in RoboMimicConnector(dataset).schema("demo_0")}
    assert schema["actions"].units is None


def test_overrides_win_over_the_convention(dataset):
    connector = RoboMimicConnector(
        dataset, schema_overrides={"robot0_eef_pos": {"units": "mm"}}
    )
    schema = {s.name: s for s in connector.schema("demo_0")}
    assert schema["robot0_eef_pos"].units == "mm"


# -- reading ---------------------------------------------------------------


def test_it_reads_the_right_rows(dataset):
    episode = RoboMimicConnector(dataset).open("demo_0")
    assert len(episode) == 10
    assert episode.array("actions").shape == (10, 7)


def test_a_window_is_sliced_in_the_file(dataset):
    episode = RoboMimicConnector(dataset).open("demo_0")
    window = episode.read(["robot0_eef_pos"], start=2, stop=6)
    assert window["robot0_eef_pos"].shape == (4, 3)
    assert np.allclose(window["robot0_eef_pos"], episode.array("robot0_eef_pos")[2:6])


def test_the_task_and_robot_come_through(dataset):
    episode = RoboMimicConnector(dataset).open("demo_0")
    assert episode.meta.task == "Lift"
    assert episode.meta.embodiment == "Panda"


def test_observations_can_be_narrowed(dataset):
    connector = RoboMimicConnector(dataset, observations=["robot0_eef_pos"])
    assert {s.name for s in connector.schema("demo_0")} == {"robot0_eef_pos", "actions"}


def test_actions_can_be_left_out(dataset):
    connector = RoboMimicConnector(dataset, include_actions=False)
    assert "actions" not in {s.name for s in connector.schema("demo_0")}


# -- refusals --------------------------------------------------------------


def test_a_file_that_is_not_robomimic_is_refused(tmp_path):
    path = tmp_path / "other.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("something", data=[1, 2, 3])
    with pytest.raises(ConfigError, match="not a RoboMimic file"):
        RoboMimicConnector(path)


def test_a_file_with_no_demos_is_refused(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("data")
    with pytest.raises(ConfigError, match="no demo_N groups"):
        RoboMimicConnector(path)


def test_asking_for_an_observation_that_is_not_there_lists_what_is(dataset):
    with pytest.raises(ConfigError, match="robot0_eef_pos"):
        RoboMimicConnector(dataset, observations=["no_such_key"])


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        RoboMimicConnector(tmp_path / "nope.hdf5")


def test_an_unknown_demo_raises(dataset):
    with pytest.raises(KeyError, match="demo_99"):
        RoboMimicConnector(dataset).open("demo_99")


# -- against the genuine article -------------------------------------------


@real_only
def test_it_reads_a_real_file():
    connector = RoboMimicConnector(REAL)
    assert len(connector.episode_ids()) == 200
    assert connector.env["env_name"] == "Lift"
    assert check_connector(connector).ok


@real_only
def test_real_outcomes_are_read_from_dones():
    connector = RoboMimicConnector(REAL)
    outcomes = [connector.open(i).labels.success for i in connector.episode_ids()[:20]]
    assert all(outcome is True for outcome in outcomes), "PH demonstrations all succeed"


@real_only
def test_a_real_episode_validates_deeply():
    connector = RoboMimicConnector(REAL)
    assert connector.open("demo_0").validate(deep=True).ok


@real_only
def test_the_real_observation_set_is_what_robosuite_writes():
    available = RoboMimicConnector(REAL).available_observations()
    assert "robot0_eef_pos" in available and "robot0_joint_pos" in available
