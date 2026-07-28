"""Read a LeRobot dataset, against a real one where there is one.

The fixtures are built from the format's own spec. Where a genuine dataset is
present the same tests run against it too, because a reader validated only
against material its author also wrote has proved that they agree with
themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gantry_connector_lerobot import SUGGESTED_SEMANTICS, LeRobotConnector

from gantry.conformance import check_connector
from gantry.errors import ConfigError

#: A real dataset, when the machine happens to have one.
REAL = Path("/tmp/gantry-real/lift_lerobot/ph")
real_only = pytest.mark.skipif(not REAL.exists(), reason="no real LeRobot dataset present")


def build_dataset(root: Path, *, episodes: int = 3, steps: int = 12, version: str = "v2.1") -> Path:
    """Write a dataset in the layout the format actually ships."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "robot_type": "franka",
                "total_episodes": episodes,
                "fps": 20,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.top": {
                        "dtype": "video",
                        "shape": [128, 128, 3],
                        "names": ["height", "width", "rgb"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [4],
                        "names": {"motors": ["x", "y", "z", "gripper"]},
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": {
                            "motors": [
                                "x", "y", "z",
                                "axis_angle1", "axis_angle2", "axis_angle3",
                                "gripper",
                            ]
                        },
                    },
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                },
            }
        )
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "lift the cube"}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "tasks": ["lift the cube"], "length": steps}) + "\n"
            for i in range(episodes)
        )
    )
    rng = np.random.default_rng(0)
    for index in range(episodes):
        table = pa.table(
            {
                "observation.state": pa.array(
                    rng.normal(size=(steps, 4)).tolist(), pa.list_(pa.float32(), 4)
                ),
                "action": pa.array(
                    rng.normal(size=(steps, 7)).tolist(), pa.list_(pa.float32(), 7)
                ),
                "timestamp": pa.array(np.arange(steps) / 20.0, pa.float32()),
                "frame_index": pa.array(np.arange(steps), pa.int64()),
                "episode_index": pa.array([index] * steps, pa.int64()),
                "index": pa.array(np.arange(steps), pa.int64()),
                "task_index": pa.array([0] * steps, pa.int64()),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
    return root


@pytest.fixture
def dataset(tmp_path):
    return build_dataset(tmp_path / "lift")


# -- the contract ----------------------------------------------------------


def test_conforms(dataset):
    verdict = check_connector(LeRobotConnector(dataset))
    assert verdict.ok, verdict.explain()


def test_the_schema_comes_from_the_declared_features(dataset):
    schema = {spec.name: spec for spec in LeRobotConnector(dataset).schema("episode_000000")}
    assert set(schema) == {"observation.state", "action", "timestamp"}
    assert schema["action"].shape == (7,)
    assert schema["observation.state"].dtype == "float32"


def test_the_frame_rate_is_carried_through(dataset):
    assert all(
        spec.rate_hz == 20.0 for spec in LeRobotConnector(dataset).schema("episode_000000")
    )


def test_per_dimension_names_become_labels(dataset):
    schema = {spec.name: spec for spec in LeRobotConnector(dataset).schema("episode_000000")}
    assert schema["action"].dim_labels == (
        "x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper",
    )


def test_a_one_wide_feature_is_a_scalar_not_a_vector_of_one(dataset):
    """LeRobot writes it as a bare column, so describing it as a vector would
    make every consumer index into an axis that is not there."""
    schema = {spec.name: spec for spec in LeRobotConnector(dataset).schema("episode_000000")}
    assert schema["timestamp"].kind == "scalar" and schema["timestamp"].shape == ()


def test_a_declared_camera_with_no_file_is_reported_and_not_promised(dataset):
    """The metadata says there is a camera; the download did not include one.

    Reporting it and leaving it out of the schema is the honest pair: nobody has
    to discover the absence by reading an empty column, and nothing plans a run
    around frames that will not arrive.
    """
    connector = LeRobotConnector(dataset)
    assert connector.video_features == ("observation.images.top",)
    assert connector.cameras == ()
    assert "observation.images.top" not in connector.open("episode_000000").channel_names
    assert connector.descriptor().metadata["video_features"] == ["observation.images.top"]
    assert "no mp4 files found" in connector.descriptor().metadata["video_unavailable"]


def test_bookkeeping_columns_are_out_by_default_and_available_on_request(dataset):
    assert "index" not in {s.name for s in LeRobotConnector(dataset).schema("episode_000000")}
    verbose = LeRobotConnector(dataset, include_bookkeeping=True)
    assert "frame_index" in {s.name for s in verbose.schema("episode_000000")}


# -- reading ---------------------------------------------------------------


def test_it_reads_the_right_rows(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    assert len(episode) == 12
    assert episode.array("action").shape == (12, 7)


def test_a_window_reads_only_that_window(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    window = episode.read(["action"], start=3, stop=7)
    assert set(window) == {"action"}
    assert window["action"].shape == (4, 7)
    assert np.allclose(window["action"], episode.array("action")[3:7])


def test_the_instruction_comes_through(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    assert episode.meta.task == "lift the cube"
    assert episode.meta.embodiment == "franka"


def test_ids_are_namespaced_by_the_dataset(dataset):
    assert LeRobotConnector(dataset).open("episode_000000").meta.uid == "lift/episode_000000"


# -- what it declines to invent --------------------------------------------


def test_units_and_meaning_are_left_undeclared(dataset):
    """The format does not say, so neither does the reader.

    ``observation.state`` being four wide with a gripper at the end is a strong
    hint and still a hint. A reader that promotes hints to declarations is how a
    millimetre comes to read as a metre.
    """
    schema = {spec.name: spec for spec in LeRobotConnector(dataset).schema("episode_000000")}
    assert schema["observation.state"].units is None
    assert schema["observation.state"].semantics is None
    assert "conformance.channel_undescribed" in check_connector(LeRobotConnector(dataset)).codes()


def test_overrides_are_how_you_say_what_you_know(dataset):
    connector = LeRobotConnector(dataset, schema_overrides=dict(SUGGESTED_SEMANTICS))
    schema = {spec.name: spec for spec in connector.schema("episode_000000")}
    assert schema["action"].semantics == "action.eef_delta_pose"
    assert schema["action"].metadata["rotation_repr"] == "axis_angle"
    assert schema["timestamp"].units == "s"


def test_it_declares_no_outcomes_and_no_milestones(dataset):
    provides = LeRobotConnector(dataset).descriptor().provides
    assert provides["outcomes"] is False and provides["stage_events"] is False


# -- refusals --------------------------------------------------------------


def test_a_directory_that_is_not_a_dataset_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not a LeRobot dataset root"):
        LeRobotConnector(tmp_path)


def test_an_unsupported_codebase_version_is_refused_by_number(tmp_path):
    root = build_dataset(tmp_path / "old", version="v1.6")
    with pytest.raises(ConfigError, match="v1.6"):
        LeRobotConnector(root)


def test_an_unknown_episode_raises(dataset):
    with pytest.raises(KeyError, match="episode_000099"):
        LeRobotConnector(dataset).open("episode_000099")


def test_an_index_naming_no_existing_parquet_is_refused(tmp_path):
    root = build_dataset(tmp_path / "empty")
    for parquet in (root / "data" / "chunk-000").glob("*.parquet"):
        parquet.unlink()
    with pytest.raises(ConfigError, match="no episode whose parquet exists"):
        LeRobotConnector(root)


# -- against a genuine dataset ---------------------------------------------


@real_only
def test_it_reads_a_real_dataset():
    connector = LeRobotConnector(REAL)
    assert len(connector.episode_ids()) == 200
    assert check_connector(connector).ok


@real_only
def test_a_real_episode_has_the_shape_the_metadata_promised():
    connector = LeRobotConnector(REAL)
    episode = connector.open(connector.episode_ids()[0])
    assert episode.array("action").shape[1] == 7
    assert episode.array("observation.state").shape[1] == 8
    assert episode.validate(deep=True).ok


@real_only
def test_a_real_dataset_with_duplicate_dimension_names_drops_them():
    """Found in genuine data: the state names end ('gripper', 'gripper').

    A repeated label cannot address a dimension, so keeping it would offer an
    addressing scheme that silently resolves two columns to one.
    """
    connector = LeRobotConnector(REAL)
    schema = {spec.name: spec for spec in connector.schema(connector.episode_ids()[0])}
    names = connector.info["features"]["observation.state"]["names"]["motors"]
    assert len(names) != len(set(names))
    assert schema["observation.state"].dim_labels is None
    assert schema["action"].dim_labels is not None
