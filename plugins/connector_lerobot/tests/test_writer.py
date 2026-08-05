"""Writing LeRobot, and refusing to do it quietly.

The round trip is the easy half. The half that matters is the refusal: this
format cannot hold an outcome, and a conversion that discovers that later has
already produced a dataset nobody can screen.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gantry_connector_lerobot import LeRobotConnector, survey, write_episodes

from gantry.conformance import check_connector
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec, EpisodeLabels, StageEvent, episode_from_arrays

STATE = ChannelSpec(
    "observation.state",
    "vector",
    (4,),
    "float32",
    rate_hz=20.0,
    dim_labels=("x", "y", "z", "gripper"),
)
ACTION = ChannelSpec(
    "action",
    "vector",
    (3,),
    "float32",
    rate_hz=20.0,
    dim_labels=("dx", "dy", "dz"),
)


def episode(index: int, steps: int = 8, **labels):
    rng = np.random.default_rng(index)
    return episode_from_arrays(
        {
            "observation.state": rng.normal(size=(steps, 4)).astype("float32"),
            "action": rng.normal(size=(steps, 3)).astype("float32"),
        },
        (STATE, ACTION),
        id=f"demo_{index}",
        source="fixture",
        task="lift the cube",
        embodiment="franka",
        labels=EpisodeLabels(**labels),
    )


@pytest.fixture
def plain():
    return [episode(i) for i in range(3)]


# -- the round trip ---------------------------------------------------------


def test_what_is_written_can_be_read_back(tmp_path, plain):
    report = write_episodes(plain, tmp_path / "out")
    assert report.lossless
    assert (report.episodes, report.frames) == (3, 24)

    connector = LeRobotConnector(tmp_path / "out")
    assert len(connector.episode_ids()) == 3
    assert check_connector(connector).ok


def test_the_numbers_survive(tmp_path, plain):
    write_episodes(plain, tmp_path / "out")
    back = LeRobotConnector(tmp_path / "out").open("episode_000000")
    assert np.allclose(back.array("action"), plain[0].array("action"), atol=1e-6)
    assert np.allclose(
        back.array("observation.state"), plain[0].array("observation.state"), atol=1e-6
    )


def test_dimension_labels_survive_as_spans(tmp_path, plain):
    """info.json's names list cannot express a two-column field without
    repeating a label; modality.json's spans always can, so that is what is
    written and what the reader prefers on the way back."""
    write_episodes(plain, tmp_path / "out")
    modality = json.loads((tmp_path / "out" / "meta" / "modality.json").read_text())
    assert modality["state"]["x"] == {"start": 0, "end": 1}
    assert modality["action"]["dz"] == {"start": 2, "end": 3}
    back = LeRobotConnector(tmp_path / "out").schema("episode_000000")
    assert {s.name: s.dim_labels for s in back}["observation.state"] == ("x", "y", "z", "gripper")


def test_a_multi_column_field_round_trips_as_one_span(tmp_path):
    wide = ChannelSpec(
        "observation.state",
        "vector",
        (4,),
        "float32",
        rate_hz=20.0,
        dim_labels=("x", "y", "gripper.0", "gripper.1"),
    )
    record = episode_from_arrays(
        {
            "observation.state": np.zeros((5, 4), dtype="float32"),
            "action": np.zeros((5, 3), dtype="float32"),
        },
        (wide, ACTION),
        id="d0",
        source="f",
        task="t",
    )
    write_episodes([record], tmp_path / "out")
    spans = json.loads((tmp_path / "out" / "meta" / "modality.json").read_text())["state"]
    assert spans["gripper"] == {"start": 2, "end": 4}
    labels = {
        s.name: s.dim_labels for s in LeRobotConnector(tmp_path / "out").schema("episode_000000")
    }
    assert labels["observation.state"] == ("x", "y", "gripper.0", "gripper.1")


def test_the_task_and_rate_come_through(tmp_path, plain):
    write_episodes(plain, tmp_path / "out")
    connector = LeRobotConnector(tmp_path / "out")
    assert connector.info["fps"] == 20.0
    assert connector.info["robot_type"] == "franka"
    assert connector.open("episode_000000").meta.task == "lift the cube"


def test_bookkeeping_columns_are_written_so_it_is_a_real_dataset(tmp_path, plain):
    write_episodes(plain, tmp_path / "out")
    verbose = LeRobotConnector(tmp_path / "out", include_bookkeeping=True)
    names = {s.name for s in verbose.schema("episode_000000")}
    assert {"frame_index", "episode_index", "index", "task_index"} <= names


# -- the refusal ------------------------------------------------------------


def test_outcomes_cannot_be_carried_and_the_writer_says_so(tmp_path):
    """The exact loss that happened to this project's own lift conversion."""
    graded = [episode(i, success=i % 2 == 0) for i in range(4)]
    losses = survey(graded)
    assert [loss.what for loss in losses] == ["outcomes"]
    assert "4 of 4" in losses[0].detail

    with pytest.raises(ConfigError, match="would drop"):
        write_episodes(graded, tmp_path / "out")


def test_the_refusal_lists_every_thing_being_dropped(tmp_path):
    rich = [
        episode(
            0,
            success=True,
            stage_events=(StageEvent("grasp", 2), StageEvent("lift", 5)),
            annotations={"operator": "kai"},
        )
    ]
    with pytest.raises(ConfigError) as raised:
        write_episodes(rich, tmp_path / "out")
    message = str(raised.value)
    assert "outcomes" in message and "stage_events" in message and "annotations" in message
    assert "grasp" in message


def test_the_loss_can_be_accepted_and_is_then_reported(tmp_path):
    graded = [episode(i, success=True) for i in range(2)]
    report = write_episodes(graded, tmp_path / "out", accept_loss=True)
    assert not report.lossless
    assert "outcomes" in report.explain()
    # And the reader agrees the outcomes are gone, rather than inventing any.
    assert LeRobotConnector(tmp_path / "out").descriptor().provides["outcomes"] is False


def test_images_are_reported_as_dropped_rather_than_half_written(tmp_path):
    camera = ChannelSpec("observation.images.top", "image", (4, 4, 3), "uint8")
    record = episode_from_arrays(
        {
            "observation.state": np.zeros((3, 4), dtype="float32"),
            "action": np.zeros((3, 3), dtype="float32"),
            "observation.images.top": np.zeros((3, 4, 4, 3), dtype="uint8"),
        },
        (STATE, ACTION, camera),
        id="d0",
        source="f",
        task="t",
    )
    losses = {loss.what: loss.detail for loss in survey([record])}
    assert "observation.images.top" in losses["channels"]
    report = write_episodes([record], tmp_path / "out", accept_loss=True)
    assert "observation.images.top" not in report.channels


def test_nothing_to_write_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="nothing to write"):
        write_episodes([], tmp_path / "out")


# -- the point of the exercise ---------------------------------------------


def test_any_connector_can_be_the_source(tmp_path):
    """There is no robomimic-to-lerobot converter, and there should not be one.

    Read with whatever reads the source, write with this. Here the source is
    the CSV reader, and nothing in either plugin knows about the other.
    """
    csv = pytest.importorskip("gantry_connector_csv")
    written = csv.write_episodes([episode(0), episode(1)], tmp_path / "mid.csv")
    source = csv.CsvConnector(written)
    episodes = [source.open(i) for i in source.episode_ids()]

    report = write_episodes(episodes, tmp_path / "out", fps=20.0, accept_loss=True)
    assert report.episodes == 2
    back = LeRobotConnector(tmp_path / "out")
    assert len(back.episode_ids()) == 2
    assert np.allclose(
        back.open("episode_000000").array("action"), episode(0).array("action"), atol=1e-5
    )


def test_what_lerobot_cannot_hold_is_kept_beside_it_and_read_back():
    """v2.1 records no outcome, no milestones and no annotations, and
    accept_loss=True says drop them. That is right for a dataset somebody else
    will read, and it silently destroyed the entire input to the feedback layer
    the first time a report was generated from a written dataset -- it correctly
    announced that no component had declared a licence and that the footage came
    from zero locations.

    An extra file in meta/ is ignored by every other LeRobot tool and read back
    here, which makes the round trip lossless without making the dataset
    non-standard.
    """
    import tempfile
    from pathlib import Path

    from gantry_connector_lerobot import LeRobotConnector

    from gantry.spine import StageEvent

    specs = [
        ChannelSpec("observation.state", "vector", (4,), "float32", rate_hz=10.0),
        ChannelSpec("action", "vector", (4,), "float32", semantics="actuation", rate_hz=10.0),
    ]
    original = episode_from_arrays(
        {
            "observation.state": np.zeros((5, 4), dtype="float32"),
            "action": np.zeros((5, 4), dtype="float32"),
        },
        specs,
        id="ego/a",
        source="ego",
        task="wash the dishes",
        embodiment="bimanual_2arm",
        license="Apache-2.0 (MediaPipe + OpenCV)",
        labels=EpisodeLabels(
            success=True,
            stage_events=(StageEvent("grasped", 2),),
            annotations={"scene": "kitchen-P04", "hands_visible": 0.93},
        ),
    )
    root = Path(tempfile.mkdtemp()) / "ds"
    LeRobotConnector.write([original], str(root), fps=10, accept_loss=True)

    assert (root / "meta" / "gantry.jsonl").exists()
    back = LeRobotConnector(str(root)).open(
        "ego/a"
        if "ego/a" in LeRobotConnector(str(root)).episode_ids()
        else LeRobotConnector(str(root)).episode_ids()[0]
    )
    assert back.meta.license == "Apache-2.0 (MediaPipe + OpenCV)"
    assert back.labels.success is True
    assert back.labels.annotations["scene"] == "kitchen-P04"
    assert back.labels.annotations["hands_visible"] == 0.93
    assert [(s.name, s.step) for s in back.labels.stage_events] == [("grasped", 2)]


def test_a_dataset_from_another_producer_has_no_sidecar_and_says_nothing():
    """Absent rather than assumed. A LeRobot dataset from anywhere else genuinely
    has no outcome, and inventing one would be worse than not having it."""
    import tempfile
    from pathlib import Path

    from gantry_connector_lerobot import LeRobotConnector

    specs = [ChannelSpec("action", "vector", (2,), "float32", semantics="actuation", rate_hz=5.0)]
    made = episode_from_arrays(
        {"action": np.zeros((4, 2), dtype="float32")},
        specs,
        id="x",
        source="other",
        task="do a thing",
    )
    root = Path(tempfile.mkdtemp()) / "ds"
    LeRobotConnector.write([made], str(root), fps=5)
    (root / "meta" / "gantry.jsonl").unlink()

    back = LeRobotConnector(str(root))
    episode = back.open(back.episode_ids()[0])
    assert episode.labels.success is None
    assert episode.meta.license is None
