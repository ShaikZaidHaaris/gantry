"""Every capability is a claim the resolver plans from. These check each one.

``docs/writing-a-plugin.md``: "A wrong claim does not produce an error — it
produces a run that looks fine."
"""

from __future__ import annotations

import pytest
from gantry_connector_so101 import FOLLOWER_REF, SO101Connector
from gantry_connector_so101.fixtures import write_corpus

from gantry.errors import ConfigError
from gantry.fixtures import make_clean
from gantry.resolve import ENTRY_POINT_GROUPS, Registry
from gantry.spine import StageEvent


@pytest.fixture
def root(tmp_path):
    return write_corpus(make_clean(n=4).episodes, tmp_path / "corpus")


def test_it_is_installed_under_the_connector_entry_point(root):
    registry = Registry()
    registry.discover()
    assert ENTRY_POINT_GROUPS()["gantry.connectors"] == "dataset"
    registration = registry.get("dataset", "so101")
    built = registration.build({"path": str(root), "embodiment": FOLLOWER_REF})
    assert isinstance(built, SO101Connector)
    assert built.descriptor().plane == "dataset"


# -- lazy ------------------------------------------------------------------


def test_schema_answers_without_reading_a_single_step(root, monkeypatch):
    """The lazy claim is inherited, so it has to survive the composition."""
    import pyarrow.parquet as pq

    connector = SO101Connector(root, embodiment=FOLLOWER_REF)

    def refuse(*args, **kwargs):  # pragma: no cover - only runs if the claim is false
        raise AssertionError("schema() read step data")

    monkeypatch.setattr(pq, "read_table", refuse)
    assert len(connector.schema("episode_000000")) == 3
    assert len(connector.open("episode_000000")) > 0


# -- media -----------------------------------------------------------------


def test_media_is_false_when_the_mp4s_are_not_there(tmp_path):
    """The metadata declares three cameras and nobody downloaded any of them.

    Delegated to the LeRobot reader, which distinguishes "this dataset has no
    images" from "this install cannot open them" — and says which.
    """
    root = write_corpus(
        make_clean(n=3).episodes, tmp_path / "cams", cameras=("top", "front", "wrist")
    )
    connector = SO101Connector(root, embodiment=FOLLOWER_REF)
    assert connector.descriptor().provides["media"] is False
    unavailable = connector.descriptor().metadata["lerobot"]["video_unavailable"]
    assert "observation.images.top" in unavailable or "no decoder installed" in unavailable
    assert "observation.images.top" not in connector.open("episode_000000").channel_names


# -- stage events ----------------------------------------------------------


def test_stage_events_are_false_for_a_teleop_recording(root):
    """The rig has no perception and the sidecar has one pose per episode.

    Claiming True here would hand a funnel diagnosis an empty table, and an
    empty funnel reads as a finished analysis rather than as a missing input.
    """
    connector = SO101Connector(root, embodiment=FOLLOWER_REF)
    assert connector.descriptor().provides["stage_events"] is False
    assert connector.open("episode_000000").labels.stage_events == ()


def test_stage_events_are_a_per_corpus_fact_and_can_be_supplied(root):
    """A sim recording does have object pose. That is a fact about the corpus."""
    events = {"episode_000000": (StageEvent("grasped", 12), StageEvent("lifted", 20))}
    connector = SO101Connector(
        root,
        embodiment=FOLLOWER_REF,
        stage_events=events,
        stage_events_source=(
            "sim ground-truth object pose, tracked against the gripper pose for 10 "
            "consecutive frames (CORPUS-PROTOCOL build step 3)"
        ),
    )
    assert connector.descriptor().provides["stage_events"] is True
    assert connector.open("episode_000000").labels.stages == ("grasped", "lifted")
    assert connector.open("episode_000001").labels.stage_events == ()
    assert "sim ground-truth" in connector.descriptor().metadata["stage_events_source"]


def test_stage_events_without_a_written_source_are_refused(root):
    with pytest.raises(ConfigError, match="stage_events_source"):
        SO101Connector(
            root, embodiment=FOLLOWER_REF, stage_events={"episode_000000": (StageEvent("x", 1),)}
        )


def test_stage_events_naming_an_absent_episode_are_refused(root):
    with pytest.raises(ConfigError, match="does not contain"):
        SO101Connector(
            root,
            embodiment=FOLLOWER_REF,
            stage_events={"episode_000099": (StageEvent("grasped", 1),)},
            stage_events_source="a detector",
        )


# -- what varies where it should not ---------------------------------------


def test_a_mixed_recording_horizon_is_reported(tmp_path):
    """Reported, not refused: this is a reader, and not knowing is the problem.

    An unequal recording horizon between conditions breaks frame-matching, which
    is the defect that invalidated three published results. A corpus deliberately
    assembled from two horizons is still a legitimate thing to look at — as long
    as the disagreement is on the descriptor where a manifest can see it.
    """
    import csv

    root = write_corpus(make_clean(n=4).episodes, tmp_path / "mixed")
    path = root / "episodes.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["episode_time_s"] = "25.0"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    connector = SO101Connector(root, embodiment=FOLLOWER_REF)
    assert connector.descriptor().metadata["mixed"]["episode_time_s"] == [1.4, 25.0]
