from __future__ import annotations

import numpy as np
import pytest

from gantry.spine import (
    ChannelSpec,
    EpisodeLabels,
    StageEvent,
    episode_from_arrays,
)


def make(steps: int = 12, **kwargs):
    arrays = {
        "state": np.zeros((steps, 7), dtype="float32"),
        "action": np.zeros((steps, 4), dtype="float32"),
    }
    schema = [
        ChannelSpec("state", "vector", (7,), "float32", units="m", semantics="position"),
        ChannelSpec("action", "vector", (4,), "float32", semantics="actuation"),
    ]
    return episode_from_arrays(arrays, schema, id="e0", source="unit-test", **kwargs)


def test_length_and_channels():
    episode = make()
    assert len(episode) == 12
    assert episode.channel_names == ("state", "action")
    assert episode.channel("state").units == "m"


def test_uid_is_namespaced_by_source():
    assert make().meta.uid == "unit-test/e0"


def test_read_slices_lazily_through_the_source():
    episode = make(steps=20)
    window = episode.read(["state"], start=5, stop=9)
    assert window["state"].shape == (4, 7)


def test_unknown_channel_raises_with_the_uid():
    with pytest.raises(KeyError, match="unit-test/e0"):
        make().channel("nope")


def test_validate_passes_on_a_coherent_record():
    assert make().validate(deep=True).ok


def test_declared_but_absent_channel_is_refused():
    arrays = {"state": np.zeros((5, 7), dtype="float32")}
    schema = [
        ChannelSpec("state", "vector", (7,), "float32"),
        ChannelSpec("action", "vector", (4,), "float32"),
    ]
    episode = episode_from_arrays(arrays, schema, id="e", source="s")
    assert "schema.missing" in episode.validate().codes()


def test_optional_channel_may_be_absent():
    arrays = {"state": np.zeros((5, 7), dtype="float32")}
    schema = [
        ChannelSpec("state", "vector", (7,), "float32"),
        ChannelSpec("wrist", "image", (None, None, 3), "uint8", optional=True),
    ]
    assert episode_from_arrays(arrays, schema, id="e", source="s").validate().ok


def test_undeclared_channel_is_a_visible_note():
    arrays = {
        "state": np.zeros((5, 7), dtype="float32"),
        "secret": np.zeros((5,), dtype="float32"),
    }
    schema = [ChannelSpec("state", "vector", (7,), "float32")]
    verdict = episode_from_arrays(arrays, schema, id="e", source="s").validate()
    assert verdict.ok
    assert "schema.undeclared" in verdict.codes()


def test_deep_validation_catches_a_lying_schema():
    arrays = {"state": np.zeros((5, 3), dtype="float32")}
    schema = [ChannelSpec("state", "vector", (7,), "float32")]
    episode = episode_from_arrays(arrays, schema, id="e", source="s")
    assert episode.validate(deep=False).ok
    assert "data.shape" in episode.validate(deep=True).codes()


def test_source_refuses_ragged_channels():
    with pytest.raises(ValueError, match="disagree on length"):
        episode_from_arrays(
            {"a": np.zeros((5, 1)), "b": np.zeros((6, 1))},
            [ChannelSpec("a", "vector", (1,)), ChannelSpec("b", "vector", (1,))],
            id="e",
            source="s",
        )


# -- stage events ----------------------------------------------------------


def test_stage_events_are_free_vocabulary():
    labels = EpisodeLabels(
        success=True,
        stage_events=(StageEvent("incise", 3), StageEvent("suture", 8)),
    )
    episode = make(labels=labels)
    assert episode.validate().ok
    assert episode.labels.stages == ("incise", "suture")
    assert episode.labels.reached("suture")
    assert episode.labels.step_of("suture") == 8
    assert not episode.labels.reached("close")


def test_stage_outside_the_episode_is_refused():
    episode = make(steps=5, labels=EpisodeLabels(stage_events=(StageEvent("late", 99),)))
    assert "stage.range" in episode.validate().codes()


def test_repeated_stage_is_refused():
    labels = EpisodeLabels(stage_events=(StageEvent("grasp", 1), StageEvent("grasp", 4)))
    assert "stage.duplicate" in make(labels=labels).validate().codes()
