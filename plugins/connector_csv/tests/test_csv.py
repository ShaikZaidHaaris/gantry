"""Behaviour specific to this format, beyond what the contract requires."""

from __future__ import annotations

import numpy as np
import pytest
from gantry.fixtures import make_clean, make_defective
from gantry.spine import compatible
from gantry_connector_csv import CsvConnector, write_episodes


@pytest.fixture
def round_trip(tmp_path):
    def _go(suite, sidecar=True, name="d.csv"):
        path = write_episodes(suite.episodes, tmp_path / name, sidecar=sidecar)
        return CsvConnector(path), path

    return _go


# -- round trip ------------------------------------------------------------


def test_numbers_survive_the_round_trip(round_trip):
    suite = make_clean(n=4)
    connector, _ = round_trip(suite)
    for original in suite.episodes:
        restored = connector.open(original.meta.id)
        assert len(restored) == len(original)
        for spec in original.schema:
            assert np.allclose(
                restored.array(spec.name), original.array(spec.name), atol=1e-6
            ), spec.name


def test_meaning_survives_only_with_the_sidecar(round_trip):
    suite = make_clean(n=3)
    original = suite.episodes[0].channel("position")

    with_sidecar, _ = round_trip(suite, sidecar=True, name="rich.csv")
    assert compatible(with_sidecar.open("ep_0000").channel("position"), original).ok

    without, _ = round_trip(suite, sidecar=False, name="bare.csv")
    bare = without.open("ep_0000").channel("position")
    assert bare.units is None and bare.frame is None
    # shape and dtype are readable off the numbers; meaning is not
    assert bare.shape == original.shape and bare.dtype == original.dtype


def test_a_sidecar_declaring_the_wrong_unit_is_caught_downstream(tmp_path):
    """The connector reports what it is told; the resolver is what objects."""
    import json

    suite = make_clean(n=2)
    path = write_episodes(suite.episodes, tmp_path / "d.csv")
    sidecar = path.with_suffix(path.suffix + ".schema.json")
    declared = json.loads(sidecar.read_text())
    declared["position"]["units"] = "mm"
    sidecar.write_text(json.dumps(declared))

    mislabelled = CsvConnector(path).open("ep_0000").channel("position")
    verdict = compatible(mislabelled, suite.episodes[0].channel("position"))
    assert "units.scale" in verdict.codes()


def test_labels_survive(round_trip):
    suite = make_defective("never_completes", n=6, fraction=0.5)
    connector, _ = round_trip(suite)
    for original in suite.episodes:
        restored = connector.open(original.meta.id)
        assert restored.labels.success is original.labels.success
        assert restored.labels.stages == original.labels.stages
        for stage in original.labels.stages:
            assert restored.labels.step_of(stage) == original.labels.step_of(stage)


def test_stage_vocabulary_is_carried_not_assumed(tmp_path):
    suite = make_clean(n=3, stages=("incise", "grip", "suture", "close"))
    path = write_episodes(suite.episodes, tmp_path / "surgery.csv")
    assert CsvConnector(path).open("ep_0000").labels.stages == (
        "incise",
        "grip",
        "suture",
        "close",
    )


# -- laziness --------------------------------------------------------------


def test_enumerating_does_not_read_step_data(round_trip, monkeypatch):
    connector, _ = round_trip(make_clean(n=5))
    opened = [connector.open(i) for i in connector.episode_ids()]

    reads = {"count": 0}
    for episode in opened:
        original = episode.source._rows

        def counted(start, stop, _original=original):
            reads["count"] += 1
            return _original(start, stop)

        episode.source._rows = counted

    assert reads["count"] == 0
    _ = [len(episode) for episode in opened]
    assert reads["count"] == 0, "length came from the index, not from parsing rows"

    opened[0].read(["position"], start=0, stop=2)
    assert reads["count"] == 1


def test_a_window_read_parses_only_that_window(round_trip):
    connector, _ = round_trip(make_clean(n=2))
    episode = connector.open("ep_0000")
    window = episode.read(["position"], start=3, stop=7)
    assert set(window) == {"position"}
    assert window["position"].shape == (4, 3)
    assert np.allclose(window["position"], episode.array("position")[3:7])


# -- malformed input -------------------------------------------------------


def test_missing_episode_column_says_what_it_found(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("t,x,y\n0,1,2\n")
    with pytest.raises(ValueError, match="no 'episode' column"):
        CsvConnector(path)


def test_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="file is empty"):
        CsvConnector(path)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        CsvConnector(tmp_path / "nope.csv")


def test_source_defaults_to_the_filename_and_namespaces_ids(round_trip):
    connector, _ = round_trip(make_clean(n=2), name="vendor_a.csv")
    assert connector.open("ep_0000").meta.uid == "vendor_a/ep_0000"


def test_two_files_with_colliding_ids_stay_distinct(tmp_path):
    """The failure that once collapsed two datasets into one."""
    left = write_episodes(make_clean(n=3, seed=1).episodes, tmp_path / "left.csv")
    right = write_episodes(make_clean(n=3, seed=2).episodes, tmp_path / "right.csv")

    a, b = CsvConnector(left), CsvConnector(right)
    assert a.episode_ids() == b.episode_ids()  # same ids in both files
    uids = {e.meta.uid for e in a} | {e.meta.uid for e in b}
    assert len(uids) == 6  # but six distinct episodes, not three
