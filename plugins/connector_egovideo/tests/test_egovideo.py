"""The ego front door, checked without decoding a single frame.

The prober and decoder are injected, so the manifest discipline — which is the
part that actually protects a user — is tested at full speed. One test at the
bottom uses a real encoded file to prove the header path works.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gantry_connector_egovideo import RGB, EgoVideoConnector

from gantry.conformance import check_connector
from gantry.errors import ConfigError


def clips(tmp_path, entries):
    for entry in entries:
        if "path" in entry:
            (tmp_path / entry["path"]).write_bytes(b"not really video")
    (tmp_path / "clips.json").write_text(json.dumps(entries))
    return tmp_path


def header(path, **over):
    return {"frames": 30, "height": 8, "width": 8, "rate_hz": 30.0, "seconds": 1.0, **over}


def frames(path, stride=1, limit=None):
    return np.zeros((30 // max(1, stride), 8, 8, 3), dtype="uint8")


def connector(tmp_path, entries=None, **kwargs):
    entries = entries or [
        {"path": "a.mp4", "instruction": "pick up the mug", "scene": "kitchen-1"},
        {"path": "b.mp4", "instruction": "put the mug in the sink", "scene": "kitchen-1"},
    ]
    root = clips(tmp_path, entries)
    return EgoVideoConnector(root, prober=header, decoder=frames, **kwargs)


# -- the manifest is the input contract --------------------------------------


def test_a_clip_with_no_instruction_is_refused_by_name(tmp_path):
    """A language-conditioned policy trains on the sentence, so a clip without
    one cannot be used for what is being measured. Dropping it silently would
    shrink somebody's dataset without telling them."""
    root = clips(tmp_path, [{"path": "a.mp4", "scene": "kitchen-1"}])
    with pytest.raises(ConfigError) as caught:
        EgoVideoConnector(root, prober=header, decoder=frames)
    assert "no instruction" in str(caught.value)
    assert "trains on this sentence" in str(caught.value)


def test_a_clip_with_no_scene_is_refused(tmp_path):
    """'Forty clips' and 'forty clips in one kitchen' are different datasets, and
    without a scene id nobody can tell them apart afterwards."""
    root = clips(tmp_path, [{"path": "a.mp4", "instruction": "pick up the mug"}])
    with pytest.raises(ConfigError, match="no scene"):
        EgoVideoConnector(root, prober=header, decoder=frames)


def test_every_problem_is_reported_at_once(tmp_path):
    """Somebody uploading forty clips through a GUI gets one list of what to fix,
    not forty rounds of fix-one-reupload."""
    root = clips(
        tmp_path,
        [
            {"path": "a.mp4", "scene": "k"},
            {"path": "b.mp4", "instruction": "do a thing"},
            {"instruction": "x", "scene": "y"},
        ],
    )
    with pytest.raises(ConfigError) as caught:
        EgoVideoConnector(root, prober=header, decoder=frames)
    message = str(caught.value)
    assert "3 problem(s)" in message
    assert "clip 0: no instruction" in message
    assert "clip 1: no scene" in message
    assert "clip 2: no path" in message


def test_a_missing_file_is_named(tmp_path):
    (tmp_path / "clips.json").write_text(
        json.dumps([{"path": "gone.mp4", "instruction": "x", "scene": "y"}])
    )
    with pytest.raises(ConfigError, match="does not exist"):
        EgoVideoConnector(tmp_path, prober=header, decoder=frames)


def test_duplicate_ids_are_refused(tmp_path):
    root = clips(
        tmp_path,
        [
            {"path": "a.mp4", "id": "same", "instruction": "x", "scene": "y"},
            {"path": "b.mp4", "id": "same", "instruction": "x", "scene": "y"},
        ],
    )
    with pytest.raises(ConfigError, match="duplicate id"):
        EgoVideoConnector(root, prober=header, decoder=frames)


def test_a_missing_manifest_says_what_one_looks_like(tmp_path):
    with pytest.raises(ConfigError, match="what the person was doing, and where"):
        EgoVideoConnector(tmp_path, prober=header, decoder=frames)


def test_an_empty_manifest_is_refused(tmp_path):
    (tmp_path / "clips.json").write_text("[]")
    with pytest.raises(ConfigError, match="lists no clips"):
        EgoVideoConnector(tmp_path, prober=header, decoder=frames)


def test_both_manifest_shapes_are_accepted(tmp_path):
    """A bare list, or an object with a 'clips' key — GUIs emit both."""
    entry = {"path": "a.mp4", "instruction": "x", "scene": "y"}
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "clips.json").write_text(json.dumps({"clips": [entry], "uploaded_by": "z"}))
    made = EgoVideoConnector(tmp_path, prober=header, decoder=frames)
    assert len(made.clips) == 1


# -- outcomes stay tri-state -------------------------------------------------


def test_an_unlabelled_outcome_is_not_a_failure(tmp_path):
    """A dataset that is secretly all failures trains nothing — that happened
    here once, with a checkpoint fine-tuned on a split that was 0-for-52. The
    only defence is that 'unlabelled' and 'failed' stay different values."""
    made = connector(tmp_path)
    episode = made.open(made.episode_ids()[0])
    assert episode.labels.success is None
    assert made.unlabelled() == made.episode_ids()
    # And the connector does not claim a capability it cannot back.
    assert made.descriptor().provides["outcomes"] is False


def test_stated_outcomes_are_read_in_both_spellings(tmp_path):
    made = connector(
        tmp_path,
        [
            {"path": "a.mp4", "instruction": "x", "scene": "k", "outcome": "success"},
            {"path": "b.mp4", "instruction": "x", "scene": "k", "outcome": False},
            {"path": "c.mp4", "instruction": "x", "scene": "k"},
        ],
    )
    assert [made.open(i).labels.success for i in made.episode_ids()] == [True, False, None]
    assert made.descriptor().provides["outcomes"] is True
    assert made.unlabelled() == ("ego/c",)


# -- what the report reads off it --------------------------------------------


def test_scene_count_is_surfaced_because_the_ranking_turns_on_it(tmp_path):
    """A user seeing 'scenes: 1' beside 'clips: 40' understands the problem
    without reading a word of prose."""
    made = connector(
        tmp_path,
        [{"path": f"{i}.mp4", "instruction": "x", "scene": "kitchen-1"} for i in range(4)],
    )
    assert made.scenes == ("kitchen-1",)
    assert made.descriptor().metadata["scenes"] == 1
    assert made.descriptor().metadata["clips"] == 4


def test_hours_come_from_the_headers(tmp_path):
    made = connector(tmp_path)
    assert made.hours() == pytest.approx(2 / 3600)


def test_a_clip_that_cannot_be_probed_does_not_break_the_total(tmp_path):
    def sometimes(path):
        if path.name == "b.mp4":
            raise ConfigError("unreadable")
        return header(path)

    made = connector(tmp_path)
    made._probe = sometimes
    made._probed.clear()
    assert made.hours() == pytest.approx(1 / 3600)


# -- the episodes ------------------------------------------------------------


def test_the_schema_is_answered_from_the_header_without_decoding(tmp_path):
    """What lets a whole upload be validated in seconds rather than after an hour
    of decoding."""
    decoded = []

    def counting(path, stride=1, limit=None):
        decoded.append(path)
        return frames(path, stride)

    made = connector(tmp_path)
    made._decode = counting
    schema = made.schema(made.episode_ids()[0])
    assert decoded == []
    assert schema[0].name == RGB
    assert schema[0].shape == (8, 8, 3)
    assert schema[0].semantics == "ego.rgb"


def test_the_episode_says_the_body_is_a_human(tmp_path):
    """Said explicitly because everything downstream asks, and 'human' is the
    answer that makes the retargeting step visible rather than assumed."""
    made = connector(tmp_path)
    episode = made.open(made.episode_ids()[0])
    assert episode.meta.embodiment == "human"
    assert episode.meta.task == "pick up the mug"
    assert episode.meta.extra["scene"] == "kitchen-1"


def test_the_instruction_and_scene_land_in_the_annotations(tmp_path):
    made = connector(tmp_path)
    annotations = made.open(made.episode_ids()[1]).labels.annotations
    assert annotations["instruction"] == "put the mug in the sink"
    assert annotations["scene"] == "kitchen-1"


def test_extra_manifest_fields_are_carried_rather_than_dropped(tmp_path):
    made = connector(
        tmp_path,
        [
            {
                "path": "a.mp4",
                "instruction": "x",
                "scene": "k",
                "device": "aria",
                "consent": "signed",
            }
        ],
    )
    annotations = made.open("ego/a").labels.annotations
    assert annotations["device"] == "aria"
    assert annotations["consent"] == "signed"


def test_stride_thins_the_decode_and_the_declared_rate_follows(tmp_path):
    """Decoding is the expensive part: an hour at 30 Hz is a hundred thousand
    frames and a policy that sees 5 Hz has no use for ninety thousand of them."""
    made = connector(tmp_path, stride=6)
    assert made.schema("ego/a")[0].rate_hz == 5.0
    assert made.open("ego/a").array(RGB).shape[0] == 5


def test_an_unknown_episode_raises_key_error(tmp_path):
    made = connector(tmp_path)
    with pytest.raises(KeyError):
        made.open("ego/nope")


def test_ids_are_stable_and_namespaced(tmp_path):
    made = connector(tmp_path)
    assert made.episode_ids() == made.episode_ids() == ("ego/a", "ego/b")


def test_the_connector_conforms(tmp_path):
    made = connector(tmp_path)
    verdict = check_connector(made)
    assert verdict.ok, verdict.explain()


# -- the one that touches a real container -----------------------------------


def test_the_header_path_works_on_a_real_file(tmp_path):
    av = pytest.importorskip("av")
    from gantry_connector_egovideo import decode, probe

    path = tmp_path / "real.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for index in range(30):
            image = np.full((32, 32, 3), index * 8, dtype="uint8")
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    head = probe(path)
    assert head["height"] == head["width"] == 32
    assert head["rate_hz"] == 30.0
    assert head["frames"] >= 29

    decoded = decode(path, stride=10)
    assert decoded.shape[1:] == (32, 32, 3)
    assert decoded.dtype == np.uint8
    assert len(decoded) == 3
