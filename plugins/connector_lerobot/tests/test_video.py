"""Frames, and the things that go wrong with frames.

Every mp4 here is written by the test itself, with a known colour per frame, so
"is this the right frame" is a question with a numeric answer rather than a
visual one. Frame *i* is a solid image whose red channel is *i*, which survives
h264 well enough to identify a frame exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest
from gantry_connector_lerobot import LeRobotConnector, VideoSource
from gantry_connector_lerobot.testing import build_dataset

from gantry.conformance import check_connector
from gantry.errors import ComponentError, ConfigError

CAMERA = "observation.images.top"
SIZE = 128


def paint(index: int, size: int = SIZE) -> np.ndarray:
    """A frame that says which frame it is."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :, 0] = index
    frame[:, :, 1] = 255 - index
    return frame


def write_video(
    path: Path,
    frames: int,
    *,
    fps: int = 20,
    size: int = SIZE,
    codec: str = "libx264rgb",
    pix_fmt: str = "rgb24",
) -> Path:
    """An mp4 whose frames can be told apart afterwards.

    Written losslessly in RGB by default. A real dataset is yuv420p, where the
    colour-space round trip moves a pixel by a few counts -- fine for a model,
    useless for "is this frame seven". The reader converts whatever it finds to
    rgb24 either way, so the encoding is not what is under test here; one test
    below uses the realistic one to prove that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = pix_fmt
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for index in range(frames):
            image = av.VideoFrame.from_ndarray(paint(index, size), format="rgb24")
            container.mux(stream.encode(image))
        container.mux(stream.encode())
    return path


def with_video(
    root: Path, *, episodes: int = 3, steps: int = 12, frames: int | None = None
) -> Path:
    build_dataset(root, episodes=episodes, steps=steps)
    for index in range(episodes):
        write_video(
            root / "videos" / "chunk-000" / CAMERA / f"episode_{index:06d}.mp4",
            frames if frames is not None else steps,
        )
    return root


@pytest.fixture
def dataset(tmp_path):
    return with_video(tmp_path / "lift")


def red_of(frames: np.ndarray) -> list[int]:
    """The identifying value of each frame, read back from the middle pixel."""
    return [int(frame[SIZE // 2, SIZE // 2, 0]) for frame in frames]


# -- the camera joins the schema -------------------------------------------


def test_conforms_with_video_on(dataset):
    # Not strict: LeRobot declares no units or meanings and this connector
    # invents none, which strict mode reports for every numeric channel. That
    # is the existing, deliberate state of affairs and not about the cameras.
    verdict = check_connector(LeRobotConnector(dataset))
    assert verdict.ok, verdict.explain()


def test_a_camera_that_can_be_read_is_in_the_schema(dataset):
    connector = LeRobotConnector(dataset)
    assert connector.cameras == (CAMERA,)
    assert connector.video_unavailable is None
    assert connector.descriptor().provides["media"] is True
    schema = {spec.name: spec for spec in connector.schema("episode_000000")}
    assert schema[CAMERA].kind == "image"
    assert schema[CAMERA].shape == (SIZE, SIZE, 3)
    assert schema[CAMERA].dtype == "uint8"
    assert schema[CAMERA].rate_hz is None or schema[CAMERA].rate_hz > 0


def test_the_axis_names_are_not_mistaken_for_dimension_labels(dataset):
    """``["height", "width", "rgb"]`` names the axes, not one label per element."""
    schema = {spec.name: spec for spec in LeRobotConnector(dataset).schema("episode_000000")}
    assert schema[CAMERA].dim_labels is None


def test_the_episode_carries_both_the_columns_and_the_camera(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    assert set(episode.channel_names) == {"observation.state", "action", "timestamp", CAMERA}
    assert episode.validate(deep=True).ok


# -- reading frames --------------------------------------------------------


def test_the_frames_come_back_in_order_and_as_pixels(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    frames = episode.array(CAMERA)
    assert frames.shape == (12, SIZE, SIZE, 3)
    assert frames.dtype == np.uint8
    assert red_of(frames) == list(range(12))


def test_a_window_decodes_only_that_window(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    window = episode.read([CAMERA], start=4, stop=9)[CAMERA]
    assert window.shape == (5, SIZE, SIZE, 3)
    assert red_of(window) == [4, 5, 6, 7, 8]


def test_a_late_window_lands_on_the_same_frames_a_full_read_would(tmp_path):
    """Past the seek threshold the reader jumps to a keyframe and decodes forward.

    That is where an off-by-a-keyframe bug would live, so the seeking path is
    checked against the sequential one rather than against itself.
    """
    root = with_video(tmp_path / "long", episodes=1, steps=90)
    episode = LeRobotConnector(root).open("episode_000000")
    whole = episode.array(CAMERA)
    late = episode.read([CAMERA], start=70, stop=80)[CAMERA]
    assert red_of(late) == list(range(70, 80))
    assert np.array_equal(late, whole[70:80])


def test_a_yuv420p_file_reads_the_same_way(tmp_path):
    """What a real dataset actually ships: h264 in yuv420p.

    The colour-space round trip moves a pixel by a couple of counts, so each
    frame is compared against the one it should be within a tolerance. That
    still pins the thing that matters -- that frame *i* came back at index *i* --     because the painted frames are far more than a tolerance apart.
    """
    root = with_video(tmp_path / "yuv", episodes=1, steps=10)
    write_video(
        root / "videos" / "chunk-000" / CAMERA / "episode_000000.mp4",
        frames=10,
        codec="libx264",
        pix_fmt="yuv420p",
    )
    frames = LeRobotConnector(root).open("episode_000000").array(CAMERA)
    assert frames.shape == (10, SIZE, SIZE, 3)
    assert frames.dtype == np.uint8
    intended = np.stack([paint(index) for index in range(10)]).astype(int)
    assert np.abs(frames.astype(int) - intended).max() <= 4


def test_reading_the_columns_alone_touches_no_video(dataset):
    """The lazy spine has to stay true, or a screen over a hundred episodes
    turns into a few gigabytes of decoded pixels."""
    connector = LeRobotConnector(dataset)
    episode = connector.open("episode_000000")
    (Path(dataset) / "videos" / "chunk-000" / CAMERA / "episode_000000.mp4").unlink()
    assert episode.array("action").shape == (12, 7)


def test_an_empty_window_is_empty_rather_than_everything(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    assert episode.read([CAMERA], start=5, stop=5)[CAMERA].shape == (0, SIZE, SIZE, 3)


def test_channels_come_back_in_the_order_they_were_asked_for(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    out = episode.read([CAMERA, "action"], start=0, stop=3)
    assert list(out) == [CAMERA, "action"]
    assert out["action"].shape == (3, 7)
    assert out[CAMERA].shape == (3, SIZE, SIZE, 3)


def test_an_unknown_channel_still_says_so(dataset):
    episode = LeRobotConnector(dataset).open("episode_000000")
    with pytest.raises(KeyError, match="nope"):
        episode.read(["nope"])


# -- the failures this format invites --------------------------------------


def test_a_short_video_is_refused_rather_than_silently_misaligned(tmp_path):
    """The failure the format invites, and the one worth catching hardest.

    A truncated mp4 still loads: the state vector is the right length, the
    frames are simply missing from the end. Everything downstream then compares
    an image against a state from a different moment, and nothing about the
    result looks wrong.
    """
    root = with_video(tmp_path / "short", episodes=1, steps=12, frames=8)
    episode = LeRobotConnector(root).open("episode_000000")
    with pytest.raises(ComponentError, match="8 frame"):
        episode.array(CAMERA)


def test_a_video_of_the_wrong_size_is_refused(tmp_path):
    root = with_video(tmp_path / "small", episodes=1, steps=6)
    write_video(root / "videos" / "chunk-000" / CAMERA / "episode_000000.mp4", frames=6, size=64)
    episode = LeRobotConnector(root).open("episode_000000")
    with pytest.raises(ComponentError, match="declares"):
        episode.array(CAMERA)


def test_a_corrupt_file_names_itself(tmp_path):
    root = with_video(tmp_path / "corrupt", episodes=1, steps=6)
    path = root / "videos" / "chunk-000" / CAMERA / "episode_000000.mp4"
    path.write_bytes(b"not an mp4 at all")
    episode = LeRobotConnector(root).open("episode_000000")
    with pytest.raises(ComponentError, match="could not decode"):
        episode.array(CAMERA)


def test_a_camera_missing_for_one_episode_costs_that_episode_only(tmp_path):
    root = with_video(tmp_path / "partial", episodes=3, steps=6)
    (root / "videos" / "chunk-000" / CAMERA / "episode_000001.mp4").unlink()
    connector = LeRobotConnector(root)
    assert connector.cameras == (CAMERA,), "the first episode's file is there"
    assert connector.open("episode_000000").array(CAMERA).shape[0] == 6
    with pytest.raises(ComponentError, match="could not decode"):
        connector.open("episode_000001").array(CAMERA)


# -- choosing ---------------------------------------------------------------


def test_video_can_be_turned_off(dataset):
    connector = LeRobotConnector(dataset, video=False)
    assert connector.cameras == ()
    assert CAMERA not in connector.open("episode_000000").channel_names
    assert connector.video_unavailable == "video=False"


def test_without_a_decoder_the_cameras_stay_out_and_say_which_problem_it_is(dataset, monkeypatch):
    """The default install has no ffmpeg bindings, and that is a different
    problem from a dataset with no cameras. Both leave the schema the same, so
    the descriptor is the only place the difference can be told."""
    monkeypatch.setattr("gantry_connector_lerobot.connector.available", lambda: False)
    connector = LeRobotConnector(dataset)
    assert connector.cameras == ()
    assert "no decoder installed" in connector.video_unavailable
    assert "[video]" in connector.video_unavailable


def test_insisting_without_a_decoder_says_how_to_get_one(dataset, monkeypatch):
    monkeypatch.setattr("gantry_connector_lerobot.connector.available", lambda: False)
    with pytest.raises(ConfigError, match=r"connector-lerobot\[video\]"):
        LeRobotConnector(dataset, video=True)


def test_insisting_on_video_that_is_not_there_refuses_at_construction(tmp_path):
    """Better here than as an empty column three hours into a run."""
    root = build_dataset(tmp_path / "no-mp4")
    with pytest.raises(ConfigError, match="no mp4"):
        LeRobotConnector(root, video=True)


def test_insisting_on_video_a_dataset_never_had_says_that_instead(tmp_path):
    root = build_dataset(tmp_path / "plain")
    info = json.loads((root / "meta" / "info.json").read_text())
    del info["features"][CAMERA]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ConfigError, match="declares no video features"):
        LeRobotConnector(root, video=True)


def test_overrides_reach_the_camera_too(dataset):
    connector = LeRobotConnector(
        dataset, schema_overrides={CAMERA: {"frame": "camera_top", "semantics": "observation"}}
    )
    spec = {s.name: s for s in connector.schema("episode_000000")}[CAMERA]
    assert (spec.frame, spec.semantics) == ("camera_top", "observation")


# -- the source on its own --------------------------------------------------


def test_the_source_reports_what_it_holds(tmp_path):
    path = write_video(tmp_path / "solo.mp4", frames=10)
    source = VideoSource(path, CAMERA, (SIZE, SIZE, 3), 10, 20.0)
    assert source.num_steps == 10
    assert red_of(source.read(0, 10)) == list(range(10))


def test_a_window_past_the_end_is_clipped_not_invented(tmp_path):
    path = write_video(tmp_path / "solo.mp4", frames=10)
    source = VideoSource(path, CAMERA, (SIZE, SIZE, 3), 10, 20.0)
    assert source.read(8, 99).shape[0] == 2


# -- against the dataset actually on this machine ---------------------------

REAL = Path("/tmp/gantry-real/lift_lerobot/ph")
real_only = pytest.mark.skipif(not REAL.exists(), reason="no real LeRobot dataset present")


@real_only
def test_the_real_dataset_says_why_its_cameras_are_unreadable():
    """This copy declares two cameras and shipped none of the mp4s."""
    connector = LeRobotConnector(REAL)
    assert set(connector.video_features) == {
        "observation.images.image",
        "observation.images.wrist_image",
    }
    if connector.cameras:
        frames = connector.open(connector.episode_ids()[0]).array("observation.images.image")
        assert frames.dtype == np.uint8 and frames.ndim == 4
    else:
        assert "no mp4 files found" in connector.video_unavailable
