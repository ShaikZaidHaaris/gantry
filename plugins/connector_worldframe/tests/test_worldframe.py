"""Placing a hand in a room, and refusing to when the room's size is unknown."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gantry_connector_worldframe import (
    Trajectory,
    WorldFrameConnector,
    from_colmap,
    from_device,
    scale_to,
)

from gantry.conformance import check_connector
from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ArraySource,
    ChannelSpec,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
)

STEPS = 20


def straight(steps=STEPS, metric=True, source="device", move=0.05):
    """A camera walking forward along +z, not turning."""
    positions = np.zeros((steps, 3))
    positions[:, 2] = np.arange(steps) * move
    return Trajectory(
        positions=positions,
        rotations=np.tile(np.eye(3), (steps, 1, 1)),
        metric=metric,
        source=source,
    )


class CameraFrame(Connector):
    """What connector_handpose presents: metric wrists in the camera frame."""

    def __init__(self, steps=STEPS, scale="metric"):
        self._steps = steps
        self._scale = scale

    def descriptor(self):
        return connector_descriptor(
            name="handpose",
            version="0.1",
            lazy=False,
            stage_events=False,
            outcomes=False,
            media=False,
            writes=False,
            licence="Apache-2.0 (test)",
        )

    def episode_ids(self):
        return ("handpose/ego/0",)

    def schema(self, episode_id):
        return self.open(episode_id).schema

    def open(self, episode_id):
        if episode_id not in self.episode_ids():
            raise KeyError(episode_id)
        arrays, schema = {}, []
        for hand in ("left", "right"):
            w = np.zeros((self._steps, 7), dtype="float32")
            w[:, 2] = 0.4  # 40 cm in front of the camera, always
            w[:, 3] = 1.0  # identity rotation
            arrays[f"{hand}_wrist"] = w
            schema.append(
                ChannelSpec(
                    f"{hand}_wrist",
                    "vector",
                    (7,),
                    "float32",
                    frame="camera",
                    rate_hz=10.0,
                    semantics="ego.wrist_pose",
                    discriminators=("rotation_repr", "scale", "hand"),
                    metadata={"rotation_repr": "quat_wxyz", "scale": self._scale, "hand": hand},
                )
            )
        arrays["state"] = np.zeros((self._steps, 4), dtype="float32")
        schema.append(ChannelSpec("state", "vector", (4,), "float32", rate_hz=10.0))
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source="handpose",
                task="pick up the mug",
                embodiment="human",
                license="Apache-2.0 (test)",
            ),
            schema=tuple(schema),
            source=ArraySource(arrays),
            labels=EpisodeLabels(annotations={"instruction": "pick up the mug"}),
        )


def made(**kwargs):
    source = kwargs.pop("source", None) or CameraFrame()
    traj = kwargs.pop("trajectory", None) or straight()
    return WorldFrameConnector(source, trajectories={"handpose/ego/0": traj}, **kwargs)


# -- the composition ---------------------------------------------------------


def test_a_hand_held_still_relative_to_a_moving_camera_moves_in_the_world():
    """The whole point. In the camera frame the hand never moves; in the world it
    travels with the person. A policy trained on the first sees the same physical
    reach labelled a dozen different ways."""
    e = made().open("worldframe/handpose/ego/0")
    world = e.array("left_wrist")[:, :3]

    assert world[0] == pytest.approx([0, 0, 0.4], abs=1e-5)
    # The camera walked 0.05 m per step; the hand went with it.
    assert world[-1] == pytest.approx([0, 0, 0.4 + 0.05 * (STEPS - 1)], abs=1e-5)
    assert e.channel("left_wrist").frame == "world"


def test_camera_rotation_carries_into_the_world_position():
    turn = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # 90 deg about y
    traj = Trajectory(
        positions=np.zeros((STEPS, 3)), rotations=np.tile(turn, (STEPS, 1, 1)), metric=True
    )
    e = made(trajectory=traj).open("worldframe/handpose/ego/0")
    # A hand 0.4 m along the camera's +z lands along the world's +x.
    assert e.array("left_wrist")[0, :3] == pytest.approx([0.4, 0, 0], abs=1e-5)


def test_channels_that_are_not_wrists_pass_through():
    e = made().open("worldframe/handpose/ego/0")
    assert "state" in e.channel_names
    assert e.array("state").shape == (STEPS, 4)


# -- the refusals ------------------------------------------------------------


def test_no_trajectory_is_a_refusal_and_says_where_to_get_one():
    """A trajectory guessed from nothing produces a hand that moves correctly
    through a room that does not exist."""
    c = WorldFrameConnector(CameraFrame(), trajectories={})
    with pytest.raises(ConfigError, match="room that does not exist"):
        c.open("worldframe/handpose/ego/0")


def test_a_scaleless_trajectory_is_refused_when_fitting_is_off():
    c = made(trajectory=straight(metric=False, source="colmap"), fit_scale=False)
    with pytest.raises(ConfigError, match="room of the wrong size"):
        c.open("worldframe/handpose/ego/0")


def test_a_non_metric_hand_is_refused():
    c = made(source=CameraFrame(scale="normalized"))
    with pytest.raises(ConfigError, match="room of the wrong size"):
        c.open("worldframe/handpose/ego/0")


def test_a_trajectory_shorter_than_the_footage_is_refused():
    c = made(trajectory=straight(steps=5))
    with pytest.raises(ConfigError, match="cannot be aligned"):
        c.open("worldframe/handpose/ego/0")


def test_a_malformed_trajectory_is_refused_at_construction():
    with pytest.raises(ConfigError, match=r"\(steps, 3\) positions"):
        Trajectory(positions=np.zeros((5, 4)), rotations=np.tile(np.eye(3), (5, 1, 1)), metric=True)
    with pytest.raises(ConfigError, match=r"\(5, 3, 3\) rotations"):
        Trajectory(positions=np.zeros((5, 3)), rotations=np.tile(np.eye(3), (4, 1, 1)), metric=True)
    with pytest.raises(ConfigError, match="unknown trajectory source"):
        Trajectory(
            positions=np.zeros((5, 3)),
            rotations=np.tile(np.eye(3), (5, 1, 1)),
            metric=True,
            source="vibes",
        )


# -- fitting a scaleless trajectory ------------------------------------------


class Resting(CameraFrame):
    """A hand held still in the world while the camera walks past it.

    The only situation the scale fit is valid in, made explicit: the hand's
    camera-frame position changes purely because the camera moved.
    """

    def open(self, episode_id):
        e = super().open(episode_id)
        arrays = {n: e.array(n) for n in e.channel_names}
        for hand in ("left", "right"):
            w = arrays[f"{hand}_wrist"].copy()
            # camera moves +0.05/step along z in TRUE metres; a world-stationary
            # hand therefore recedes at the same rate in the camera frame.
            w[:, 2] = 0.4 - np.arange(len(w)) * 0.05
            arrays[f"{hand}_wrist"] = w
        return EpisodeRecord(
            meta=e.meta, schema=e.schema, source=ArraySource(arrays), labels=e.labels
        )


def test_fitting_is_off_by_default_because_it_is_the_weakest_link():
    c = made(trajectory=straight(metric=False, source="colmap"))
    with pytest.raises(ConfigError, match="fitting is off"):
        c.open("worldframe/handpose/ego/0")


def test_a_scaleless_trajectory_fits_against_a_world_stationary_hand():
    """The physics the obvious version got wrong: a hand's *distance* from the
    camera is a camera-frame quantity and constrains nothing. A world-stationary
    point's camera-frame *motion* does."""
    # SfM recovered the path at half its true size.
    half = Trajectory(
        positions=straight().positions * 0.5,
        rotations=straight().rotations,
        metric=False,
        source="colmap",
    )
    c = WorldFrameConnector(Resting(), trajectories={"handpose/ego/0": half}, fit_scale=True)
    e = c.open("worldframe/handpose/ego/0")
    assert c.fitted_scales()["worldframe/handpose/ego/0"] == pytest.approx(2.0, rel=0.05)
    assert e.labels.annotations["trajectory_metric"] is True
    assert e.labels.annotations["trajectory_source"] == "colmap"


def test_a_camera_that_never_moves_cannot_be_scaled():
    still = Trajectory(
        positions=np.zeros((STEPS, 3)),
        rotations=np.tile(np.eye(3), (STEPS, 1, 1)),
        metric=False,
        source="colmap",
    )
    c = WorldFrameConnector(Resting(), trajectories={"handpose/ego/0": still}, fit_scale=True)
    with pytest.raises(ConfigError, match="does not move"):
        c.open("worldframe/handpose/ego/0")


def test_too_few_solved_frames_cannot_be_scaled():
    traj = straight(steps=5, metric=False)
    with pytest.raises(ConfigError, match="fewer than eight"):
        scale_to(traj, np.zeros((5, 3)))


# -- reading trajectories ----------------------------------------------------


def test_a_device_trajectory_is_metric_by_construction(tmp_path):
    """ARKit, ARCore, Quest and Aria all track the device and throw it away on
    export. If the capture app kept it, nothing reconstructed will beat it."""
    path = tmp_path / "poses.json"
    path.write_text(
        json.dumps(
            {
                "device": "arkit",
                "frames": [
                    {"position": [0, 0, i * 0.1], "rotation": [1, 0, 0, 0]} for i in range(6)
                ],
            }
        )
    )
    t = from_device(path)
    assert t.metric is True and t.source == "device"
    assert t.positions[-1] == pytest.approx([0, 0, 0.5])
    assert t.metadata["device"] == "arkit"


def test_a_device_frame_with_no_pose_is_marked_invalid(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text(
        json.dumps(
            [
                {"position": [0, 0, 0], "rotation": [1, 0, 0, 0]},
                {},
                {"q": [1, 0, 0, 0], "t": [0, 0, 1]},
            ]
        )
    )
    t = from_device(path)
    assert list(t.valid) == [True, False, True]


def test_colmap_output_is_inverted_and_declared_scaleless(tmp_path):
    """COLMAP stores world-to-camera. Getting that backwards produces a plausible
    mirror of the truth, which is the classic way to lose a day."""
    path = tmp_path / "images.txt"
    path.write_text(
        "# comment\n"
        "1 1 0 0 0 0 0 -1 1 frame_000000.jpg\n"
        "100 200 -1\n"
        "2 1 0 0 0 0 0 -2 1 frame_000001.jpg\n"
        "100 200 -1\n"
    )
    t = from_colmap(path)
    assert t.metric is False and t.source == "colmap"
    # world-to-camera t=(0,0,-1) with identity R means the camera sits at +1.
    assert t.positions[0] == pytest.approx([0, 0, 1])
    assert t.positions[1] == pytest.approx([0, 0, 2])


def test_an_empty_colmap_file_is_refused(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# nothing here\n")
    with pytest.raises(ConfigError, match="no camera poses"):
        from_colmap(path)


# -- what lands on the record ------------------------------------------------


def test_drift_is_reported_rather_than_hidden():
    """Every trajectory drifts. A ten-minute clip can be metres out by the end,
    silently stretching every trajectory near the tail."""
    e = made().open("worldframe/handpose/ego/0")
    annotations = e.labels.annotations
    assert annotations["drift_path_length_m"] == pytest.approx(0.05 * (STEPS - 1))
    assert annotations["drift_start_to_end_m"] == pytest.approx(0.05 * (STEPS - 1))
    assert annotations["drift_unsolved_frames"] == 0.0


def test_the_trajectory_source_travels_with_the_data():
    e = made().open("worldframe/handpose/ego/0")
    assert e.meta.extra["trajectory_source"] == "device"
    assert "capture device" in e.meta.extra["trajectory_note"]


def test_the_licence_is_carried_forward():
    c = made()
    assert c.descriptor().metadata["estimator_licence"] == "Apache-2.0 (test)"
    assert c.open("worldframe/handpose/ego/0").meta.license == "Apache-2.0 (test)"


def test_lineage_reaches_back():
    e = made().open("worldframe/handpose/ego/0")
    assert e.meta.derived_from == ("handpose/ego/0",)


def test_frames_the_tracker_could_not_solve_leave_the_hand_unplaced():
    valid = np.ones(STEPS, dtype=bool)
    valid[5:8] = False
    t = Trajectory(
        positions=straight().positions, rotations=straight().rotations, metric=True, valid=valid
    )
    e = made(trajectory=t).open("worldframe/handpose/ego/0")
    world = e.array("left_wrist")[:, :3]
    assert np.all(world[5:8] == 0.0)
    assert np.any(world[0] != 0.0)


def test_the_connector_conforms():
    verdict = check_connector(made())
    assert verdict.ok, verdict.explain()
