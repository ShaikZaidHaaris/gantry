"""Hand estimation as a derived dataset, checked with a scripted estimator.

The estimator is injected, so what is under test is the thing this plugin is
responsible for: that the estimate stays traceable and stays labelled as an
estimate, and that the three counts the report later reads are right.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_connector_handpose import (
    JOINTS,
    HandPoseConnector,
    Track,
    aperture_from,
    wrist_from,
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

RGB = "ego_rgb"
SIZE = 8


def hand_at(x=0.0, y=0.0, z=0.0, spread=0.05):
    """Twenty-one joints: wrist at the origin, knuckles and tips laid out so the
    palm frame and the aperture are both well defined."""
    points = np.zeros((21, 3), dtype="float32")
    points[0] = (x, y, z)
    points[5] = (x + spread, y, z)  # index knuckle
    points[17] = (x, y + spread, z)  # pinky knuckle
    points[4] = (x + spread, y - spread, z)  # thumb tip
    points[8] = (x + spread * 2, y, z)  # index tip
    return points


class FakeSource(Connector):
    """An ego-video connector, as far as this plugin can tell."""

    def __init__(self, clips=2, steps=20, rate=30.0):
        self._clips = clips
        self._steps = steps
        self._rate = rate

    def descriptor(self):
        return connector_descriptor(
            name="ego",
            version="0.1",
            lazy=True,
            stage_events=False,
            outcomes=True,
            media=True,
            writes=False,
        )

    def episode_ids(self):
        return tuple(f"ego/{index}" for index in range(self._clips))

    def schema(self, episode_id):
        return (
            ChannelSpec(
                RGB,
                "image",
                (SIZE, SIZE, 3),
                "uint8",
                frame="camera",
                rate_hz=self._rate,
                semantics="ego.rgb",
            ),
        )

    def open(self, episode_id):
        if episode_id not in self.episode_ids():
            raise KeyError(episode_id)
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source="ego",
                task="pick up the mug",
                embodiment="human",
                extra={"scene": "kitchen-1"},
            ),
            schema=self.schema(episode_id),
            source=ArraySource({RGB: np.zeros((self._steps, SIZE, SIZE, 3), dtype="uint8")}),
            labels=EpisodeLabels(success=True, annotations={"instruction": "pick up the mug"}),
        )


class Scripted:
    """An estimator whose confidence and motion are dictated by the test."""

    def __init__(
        self,
        *,
        confidence=1.0,
        drift=0.0,
        hands=("left", "right"),
        convention="mediapipe",
        omit_confidence=False,
        world=False,
    ):
        self.world = world
        self.confidence = confidence
        self.drift = drift
        self.hands = hands
        self.convention = convention
        self.omit_confidence = omit_confidence
        self.seen = []

    def estimate(self, frames):
        self.seen.append(len(frames))
        steps = len(frames)
        points = {
            hand: np.stack([hand_at(x=index * self.drift) for index in range(steps)])
            for hand in self.hands
        }
        scores = {hand: np.full(steps, self.confidence, dtype="float32") for hand in self.hands}
        return Track(
            keypoints=points,
            world={h: v * 0.1 for h, v in points.items()} if self.world else {},
            confidence={} if self.omit_confidence else scores,
            convention=self.convention,
            metadata={"estimator": "scripted"},
        )


def derived(**kwargs):
    source = kwargs.pop("source", None) or FakeSource()
    estimator = kwargs.pop("estimator", None) or Scripted()
    made = HandPoseConnector(source, estimator=estimator, **kwargs)
    made.estimator_used = estimator
    return made


# -- the estimate stays an estimate -----------------------------------------


def test_position_is_normalized_and_only_the_hand_shape_is_metric():
    """The distinction the first real video taught, expensively.

    Image landmarks are pixel fractions; world landmarks are metres centred on
    the hand. So the hand's shape is metric and its position in the room is not,
    and calling both "unscaled" let a hand span be applied to a pixel fraction --     which produced a smooth, confident trajectory inside a 19 cm box.
    """
    episode = derived().open("handpose/ego/0")
    for name in ("left_hand", "right_hand", "left_wrist"):
        assert episode.channel(name).metadata["scale"] == "normalized"
    assert episode.meta.extra["scale"] == "normalized"


def test_the_metric_half_is_labelled_metric():
    """Aperture from world landmarks is a real distance in metres, which is what
    makes the gripper command trustworthy even when the wrist trajectory is not."""
    episode = derived(estimator=Scripted(world=True)).open("handpose/ego/0")
    assert episode.channel("left_shape").metadata["scale"] == "metric"
    assert episode.channel("left_aperture").metadata["scale"] == "metric"
    assert episode.meta.extra["shape_scale"] == "metric"


def test_without_world_landmarks_nothing_claims_to_be_metric():
    episode = derived().open("handpose/ego/0")
    assert "left_shape" not in episode.channel_names
    assert episode.channel("left_aperture").metadata["scale"] == "normalized"


def test_the_lineage_points_back_at_the_footage_it_was_guessed_from():
    episode = derived().open("handpose/ego/1")
    assert episode.meta.derived_from == ("ego/1",)
    assert episode.meta.source == "handpose"


def test_the_descriptor_names_the_estimator_and_the_dataset_it_came_from():
    """Two runs through different estimators are not comparable, and nothing else
    in the pipeline would have kept the difference."""
    metadata = derived().descriptor().metadata
    assert metadata["estimator"] == "Scripted"
    assert metadata["derived_from"] == "ego@0.1"
    assert metadata["scale"] == "normalized"


def test_the_keypoint_convention_travels_on_the_channel_and_the_episode():
    episode = derived(estimator=Scripted(convention="mano")).open("handpose/ego/0")
    assert episode.channel("left_hand").metadata["keypoints"] == "mano"
    assert episode.meta.extra["keypoints"] == "mano"


def test_an_estimator_that_returns_no_confidence_is_refused():
    """A consumer could not then tell a firmly-tracked hand from a guess, and
    monocular estimators are confidently wrong often enough that the difference
    matters more than the poses in the tail."""
    with pytest.raises(ConfigError, match="confidently wrong"):
        derived(estimator=Scripted(omit_confidence=True)).open("handpose/ego/0")


def test_confidence_is_kept_as_a_channel_rather_than_thresholded_away():
    episode = derived(estimator=Scripted(confidence=0.4)).open("handpose/ego/0")
    assert episode.array("left_confidence").shape == (20,)
    assert float(episode.array("left_confidence")[0]) == pytest.approx(0.4)
    # And the poses are still there -- the threshold is only for the count.
    assert episode.array("left_hand").shape == (20, 21, 3)


# -- the three counts the report reads --------------------------------------


def test_hands_visible_counts_frames_the_estimator_was_confident_about():
    high = derived(estimator=Scripted(confidence=0.9)).open("handpose/ego/0")
    low = derived(estimator=Scripted(confidence=0.2)).open("handpose/ego/0")
    assert high.labels.annotations["hands_visible"] == 1.0
    assert low.labels.annotations["hands_visible"] == 0.0


def test_motion_ok_counts_frames_an_arm_could_have_followed():
    """Human hands move several times faster than a robot; the fast parts are
    simultaneously blurred and unreachable as a target."""
    slow = derived(estimator=Scripted(drift=0.001)).open("handpose/ego/0")
    fast = derived(estimator=Scripted(drift=0.5)).open("handpose/ego/0")
    assert slow.labels.annotations["motion_ok"] == 1.0
    assert fast.labels.annotations["motion_ok"] == 0.0


def test_usable_length_marks_a_clip_too_short_to_hold_a_whole_attempt():
    short = derived(source=FakeSource(steps=5)).open("handpose/ego/0")
    long = derived(source=FakeSource(steps=40)).open("handpose/ego/0")
    assert short.labels.annotations["usable_length"] == 0.0
    assert long.labels.annotations["usable_length"] == 1.0


def test_the_signals_are_counts_and_not_judgements():
    """Which number is worth worrying about, and what to do, is decided somewhere
    else -- a module that both measured and graded would be scoring its own work."""
    annotations = derived().open("handpose/ego/0").labels.annotations
    assert set(annotations) >= {"hands_visible", "motion_ok", "usable_length", "steps"}
    assert not any("should" in str(value) for value in annotations.values())


def test_the_upstream_annotations_and_outcome_survive():
    episode = derived().open("handpose/ego/0")
    assert episode.labels.annotations["instruction"] == "pick up the mug"
    assert episode.meta.extra["scene"] == "kitchen-1"
    assert episode.labels.success is True


# -- the geometry ------------------------------------------------------------


def test_the_wrist_pose_is_position_then_a_wxyz_quaternion():
    points = np.stack([hand_at(x=0.1, y=0.2, z=0.3)] * 3)
    wrist = wrist_from(points)
    assert wrist.shape == (3, 7)
    assert wrist[0, :3] == pytest.approx([0.1, 0.2, 0.3])
    assert float(np.linalg.norm(wrist[0, 3:])) == pytest.approx(1.0, abs=1e-5)


def test_a_frame_where_no_hand_was_found_becomes_identity_rather_than_a_guess():
    """The confidence channel already says the frame is worthless; inventing a
    plausible rotation would hide that."""
    points = np.zeros((2, 21, 3), dtype="float32")
    wrist = wrist_from(points)
    assert wrist[0, 3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_the_aperture_is_thumb_tip_to_index_tip():
    points = np.stack([hand_at(spread=0.05)])
    # thumb tip (0.05, -0.05, 0), index tip (0.10, 0, 0) -> sqrt(0.05^2+0.05^2)
    assert float(aperture_from(points)[0]) == pytest.approx(0.0707, abs=1e-3)


def test_the_joint_table_is_per_convention_and_an_unknown_one_is_refused():
    """Exactly the table that gets silently wrong -- MANO and MediaPipe are both
    21 joints in different orders."""
    assert set(JOINTS) >= {"mediapipe", "mano"}
    assert JOINTS["mediapipe"] != JOINTS["mano"]
    with pytest.raises(ConfigError, match="wrong knuckle"):
        wrist_from(np.zeros((2, 21, 3)), convention="whatever")


def test_a_convention_indexing_past_the_end_is_refused():
    with pytest.raises(ConfigError, match="indexes joint"):
        wrist_from(np.zeros((2, 4, 3)), convention="mediapipe")


def test_badly_shaped_keypoints_are_refused():
    with pytest.raises(ConfigError, match=r"\(steps, joints, 3\)"):
        wrist_from(np.zeros((2, 21)))


# -- the plumbing ------------------------------------------------------------


def test_no_estimator_says_the_interface_is_one_method():
    made = HandPoseConnector(FakeSource())
    with pytest.raises(ConfigError, match="single estimate"):
        made.open("handpose/ego/0")


def test_a_factory_is_accepted_as_well_as_an_instance():
    made = HandPoseConnector(FakeSource(), estimator=lambda: Scripted())
    assert made.open("handpose/ego/0").array("left_hand").shape == (20, 21, 3)


def test_a_source_with_no_video_channel_is_refused_by_name():
    class Blind(FakeSource):
        def schema(self, episode_id):
            return (ChannelSpec("state", "vector", (7,), "float32"),)

        def open(self, episode_id):
            return EpisodeRecord(
                meta=EpisodeMeta(id=episode_id, source="ego"),
                schema=self.schema(episode_id),
                source=ArraySource({"state": np.zeros((4, 7), dtype="float32")}),
                labels=EpisodeLabels(),
            )

    with pytest.raises(ConfigError, match="no 'ego_rgb' channel"):
        derived(source=Blind()).open("handpose/ego/0")


def test_one_hand_only_is_handled():
    episode = derived(estimator=Scripted(hands=("right",))).open("handpose/ego/0")
    assert "right_hand" in episode.channel_names
    assert "left_hand" not in episode.channel_names


def test_the_video_can_be_dropped_to_save_space():
    with_video = derived().open("handpose/ego/0")
    without = derived(keep_video=False).open("handpose/ego/0")
    assert RGB in with_video.channel_names
    assert RGB not in without.channel_names
    assert without.descriptor if False else True


def test_the_estimator_runs_once_per_episode():
    made = derived()
    made.open("handpose/ego/0")
    made.open("handpose/ego/0")
    assert made.estimator_used.seen == [20]


def test_ids_are_namespaced_and_an_unknown_one_raises_key_error():
    made = derived()
    assert made.episode_ids() == ("handpose/ego/0", "handpose/ego/1")
    with pytest.raises(KeyError):
        made.open("handpose/ego/9")
    with pytest.raises(KeyError):
        made.open("ego/0")


def test_the_connector_conforms():
    verdict = check_connector(derived())
    assert verdict.ok, verdict.explain()
