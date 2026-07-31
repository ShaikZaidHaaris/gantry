"""Hands to state and action, and the conventions that must not be implicit."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_connector_egoactions import ACTION, STATE, EgoActionConnector
from gantry_retargeter_hands import VIPERX_300, Hand, HandToArm, Mount
from gantry_semantics_ego import aperture_channel, wrist_channel

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
STEPS = 20


class FakeHands(Connector):
    """What connector_handpose presents: a camera and metric wrists."""

    def __init__(self, steps=STEPS, unsolved=(), scale="metric"):
        self._steps = steps
        self._unsolved = set(unsolved)
        self._scale = scale

    def descriptor(self):
        return connector_descriptor(
            name="handpose",
            version="0.1",
            lazy=False,
            stage_events=False,
            outcomes=True,
            media=True,
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
        wrists, apertures, schema, arrays = {}, {}, [], {}
        for hand in ("left", "right"):
            w = np.zeros((self._steps, 7), dtype="float32")
            w[:, 0] = 0.35
            w[:, 2] = np.linspace(0.3, 0.5, self._steps)
            w[:, 3] = 1.0
            for index in self._unsolved:
                w[index] = 0.0
            arrays[f"{hand}_wrist"] = w
            arrays[f"{hand}_aperture"] = np.full(self._steps, 0.06, dtype="float32")
            schema += [
                wrist_channel(
                    f"{hand}_wrist",
                    hand=hand,
                    scale=self._scale,
                    rotation_repr="quat_wxyz",
                    frame="camera",
                    rate_hz=10.0,
                ),
                aperture_channel(f"{hand}_aperture", hand=hand, scale="metric", rate_hz=10.0),
            ]
        arrays[RGB] = np.zeros((self._steps, 8, 8, 3), dtype="uint8")
        schema.insert(
            0,
            ChannelSpec(
                RGB, "image", (8, 8, 3), "uint8", frame="camera", rate_hz=10.0, semantics="ego.rgb"
            ),
        )
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source="handpose",
                task="pick up the mug",
                embodiment="human",
                extra={"scene": "kitchen-1"},
            ),
            schema=tuple(schema),
            source=ArraySource(arrays),
            labels=EpisodeLabels(
                annotations={
                    "instruction": "pick up the mug",
                    "estimator_licence": "Apache-2.0 (test)",
                }
            ),
        )


def retargeter():
    return HandToArm(
        mount=Mount.aligned(), hand=Hand(closed=0.02, open=0.10, span=0.19), reach=VIPERX_300
    )


def made(**kwargs):
    source = kwargs.pop("source", None) or FakeHands()
    return EgoActionConnector(source, retargeter=retargeter(), **kwargs)


def test_state_and_action_come_out_at_the_declared_width():
    c = made()
    e = c.open("egoactions/handpose/ego/0")
    assert c.width == 14
    assert e.array(STATE).shape == (STEPS - 1, 14)
    assert e.array(ACTION).shape == (STEPS - 1, 14)


def test_the_action_is_the_next_state_and_the_last_step_is_dropped():
    """Repeating the final state teaches a policy that the right move in the last
    state is to stay there — a lesson about the end of a recording, not the task."""
    e = made().open("egoactions/handpose/ego/0")
    state, action = e.array(STATE), e.array(ACTION)
    assert np.allclose(action[:-1], state[1:])
    assert len(state) == STEPS - 1
    assert e.meta.extra["action_convention"] == "action[t] is state[t+1]"


def test_which_half_is_which_arm_is_written_down():
    c = made()
    labels = c.labels
    assert len(labels) == 14
    assert labels[0].startswith("left_") and labels[7].startswith("right_")
    assert c.open("egoactions/handpose/ego/0").channel(ACTION).dim_labels == labels


def test_unsolved_frames_are_cut_rather_than_interpolated():
    """Interpolating invents a hand position nobody observed and puts it in the
    training set indistinguishable from a real one."""
    c = made(source=FakeHands(unsolved=(3, 4, 5)))
    e = c.open("egoactions/handpose/ego/0")
    assert e.array(STATE).shape[0] == STEPS - 3 - 1
    assert e.labels.annotations["dropped_unsolved"] == pytest.approx(0.15)


def test_an_episode_with_too_little_left_is_refused_as_a_filming_problem():
    c = made(source=FakeHands(steps=20, unsolved=range(14)))
    with pytest.raises(ConfigError, match="filming problem"):
        c.open("egoactions/handpose/ego/0")


def test_normalized_wrists_are_refused_all_the_way_here():
    """The scale discipline survives to the last connector in the chain."""
    c = made(source=FakeHands(scale="normalized"))
    with pytest.raises(Exception, match="image_coordinates|cannot be retargeted"):
        c.open("egoactions/handpose/ego/0")


def test_the_licence_is_carried_onto_the_training_set():
    """A training set built through non-commercial weights is itself encumbered,
    and nothing else in the chain would remember."""
    c = made()
    e = c.open("egoactions/handpose/ego/0")
    assert e.meta.license == "Apache-2.0 (test)"
    assert c.descriptor().metadata["estimator_licence"] == "Apache-2.0 (test)"


def test_the_lineage_reaches_back_to_the_footage():
    e = made().open("egoactions/handpose/ego/0")
    assert e.meta.derived_from == ("handpose/ego/0",)
    assert e.meta.task == "pick up the mug"
    assert e.meta.extra["scene"] == "kitchen-1"


def test_the_video_is_cut_to_match_the_kept_steps():
    c = made(source=FakeHands(unsolved=(2, 3)))
    e = c.open("egoactions/handpose/ego/0")
    assert e.array(RGB).shape[0] == e.array(STATE).shape[0]


def test_the_yield_rate_is_measured_rather_than_predicted():
    c = made(source=FakeHands(unsolved=(1, 2, 3, 4)))
    assert c.yield_rate() == {"measured": False}
    c.open("egoactions/handpose/ego/0")
    rate = c.yield_rate()
    assert rate["measured"] is True
    assert rate["steps_in"] == STEPS
    assert rate["steps_out"] == STEPS - 4 - 1
    assert 0 < rate["kept"] < 1


def test_one_retargeter_per_hand_is_accepted():
    """People are not symmetric, and one calibration applied to both puts the
    difference straight into the gripper signal of the hand nobody measured."""
    c = EgoActionConnector(
        FakeHands(),
        retargeter={
            "left": HandToArm(mount=Mount.aligned(), hand=Hand(closed=0.02, open=0.09, span=0.19)),
            "right": HandToArm(mount=Mount.aligned(), hand=Hand(closed=0.03, open=0.11, span=0.20)),
        },
    )
    e = c.open("egoactions/handpose/ego/0")
    assert e.array(STATE).shape == (STEPS - 1, 14)
    assert set(c.descriptor().metadata["retargeters"]) == {"left", "right"}


def test_a_missing_retargeter_is_refused():
    with pytest.raises(ConfigError, match="no retargeter"):
        EgoActionConnector(FakeHands(), retargeter={"left": retargeter()})


def test_a_single_arm_set_is_half_the_width():
    c = made(hands=("right",))
    assert c.width == 7
    assert c.open("egoactions/handpose/ego/0").array(ACTION).shape == (STEPS - 1, 7)


def test_a_source_with_no_hands_is_refused_by_name():
    class Bare(FakeHands):
        def open(self, episode_id):
            return EpisodeRecord(
                meta=EpisodeMeta(id=episode_id, source="handpose"),
                schema=(ChannelSpec(RGB, "image", (8, 8, 3), "uint8"),),
                source=ArraySource({RGB: np.zeros((4, 8, 8, 3), dtype="uint8")}),
                labels=EpisodeLabels(),
            )

    with pytest.raises(ConfigError, match="no 'left_wrist' channel"):
        made(source=Bare()).open("egoactions/handpose/ego/0")


def test_the_connector_conforms():
    verdict = check_connector(made())
    assert verdict.ok, verdict.explain()


# -- holding an idle arm -----------------------------------------------------


class OneHanded(FakeHands):
    """The common real case: one hand works, the other is out of frame."""

    def open(self, episode_id):
        e = super().open(episode_id)
        arrays = {n: e.array(n) for n in e.channel_names}
        right = arrays["right_wrist"].copy()
        right[5:] = 0.0  # the right hand leaves the frame and never returns
        arrays["right_wrist"] = right
        return EpisodeRecord(
            meta=e.meta, schema=e.schema, source=ArraySource(arrays), labels=e.labels
        )


def test_strict_mode_keeps_only_steps_where_every_hand_solved():
    """The brutal reading, and the default. On real ego footage each hand solves
    in roughly half the frames and the intersection was 8%."""
    c = made(source=OneHanded())
    with pytest.raises(ConfigError, match="filming problem"):
        c.open("egoactions/handpose/ego/0")


def test_holding_an_idle_arm_recovers_the_one_handed_footage():
    """A person working one-handed leaves the other still, so a bimanual robot
    imitating them should too. A claim about the world, not a convenience."""
    c = made(source=OneHanded(), hold_missing=True)
    e = c.open("egoactions/handpose/ego/0")

    assert e.array(STATE).shape == (STEPS - 1, 14)
    annotations = e.labels.annotations
    assert annotations["both_hands_solved"] == pytest.approx(5 / STEPS)
    assert annotations["held_idle_arm"]["right"] == pytest.approx(15 / STEPS)
    assert annotations["held_idle_arm"]["left"] == 0.0


def test_a_held_arm_stays_exactly_where_it_was():
    """Forward fill, not interpolation. A held arm is a claim that it stayed put,
    which is checkable against the footage; interpolating across a gap invents
    motion that may not have happened."""
    c = made(source=OneHanded(), hold_missing=True)
    state = c.open("egoactions/handpose/ego/0").array(STATE)
    right = state[:, 7:]
    assert np.allclose(right[5:], right[4]), "the idle arm drifted"


def test_holding_is_off_by_default_and_declared_when_on():
    assert made().descriptor().metadata["hold_missing"] is False
    assert made(hold_missing=True).descriptor().metadata["hold_missing"] is True


def test_frames_before_any_hand_was_seen_cannot_be_held_and_are_dropped():
    c = made(source=FakeHands(unsolved=(0, 1, 2)), hold_missing=True)
    e = c.open("egoactions/handpose/ego/0")
    assert e.array(STATE).shape[0] == STEPS - 3 - 1


def test_stored_frames_can_be_shrunk_to_what_a_policy_actually_sees():
    """Estimation wants every pixel; a policy does not. openpi resizes to 224
    internally regardless, so keeping 1080p in the training set costs disk,
    encode time and — the way this was found — 44 GB of RAM across a two-dozen
    clip build, which wedged the machine."""
    c = made(size=(64, 64))
    e = c.open("egoactions/handpose/ego/0")
    assert e.array(RGB).shape[1:] == (64, 64, 3)
    assert e.channel(RGB).shape == (64, 64, 3)
    # and the row counts still line up with the actions
    assert len(e.array(RGB)) == len(e.array(STATE))


def test_caching_can_be_turned_off_for_a_large_build():
    c = made(cache=False)
    first = c.open("egoactions/handpose/ego/0")
    second = c.open("egoactions/handpose/ego/0")
    assert first is not second
    assert np.array_equal(first.array(STATE), second.array(STATE))
