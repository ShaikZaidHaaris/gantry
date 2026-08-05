"""Two hands cannot be in the same place, and other things ego video breaks.

Whole-body pose models assume a third-person view of a whole person. Egocentric
footage has no body and no face, so the model fits its skeleton to a hand
close-up and puts both hand sets on the one visible hand. That is not an edge
case here -- measured on real EPIC footage it happened in 54% of frames.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_connector_handpose import APART, centroid, separate

STEPS = 6


def hand_at(x, y, size=100.0):
    """Twenty-one keypoints in a blob of the given size at the given place."""
    rng = np.random.default_rng(0)
    return rng.uniform(-size / 2, size / 2, size=(21, 2)) + np.array([x, y])


def pair(left_xy, right_xy, left_score=0.7, right_score=0.5, steps=STEPS):
    left = np.stack([hand_at(*left_xy) for _ in range(steps)])
    right = np.stack([hand_at(*right_xy) for _ in range(steps)])
    return {
        "left": (left, np.full(steps, left_score, dtype="float32")),
        "right": (right, np.full(steps, right_score, dtype="float32")),
    }


def test_two_hands_in_the_same_place_become_one():
    """The bug a person spotted by watching the render: both skeletons drawn on
    one hand. Keeping both would feed a phantom trajectory into training as
    though a second arm had been doing something."""
    found = separate(pair((500, 500), (505, 502)))
    left, left_score = found["left"]
    right, right_score = found["right"]

    assert np.all(left_score > 0), "the more confident hand survives"
    assert np.all(right_score == 0), "the phantom is dropped"
    assert np.all(right == 0)


def test_the_more_confident_hand_is_the_one_kept():
    found = separate(pair((500, 500), (505, 502), left_score=0.2, right_score=0.9))
    assert np.all(found["left"][1] == 0)
    assert np.all(found["right"][1] > 0)


def test_two_hands_genuinely_apart_are_both_kept():
    """A person working with both hands is the case this must not break."""
    found = separate(pair((300, 500), (1200, 500)))
    assert np.all(found["left"][1] > 0)
    assert np.all(found["right"][1] > 0)


def test_the_threshold_is_relative_to_hand_size():
    """A 200 px gap is two hand-widths apart for a small hand and half a
    hand-width for a large one, and only the second is a collision."""
    small = {
        "left": (
            np.stack([hand_at(500, 500, size=60) for _ in range(2)]),
            np.full(2, 0.7, dtype="float32"),
        ),
        "right": (
            np.stack([hand_at(700, 500, size=60) for _ in range(2)]),
            np.full(2, 0.5, dtype="float32"),
        ),
    }
    big = {
        "left": (
            np.stack([hand_at(500, 500, size=600) for _ in range(2)]),
            np.full(2, 0.7, dtype="float32"),
        ),
        "right": (
            np.stack([hand_at(700, 500, size=600) for _ in range(2)]),
            np.full(2, 0.5, dtype="float32"),
        ),
    }
    assert np.all(separate(small)["right"][1] > 0), "small hands 200px apart are two hands"
    assert np.all(separate(big)["right"][1] == 0), "big hands 200px apart are one hand"


def test_a_frame_where_only_one_hand_was_detected_is_left_alone():
    found = pair((500, 500), (505, 502))
    found["right"][1][:] = 0.0
    found["right"][0][:] = 0.0
    out = separate(found)
    assert np.all(out["left"][1] > 0)


def test_centroid_ignores_absent_keypoints():
    points = np.zeros((21, 2))
    points[:5] = [100.0, 200.0]
    assert centroid(points) == pytest.approx([100.0, 200.0])
    assert np.isnan(centroid(np.zeros((21, 2)))).all()


def test_the_threshold_is_a_fact_about_hands_not_a_tuned_number():
    # Half a hand-width. Two hands closer than that are one hand.
    assert APART == 0.5
