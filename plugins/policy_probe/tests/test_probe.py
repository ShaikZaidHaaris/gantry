"""The one thing the probe has to do: separate real data from its own shuffle.

Everything else about it is a means to that. A probe that cannot tell a
correspondence from no correspondence is not a cheap screening gate, it is a
random number with a price attached.
"""

from __future__ import annotations

import numpy as np
import pytest

from gantry.contracts.policy import Learns, Policy
from gantry.spine import ChannelSpec, EpisodeMeta, EpisodeRecord
from gantry.spine.episode import ArraySource
from gantry_curate_control import shuffled
from gantry_policy_probe import ProbeLearner, features


def episode(uid: str, frames: np.ndarray, actions: np.ndarray) -> EpisodeRecord:
    return EpisodeRecord(
        meta=EpisodeMeta(id=uid, source="t"),
        schema=(
            ChannelSpec("observation.images.cam", "image", frames.shape[1:]),
            ChannelSpec("action", "control", (actions.shape[1],)),
        ),
        source=ArraySource({"observation.images.cam": frames, "action": actions}),
    )


def corpus(count: int = 8, steps: int = 60, seed: int = 0):
    """Episodes where the image genuinely determines the action.

    A bright square moves across the frame and the action is where it is. Crude,
    and exactly the kind of coarse structure a linear probe on eight-by-eight
    pixels is supposed to find.
    """
    rng = np.random.default_rng(seed)
    out = []
    for e in range(count):
        frames = np.zeros((steps, 32, 32, 3), np.uint8)
        actions = np.zeros((steps, 2), np.float32)
        x0, y0 = rng.uniform(2, 20, 2)
        dx, dy = rng.uniform(-0.15, 0.15, 2)
        for t in range(steps):
            x = int(np.clip(x0 + dx * t, 0, 27))
            y = int(np.clip(y0 + dy * t, 0, 27))
            frames[t, y : y + 5, x : x + 5] = 255
            actions[t] = (x / 32.0, y / 32.0)
        out.append(episode(f"e{e}", frames, actions))
    return out


def held_out_error(learner: ProbeLearner, train, test) -> float:
    """Mean squared error of a fit on episodes it never saw."""
    fitted = learner.fit(train, seed=0)
    names = learner.observation_names(train)
    total, steps = 0.0, 0
    for ep in test:
        x = features(ep, names)
        predicted = x @ fitted._weights + fitted._bias
        actual = np.asarray(ep.array("action"), np.float64)
        total += float(((predicted - actual) ** 2).sum())
        steps += len(actual) * actual.shape[1]
    return total / max(1, steps)


def test_it_is_a_learner_and_produces_a_policy():
    assert isinstance(ProbeLearner(), Learns)
    assert isinstance(ProbeLearner().fit(corpus(3, 20)), Policy)


def test_real_data_beats_its_own_shuffled_control():
    """The whole point. Same frames, same actions, only the pairing differs."""
    data = corpus(10, 60, seed=1)
    train, test = data[:8], data[8:]
    learner = ProbeLearner()

    real = held_out_error(learner, train, test)
    control = held_out_error(learner, shuffled(train, seed=3), test)
    assert real < control / 2, f"real {real:.5f} vs shuffled {control:.5f}"


def test_data_with_no_correspondence_does_not_beat_its_control():
    """Actions unrelated to the frames: neither arm should be able to predict,
    so the gate this feeds must not report signal."""
    rng = np.random.default_rng(5)
    data = corpus(10, 60, seed=2)
    scrambled = [
        episode(e.meta.id, e.array("observation.images.cam"), rng.normal(0, 1, (60, 2)).astype(np.float32))
        for e in data
    ]
    train, test = scrambled[:8], scrambled[8:]
    learner = ProbeLearner()

    real = held_out_error(learner, train, test)
    control = held_out_error(learner, shuffled(train, seed=3), test)
    assert real > control / 2, f"real {real:.5f} vs shuffled {control:.5f}"


def test_the_same_episodes_give_the_same_fit():
    """Determinism is what lets the two arms differ in nothing but the pairing.
    A fit with an initialisation or a schedule would introduce a dozen ways for
    them to differ by accident, all indistinguishable from the signal."""
    data = corpus(6, 40, seed=4)
    a = ProbeLearner().fit(data, seed=0)
    b = ProbeLearner().fit(data, seed=0)
    assert np.array_equal(a._weights, b._weights)
    assert np.array_equal(a._bias, b._bias)


def test_fitting_twice_from_one_learner_does_not_contaminate():
    """The treatment arm and its control are two fits from one learner."""
    data = corpus(8, 40, seed=6)
    learner = ProbeLearner()
    first = learner.fit(data, seed=0)
    learner.fit(shuffled(data, seed=1), seed=0)
    again = learner.fit(data, seed=0)
    assert np.array_equal(first._weights, again._weights)


def test_channels_are_read_in_a_fixed_order():
    """Two episodes whose columns disagree make the fit meaningless in a way
    nothing downstream could detect."""
    data = corpus(4, 20, seed=7)
    names = ProbeLearner().observation_names(data)
    assert names == tuple(sorted(names))
    assert "action" not in names


def test_a_channel_only_some_episodes_carry_is_not_used():
    data = corpus(4, 20, seed=8)
    extra = EpisodeRecord(
        meta=EpisodeMeta(id="odd", source="t"),
        schema=(
            ChannelSpec("observation.images.cam", "image", (32, 32, 3)),
            ChannelSpec("observation.state", "state", (4,)),
            ChannelSpec("action", "control", (2,)),
        ),
        source=ArraySource(
            {
                "observation.images.cam": np.zeros((20, 32, 32, 3), np.uint8),
                "observation.state": np.zeros((20, 4), np.float32),
                "action": np.zeros((20, 2), np.float32),
            }
        ),
    )
    assert ProbeLearner().observation_names([*data, extra]) == ("observation.images.cam",)


def test_it_refuses_rather_than_fitting_nothing():
    with pytest.raises(ValueError):
        ProbeLearner().fit([])


def test_long_clips_do_not_decide_the_answer_for_the_upload():
    """A thousand-frame clip contributes a thousand near-identical rows
    otherwise, and the long clips speak for everyone."""
    short = corpus(2, 20, seed=9)
    long_one = corpus(1, 900, seed=10)
    fitted = ProbeLearner(per_episode=50).fit([*short, *long_one], seed=0)
    assert fitted.descriptor().metadata["fitted_on_steps"] <= 20 + 20 + 50
