"""What the shuffled control promises, held to.

The control's whole value is three properties at once: every episode LOSES its
own actions (a fixed point is training data hiding inside the control arm),
the cohort's action set is UNCHANGED (only the pairing is destroyed), and the
swap is TRACEABLE (a control that cannot be traced is indistinguishable from a
bug). Each is tested on its own, because each fails on its own.
"""

import numpy as np
import pytest

from gantry.spine import ChannelSpec, EpisodeMeta, EpisodeRecord
from gantry.spine.episode import ArraySource
from gantry_curate_control import DetachedSource, shuffled


def episode(index: int, steps: int = 20) -> EpisodeRecord:
    """Frames and actions each stamped with the episode's index, so after a
    shuffle the contents say exactly which parent each channel came from."""
    frames = np.full((steps, 4, 4, 3), index, np.uint8)
    actions = np.full((steps, 2), float(index), np.float32)
    return EpisodeRecord(
        meta=EpisodeMeta(id=f"ep{index}", source="t"),
        schema=(
            ChannelSpec("observation.images.cam", "image", frames.shape[1:]),
            ChannelSpec("action", "control", (actions.shape[1],)),
        ),
        source=ArraySource({"observation.images.cam": frames, "action": actions}),
    )


def donors_of(controls):
    """Which episode's actions each control carries, read from the data."""
    return [int(c.array("action")[0, 0]) for c in controls]


def test_every_episode_loses_its_own_actions():
    for count in (2, 3, 5, 9):
        for seed in (0, 1, 7):
            controls = shuffled([episode(i) for i in range(count)], seed=seed)
            donors = donors_of(controls)
            assert all(d != i for i, d in enumerate(donors)), (
                "a fixed point is not a control, it is training data inside "
                "the control arm", count, seed, donors)


def test_the_cohorts_action_set_is_unchanged():
    controls = shuffled([episode(i) for i in range(6)], seed=2)
    assert sorted(donors_of(controls)) == list(range(6)), (
        "the shuffle must move actions, never add, drop or duplicate them")


def test_frames_stay_with_their_episode():
    controls = shuffled([episode(i) for i in range(4)], seed=0)
    for i, c in enumerate(controls):
        assert int(c.array("observation.images.cam")[0, 0, 0, 0]) == i


def test_two_episodes_can_only_swap():
    donors = donors_of(shuffled([episode(0), episode(1)], seed=5))
    assert donors == [1, 0]


def test_fewer_than_two_episodes_refuses():
    with pytest.raises(ValueError, match="at least two"):
        shuffled([])
    with pytest.raises(ValueError, match="at least two"):
        shuffled([episode(0)])


def test_lengths_truncate_to_the_shorter_never_pad():
    long, short = episode(0, steps=30), episode(1, steps=12)
    controls = shuffled([long, short], seed=0)
    # ep0 keeps 30 frames but borrows 12 actions; inventing the other 18
    # steps would build the control out of data nobody recorded
    assert controls[0].source.num_steps == 12
    assert controls[1].source.num_steps == 12
    assert controls[0].array("action").shape[0] == 12
    # and an out-of-range read clamps rather than fabricates
    reads = controls[0].source.read(["action"], start=0, stop=999)
    assert reads["action"].shape[0] == 12


def test_the_swap_is_traceable_to_both_parents():
    eps = [episode(i) for i in range(4)]
    controls = shuffled(eps, seed=3)
    donors = donors_of(controls)
    for i, c in enumerate(controls):
        assert c.meta.id == f"ep{i}+shuffled"
        assert c.meta.extra["control"] == "shuffled"
        expected = str(getattr(eps[donors[i]].meta, "uid", "?"))
        assert c.meta.extra["actions_from"] == expected, (
            "the recorded donor must be the episode the actions actually "
            "came from", i, donors[i])


def test_same_seed_same_control_and_seeds_differ():
    eps = [episode(i) for i in range(6)]
    first = donors_of(shuffled(eps, seed=4))
    assert donors_of(shuffled(eps, seed=4)) == first, "a control must replay"
    assert any(donors_of(shuffled(eps, seed=s)) != first for s in range(8)), (
        "every seed producing one assignment means the derangement is a "
        "constant an upload could be ordered to exploit")


def test_channel_reads_route_to_the_right_parent():
    keeps, donor = episode(0), episode(1)
    source = DetachedSource(keeps.source, donor.source)
    assert set(source.channels) == {"observation.images.cam", "action"}
    only_action = source.read(["action"])
    assert list(only_action) == ["action"] and only_action["action"][0, 0] == 1.0
    only_frames = source.read(["observation.images.cam"])
    assert only_frames["observation.images.cam"][0, 0, 0, 0] == 0
    both = source.read()
    assert both["action"][0, 0] == 1.0
    assert both["observation.images.cam"][0, 0, 0, 0] == 0
