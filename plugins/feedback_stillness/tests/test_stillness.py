"""The defect this module exists for, and the ways it must not overreach."""

from __future__ import annotations

import numpy as np
import pytest

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeMeta, EpisodeRecord
from gantry.spine.episode import ArraySource
from gantry_feedback_stillness import Stillness, blocks_in, still_dims


def episode(values: np.ndarray, uid: str = "e", labels=None) -> EpisodeRecord:
    spec = ChannelSpec(name="action", kind="control", shape=(values.shape[1],), dim_labels=labels)
    return EpisodeRecord(
        meta=EpisodeMeta(id=uid, source="test"),
        schema=(spec,),
        source=ArraySource({"action": values.astype(np.float32)}),
    )


def moving(n: int = 40, width: int = 14, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.05, (n, width)), axis=0)


def freeze(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = values.copy()
    out[:, start:stop] = out[0, start:stop]
    return out


def test_finds_a_held_block_and_names_the_clips():
    clips = [episode(moving(seed=i), f"clip{i}") for i in range(10)]
    clips[2] = episode(freeze(moving(seed=2), 0, 7), "clip2")
    clips[5] = episode(freeze(moving(seed=5), 0, 7), "clip5")
    report = Stillness().analyse([Cohort(name="yours", episodes=tuple(clips))])

    frozen = [f for f in report.findings if f.code == "stillness.frozen_block"]
    assert len(frozen) == 1
    # Namespaced uids, because two datasets that both number from zero collapse
    # into each other otherwise, and a named clip has to be findable.
    assert frozen[0].evidence["episodes"] == ["test/clip2", "test/clip5"]
    # The clips are named in the prescription, not just counted. "Re-film some
    # of your clips" is not something anyone can act on.
    assert "clip2" in frozen[0].prescription and "clip5" in frozen[0].prescription
    assert frozen[0].measurements["clips"].n == 10


def test_a_clean_cohort_says_so_rather_than_staying_silent():
    report = Stillness().analyse(
        [Cohort(name="yours", episodes=tuple(episode(moving(seed=i), f"c{i}") for i in range(6)))]
    )
    assert [f.code for f in report.findings] == ["stillness.none"]


def test_one_constant_dimension_is_not_a_finding():
    """An unused gripper is constant and correct. Only limb-width runs count."""
    clips = [episode(freeze(moving(seed=i), 6, 7), f"c{i}") for i in range(8)]
    report = Stillness().analyse([Cohort(name="yours", episodes=tuple(clips))])
    assert [f.code for f in report.findings] == ["stillness.none"]


def test_a_single_affected_clip_is_below_the_bar():
    """One still clip in twenty is a clip; a third of them is a filming problem."""
    clips = [episode(moving(seed=i), f"c{i}") for i in range(20)]
    clips[0] = episode(freeze(moving(seed=0), 0, 7), "c0")
    report = Stillness().analyse([Cohort(name="yours", episodes=tuple(clips))])
    assert not [f for f in report.findings if f.code == "stillness.frozen_block"]
    # Measured all the same, so the number exists even where the finding does not.
    assert any("frozen[" in key for key in report.measurements)


def test_a_wholly_motionless_channel_is_its_own_defect():
    flat = np.tile(moving(seed=1)[0], (40, 1))
    clips = [episode(flat, f"c{i}") for i in range(4)]
    codes = {f.code for f in Stillness().analyse([Cohort(name="yours", episodes=tuple(clips))]).findings}
    assert "stillness.nothing_moved" in codes
    assert "stillness.frozen_block" not in codes


def test_declared_labels_are_read_rather_than_the_position_guessed():
    labels = tuple(f"left_{i}" for i in range(7)) + tuple(f"right_{i}" for i in range(7))
    clips = [episode(freeze(moving(seed=i), 0, 7), f"c{i}", labels) for i in range(8)]
    frozen = [
        f
        for f in Stillness().analyse([Cohort(name="y", episodes=tuple(clips))]).findings
        if f.code == "stillness.frozen_block"
    ]
    assert "left" in frozen[0].summary


def test_undeclared_labels_report_indices_rather_than_inventing_a_limb():
    clips = [episode(freeze(moving(seed=i), 0, 7), f"c{i}") for i in range(8)]
    frozen = [
        f
        for f in Stillness().analyse([Cohort(name="y", episodes=tuple(clips))]).findings
        if f.code == "stillness.frozen_block"
    ]
    assert "dimensions 0-6" in frozen[0].summary


def test_it_refuses_rather_than_reporting_nothing_found():
    """No channel to read and no frozen blocks are the same empty list."""
    bare = EpisodeRecord(
        meta=EpisodeMeta(id="x", source="test"),
        schema=(ChannelSpec(name="observation.images.cam", kind="image", shape=(4, 4, 3)),),
        source=ArraySource({"observation.images.cam": np.zeros((5, 4, 4, 3), dtype=np.uint8)}),
    )
    verdict = Stillness().check_inputs([Cohort(name="y", episodes=(bare,))])
    assert not verdict.ok
    assert [r.code for r in verdict.reasons] == ["stillness.no_control_channel"]


def test_an_unreadable_clip_is_a_note_not_a_pass():
    class Broken(ArraySource):
        def read(self, channels=None, start=0, stop=None):
            raise OSError("truncated")

    good = [episode(moving(seed=i), f"c{i}") for i in range(4)]
    bad = EpisodeRecord(
        meta=EpisodeMeta(id="torn", source="test"),
        schema=(ChannelSpec(name="action", kind="control", shape=(14,)),),
        source=Broken({"action": np.zeros((4, 14), dtype=np.float32)}),
    )
    report = Stillness().analyse([Cohort(name="y", episodes=(*good, bad))])
    assert any("not judged either way" in note for note in report.notes)


@pytest.mark.parametrize(
    "mask, expect",
    [
        ([1, 1, 1, 0, 0], [(0, 3)]),
        ([0, 0, 1, 1, 1], [(2, 5)]),
        ([1, 1, 0, 1, 1, 1], [(3, 6)]),          # the 2-wide run is below MIN_BLOCK
        ([1, 1, 1, 0, 1, 1, 1], [(0, 3), (4, 7)]),
    ],
)
def test_blocks_are_maximal_runs(mask, expect):
    found = blocks_in(np.array(mask, dtype=bool))
    assert [(b.start, b.stop) for b in found] == expect


def test_spread_not_step_size():
    """A dimension alternating between two values has moved; one held has not."""
    values = np.zeros((10, 2))
    values[:, 0] = np.tile([0.0, 1.0], 5)
    assert still_dims(values).tolist() == [False, True]
