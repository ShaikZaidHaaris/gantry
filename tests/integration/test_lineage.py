"""Identity across a format conversion, forwards and backwards.

Forwards: a converter records where each episode came from, so a plan written
about one format applies in the other.

Backwards: for datasets converted before anyone recorded it, the connection is
recovered from content — the actions are the same numbers whatever the copy
calls them.

The thing being defended is narrow and expensive. Without it, a drop-list built
from success labels — which usually survive only in a collection's native
format — can never be applied to the converted copy a trainer actually reads.
The two datasets contain the same demonstrations and share no names.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_connector_lerobot import LeRobotConnector
from gantry_connector_lerobot.testing import build_dataset

from gantry.contracts.curation import CurationAction, CurationPlan, Prediction
from gantry.curate import apply, names_of
from gantry.lineage import fingerprint, relink, rename
from gantry.spine import EpisodeMeta, IncompatibleError


def converted(tmp_path, name="a", episodes=5, steps=6):
    build_dataset(tmp_path / name, episodes=episodes, steps=steps)
    return LeRobotConnector(tmp_path / name)


def plan_dropping(uids):
    return CurationPlan(
        actions=(CurationAction("drop", episodes=tuple(uids)),),
        signal="labels", rung="screening",
        predicted=Prediction(magnitude=0.1, tasks=("lift_cube",)),
    )


# -- the meta itself ---------------------------------------------------------


def test_an_episode_answers_to_the_names_it_used_to_have():
    meta = EpisodeMeta(id="episode_000000", source="curated").descended("mg/demo_1")
    assert meta.uid == "curated/episode_000000"
    assert meta.lineage == ("mg/demo_1", "curated/episode_000000")
    assert meta.known_as("mg/demo_1")
    assert meta.known_as("curated/episode_000000")
    assert not meta.known_as("somewhere/else")


def test_a_chain_of_conversions_stays_walkable_end_to_end():
    meta = (
        EpisodeMeta(id="e0", source="c")
        .descended("mg/demo_1")
        .descended("lerobot/episode_000000")
    )
    assert meta.lineage == ("mg/demo_1", "lerobot/episode_000000", "c/e0")
    assert meta.known_as("mg/demo_1")


def test_recording_the_same_ancestor_twice_does_not_grow_the_chain():
    meta = EpisodeMeta(id="e0", source="c").descended("mg/demo_1").descended("mg/demo_1")
    assert meta.lineage == ("mg/demo_1", "c/e0")


def test_an_episode_from_a_producer_that_recorded_nothing_has_only_its_own_name():
    """The default is empty rather than a guess. Most datasets are like this."""
    meta = EpisodeMeta(id="e0", source="c")
    assert meta.derived_from == ()
    assert meta.lineage == ("c/e0",)


# -- forwards: the converter records it --------------------------------------


def test_a_written_copy_remembers_what_it_was_called(tmp_path):
    original = list(converted(tmp_path, "a", episodes=3))
    LeRobotConnector.write(original, tmp_path / "b", accept_loss=True)
    copy = list(LeRobotConnector(tmp_path / "b"))
    assert copy[0].meta.uid == "b/episode_000000"
    assert copy[0].meta.known_as(original[0].meta.uid)


def test_a_plan_written_about_the_source_applies_to_the_copy(tmp_path):
    """The whole point. Same demonstrations, different names, one plan."""
    original = list(converted(tmp_path, "a", episodes=5))
    LeRobotConnector.write(original, tmp_path / "b", accept_loss=True)
    copy = list(LeRobotConnector(tmp_path / "b"))

    plan = plan_dropping([original[0].meta.uid, original[1].meta.uid])
    survivors, applied = apply(plan, copy)
    assert len(survivors) == 3
    assert not applied.missing


def test_a_plan_naming_something_neither_copy_ever_was_is_still_refused(tmp_path):
    original = list(converted(tmp_path, "a", episodes=3))
    LeRobotConnector.write(original, tmp_path / "b", accept_loss=True)
    copy = list(LeRobotConnector(tmp_path / "b"))
    with pytest.raises(IncompatibleError, match="curation.unactionable"):
        apply(plan_dropping(["nowhere/demo_9"]), copy)


def test_names_of_falls_back_gracefully_for_anything_without_a_lineage():
    class Bare:
        uid = "x/1"

    assert names_of(Bare()) == ("x/1",)


# -- backwards: recovering it from content -----------------------------------


def test_two_copies_of_the_same_demonstration_have_the_same_fingerprint(tmp_path):
    original = list(converted(tmp_path, "a", episodes=3))
    LeRobotConnector.write(original, tmp_path / "b", accept_loss=True)
    copy = list(LeRobotConnector(tmp_path / "b"))
    assert fingerprint(original[0]) == fingerprint(copy[0])
    assert fingerprint(original[0]) != fingerprint(original[1])


def test_a_converted_copy_is_reconnected_to_its_source(tmp_path):
    original = list(converted(tmp_path, "a", episodes=5))
    LeRobotConnector.write(original[:3], tmp_path / "b", accept_loss=True)
    subset = list(LeRobotConnector(tmp_path / "b"))

    found = relink(original, subset)
    assert len(found.links) == 3
    assert found.links["b/episode_000000"] == "a/episode_000000"
    assert not found.unmatched


def test_a_plan_is_translated_into_the_targets_vocabulary(tmp_path):
    """A drop-list built where the labels are, applied where the trainer reads."""
    original = list(converted(tmp_path, "a", episodes=5))
    LeRobotConnector.write(original, tmp_path / "b", accept_loss=True)
    copy = list(LeRobotConnector(tmp_path / "b"))

    found = relink(original, copy)
    doomed = [original[0].meta.uid, original[3].meta.uid]
    translated = rename(doomed, found.links)
    assert translated == ("b/episode_000000", "b/episode_000003")

    survivors, applied = apply(plan_dropping(translated), copy)
    assert len(survivors) == 3
    assert not applied.missing


def test_an_episode_with_no_match_is_reported_rather_than_guessed_at(tmp_path):
    original = list(converted(tmp_path, "a", episodes=3))
    other = list(converted(tmp_path, "c", episodes=2, steps=9))
    found = relink(original, other)
    assert not found.links
    assert len(found.unmatched) == 2
    verdict = found.validate(target_size=2)
    assert not verdict.ok
    assert "lineage.nothing_matched" in verdict.codes()


def test_identical_demonstrations_are_left_unlinked_rather_than_linked_wrongly(tmp_path):
    """A wrong link attaches a measurement to the wrong demonstration, and
    everything downstream is quietly about something else."""

    class Fake:
        def __init__(self, uid, values):
            self.meta = EpisodeMeta(id=uid, source="s")
            self._values = np.asarray(values, dtype=np.float64)

        @property
        def schema(self):
            class Spec:
                name = "actions"

            return (Spec(),)

        def read(self, channels=None, start=0, stop=None):
            return {"actions": self._values}

    twins = [Fake("a", [[1.0, 2.0]]), Fake("b", [[1.0, 2.0]])]
    target = [Fake("copy", [[1.0, 2.0]])]
    found = relink(twins, target)
    assert not found.links
    assert found.ambiguous == ("s/copy",)
    assert "lineage.ambiguous" in found.validate(target_size=1).codes()


def test_a_float_width_change_does_not_break_the_match(tmp_path):
    """A conversion writing float32 where the source held float64 changes the
    last bits without changing the demonstration."""
    original = list(converted(tmp_path, "a", episodes=2))
    LeRobotConnector.write(original, tmp_path / "b", accept_loss=True)
    copy = list(LeRobotConnector(tmp_path / "b"))
    assert len(relink(original, copy).links) == 2
