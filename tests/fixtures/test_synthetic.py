from __future__ import annotations

import numpy as np
import pytest

from gantry.fixtures import (
    known_defects,
    make_clean,
    make_defective,
    make_duration_confound,
    make_suite,
)
from gantry.fixtures import statistics as stats
from gantry.spine import compatible

BEHAVIOURAL = ["path_detour", "actuation_chatter", "actuation_jerk", "late_engagement", "never_completes"]
SCHEMA = ["unit_drift", "frame_drift"]


def paired(defect: str, n: int = 8, seed: int = 3):
    """Same seeds, defect on or off: episode i differs only by the mutation.

    Comparing the defective half of a suite against its clean half compares
    different random trajectories, so a difference proves nothing. Drafting is
    seeded before any mutation runs, so this pairing isolates the defect.
    """
    hurt = make_defective(defect, n=n, fraction=1.0, seed=seed)
    well = make_clean(n=n, seed=seed)
    return list(zip(hurt.episodes, well.episodes))


# -- the records themselves ------------------------------------------------


def test_clean_suite_is_valid_all_the_way_down():
    suite = make_clean(n=8)
    assert suite.validate(deep=True).ok


def test_episodes_carry_stage_events_in_order():
    episode = make_clean(n=1).episodes[0]
    assert episode.labels.stages == ("approach", "engage", "transport", "release")
    steps = [episode.labels.step_of(name) for name in episode.labels.stages]
    assert steps == sorted(steps)


def test_stage_vocabulary_is_not_baked_in():
    suite = make_suite(n=4, stages=("incise", "grip", "suture", "close"))
    assert suite.episodes[0].labels.stages == ("incise", "grip", "suture", "close")
    assert suite.validate(deep=True).ok


def test_image_channel_is_exercised_when_asked():
    episode = make_clean(n=2, include_view=True).episodes[0]
    assert episode.channel("view").kind == "image"
    assert episode.array("view").shape[1:] == (8, 8, 3)
    assert episode.validate(deep=True).ok


# -- determinism -----------------------------------------------------------


def test_same_seed_gives_byte_identical_arrays():
    left = make_clean(n=4, seed=7).episodes[2].array("position")
    right = make_clean(n=4, seed=7).episodes[2].array("position")
    assert np.array_equal(left, right)


def test_different_seeds_differ():
    left = make_clean(n=4, seed=1).episodes[0].array("position")
    right = make_clean(n=4, seed=2).episodes[0].array("position")
    assert not np.array_equal(left, right)


def test_episode_ids_are_namespaced_by_source():
    suite = make_clean(n=2, source="vendor-a")
    assert suite.episodes[0].meta.uid == "vendor-a/ep_0000"


# -- the answer key is honest ---------------------------------------------


@pytest.mark.parametrize("defect", BEHAVIOURAL + SCHEMA)
def test_every_catalogued_defect_is_actually_planted(defect):
    suite = make_defective(defect, n=16, fraction=0.5)
    verdict = suite.verify()
    assert verdict.ok, verdict.explain()


@pytest.mark.parametrize("defect", BEHAVIOURAL + SCHEMA)
def test_ground_truth_partitions_the_suite(defect):
    suite = make_defective(defect, n=16, fraction=0.25)
    assert len(suite.with_defect(defect)) == 4
    assert len(suite.clean()) == 12
    assert suite.truth.defects == (defect,)


def test_clean_suite_claims_nothing():
    suite = make_clean(n=6)
    assert suite.truth.defects == ()
    assert len(suite.clean()) == 6
    assert suite.verify().ok


def test_verify_catches_a_lying_answer_key():
    """A ground truth that claims a defect nobody carries must fail loudly."""
    import dataclasses

    suite = make_clean(n=6)
    liar = dataclasses.replace(
        suite,
        truth=dataclasses.replace(
            suite.truth, planted={**suite.truth.planted, "fixture/ep_0000": ("path_detour",)}
        ),
    )
    verdict = liar.verify()
    assert not verdict.ok
    assert "fixture.not_planted" in verdict.codes()


def test_unknown_defect_names_the_alternatives():
    with pytest.raises(KeyError, match="path_detour"):
        make_defective("wishful_thinking", n=4)


# -- the defects are what they say -----------------------------------------


def test_detour_lengthens_the_path_without_moving_the_endpoints():
    for hurt, well in paired("path_detour"):
        assert stats.path_efficiency(hurt) < stats.path_efficiency(well)
        # the endpoints are untouched: only the route between them changed
        assert hurt.array("position")[-1] == pytest.approx(well.array("position")[-1], abs=1e-5)


def test_jerk_lives_in_the_command_not_the_motion():
    for hurt, well in paired("actuation_jerk"):
        assert stats.action_jerk(hurt) > stats.action_jerk(well)
        # the path itself is identical, which is what makes this defect distinct
        assert stats.path_efficiency(hurt) == pytest.approx(stats.path_efficiency(well), abs=1e-6)


def test_chatter_only_touches_the_actuator():
    for hurt, well in paired("actuation_chatter"):
        assert stats.engagement_transition_rate(hurt) > stats.engagement_transition_rate(well)
        assert stats.path_efficiency(hurt) == pytest.approx(stats.path_efficiency(well), abs=1e-6)


def test_incomplete_episodes_are_labelled_unsuccessful():
    suite = make_defective("never_completes", n=8, fraction=0.5)
    assert all(e.labels.success is False for e in suite.with_defect("never_completes"))
    assert all(e.labels.success is True for e in suite.clean())


def test_truncated_episodes_drop_the_stages_they_never_reached():
    suite = make_defective("never_completes", n=8, fraction=0.5)
    hurt = suite.with_defect("never_completes")[0]
    assert "release" not in hurt.labels.stages
    assert hurt.validate(deep=True).ok


# -- schema defects are invisible to statistics, visible to the resolver ---


def test_unit_drift_is_caught_by_compatibility_not_by_any_statistic():
    pairs = paired("unit_drift")
    for hurt, well in pairs:
        # the motion is the same shape; only the declared unit differs, and no
        # scale-invariant statistic can possibly see it
        assert stats.path_efficiency(hurt) == pytest.approx(stats.path_efficiency(well), abs=1e-6)
    hurt, well = pairs[0]
    verdict = compatible(hurt.channel("position"), well.channel("position"))
    assert "units.scale" in verdict.codes()
    assert verdict.because("units.scale")[0].detail["factor"] == pytest.approx(1e-3)


def test_frame_drift_refuses_with_a_frame_code():
    hurt, well = paired("frame_drift")[0]
    assert np.array_equal(hurt.array("position"), well.array("position"))
    assert "frame.mismatch" in compatible(hurt.channel("position"), well.channel("position")).codes()


# -- the decoy: what a detector must NOT report ----------------------------


def test_duration_confound_traps_raw_counts_and_spares_rates():
    """The regression test for a real false positive.

    Every episode behaves identically per step; only length varies. A statistic
    computed as a raw count tracks duration almost perfectly, and reporting it
    would be reporting nothing. The per-step rate does not move.
    """
    suite = make_duration_confound(n=20)
    assert suite.truth.decoys == ("duration_spread",)
    assert suite.truth.defects == ()

    durations = [stats.duration(e) for e in suite]
    assert max(durations) > 2 * min(durations)

    counts = [stats.direction_changes(e) for e in suite]
    rates = [stats.direction_change_rate(e) for e in suite]

    assert abs(stats.correlation(durations, counts)) > 0.8
    assert abs(stats.correlation(durations, rates)) < 0.3


def test_decoy_suite_still_verifies_as_clean():
    suite = make_duration_confound(n=12)
    assert suite.verify().ok


# -- reporting -------------------------------------------------------------


def test_summary_is_json_able_and_names_the_decoys():
    import json

    summary = make_duration_confound(n=6).summary()
    assert json.loads(json.dumps(summary))["decoys"] == ["duration_spread"]
    assert "raw counts will correlate" in summary["decoy_descriptions"]["duration_spread"]


def test_catalogue_is_discoverable():
    assert set(known_defects()) == set(BEHAVIOURAL + SCHEMA)
