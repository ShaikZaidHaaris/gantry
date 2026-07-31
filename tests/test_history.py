"""What history remembers, and what it refuses to invent.

The fixtures are this session's real measurements — ph 28/50 and mh 33/50 on the
training region, 9/50 and 29/50 on the wider one, mg at zero. Using them rather
than round numbers keeps the tests honest about the regime the code runs in: at
these sample sizes the difference between remembering and guessing is the
difference between a finding and a mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from gantry.history import History, matrix_of, summarise
from gantry.spine import ComponentRef, Provenance, proportion
from gantry.spine.inference import iqm, trials_needed


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)


@dataclass
class Meta:
    id: str
    task: str | None = None


@dataclass
class Ep:
    meta: Meta
    labels: Labels


@dataclass
class Run:
    provenance: Provenance
    episodes: tuple
    metrics: dict


def run(task, policy, outcomes, *, scorer="machine", embodiment="panda", version="1.0"):
    components = [
        ComponentRef("policy", policy, version),
        ComponentRef("embodiment", embodiment, "1.0"),
    ]
    if scorer:
        components.append(ComponentRef("scorer", scorer, "1.0"))
    episodes = tuple(
        Ep(Meta(f"seed_{i}", task), Labels(outcome)) for i, outcome in enumerate(outcomes)
    )
    scored = [o for o in outcomes if o is not None]
    return Run(
        Provenance(components=tuple(components)),
        episodes,
        {"success_rate": proportion(sum(1 for o in scored if o), len(scored))} if scored else {},
    )


def measured(tmp_path) -> tuple[History, dict[str, str]]:
    """This session's Lift results, as recorded history."""
    h = History(tmp_path / "history")
    keys = {
        "ph": h.put(run("lift_cube", "ph_official", [True] * 28 + [False] * 22), keep_record=False),
        "mh": h.put(run("lift_cube", "mh_official", [True] * 33 + [False] * 17), keep_record=False),
        "mg": h.put(run("lift_cube", "mg_official", [False] * 50), keep_record=False),
        "ph_wide": h.put(
            run("lift_cube_wide", "ph_official", [True] * 9 + [False] * 41), keep_record=False
        ),
        "mh_wide": h.put(
            run("lift_cube_wide", "mh_official", [True] * 29 + [False] * 21), keep_record=False
        ),
    }
    return h, keys


# -- recording ---------------------------------------------------------------


def test_a_run_becomes_one_row(tmp_path):
    h, keys = measured(tmp_path)
    assert len(h) == 5
    row = h.get(keys["ph"])
    assert row.task == "lift_cube"
    assert row.policy == "ph_official"
    assert row.rate == pytest.approx(0.56)
    assert row.trials == 50


def test_recording_the_same_run_twice_is_one_row(tmp_path):
    """Content-addressed, so a rerun of the same experiment does not inflate
    the count of evidence it appears to rest on."""
    h = History(tmp_path / "history")
    record = run("lift_cube", "ph_official", [True] * 28 + [False] * 22)
    first = h.put(record, keep_record=False)
    again = h.put(record, keep_record=False)
    assert first == again
    assert len(h) == 1


def test_the_full_record_is_kept_beside_the_summary(tmp_path):
    """Uses the real record type, because the archival path goes through the
    versioned writer and a stand-in would not exercise it."""
    from gantry.spine import EpisodeLabels, RunRecord, episode_from_labels

    h = History(tmp_path / "history")
    record = RunRecord(
        provenance=Provenance(
            components=(
                ComponentRef("policy", "ph_official", "1.0"),
                ComponentRef("scorer", "machine", "1.0"),
            )
        ),
        episodes=tuple(
            episode_from_labels(
                id=f"seed_{i}",
                source="lift_cube",
                labels=EpisodeLabels(success=outcome),
                task="lift_cube",
            )
            for i, outcome in enumerate([True, False, True])
        ),
        metrics={"success_rate": proportion(2, 3)},
    )
    key = h.put(record)
    back = h.record_for(key)
    assert back is not None
    assert len(back.episodes) == 3
    assert h.get(key).scorer == "machine"


def test_a_summary_alone_can_be_stored_when_the_arrays_are_not_wanted(tmp_path):
    h = History(tmp_path / "history")
    key = h.put(run("lift_cube", "ph", [True]), keep_record=False)
    assert h.get(key) is not None
    assert h.record_for(key) is None


# -- the query surface -------------------------------------------------------


def test_it_filters_on_bare_names_not_version_strings(tmp_path):
    """Nobody remembers a version string, and a filter that silently matches
    nothing is worse than one that errors."""
    h, _ = measured(tmp_path)
    assert len(h.query(policy="ph_official")) == 2
    assert len(h.query(task="lift_cube")) == 3
    assert len(h.query(policy="mh_official", task="lift_cube_wide")) == 1


def test_the_version_is_still_recorded(tmp_path):
    h = History(tmp_path / "history")
    key = h.put(run("lift_cube", "ph_official", [True], version="2.3"), keep_record=False)
    assert h.get(key).metadata["policy_ref"] == "ph_official@2.3"


def test_a_typo_in_a_filter_is_refused_rather_than_matching_nothing(tmp_path):
    h, _ = measured(tmp_path)
    with pytest.raises(ValueError, match="no field"):
        h.query(polcy="ph_official")


def test_an_empty_history_answers_without_raising(tmp_path):
    h = History(tmp_path / "nothing")
    assert len(h) == 0
    assert h.query(task="lift_cube") == ()
    assert h.rate_for("lift_cube") is None
    assert h.baseline_for("lift_cube") is None
    assert "empty" in h.report()


# -- baselines stop being folklore -------------------------------------------


def test_a_pinned_run_is_the_baseline(tmp_path):
    h, keys = measured(tmp_path)
    h.pin(keys["ph"], task="lift_cube")
    assert h.baseline_for("lift_cube").policy == "ph_official"


def test_a_pin_survives_later_runs(tmp_path):
    """A gate that silently re-baselines on the newest run is not a gate."""
    h, keys = measured(tmp_path)
    h.pin(keys["ph"], task="lift_cube")
    h.put(run("lift_cube", "something_new", [True] * 40 + [False] * 10), keep_record=False)
    assert h.baseline_for("lift_cube").policy == "ph_official"


def test_a_body_specific_pin_beats_a_general_one(tmp_path):
    h, keys = measured(tmp_path)
    h.pin(keys["mg"], task="lift_cube")  # any body
    h.pin(keys["ph"], task="lift_cube", embodiment="panda")  # this body
    assert h.baseline_for("lift_cube", "panda").policy == "ph_official"
    assert h.baseline_for("lift_cube", "sawyer").policy == "mg_official"


# -- sizing stops needing a guess --------------------------------------------


def test_the_historical_rate_comes_from_evidence(tmp_path):
    h, _ = measured(tmp_path)
    rate = h.rate_for("lift_cube")
    # (0.56 + 0.66 + 0.00) / 3
    assert rate.value == pytest.approx(0.4067, abs=1e-3)
    assert rate.detail["runs"] == 3
    assert rate.detail["min"] == 0.0 and rate.detail["max"] == pytest.approx(0.66)
    assert rate.n == 150


def test_an_untested_task_returns_nothing_rather_than_a_default(tmp_path):
    """An invented baseline rate produces an invented trial count, and an
    invented trial count is how an underpowered run gets approved."""
    h, _ = measured(tmp_path)
    assert h.rate_for("open_door") is None


def test_the_spread_is_reported_so_a_single_run_is_distinguishable(tmp_path):
    h = History(tmp_path / "history")
    h.put(run("lift_cube", "only_one", [True] * 25 + [False] * 25), keep_record=False)
    assert h.rate_for("lift_cube").detail["spread"] is None
    h.put(run("lift_cube", "second", [True] * 10 + [False] * 40), keep_record=False)
    assert h.rate_for("lift_cube").detail["spread"] > 0


def test_sizing_can_be_answered_from_history(tmp_path):
    """The join that makes the loop close: measured rate in, trial count out."""
    h, _ = measured(tmp_path)
    rate = h.rate_for("lift_cube")
    assert trials_needed(rate.value, 0.10) > 20


# -- selection corrects itself -----------------------------------------------


def test_it_counts_attempts_so_nobody_has_to_remember(tmp_path):
    h, _ = measured(tmp_path)
    assert h.attempts(task="lift_cube") == 3
    assert h.attempts(policy="ph_official") == 2
    assert h.attempts(task="open_door") == 0


# -- pairing -----------------------------------------------------------------


def test_pairing_is_by_scene_and_reports_both_directions(tmp_path):
    h, keys = measured(tmp_path)
    right_wins, left_wins, shared = h.paired(keys["ph"], keys["mh"])
    assert shared == 50
    assert right_wins == 5 and left_wins == 0


def test_pairing_only_counts_scenes_both_runs_attempted(tmp_path):
    h = History(tmp_path / "history")
    a = h.put(run("t", "a", [True, False, True]), keep_record=False)
    b = h.put(run("t", "b", [True, True]), keep_record=False)
    assert h.paired(a, b)[2] == 2


def test_pairing_skips_trials_that_errored(tmp_path):
    h = History(tmp_path / "history")
    a = h.put(run("t", "a", [True, None, False]), keep_record=False)
    b = h.put(run("t", "b", [True, True, True]), keep_record=False)
    assert h.paired(a, b)[2] == 2


def test_pairing_an_unknown_run_reports_nothing_rather_than_raising(tmp_path):
    h, keys = measured(tmp_path)
    assert h.paired(keys["ph"], "not-a-key") == (0, 0, 0)


# -- feeding the aggregates --------------------------------------------------


def test_a_policys_runs_group_into_the_matrix_the_aggregates_take(tmp_path):
    h, _ = measured(tmp_path)
    matrix = h.matrix_for("mh_official")
    assert len(matrix) == 2  # two tasks
    flat = [v for row in matrix for v in row]
    assert iqm(flat) == pytest.approx(0.62)


def test_the_aggregate_ranks_the_three_the_way_the_evidence_does(tmp_path):
    h, _ = measured(tmp_path)
    scores = {
        policy: iqm([v for row in h.matrix_for(policy) for v in row])
        for policy in ("ph_official", "mh_official", "mg_official")
    }
    assert scores["mh_official"] > scores["ph_official"] > scores["mg_official"]


def test_rows_can_be_grouped_without_going_through_history(tmp_path):
    h, _ = measured(tmp_path)
    assert len(matrix_of(h.query(policy="ph_official"))) == 2


# -- the scorer field --------------------------------------------------------


def test_a_run_records_which_judge_decided_its_labels(tmp_path):
    h = History(tmp_path / "history")
    key = h.put(run("lift_cube", "ph", [True], scorer="machine"), keep_record=False)
    assert h.get(key).scorer == "machine"


def test_a_run_with_no_scorer_is_distinguishable_from_one_judged_by_the_sim(tmp_path):
    """A null here means nobody said, which is not the same as the simulator
    saying — and the report says so rather than assuming."""
    h = History(tmp_path / "history")
    h.put(run("lift_cube", "old_run", [True], scorer=None), keep_record=False)
    row = next(iter(h))
    assert row.scorer is None
    assert "no scorer" in h.report()


def test_summarise_works_on_a_record_with_no_metrics(tmp_path):
    empty = Run(Provenance(), (), {})
    out = summarise(empty)
    assert out["rate"] is None and out["trials"] == 0
