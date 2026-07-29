"""Applying a plan to a real dataset on disk, and reading the result back.

Not a mock. A LeRobot dataset is written to a temporary directory, curated
through the same command line a person would use, and re-opened with the same
connector — because the failure this step exists to prevent only happens at the
boundary between the plan's identifiers and the format's, and a test that keeps
episodes in memory never crosses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gantry_connector_lerobot import LeRobotConnector
from gantry_connector_lerobot.testing import build_dataset

from gantry.cli import main
from gantry.contracts.curation import (
    CurationAction,
    CurationPlan,
    Prediction,
)
from gantry.curate import apply, mix_config
from gantry.plan_io import plan_to_dict, read_plan, write_plan
from gantry.spine import IncompatibleError


def source(tmp_path: Path, episodes: int = 6) -> LeRobotConnector:
    build_dataset(tmp_path / "src", episodes=episodes, steps=8)
    return LeRobotConnector(tmp_path / "src")


def plan_dropping(uids, *, signal="labels", magnitude=0.1):
    return CurationPlan(
        actions=(CurationAction("drop", episodes=tuple(uids)),),
        signal=signal,
        rung="screening",
        predicted=Prediction(magnitude=magnitude, tasks=("lift_cube",)),
    )


# -- selection ---------------------------------------------------------------


def test_dropping_removes_exactly_those_episodes(tmp_path):
    connector = source(tmp_path)
    episodes = list(connector)
    doomed = [e.meta.uid for e in episodes[:2]]
    survivors, applied = apply(plan_dropping(doomed), episodes)
    assert len(survivors) == len(episodes) - 2
    assert set(applied.dropped) == set(doomed)
    assert not applied.missing


def test_keep_is_a_whitelist_not_a_hint(tmp_path):
    """'Keep the top third' has to mean the other two thirds go."""
    episodes = list(source(tmp_path))
    chosen = [e.meta.uid for e in episodes[:2]]
    plan = CurationPlan(
        actions=(CurationAction("keep", episodes=tuple(chosen)),),
        signal="topk", rung="influence",
        predicted=Prediction(magnitude=0.1), evidence_seeds=(1, 2),
    )
    survivors, applied = apply(plan, episodes)
    assert [e.meta.uid for e in survivors] == chosen
    assert len(applied.dropped) == len(episodes) - 2


def test_order_is_preserved(tmp_path):
    episodes = list(source(tmp_path))
    survivors, _ = apply(plan_dropping([episodes[2].meta.uid]), episodes)
    kept = [e.meta.uid for e in episodes if e.meta.uid != episodes[2].meta.uid]
    assert [e.meta.uid for e in survivors] == kept


# -- what it refuses ---------------------------------------------------------


def test_a_plan_naming_episodes_that_are_not_there_is_refused(tmp_path):
    """The expensive silence: a drop-list that matches nothing applies cleanly,
    does nothing, and the retrain afterwards still looks like a measurement."""
    episodes = list(source(tmp_path))
    with pytest.raises(IncompatibleError, match="curation.unactionable"):
        apply(plan_dropping(["nope/1", "nope/2"]), episodes)


def test_a_plan_that_would_empty_the_dataset_is_refused(tmp_path):
    episodes = list(source(tmp_path))
    everything = [e.meta.uid for e in episodes]
    with pytest.raises(IncompatibleError, match="empties_the_dataset"):
        apply(plan_dropping(everything), episodes)


def test_a_plan_that_changes_nothing_is_noted(tmp_path):
    episodes = list(source(tmp_path))
    plan = CurationPlan(
        actions=(CurationAction("keep", episodes=tuple(e.meta.uid for e in episodes)),),
        signal="labels", rung="screening",
    )
    _, applied = apply(plan, episodes)
    verdict = applied.validate(source_size=len(episodes))
    assert verdict.ok  # a note: it is legal, just pointless to verify
    assert "curation.no_change" in verdict.codes()


def test_force_applies_a_partly_matching_plan_and_still_says_so(tmp_path):
    episodes = list(source(tmp_path))
    plan = plan_dropping([episodes[0].meta.uid, "nope/9"])
    survivors, applied = apply(plan, episodes, strict=False)
    assert len(survivors) == len(episodes) - 1
    assert applied.missing == ("nope/9",)


# -- weights -----------------------------------------------------------------


def test_weights_are_configuration_not_a_change_to_the_files(tmp_path):
    episodes = list(source(tmp_path))
    plan = CurationPlan(
        actions=(
            CurationAction("weight", cohort="mh", weight=2.0),
            CurationAction("weight", cohort="mg", weight=0.0),
        ),
        signal="mixture", rung="leave_out",
        predicted=Prediction(magnitude=0.05),
    )
    survivors, applied = apply(plan, episodes)
    assert len(survivors) == len(episodes)      # nothing removed
    assert applied.weights == {"mh": 2.0, "mg": 0.0}
    assert mix_config(plan)["datasets"][0] == {"cohort": "mg", "mix_ratio": 0.0}


# -- the plan is a file ------------------------------------------------------


def test_a_plan_round_trips_through_a_file(tmp_path):
    plan = CurationPlan(
        actions=(
            CurationAction("drop", episodes=("a/1", "a/2"), detail={"score": 0.3}),
            CurationAction("weight", cohort="mh", weight=2.0),
        ),
        signal="cupid", rung="influence",
        predicted=Prediction(magnitude=0.07, tasks=("lift_cube",)),
        evidence_seeds=(11, 22, 33),
        metadata={"note": "hello"},
    )
    path = write_plan(plan, tmp_path / "plan.json")
    back = read_plan(path)
    assert plan_to_dict(back) == plan_to_dict(plan)
    assert back.drops == ("a/1", "a/2")
    assert back.weights == {"mh": 2.0}


def test_the_evidence_seeds_survive_the_round_trip(tmp_path):
    """Losing them turns a leakage refusal into a clean pass."""
    plan = CurationPlan(
        actions=(CurationAction("drop", episodes=("a/1",)),),
        signal="cupid", rung="influence",
        predicted=Prediction(magnitude=0.1), evidence_seeds=(7, 8, 9),
    )
    assert read_plan(write_plan(plan, tmp_path / "p.json")).evidence_seeds == (7, 8, 9)


def test_a_collection_order_survives_the_round_trip(tmp_path):
    from gantry.contracts.curation import CollectionOrder

    order = CollectionOrder(task="lift_cube", n=40, seeds=(1, 2), stage="grasp", note="east")
    plan = CurationPlan(
        actions=(CurationAction("collect", order=order),),
        signal="collect", rung="screening",
        predicted=Prediction(magnitude=0.08),
    )
    back = read_plan(write_plan(plan, tmp_path / "p.json"))
    assert back.orders[0] == order


# -- through the command line, onto disk -------------------------------------


def test_the_curated_dataset_is_written_and_reads_back(tmp_path, capsys):
    connector = source(tmp_path, episodes=6)
    episodes = list(connector)
    doomed = [e.meta.uid for e in episodes[:2]]
    write_plan(plan_dropping(doomed), tmp_path / "plan.json")

    code = main([
        "curate-apply", str(tmp_path / "plan.json"), str(tmp_path / "src"), "--reader", "lerobot",
        "-o", str(tmp_path / "curated"), "--write", "--accept-loss",
    ])
    assert code == 0

    # The real check: re-open it with the same connector.
    curated = LeRobotConnector(tmp_path / "curated")
    assert len(list(curated)) == 4
    out = capsys.readouterr().out
    assert "4/6 episodes kept" in out


def test_the_curated_dataset_carries_what_produced_it(tmp_path):
    """A checkpoint trained on this is traceable to the curation, not a memory."""
    episodes = list(source(tmp_path, episodes=6))
    write_plan(plan_dropping([episodes[0].meta.uid]), tmp_path / "plan.json")
    main([
        "curate-apply", str(tmp_path / "plan.json"), str(tmp_path / "src"), "--reader", "lerobot",
        "-o", str(tmp_path / "curated"), "--write", "--accept-loss",
    ])
    stamped = json.loads((tmp_path / "curated" / "curation.json").read_text())
    assert stamped["signal"] == "labels"
    assert stamped["rung"] == "screening"
    assert stamped["dropped"] == 1
    assert stamped["predicted"] == "success_rate +0.1 on lift_cube"


def test_a_dry_run_writes_nothing(tmp_path, capsys):
    episodes = list(source(tmp_path))
    write_plan(plan_dropping([episodes[0].meta.uid]), tmp_path / "plan.json")
    main([
        "curate-apply", str(tmp_path / "plan.json"), str(tmp_path / "src"), "--reader", "lerobot",
        "-o", str(tmp_path / "curated"),
    ])
    assert not (tmp_path / "curated").exists()
    assert "pass --write" in capsys.readouterr().out


def test_the_command_refuses_an_unactionable_plan_rather_than_writing_nothing_useful(tmp_path):
    source(tmp_path)
    write_plan(plan_dropping(["nope/1"]), tmp_path / "plan.json")
    with pytest.raises(IncompatibleError, match="curation.unactionable"):
        main([
            "curate-apply", str(tmp_path / "plan.json"), str(tmp_path / "src"), "--reader", "lerobot",
            "-o", str(tmp_path / "curated"), "--write",
        ])
