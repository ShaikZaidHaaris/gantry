from __future__ import annotations

import numpy as np
import pytest

from gantry.spine import (
    AdapterStep,
    ChannelSpec,
    ComponentRef,
    EpisodeLabels,
    Measurement,
    Provenance,
    RunRecord,
    StageEvent,
    episode_from_arrays,
    runset,
)


def episode(uid: str, stages=(), success=None, source="ds"):
    return episode_from_arrays(
        {"state": np.zeros((6, 3), dtype="float32")},
        [ChannelSpec("state", "vector", (3,), "float32")],
        id=uid,
        source=source,
        labels=EpisodeLabels(
            success=success,
            stage_events=tuple(StageEvent(name, i) for i, name in enumerate(stages)),
        ),
    )


def provenance(policy="p", embodiment="arm", evaluator="sim", **protocol):
    return Provenance(
        components=(
            ComponentRef("policy", policy, "1.0", artifact_digest="abc"),
            ComponentRef("embodiment", embodiment, "1.0"),
            ComponentRef("evaluation", evaluator, "1.0"),
        ),
        protocol={"n": 100, "chunk": 8, **protocol},
    )


def test_stage_vocabulary_comes_from_the_data():
    run = RunRecord(
        provenance(),
        episodes=(
            episode("a", stages=("approach", "contact")),
            episode("b", stages=("approach",)),
            episode("c", stages=("approach", "contact", "retract")),
        ),
    )
    assert run.stages == ("approach", "contact", "retract")
    assert run.has_stage_events


def test_a_run_without_stage_events_says_so():
    run = RunRecord(provenance(), episodes=(episode("a", success=True),))
    assert not run.has_stage_events
    assert run.has_outcomes
    assert run.outcomes() == (True,)


def test_duplicate_episode_uids_are_refused():
    run = RunRecord(provenance(), episodes=(episode("a"), episode("a")))
    verdict = run.validate()
    assert "run.duplicate_episode" in verdict.codes()


def test_same_id_from_different_sources_is_fine():
    run = RunRecord(
        provenance(),
        episodes=(episode("demo_0", source="ph"), episode("demo_0", source="mh")),
    )
    assert run.validate().ok


def test_protocol_is_part_of_run_identity():
    assert provenance(chunk=8).digest != provenance(chunk=4).digest


def test_digest_ignores_when_and_where_it_ran():
    import dataclasses

    base = provenance()
    stamped = dataclasses.replace(base, created_at="2026-07-28T00:00:00Z", host="box-1")
    assert base.digest == stamped.digest


def test_adapter_losses_travel_with_the_run():
    import dataclasses

    lossy = dataclasses.replace(
        provenance(),
        adapters=(AdapterStep("resample", "1.0", ("30 Hz -> 20 Hz",)),),
    )
    assert lossy.lossy
    assert lossy.losses == ("30 Hz -> 20 Hz",)


def test_metrics_carry_their_own_shape():
    run = RunRecord(provenance()).with_metrics(
        {"progress": Measurement(34.83, n=100, ci=(28.7, 41.7), units="percent")}
    )
    assert "n=100" in str(run.metrics["progress"])


# -- aggregation guards ----------------------------------------------------


def test_runs_on_different_embodiments_are_not_comparable():
    runs = runset([RunRecord(provenance(embodiment="arm")), RunRecord(provenance(embodiment="humanoid"))])
    verdict = runs.comparable(["embodiment", "evaluation"])
    assert not verdict.ok
    assert verdict.because("runset.incomparable")[0].detail["planes"] == ["embodiment"]


def test_runs_differing_only_on_policy_are_comparable_when_policy_is_the_variable():
    runs = runset([RunRecord(provenance(policy="a")), RunRecord(provenance(policy="b"))])
    assert runs.comparable(["embodiment", "evaluation"]).ok
    assert not runs.comparable(["policy"]).ok


def test_grouping_partitions_by_plane_ref():
    runs = runset(
        [
            RunRecord(provenance(embodiment="arm", policy="a")),
            RunRecord(provenance(embodiment="arm", policy="b")),
            RunRecord(provenance(embodiment="humanoid", policy="a")),
        ]
    )
    groups = runs.grouped_by("embodiment")
    assert len(groups) == 2
    assert sorted(len(value) for value in groups.values()) == [1, 2]


def test_there_is_no_blind_mean():
    assert not hasattr(runset([]), "mean")


def test_unknown_plane_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown plane"):
        ComponentRef("wishful", "x", "1.0")
