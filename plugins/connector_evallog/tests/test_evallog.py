"""Reading a real-shaped eval log, and refusing the parts that are not there."""

from __future__ import annotations

import json

import pytest
from gantry.conformance import check_connector
from gantry.contracts.feedback import Cohort
from gantry.errors import ConfigError
from gantry.resolve import Registry, requires_channels, resolve
from gantry.spine import ChannelSpec

from gantry_connector_evallog import STAGE_KEY, EvalLogConnector, read_run, stage_metadata


def log_payload(*, scenes=None, stages=False, epochs=1, status="success"):
    """A log in the Inspect Robots schema-v1 shape."""
    scenes = scenes if scenes is not None else [("layout-0", True), ("layout-1", False)]
    samples = []
    for index, (scene_id, ok) in enumerate(scenes):
        per_epoch = [{"success_at_end": 1.0 if ok else 0.0} for _ in range(epochs)]
        metadata = []
        for _ in range(epochs):
            if not stages:
                metadata.append({})
            elif ok:
                metadata.append(stage_metadata(reached=8, grasped=19, lifted=27))
            else:
                metadata.append(stage_metadata(reached=8))
        samples.append(
            {
                "scene_id": scene_id,
                "status": "success",
                "reduced": {"success_at_end": 1.0 if ok else 0.0},
                "epochs": per_epoch,
                "error": None,
                "instruction": "place the fork on the plate",
                "operator_judgements": ["clean" if ok else "missed grasp"] * epochs,
                "operator_notes": [None] * epochs,
                "trial_metadata": metadata,
                "termination_reasons": ["scored" if ok else "max_steps"] * epochs,
                "policy_transcripts": [None] * epochs,
            }
        )
    return {
        "version": 1,
        "status": status,
        "eval": {
            "task": "kitchenbench/pour_pasta",
            "policy": "molmoact2",
            "embodiment": "yam_arms",
            "created": "2026-07-28T09:15:00Z",
            "inspect_robots_version": "0.4.1",
            "git_commit": "abc1234",
            "policy_config": {"replan_interval": 8},
            "embodiment_info": {"control_hz": 20.0},
            "seed": 7,
            "max_steps": 300,
            "max_seconds": None,
        },
        "results": {
            "total_scenes": len(scenes),
            "total_trials": len(scenes) * epochs,
            "metrics": {"success_at_end": sum(ok for _, ok in scenes) / len(scenes)},
            "errored_trials": 0,
        },
        "stats": {
            "started_at": "2026-07-28T09:15:00Z",
            "completed_at": "2026-07-28T09:41:00Z",
            "duration_s": 1560.0,
            "total_steps": 4200,
            "mean_inference_latency_s": 0.11,
            "frames_dir": None,
        },
        "samples": samples,
        "error": None,
    }


@pytest.fixture
def write(tmp_path):
    def _write(payload=None, name="run.json"):
        path = tmp_path / name
        path.write_text(json.dumps(payload if payload is not None else log_payload()))
        return path

    return _write


# -- the contract ----------------------------------------------------------


def test_conforms(write):
    verdict = check_connector(EvalLogConnector(write()), strict=True)
    assert verdict.ok, verdict.explain()


def test_one_episode_per_trial_not_per_scene(write):
    """Three epochs of a scene are three attempts with three outcomes."""
    connector = EvalLogConnector(write(log_payload(epochs=3)))
    assert len(connector.episode_ids()) == 6
    assert "layout-0#0" in connector.episode_ids()


def test_a_single_epoch_keeps_the_plain_scene_id(write):
    assert EvalLogConnector(write()).episode_ids() == ("layout-0", "layout-1")


def test_outcomes_are_read_from_the_score(write):
    connector = EvalLogConnector(write())
    assert connector.open("layout-0").labels.success is True
    assert connector.open("layout-1").labels.success is False


def test_operator_context_survives(write):
    labels = EvalLogConnector(write()).open("layout-1").labels
    assert labels.annotations["operator_judgement"] == "missed grasp"
    assert labels.annotations["termination_reason"] == "max_steps"
    assert labels.annotations["instruction"] == "place the fork on the plate"


def test_ids_are_namespaced_by_the_task(write):
    assert EvalLogConnector(write()).open("layout-0").meta.uid.startswith("kitchenbench/pour_pasta/")


# -- what the log genuinely does not contain -------------------------------


def test_episodes_are_label_only(write):
    """A log records what happened, not how. Saying so is the point."""
    episode = EvalLogConnector(write()).open("layout-0")
    assert episode.schema == ()
    assert episode.read() == {}
    assert episode.validate(deep=True).ok


def test_asking_for_a_channel_says_there_are_none(write):
    with pytest.raises(KeyError, match="this record has none"):
        EvalLogConnector(write()).open("layout-0").array("position")


def test_the_descriptor_admits_it_has_no_stage_events(write):
    assert EvalLogConnector(write()).descriptor().provides["stage_events"] is False


# -- stage events, when a scorer records them ------------------------------


def test_a_scorer_can_supply_milestones(write):
    connector = EvalLogConnector(write(log_payload(stages=True)))
    assert connector.descriptor().provides["stage_events"] is True
    assert connector.open("layout-0").labels.stages == ("reached", "grasped", "lifted")
    assert connector.open("layout-0").labels.step_of("grasped") == 19


def test_a_failed_trial_records_only_the_milestones_it_reached(write):
    """Absence is the signal a funnel reads."""
    episode = EvalLogConnector(write(log_payload(stages=True))).open("layout-1")
    assert episode.labels.stages == ("reached",)
    assert not episode.labels.reached("grasped")


def test_the_horizon_is_what_the_milestones_prove(write):
    """Not the configured max_steps, which would be inventing a number."""
    episode = EvalLogConnector(write(log_payload(stages=True))).open("layout-0")
    assert len(episode) == 28  # last milestone at step 27


def test_stage_metadata_refuses_a_negative_step():
    with pytest.raises(ValueError, match="cannot be reached at step"):
        stage_metadata(reached=-1)


def test_malformed_stage_metadata_is_ignored_not_crashed(write):
    payload = log_payload()
    payload["samples"][0]["trial_metadata"] = [{STAGE_KEY: "not a mapping"}]
    assert EvalLogConnector(write(payload)).open("layout-0").labels.stage_events == ()


# -- the run behind the episodes -------------------------------------------


def test_provenance_carries_what_produced_the_numbers(write):
    provenance = read_run(write()).provenance
    assert provenance.component("policy").name == "molmoact2"
    assert provenance.component("evaluation").name == "yam_arms"
    assert provenance.protocol["seed"] == 7
    assert provenance.protocol["policy_config"]["replan_interval"] == 8
    assert any("git abc1234" in note for note in provenance.notes)


def test_the_protocol_is_part_of_run_identity(write):
    """Chunking changes results, so a run that changed it is a different run."""
    other = log_payload()
    other["eval"]["policy_config"] = {"replan_interval": 4}
    assert read_run(write()).digest != read_run(write(other, "b.json")).digest


def test_reported_metrics_come_through_as_measurements(write):
    run = read_run(write())
    assert run.metrics["success_at_end"].value == pytest.approx(0.5)
    assert run.metrics["success_at_end"].n == 2


def test_two_runs_on_different_embodiments_are_not_comparable(write):
    from gantry.spine import runset

    other = log_payload()
    other["eval"]["embodiment"] = "franka"
    runs = runset([read_run(write()), read_run(write(other, "b.json"))])
    assert not runs.comparable(["evaluation"]).ok


# -- malformed logs --------------------------------------------------------


def test_an_unsupported_schema_version_is_refused_by_number(write):
    payload = log_payload()
    payload["version"] = 99
    with pytest.raises(ConfigError, match="schema version 99"):
        EvalLogConnector(write(payload))


def test_a_truncated_log_names_the_missing_section(write, tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"version": 1, "status": "error", "eval": {}}))
    with pytest.raises(ConfigError, match="missing its 'results' section"):
        EvalLogConnector(path)


def test_invalid_json_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        EvalLogConnector(path)


# -- what this unlocks, and what it does not -------------------------------


def test_the_funnel_is_refused_on_an_outcome_only_log(write):
    """The refusal that names what would run instead."""
    registry = Registry()
    registry.register(
        "dataset", "evallog", lambda **config: EvalLogConnector(**config)
    )
    resolution = resolve(
        registry,
        components={"dataset": {"name": "evallog", "config": {"path": str(write())}}},
        consumers=[
            requires_channels("funnel", "feedback", capabilities={"stage_events": True}),
            requires_channels("outcome_rate", "feedback", capabilities={"outcomes": True}),
        ],
        provided_channels=[],
    )
    assert not resolution.ok
    assert "funnel needs stage_events" in resolution.explain()
    assert resolution.alternatives == ("outcome_rate",)


def test_the_funnel_runs_once_a_scorer_records_milestones(write):
    from gantry_feedback_core import Funnel

    connector = EvalLogConnector(write(log_payload(stages=True, epochs=6)))
    report = Funnel().run([Cohort("kitchen", tuple(connector))])
    finding = report.by_code("funnel.bottleneck")[0]
    assert "grasped is the weak link" in finding.summary


def test_a_trajectory_analysis_still_cannot_run_on_a_log(write):
    """Being able to read a log does not make it into data it never held."""
    from gantry_feedback_core import Attribution

    connector = EvalLogConnector(write(log_payload(stages=True, epochs=6)))
    report = Attribution().run([Cohort("kitchen", tuple(connector))])
    assert report.prescriptions == ()
