"""A manifest is the unit of reproducibility. This is it working end to end."""

from __future__ import annotations

import json

import pytest

from gantry.errors import ConfigError
from gantry.manifest import Manifest
from gantry.resolve import Registry
from gantry.runner import check_manifest, run_manifest

# -- parsing ---------------------------------------------------------------


def test_a_component_may_be_a_bare_name_or_an_object():
    manifest = Manifest.from_dict(
        {"name": "m", "cohorts": {"a": "csv", "b": {"name": "csv", "config": {"path": "x"}}}}
    )
    assert manifest.cohorts["a"].name == "csv" and manifest.cohorts["a"].config == {}
    assert manifest.cohorts["b"].config == {"path": "x"}


def test_a_single_feedback_module_need_not_be_a_list():
    assert len(Manifest.from_dict({"name": "m", "feedback": "screen"}).feedback) == 1


def test_it_round_trips():
    original = Manifest.from_dict(
        {"name": "m", "cohorts": {"a": "csv"}, "feedback": ["screen"], "protocol": {"epochs": 2}}
    )
    assert Manifest.from_dict(json.loads(original.to_json())) == original


def test_an_unsupported_version_is_refused_by_number():
    with pytest.raises(ConfigError, match="version 99"):
        Manifest.from_dict({"version": 99, "name": "m"})


def test_a_nameless_manifest_is_refused():
    with pytest.raises(ConfigError, match="non-empty 'name'"):
        Manifest.from_dict({"cohorts": {}})


def test_a_nameless_component_says_where():
    with pytest.raises(ConfigError, match="cohorts.a: needs a non-empty 'name'"):
        Manifest.from_dict({"name": "m", "cohorts": {"a": {"config": {}}}})


def test_half_an_evaluation_is_refused():
    manifest = Manifest.from_dict({"name": "m", "cohorts": {"a": "csv"}, "policy": "replay"})
    verdict = manifest.validate()
    assert "manifest.half_an_evaluation" in verdict.codes()
    reason = verdict.because("manifest.half_an_evaluation")[0]
    assert "'evaluation' is missing" in reason.message
    assert "drop both" in reason.hint


def test_no_feedback_is_a_note_not_a_failure():
    verdict = Manifest.from_dict({"name": "m", "cohorts": {"a": "csv"}}).validate()
    assert verdict.ok
    assert "manifest.no_feedback" in verdict.codes()


def test_yaml_is_refused_with_the_way_round_it(tmp_path):
    """Core takes no parser dependency, and says what to do instead."""
    path = tmp_path / "m.yaml"
    path.write_text("name: m\n")
    with pytest.raises(ConfigError, match="Manifest.from_dict"):
        Manifest.load(path)


def test_from_dict_is_the_public_seam_for_any_other_format():
    assert Manifest.from_dict({"name": "parsed-elsewhere"}).name == "parsed-elsewhere"


def test_a_missing_manifest_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no manifest at"):
        Manifest.load(tmp_path / "nope.json")


# -- checking before running -----------------------------------------------


def test_an_uninstalled_component_is_named_before_anything_is_built():
    manifest = Manifest.from_dict({"name": "m", "cohorts": {"a": "no-such-connector"}})
    verdict = check_manifest(manifest, Registry())
    assert "manifest.not_installed" in verdict.codes()
    assert verdict.because("manifest.not_installed")[0].detail["name"] == "no-such-connector"


# -- running ---------------------------------------------------------------


@pytest.fixture
def csv_manifest(tmp_path):
    """Two cohorts written to CSV, one clean and one not."""
    from gantry_connector_csv import write_episodes

    from gantry.fixtures import make_clean, make_defective

    clean = write_episodes(make_clean(n=20, seed=1).episodes, tmp_path / "clean.csv")
    broken = write_episodes(
        make_defective("never_completes", n=20, fraction=0.5, seed=2).episodes,
        tmp_path / "broken.csv",
    )
    return Manifest.from_dict(
        {
            "name": "clean-vs-broken",
            "cohorts": {
                "clean": {"name": "csv", "config": {"path": str(clean)}},
                "broken": {"name": "csv", "config": {"path": str(broken)}},
            },
            "feedback": [
                {"name": "screen", "config": {"mode": "comparative"}},
                {"name": "funnel"},
            ],
        }
    )


def test_a_manifest_runs_and_reports(csv_manifest):
    outcome = run_manifest(csv_manifest)
    assert outcome.ok
    assert {report.module for report in outcome.reports} == {"screen", "funnel"}


def test_the_outcome_knows_everything_that_produced_it(csv_manifest):
    provenance = run_manifest(csv_manifest).provenance
    planes = {ref.plane for ref in provenance.components}
    assert planes == {"dataset", "feedback"}
    assert provenance.created_at and provenance.host


def test_the_outcome_is_json_able(csv_manifest):
    payload = json.loads(json.dumps(run_manifest(csv_manifest).as_dict(), default=str))
    assert payload["manifest"]["name"] == "clean-vs-broken"


def test_the_funnel_finds_the_broken_cohort(csv_manifest):
    outcome = run_manifest(csv_manifest)
    funnel = next(r for r in outcome.reports if r.module == "funnel")
    assert any("broken" in f.summary and "weak link" in f.summary for f in funnel.findings)


def test_a_feedback_module_that_refuses_fails_only_itself(tmp_path):
    """Harden needs two cohorts; giving it one must not sink the whole run."""
    from gantry_connector_csv import write_episodes

    from gantry.fixtures import make_defective

    path = write_episodes(
        make_defective("never_completes", n=12, fraction=0.5).episodes, tmp_path / "one.csv"
    )
    manifest = Manifest.from_dict(
        {
            "name": "one-cohort",
            "cohorts": {"only": {"name": "csv", "config": {"path": str(path)}}},
            "feedback": ["funnel", "harden"],
        }
    )
    outcome = run_manifest(manifest)
    assert [r.module for r in outcome.reports] == ["funnel"]
    assert any("harden" in failure for failure in outcome.failures)
    assert not outcome.ok


# -- the whole frame: data -> policy -> evaluation -> feedback -------------


def test_an_evaluating_manifest_runs_every_plane(tmp_path):
    from gantry_connector_csv import write_episodes
    from gantry_evaluator_offline import OfflineEvaluator
    from gantry_policy_basic import NoisyReplayPolicy

    from gantry.fixtures import make_clean

    suite = make_clean(n=10, seed=3)
    path = write_episodes(suite.episodes, tmp_path / "held_out.csv")
    action = suite.episodes[0].channel("action")

    registry = Registry()
    registry.discover()
    registry.register(
        "policy", "noisy", lambda **_: NoisyReplayPolicy(action, sigma=0.1), replace=True
    )
    registry.register(
        "evaluation",
        "offline",
        lambda **_: OfflineEvaluator(suite.episodes, "action"),
        replace=True,
    )

    manifest = Manifest.from_dict(
        {
            "name": "offline-eval",
            "cohorts": {"held_out": {"name": "csv", "config": {"path": str(path)}}},
            "policy": "noisy",
            "evaluation": "offline",
            "protocol": {"epochs": 1, "execute": 1},
            "feedback": [],
        }
    )
    outcome = run_manifest(manifest, registry)
    assert outcome.ok
    assert len(outcome.runs) == 1
    assert outcome.runs[0].metrics["action_mse"].value == pytest.approx(0.01, rel=0.3)
    assert any("compounding error is not measured" in n for n in outcome.notes)


# -- the command line ------------------------------------------------------


def test_list_names_every_plane(capsys):
    from gantry.cli import main

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for plane in ("dataset", "policy", "evaluation", "feedback", "adapter"):
        assert plane in out


def test_checks_enumerates_the_specification(capsys):
    from gantry.cli import main

    assert main(["checks", "policy"]) == 0
    assert "determinism" in capsys.readouterr().out


def test_plan_refuses_a_manifest_naming_something_uninstalled(tmp_path, capsys):
    from gantry.cli import main

    path = tmp_path / "m.json"
    path.write_text(json.dumps({"name": "m", "cohorts": {"a": "ghost"}}))
    assert main(["plan", str(path)]) == 1
    assert "dataset:ghost is not installed" in capsys.readouterr().out


def test_plan_accepts_a_good_manifest(tmp_path, capsys, csv_manifest):
    from gantry.cli import main

    path = csv_manifest.save(tmp_path / "good.json")
    assert main(["plan", str(path)]) == 0
    assert "would analyse as recorded" in capsys.readouterr().out


def test_run_writes_its_outcome(tmp_path, csv_manifest, capsys):
    from gantry.cli import main

    path = csv_manifest.save(tmp_path / "m.json")
    out = tmp_path / "outcome.json"
    assert main(["run", str(path), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["manifest"]["name"] == "clean-vs-broken"


def test_a_malformed_manifest_exits_as_misuse(tmp_path, capsys):
    from gantry.cli import main

    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert main(["plan", str(path)]) == 2


# -- resolution is on the execution path -----------------------------------


def test_a_module_that_does_not_fit_is_refused_and_the_rest_still_run(tmp_path):
    """The funnel needs milestones; a log without them refuses at plan time.

    Before resolution was wired into the runner this ran anyway and returned an
    empty funnel with a note attached, which reads exactly like a finished
    analysis of a policy that never got anywhere.
    """
    from gantry_connector_csv import write_episodes

    from gantry.fixtures import make_clean
    from gantry.spine import EpisodeLabels

    stripped = [
        e.with_labels(EpisodeLabels(success=e.labels.success))
        for e in make_clean(n=12, seed=1).episodes
    ]
    path = write_episodes(stripped, tmp_path / "no_stages.csv")
    manifest = Manifest.from_dict(
        {
            "name": "outcome-only",
            "cohorts": {"logs": {"name": "csv", "config": {"path": str(path)}}},
            "feedback": ["funnel", {"name": "screen", "config": {"mode": "absolute"}}],
        }
    )
    outcome = run_manifest(manifest)
    assert [r.module for r in outcome.reports] == ["screen"]
    assert any("funnel" in r and "stage_events" in r for r in outcome.refusals)
    assert any("runnable as requested: screen" in n for n in outcome.notes)


def test_plan_reports_the_refusal_without_running_anything(tmp_path):
    from gantry_connector_csv import write_episodes

    from gantry.fixtures import make_clean
    from gantry.runner import plan_manifest
    from gantry.spine import EpisodeLabels

    stripped = [
        e.with_labels(EpisodeLabels(success=e.labels.success))
        for e in make_clean(n=8, seed=1).episodes
    ]
    path = write_episodes(stripped, tmp_path / "no_stages.csv")
    manifest = Manifest.from_dict(
        {
            "name": "planned",
            "cohorts": {"logs": {"name": "csv", "config": {"path": str(path)}}},
            "feedback": ["funnel"],
        }
    )
    verdict = plan_manifest(manifest)
    assert not verdict.ok
    assert "plan.module_refused" in verdict.codes()


def test_a_lossy_adapter_used_by_a_run_reaches_its_provenance(tmp_path):
    """A loss stated by an adapter has to survive all the way to the record."""
    from gantry_connector_csv import write_episodes

    from gantry.contracts.feedback import FeedbackModule, Report, feedback_descriptor
    from gantry.fixtures import make_clean
    from gantry.resolve import requires_channels
    from gantry.spine import ChannelSpec

    slow = ChannelSpec(
        "position",
        "vector",
        (3,),
        "float32",
        units="m",
        frame="world",
        rate_hz=10.0,
        semantics="position",
    )

    class Downsampler(FeedbackModule):
        def descriptor(self):
            return feedback_descriptor("downsampled", "1.0", min_cohorts=1, prescribes=False)

        def requirement(self):
            return requires_channels("downsampled", "feedback", slow)

        def analyse(self, cohorts):
            return Report("downsampled", (), {}, cohorts=tuple(c.name for c in cohorts))

    path = write_episodes(make_clean(n=6, seed=1).episodes, tmp_path / "fast.csv")
    registry = Registry()
    registry.discover()
    registry.register("feedback", "downsampled", lambda **_: Downsampler(), replace=True)

    manifest = Manifest.from_dict(
        {
            "name": "resampled",
            "cohorts": {"fast": {"name": "csv", "config": {"path": str(path)}}},
            "feedback": ["downsampled"],
        }
    )
    outcome = run_manifest(manifest, registry)
    assert outcome.ok, outcome.explain()
    assert any("resample" in step.name for step in outcome.provenance.adapters)
    assert any("detail faster than 10 Hz is gone" in loss for loss in outcome.provenance.losses)


def test_the_whole_loop_runs_from_one_manifest(tmp_path):
    """Data, policy, closed-loop world, diagnosis — no GPU, no simulator."""
    from gantry_connector_csv import write_episodes
    from gantry_evaluator_waypoint import GreedyPolicy, WaypointWorld

    from gantry.fixtures import make_clean

    path = write_episodes(make_clean(n=4, seed=1).episodes, tmp_path / "seed.csv")
    registry = Registry()
    registry.discover()
    registry.register("policy", "greedy", lambda **_: GreedyPolicy(skill=0.75), replace=True)
    registry.register(
        "evaluation",
        "waypoint",
        lambda **_: WaypointWorld(tolerance=[0.08, 0.03, 0.08, 0.08]),
        replace=True,
    )

    manifest = Manifest.from_dict(
        {
            "name": "full-loop",
            "cohorts": {"scenes": {"name": "csv", "config": {"path": str(path)}}},
            "policy": "greedy",
            "evaluation": "waypoint",
            "feedback": ["funnel"],
            "protocol": {"epochs": 1},
        }
    )
    outcome = run_manifest(manifest, registry)
    assert outcome.ok, outcome.explain()
    assert len(outcome.runs) == 1

    funnel = outcome.reports[0]
    assert "engage is the weak link" in funnel.findings[0].summary


# -- cohorts on any plane --------------------------------------------------


def test_cohorts_default_to_the_dataset_plane():
    """Unchanged behaviour: a manifest that says nothing varies datasets."""
    from gantry.manifest import Manifest

    m = Manifest.from_dict(
        {
            "version": 1,
            "name": "x",
            "cohorts": {"a": {"name": "csv", "config": {"path": "a.csv"}}},
        }
    )
    assert m.varies == "dataset"
    assert m.provides("dataset") and not m.provides("policy")


def test_cohorts_can_vary_the_policy_instead():
    """One world, three checkpoints — the shape the manifest could not express.

    The dataset plane was hard-coded as the axis, which meant a manifest could
    say 'three datasets, one policy' and not its mirror image. That was the
    dataset plane getting special treatment in the one file meant to have none.
    """
    from gantry.manifest import Manifest

    m = Manifest.from_dict(
        {
            "version": 1,
            "name": "three-checkpoints",
            "varies": "policy",
            "cohorts": {
                "ph": {"name": "constant", "config": {}},
                "mg": {"name": "replay", "config": {}},
            },
            "dataset": {"name": "csv", "config": {"path": "held_out.csv"}},
            "evaluation": {"name": "offline", "config": {"action_name": "action"}},
        }
    )
    assert m.varies == "policy"
    assert m.provides("policy"), "supplied by the cohorts, not by a single component"
    assert m.evaluates
    assert m.spec_for("policy", "ph").name == "constant"
    assert m.spec_for("evaluation", "ph").name == "offline"
    assert m.spec_for("evaluation", "mg").name == "offline", "held planes are the same one"


def test_the_explicit_form_reads_the_same():
    from gantry.manifest import Manifest

    m = Manifest.from_dict(
        {
            "version": 1,
            "name": "x",
            "cohorts": {"plane": "evaluation", "of": {"sim": {"name": "waypoint"}}},
            "dataset": {"name": "csv", "config": {"path": "a.csv"}},
        }
    )
    assert m.varies == "evaluation"
    assert m.spec_for("evaluation", "sim").name == "waypoint"


def test_a_plane_cannot_both_vary_and_be_fixed():
    from gantry.errors import ConfigError
    from gantry.manifest import Manifest

    with pytest.raises(ConfigError, match="one or the other"):
        Manifest.from_dict(
            {
                "version": 1,
                "name": "x",
                "varies": "policy",
                "cohorts": {"a": {"name": "constant"}},
                "policy": {"name": "replay"},
            }
        )


def test_varying_on_something_that_is_not_a_plane_is_refused():
    from gantry.errors import ConfigError
    from gantry.manifest import Manifest

    with pytest.raises(ConfigError, match="not a plane"):
        Manifest.from_dict(
            {
                "version": 1,
                "name": "x",
                "varies": "checkpoint",
                "cohorts": {"a": {"name": "constant"}},
            }
        )


def test_a_policy_varying_run_evaluates_each_cohort_against_one_dataset(tmp_path):
    """End to end through the runner, with the axis moved off the dataset."""
    from gantry_connector_csv import write_episodes

    from gantry.fixtures import make_clean
    from gantry.manifest import Manifest
    from gantry.runner import run_manifest

    suite = make_clean(n=3, seed=1)
    path = write_episodes(suite.episodes, tmp_path / "held_out.csv")
    action = suite.episodes[0].channel("action")

    m = Manifest.from_dict(
        {
            "version": 1,
            "name": "two-policies",
            "varies": "policy",
            "cohorts": {
                "zero": {"name": "constant", "config": {"action": _spec_dict(action)}},
                "copy": {"name": "replay", "config": {"action": _spec_dict(action)}},
            },
            "dataset": {"name": "csv", "config": {"path": str(path)}},
            "evaluation": {"name": "offline", "config": {"action_name": "action"}},
            "feedback": [],
        }
    )
    outcome = run_manifest(m)
    assert not outcome.refusals, outcome.explain()
    assert len(outcome.runs) == 2, "one run per policy, same dataset"
    errors = [r.metrics["action_mse"].value for r in outcome.runs]
    # Replay reproduces the recording exactly; a constant does not.
    assert min(errors) == pytest.approx(0.0, abs=1e-9)
    assert max(errors) > 0.0


def _spec_dict(spec):
    from gantry.serial import spec_to_dict

    return spec_to_dict(spec)
