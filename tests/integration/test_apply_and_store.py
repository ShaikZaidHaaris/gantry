"""Adapters actually applied, and records that survive the process."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_adapters_core import default_registry

from gantry.errors import ConfigError
from gantry.fixtures import make_clean, make_defective
from gantry.resolve import AdapterRegistry, adapt_episode, bind, requires_channels
from gantry.spine import (
    ChannelSpec,
    EpisodeLabels,
    Provenance,
    RunRecord,
    StageEvent,
    episode_from_labels,
)
from gantry.store import read_run, same_run, write_run

SUITE = make_clean(n=4, seed=3)
EPISODE = SUITE.episodes[0]


def want(**over) -> ChannelSpec:
    base = dict(
        name="position",
        kind="vector",
        shape=(3,),
        dtype="float32",
        units="m",
        frame="world",
        rate_hz=20.0,
        semantics="position",
    )
    return ChannelSpec(**{**base, **over})


def wire(consumer_spec, adapters=None):
    requirement = requires_channels("consumer", "feedback", consumer_spec)
    wiring, verdict = bind(requirement, EPISODE.schema, adapters or AdapterRegistry())
    assert verdict.ok, verdict.explain()
    return wiring


# -- applying ---------------------------------------------------------------


def test_a_direct_binding_leaves_the_numbers_alone():
    adapted = adapt_episode(EPISODE, wire(want()))
    assert np.array_equal(adapted.array("position"), EPISODE.array("position"))


def test_a_unit_conversion_is_actually_performed():
    """The gap this closes: a plan that reported a conversion and never did it."""
    adapted = adapt_episode(EPISODE, wire(want(units="mm"), default_registry()))
    assert np.allclose(adapted.array("position"), EPISODE.array("position") * 1000.0)
    assert adapted.channel("position").units == "mm"


def test_conversion_stays_lazy_and_honours_a_window():
    adapted = adapt_episode(EPISODE, wire(want(units="mm"), default_registry()))
    assert len(adapted) == len(EPISODE)
    window = adapted.read(["position"], start=2, stop=5)
    assert window["position"].shape == (3, 3)
    assert np.allclose(window["position"], EPISODE.array("position")[2:5] * 1000.0)


def test_a_resample_materialises_because_a_window_would_lie():
    """After resampling, step 2 is a different moment. Slicing lazily would
    return the right count of the wrong rows."""
    adapted = adapt_episode(EPISODE, wire(want(rate_hz=10.0), default_registry()))
    assert len(adapted) != len(EPISODE)
    assert len(adapted) == pytest.approx(len(EPISODE) / 2, abs=1)


def test_a_renamed_binding_presents_the_consumer_its_own_name():
    """Matched by meaning, delivered under the name the consumer asked for."""
    requirement = requires_channels(
        "consumer", "feedback", want(name="eef_xyz"), aliases={"eef_xyz": ("position",)}
    )
    wiring, verdict = bind(requirement, EPISODE.schema, AdapterRegistry())
    assert verdict.ok
    adapted = adapt_episode(EPISODE, wiring)
    assert adapted.channel_names == ("eef_xyz",)
    assert np.array_equal(adapted.array("eef_xyz"), EPISODE.array("position"))


def test_an_unknown_channel_still_raises_after_adaptation():
    adapted = adapt_episode(EPISODE, wire(want(units="mm"), default_registry()))
    with pytest.raises(KeyError):
        adapted.array("nope")


def test_an_adapter_that_cannot_be_applied_cannot_be_registered():
    """Caught at registration, not on the first read halfway through a run."""
    from gantry.resolve import Adapter

    planner_only = Adapter("paper", "1.0", closes=("units.scale",), cost=lambda a, b: ())
    with pytest.raises(ValueError, match="on paper and not in the data"):
        AdapterRegistry([planner_only])


# -- storing ----------------------------------------------------------------


def run_with_arrays():
    return RunRecord(
        Provenance(protocol={"epochs": 2}, notes=("a note",)),
        tuple(make_defective("never_completes", n=4, fraction=0.5, seed=1).episodes),
    )


def run_label_only():
    return RunRecord(
        Provenance(protocol={"epochs": 1}),
        (
            episode_from_labels(
                id="s0",
                source="log",
                labels=EpisodeLabels(
                    success=True,
                    stage_events=(StageEvent("reached", 3), StageEvent("grasped", 9)),
                    annotations={"operator": "clean"},
                ),
                steps=10,
            ),
        ),
    )


@pytest.mark.parametrize("record", [run_with_arrays(), run_label_only()], ids=["arrays", "labels"])
def test_a_record_reads_back_as_the_same_thing(tmp_path, record):
    """The claim that matters: not "we serialise it" but "we can read it back"."""
    path = write_run(record, tmp_path / "run")
    assert same_run(read_run(path), record)


def test_labels_milestones_and_annotations_all_survive(tmp_path):
    restored = read_run(write_run(run_label_only(), tmp_path / "r"))
    labels = restored.episodes[0].labels
    assert labels.success is True
    assert labels.stages == ("reached", "grasped")
    assert labels.step_of("grasped") == 9
    assert labels.annotations["operator"] == "clean"


def test_provenance_and_metrics_survive(tmp_path):
    from gantry.spine import Measurement

    record = run_with_arrays().with_metrics(
        {"progress": Measurement(34.83, n=100, ci=(28.7, 41.7), units="percent")}
    )
    restored = read_run(write_run(record, tmp_path / "r"))
    assert restored.digest == record.digest
    assert restored.metrics["progress"].ci == (28.7, 41.7)


def test_a_load_bearing_metadata_key_survives_the_round_trip(tmp_path):
    """Otherwise storing a record disarms the refusal it was declared to cause.

    Two channels of four floats, one scalar-first and one scalar-last, are
    indistinguishable by shape, dtype, units and meaning. The only thing that
    keeps them apart is the discriminator, so a serialiser that quietly drops it
    makes a stored episode more compatible than the one it came from.
    """
    from gantry.serial import spec_from_dict, spec_to_dict
    from gantry.spine import compatible

    provider = ChannelSpec(
        "rotation",
        "vector",
        (4,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_xyzw"},
    )
    restored = spec_from_dict(spec_to_dict(provider))
    assert restored == provider

    consumer = ChannelSpec(
        "rotation",
        "vector",
        (4,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_wxyz"},
    )
    assert "metadata.mismatch" in compatible(restored, consumer).codes()


def test_a_label_only_record_writes_no_sidecar(tmp_path):
    """An evaluation log stays one small file."""
    path = write_run(run_label_only(), tmp_path / "r")
    assert not path.with_suffix(".npz").exists()


def test_a_record_with_channels_writes_one(tmp_path):
    path = write_run(run_with_arrays(), tmp_path / "r")
    assert path.with_suffix(".npz").exists()


def test_an_unknown_schema_version_is_refused_by_number(tmp_path):
    import json

    path = write_run(run_label_only(), tmp_path / "r")
    payload = json.loads(path.read_text())
    payload["version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="version 99"):
        read_run(path)


def test_a_missing_sidecar_is_refused_rather_than_half_read(tmp_path):
    path = write_run(run_with_arrays(), tmp_path / "r")
    path.with_suffix(".npz").unlink()
    with pytest.raises(ConfigError, match="travel together"):
        read_run(path)


def test_a_missing_record_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no record at"):
        read_run(tmp_path / "nope.json")


# -- holding a plane fixed, and knowing when you cannot --------------------


def _cohort(name, *, policy=None, evaluation=None, provenance=True):
    from gantry.contracts.feedback import Cohort
    from gantry.spine import ComponentRef

    components = []
    if policy:
        components.append(ComponentRef("policy", policy, "1.0"))
    if evaluation:
        components.append(ComponentRef("evaluation", evaluation, "1.0"))
    episodes = (episode_from_labels(id="s0", source=name, labels=EpisodeLabels(success=True)),)
    return Cohort(
        name,
        episodes,
        provenance=Provenance(components=tuple(components)) if provenance else None,
    )


def _holder(planes=("policy",)):
    from gantry.contracts.feedback import FeedbackModule, Report, feedback_descriptor
    from gantry.resolve import requires_channels

    class Holding(FeedbackModule):
        def descriptor(self):
            return feedback_descriptor(
                "holding", "1", min_cohorts=2, prescribes=False, holds=planes
            )

        def requirement(self):
            return requires_channels("holding", "feedback")

        def analyse(self, cohorts):
            return Report("holding", cohorts=tuple(c.name for c in cohorts))

    return Holding()


def test_two_runs_naming_the_same_policy_agree():
    verdict = _holder().check_comparability([_cohort("a", policy="p"), _cohort("b", policy="p")])
    assert verdict.ok and not verdict.codes()


def test_two_runs_naming_different_policies_are_refused():
    """The confound the whole design exists to catch."""
    verdict = _holder().check_comparability([_cohort("a", policy="p"), _cohort("b", policy="q")])
    assert "feedback.incomparable" in verdict.codes()
    assert not verdict.ok


def test_one_side_naming_a_policy_and_the_other_not_is_unverifiable():
    """Different from a confound: somebody may have changed it and not said so."""
    verdict = _holder().check_comparability([_cohort("a", policy="p"), _cohort("b")])
    assert "feedback.unverifiable" in verdict.codes()
    assert not verdict.ok


def test_recordings_with_no_policy_at_all_are_vacuously_held():
    """A screen over raw datasets. Nothing downstream existed, so nothing varied.

    This used to read as suspicious, because a bare boolean cannot tell "they
    differ" from "there was never one".
    """
    verdict = _holder().check_comparability([_cohort("a"), _cohort("b"), _cohort("c")])
    assert verdict.ok
    assert "feedback.nothing_held" in verdict.codes()


def test_a_cohort_with_no_provenance_counts_as_naming_nothing():
    verdict = _holder().check_comparability(
        [_cohort("a", provenance=False), _cohort("b", provenance=False)]
    )
    assert verdict.ok and "feedback.nothing_held" in verdict.codes()


def test_provenance_answers_four_ways_not_two():
    from gantry.spine import ComponentRef

    p = Provenance(components=(ComponentRef("policy", "p", "1.0"),))
    q = Provenance(components=(ComponentRef("policy", "q", "1.0"),))
    empty = Provenance()
    assert p.agreement(p, "policy") == "same"
    assert p.agreement(q, "policy") == "differs"
    assert p.agreement(empty, "policy") == "one_sided"
    assert empty.agreement(empty, "policy") == "absent"
    # The old bare-boolean answer collapses the last three into one.
    assert not p.comparable_to(empty, ["policy"])
    assert not empty.comparable_to(empty, ["policy"])


def test_a_seed_derived_from_data_is_the_same_in_every_process():
    """The builtin hash() is salted per process for strings.

    Both reference policies seeded from it, so they were deterministic within a
    run and not across two — while still declaring determinism, and still
    passing a conformance check that can only look inside one interpreter. Two
    runs of the same seeded experiment on different days simply disagreed.
    """
    import subprocess
    import sys

    from gantry.spine import seed_from

    here = seed_from(0, "scene-1", 3)
    elsewhere = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gantry.spine import seed_from; print(seed_from(0, 'scene-1', 3))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert str(here) == elsewhere

    salted = subprocess.run(
        [sys.executable, "-c", "print(abs(hash((0, 'scene-1', 3))) % (2**32))"],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "random", "PATH": ""},
    ).stdout.strip()
    assert salted != elsewhere, "the builtin would have agreed by luck; rerun"


def test_the_seed_is_stable_and_well_spread():
    from gantry.spine import seed_from

    assert seed_from("a") == seed_from("a")
    assert seed_from("a") != seed_from("b")
    assert 0 <= seed_from(1, 2, 3) < 2**32
