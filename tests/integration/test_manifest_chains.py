"""Connectors composed in a manifest, which is what makes a pipeline declarative.

The ego pipeline is three connectors stacked, and until a manifest could say so
the whole thing lived in hand-written scripts — declarative in principle and
bespoke in practice. These check that a chain resolves, that each stage's own
refusals fire at plan time rather than an hour into a run, and that the record
says what was actually built.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.manifest import ComponentSpec, Manifest
from gantry.resolve import Registry
from gantry.runner import build_component
from gantry.spine import (
    ArraySource,
    ChannelSpec,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
)

SPEC = ChannelSpec("frames", "vector", (2,), "float32")


class Root(Connector):
    """A connector that reads from disk — the first link in any chain."""

    def __init__(self, *, path: str = "somewhere", episodes: int = 2):
        self.path = path
        self._n = episodes

    def descriptor(self):
        return connector_descriptor(
            name="root",
            version="0.1",
            lazy=True,
            stage_events=False,
            outcomes=False,
            media=False,
            writes=False,
            path=self.path,
        )

    def episode_ids(self):
        return tuple(f"root/{i}" for i in range(self._n))

    def schema(self, episode_id):
        return (SPEC,)

    def open(self, episode_id):
        if episode_id not in self.episode_ids():
            raise KeyError(episode_id)
        return EpisodeRecord(
            meta=EpisodeMeta(id=episode_id, source="root"),
            schema=(SPEC,),
            source=ArraySource({"frames": np.ones((3, 2), dtype="float32")}),
            labels=EpisodeLabels(),
        )


class Doubles(Connector):
    """A stage built on another. Source first and positional, per the contract."""

    def __init__(self, source, *, times: float = 2.0, needs: str = "frames"):
        self._source = source
        self._times = float(times)
        self._needs = needs

    def descriptor(self):
        upstream = self._source.descriptor()
        return connector_descriptor(
            name="doubles",
            version="0.1",
            lazy=False,
            stage_events=False,
            outcomes=False,
            media=False,
            writes=False,
            derived_from=f"{upstream.name}@{upstream.version}",
            times=self._times,
        )

    def episode_ids(self):
        return tuple(f"doubles/{i}" for i in self._source.episode_ids())

    def schema(self, episode_id):
        return self.open(episode_id).schema

    def open(self, episode_id):
        upstream_id = episode_id.removeprefix("doubles/")
        if upstream_id not in self._source.episode_ids():
            raise KeyError(episode_id)
        original = self._source.open(upstream_id)
        if not original.has(self._needs):
            # The refusal that must fire at PLAN time, not an hour in.
            raise ConfigError(
                f"{episode_id}: no {self._needs!r} channel to read; upstream has "
                f"{list(original.channel_names)}"
            )
        return EpisodeRecord(
            meta=EpisodeMeta(id=episode_id, source="doubles", derived_from=(upstream_id,)),
            schema=original.schema,
            source=ArraySource({"frames": original.array("frames") * self._times}),
            labels=original.labels,
        )


class Sourceless(Connector):
    """A component that does not take a source. Chaining onto it must refuse."""

    def __init__(self, **_):
        pass

    def descriptor(self):
        return connector_descriptor(
            name="sourceless",
            version="0.1",
            lazy=True,
            stage_events=False,
            outcomes=False,
            media=False,
            writes=False,
        )

    def episode_ids(self):
        return ()

    def schema(self, episode_id):
        return ()

    def open(self, episode_id):
        raise KeyError(episode_id)


def registry():
    made = Registry()
    made.register("dataset", "root", Root)
    made.register("dataset", "doubles", Doubles)
    made.register("dataset", "sourceless", Sourceless)
    return made


def chain(*links):
    return ComponentSpec.parse({"chain": list(links)}, "dataset")


# -- the thing this exists for ----------------------------------------------


def test_a_chain_builds_innermost_first_and_each_stage_reads_the_one_before():
    spec = chain(
        {"name": "root", "config": {"path": "uploads/42", "episodes": 2}},
        {"name": "doubles", "config": {"times": 3.0}},
    )
    component, ref, verdict = build_component(registry(), "dataset", spec)

    assert component.episode_ids() == ("doubles/root/0", "doubles/root/1")
    episode = component.open("doubles/root/0")
    assert np.allclose(episode.array("frames"), 3.0)
    # lineage falls out of the composition rather than being wired up
    assert episode.meta.derived_from == ("root/0",)
    assert verdict.ok


def test_three_deep_composes_in_order():
    spec = chain(
        {"name": "root"},
        {"name": "doubles", "config": {"times": 2.0}},
        {"name": "doubles", "config": {"times": 5.0}},
    )
    component, _ref, _v = build_component(registry(), "dataset", spec)
    episode = component.open("doubles/doubles/root/0")
    assert np.allclose(episode.array("frames"), 10.0)


def test_the_record_says_what_was_actually_built():
    spec = chain({"name": "root"}, {"name": "doubles"})
    _c, ref, _v = build_component(registry(), "dataset", spec)
    assert "root" in ref and "doubles" in ref
    assert "->" in ref


# -- refusals happen at plan time -------------------------------------------


def test_a_stage_whose_input_is_missing_refuses_before_any_work():
    """A chain that cannot work is refused before a frame is decoded, rather
    than an hour into a run."""
    spec = chain({"name": "root"}, {"name": "doubles", "config": {"needs": "hands"}})
    component, _ref, _v = build_component(registry(), "dataset", spec)
    with pytest.raises(ConfigError, match="no 'hands' channel"):
        component.open(component.episode_ids()[0])


def test_chaining_onto_something_that_takes_no_source_is_named_clearly():
    """The alternative is a TypeError from inside somebody's __init__ that reads
    as a config error."""
    spec = chain({"name": "root"}, {"name": "sourceless"})
    with pytest.raises(Exception, match="does not take one positionally"):
        build_component(registry(), "dataset", spec)


def test_an_unknown_stage_is_refused_by_name():
    spec = chain({"name": "root"}, {"name": "nope"})
    with pytest.raises(ConfigError):
        build_component(registry(), "dataset", spec)


# -- optional stages ---------------------------------------------------------


def test_an_optional_stage_that_cannot_build_is_skipped_and_recorded():
    """For stages that genuinely may not apply — lifting to a world frame needs
    a camera trajectory and half of real uploads have none. A silently shorter
    chain is a different pipeline wearing the same name."""
    spec = chain(
        {"name": "root"},
        {"name": "doubles", "config": {"times": 2.0}},
        {"name": "sourceless", "optional": True},
    )
    component, ref, verdict = build_component(registry(), "dataset", spec)

    # carried on from the stage before it
    assert np.allclose(component.open("doubles/root/0").array("frames"), 2.0)
    assert "skipped" in ref
    assert any(r.code == "chain.skipped" for r in verdict.reasons)


def test_a_required_stage_that_cannot_build_fails_the_run():
    spec = chain({"name": "root"}, {"name": "sourceless"})
    with pytest.raises(Exception):
        build_component(registry(), "dataset", spec)


def test_the_first_stage_cannot_be_optional():
    with pytest.raises(ConfigError, match="cannot be optional"):
        chain({"name": "root", "optional": True}, {"name": "doubles"})


# -- parsing and round-tripping ---------------------------------------------


def test_a_chain_survives_a_round_trip_through_json():
    spec = chain(
        {"name": "root", "config": {"path": "x"}},
        {"name": "doubles", "config": {"times": 4.0}},
        {"name": "sourceless", "optional": True},
    )
    again = ComponentSpec.parse(json.loads(json.dumps(spec.as_dict())), "dataset")
    assert again.ref == spec.ref
    assert [link.optional for link in again.chain] == [False, False, True]
    assert again.chain[1].config == {"times": 4.0}


def test_a_plain_component_still_parses_and_is_not_a_chain():
    spec = ComponentSpec.parse({"name": "root", "config": {"path": "x"}}, "dataset")
    assert not spec.chained
    assert spec.ref == "root"
    assert spec.as_dict() == {"name": "root", "config": {"path": "x"}}


def test_an_empty_chain_is_refused_with_the_ordering_explained():
    with pytest.raises(ConfigError, match="innermost first"):
        ComponentSpec.parse({"chain": []}, "dataset")
    with pytest.raises(ConfigError, match="non-empty list"):
        ComponentSpec.parse({"chain": "egovideo"}, "dataset")


def test_a_chain_inside_a_whole_manifest_parses():
    payload = {
        "version": 1,
        "name": "ego-run",
        "varies": "dataset",
        "cohorts": {
            "ego": {
                "chain": [
                    {"name": "root", "config": {"path": "uploads/42"}},
                    {"name": "doubles", "config": {"times": 2.0}},
                ]
            }
        },
    }
    manifest = Manifest.from_dict(payload)
    spec = manifest.cohorts["ego"]
    assert spec.ref == "root -> doubles"
    assert manifest.as_dict()["cohorts"]["ego"]["chain"][0]["name"] == "root"


# -- the real ego chain, resolved from the shipped manifest ------------------


def test_the_shipped_ego_manifest_resolves_to_the_real_pipeline():
    """The one that matters: the flagship pipeline, from the JSON that ships,
    with no Python in between."""
    from pathlib import Path

    from gantry.manifest import Manifest

    manifest = Manifest.load(Path(__file__).parents[2] / "manifests/ego/upload.json")
    spec = manifest.cohorts["ego"]

    assert spec.ref == "egovideo -> handpose -> egoactions -> worldframe"
    assert [link.optional for link in spec.chain] == [False, False, False, True]
    assert manifest.validate().ok


def test_every_argument_that_crosses_the_manifest_accepts_plain_data():
    """A component whose key argument is only constructible in Python is a
    component no manifest can use, which means the pipeline that uses it is a
    script."""
    from gantry_connector_egoactions import retargeter_from
    from gantry_connector_handpose import estimator_from

    rt = retargeter_from(
        {
            "mount": "aligned",
            "hand": {"closed": 0.02, "open": 0.10, "span": 0.19},
            "reach": "viperx_300",
        }
    )
    assert rt.provenance()["reach"] == "viperx_300"

    pair = retargeter_from(
        {
            "mount": "aligned",
            "reach": "viperx_300",
            "hand": {
                "left": {"closed": 0.02, "open": 0.09, "span": 0.19},
                "right": {"closed": 0.03, "open": 0.11, "span": 0.20},
            },
        }
    )
    assert sorted(pair) == ["left", "right"]

    # an already-built object passes through untouched
    assert retargeter_from(rt) is rt
    assert estimator_from(None) is None


def test_being_reachable_from_json_does_not_make_the_measurements_optional():
    """The refusals are the point and they survive the plain-data path."""
    from gantry_connector_egoactions import retargeter_from
    from gantry_connector_handpose import estimator_from

    with pytest.raises(ConfigError, match="cannot be defaulted"):
        retargeter_from({"mount": "aligned"})
    with pytest.raises(ConfigError, match="needs the camera's intrinsics"):
        estimator_from({"wire": "metric"})
    with pytest.raises(ConfigError, match="unknown estimator wire"):
        estimator_from({"wire": "magic"})
    with pytest.raises(ConfigError, match="unknown reach"):
        retargeter_from(
            {"mount": "aligned", "reach": "some_arm", "hand": {"closed": 0.02, "open": 0.10}}
        )


def test_a_rig_without_a_resolution_is_refused():
    """Intrinsics are in pixels; a focal length without a resolution is not a
    camera."""
    from gantry_connector_handpose import estimator_from

    with pytest.raises(ConfigError, match="width and height"):
        estimator_from({"wire": "metric", "rig": "gopro_hero5_wide"})


def test_planning_costs_one_episode_per_cohort_not_one_per_module():
    """A plan is a check, not a job.

    `fit_consumer` only ever wanted the first episode's schema, so reading the
    whole cohort to get it made planning a chain that decodes video cost the
    same as running it — and cost that again for every feedback module."""
    from gantry.runner import _cohort_schema

    class Computing(Root):
        """What connector_handpose is: the schema is not knowable until an
        episode has actually been produced."""

        def __init__(self):
            super().__init__(episodes=24)
            self.opened = []

        def open(self, episode_id):
            self.opened.append(episode_id)
            return super().open(episode_id)

        def schema(self, episode_id):
            return self.open(episode_id).schema

    c = Computing()
    schema = _cohort_schema(c)
    assert schema == c.open("root/0").schema
    # Twenty-four episodes upstream, and a plan that looks at one of them.
    assert set(c.opened) == {"root/0"}


def test_a_cohort_with_no_episodes_plans_without_pretending_to_have_channels():
    from gantry.runner import _cohort_schema

    class Empty(Root):
        def episode_ids(self):
            return ()

    assert _cohort_schema(Empty()) == ()
