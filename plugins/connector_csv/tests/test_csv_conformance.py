"""The kit is the acceptance test. Everything else is detail."""

from __future__ import annotations

import pytest
from gantry_connector_csv import CsvConnector, write_episodes

from gantry.conformance import check_connector, connector_checks
from gantry.fixtures import make_clean, make_defective, make_duration_confound


@pytest.fixture
def written(tmp_path):
    def _write(suite, name="data.csv", sidecar=True):
        path = write_episodes(suite.episodes, tmp_path / name, sidecar=sidecar)
        return CsvConnector(path)

    return _write


def test_passes_conformance_strictly(written):
    connector = written(make_clean(n=6))
    verdict = check_connector(connector, strict=True)
    assert verdict.ok, verdict.explain()


def test_passes_conformance_on_every_fixture_shape(written):
    for index, suite in enumerate(
        [
            make_clean(n=4),
            make_defective("never_completes", n=6, fraction=0.5),
            make_defective("actuation_chatter", n=6, fraction=0.5),
            make_duration_confound(n=6),
        ]
    ):
        connector = written(suite, name=f"suite_{index}.csv")
        verdict = check_connector(connector, strict=True, sample=4)
        assert verdict.ok, verdict.explain()


def test_without_a_sidecar_the_data_is_undescribed_and_says_so(written):
    """A bare CSV carries numbers with no stated meaning.

    The connector cannot invent units it was not given, and must not. What it
    can do is be honest, so the gap shows up as a note by default and as a
    refusal under strict, rather than as a channel that quietly matches
    anything of the same width.
    """
    connector = written(make_clean(n=4), name="bare.csv", sidecar=False)

    lenient = check_connector(connector, strict=False)
    assert lenient.ok
    assert "conformance.channel_undescribed" in lenient.codes()

    strict = check_connector(connector, strict=True)
    assert not strict.ok
    assert "conformance.channel_undescribed" in strict.codes()


def test_the_kit_is_discoverable():
    codes = dict(connector_checks())
    assert "reads_selective" in codes
    assert codes["unknown_id"] == "an unknown id raises KeyError"


def test_the_kit_can_run_a_subset(written):
    connector = written(make_clean(n=3))
    assert check_connector(connector, only=["ids_unique", "unknown_id"]).ok


# -- the kit must actually catch things ------------------------------------


class _Broken(CsvConnector):
    """A connector that violates one invariant, to prove the kit notices."""

    def __init__(self, path, breakage: str):
        super().__init__(path)
        self._breakage = breakage

    def episode_ids(self):
        ids = super().episode_ids()
        if self._breakage == "unstable_order":
            return tuple(reversed(ids)) if not hasattr(self, "_seen") else ids
        if self._breakage == "duplicate_ids":
            return ids + ids[:1]
        return ids

    def open(self, episode_id):
        if self._breakage == "silent_unknown" and episode_id not in super().episode_ids():
            return super().open(super().episode_ids()[0])
        return super().open(episode_id)

    def descriptor(self):
        descriptor = super().descriptor()
        if self._breakage == "understated":
            import dataclasses

            return dataclasses.replace(
                descriptor, provides={**descriptor.provides, "stage_events": False}
            )
        return descriptor


@pytest.mark.parametrize(
    "breakage,code",
    [
        ("duplicate_ids", "conformance.ids_duplicate"),
        ("silent_unknown", "conformance.unknown_id_silent"),
        ("understated", "conformance.capability_understated"),
    ],
)
def test_kit_catches_a_broken_connector(tmp_path, breakage, code):
    path = write_episodes(make_clean(n=4).episodes, tmp_path / "d.csv")
    verdict = check_connector(_Broken(path, breakage))
    assert not verdict.ok
    assert code in verdict.codes(), verdict.explain()


def test_a_check_that_raises_is_reported_not_swallowed(tmp_path):
    path = write_episodes(make_clean(n=3).episodes, tmp_path / "d.csv")

    class Exploding(CsvConnector):
        def episode_ids(self):
            if getattr(self, "_armed", False):
                raise RuntimeError("boom")
            return super().episode_ids()

    connector = Exploding(path)
    connector._armed = True
    verdict = check_connector(connector)
    assert not verdict.ok
    assert "conformance.check_raised" in verdict.codes()
