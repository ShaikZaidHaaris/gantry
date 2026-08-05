"""The kit is the acceptance test. Everything else is detail.

Run against corpora written by :mod:`gantry_connector_so101.fixtures` from
``gantry.fixtures`` suites, so the trajectories are material this package did
not write and the file layout is the one ``lerobot-record`` produces.
"""

from __future__ import annotations

import pytest
from gantry_connector_so101 import BIMANUAL_REF, FOLLOWER_REF, SO101Connector
from gantry_connector_so101.fixtures import write_corpus

from gantry.conformance import check_connector, connector_checks
from gantry.fixtures import make_clean, make_defective, make_duration_confound


@pytest.fixture
def corpus(tmp_path):
    def _write(suite, name="corpus", **kwargs):
        root = write_corpus(suite.episodes, tmp_path / name, **kwargs)
        return SO101Connector(root, embodiment=kwargs.get("embodiment", FOLLOWER_REF))

    return _write


def test_passes_conformance_strictly(corpus):
    verdict = check_connector(corpus(make_clean(n=6)), strict=True)
    assert verdict.ok, verdict.explain()


def test_passes_conformance_on_every_fixture_shape(corpus):
    suites = [
        make_clean(n=4),
        make_defective("never_completes", n=6, fraction=0.5),
        make_defective("actuation_chatter", n=6, fraction=0.5),
        make_duration_confound(n=6),
    ]
    for index, suite in enumerate(suites):
        connector = corpus(suite, name=f"suite_{index}")
        verdict = check_connector(connector, strict=True, sample=4)
        assert verdict.ok, verdict.explain()


def test_the_bimanual_binding_conforms_too(corpus):
    """Twelve wide, and described as twelve wide, all the way through."""
    connector = corpus(make_clean(n=4), name="bi", embodiment=BIMANUAL_REF)
    assert check_connector(connector, strict=True).ok
    assert connector.open("episode_000000").channel("action").width == 12


def test_a_corpus_with_no_metadata_still_conforms(corpus):
    """The sidecar is what makes a corpus analysable, not what makes it readable."""
    connector = corpus(make_clean(n=4), name="bare", sidecar=False)
    assert check_connector(connector, strict=True).ok
    assert connector.descriptor().provides["outcomes"] is False


def test_the_kit_is_discoverable():
    codes = dict(connector_checks())
    assert codes["unknown_id"] == "an unknown id raises KeyError"


def test_an_unknown_episode_raises_keyerror(corpus):
    connector = corpus(make_clean(n=3))
    with pytest.raises(KeyError, match="episode_000099"):
        connector.open("episode_000099")


def test_reads_are_selective_and_lazy(corpus):
    """The lazy claim is delegated, so it has to be true of the composed reader."""
    episode = corpus(make_clean(n=3)).open("episode_000000")
    window = episode.read(["action"], start=3, stop=7)
    assert set(window) == {"action"}
    assert window["action"].shape == (4, 6)


def test_the_kit_catches_an_understated_capability(corpus, monkeypatch):
    """Understating is caught too: it denies analyses the data could support."""
    import dataclasses

    connector = corpus(make_clean(n=4))
    honest = connector.descriptor()
    assert honest.provides["outcomes"] is True
    monkeypatch.setattr(
        connector,
        "descriptor",
        lambda: dataclasses.replace(honest, provides={**honest.provides, "outcomes": False}),
    )
    verdict = check_connector(connector)
    assert not verdict.ok
    assert "conformance.capability_understated" in verdict.codes()
