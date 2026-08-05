"""Where the sample datasets are, in each layout that actually exists.

The download link worked on a laptop and returned "not in this checkout" on the
live site, from one line of arithmetic. The path was three parents up from this
module plus ``samples`` -- correct in a checkout, where the file is
``bench/api/app/main.py`` and three parents up is the repository root. Deployed,
the file is ``gantry_bench/api/app/main.py`` and three parents up is
``/home/ubuntu``, while the samples sit in a sibling tree at ``gantry/samples``.
Same arithmetic, different tree, no error anywhere until a visitor clicks
download.

Nothing in a single-layout test could have caught that, so these build both
layouts on disk and ask the resolver which directory it would use.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="bench-samples-test-"))
os.environ["BENCH_DATA"] = str(_TMP)

from app import main as mainmod  # noqa: E402


def resolver_for(app_file: Path):
    """Run the real resolver as though this module lived at ``app_file``."""
    real = mainmod.__file__
    try:
        mainmod.__file__ = str(app_file)
        return mainmod._samples_dir()
    finally:
        mainmod.__file__ = real


def test_a_checkout_layout_finds_samples_at_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_SAMPLES", raising=False)
    app = tmp_path / "bench" / "api" / "app"
    app.mkdir(parents=True)
    (tmp_path / "samples").mkdir()
    assert resolver_for(app / "main.py") == tmp_path / "samples"


def test_the_deployed_layout_finds_them_in_the_sibling_tree(tmp_path, monkeypatch):
    """The case that shipped broken.

    The API is unpacked to ``gantry_bench/`` and the pipeline, with its samples,
    to ``gantry/``. Counting parents lands on the directory above both.
    """
    monkeypatch.delenv("BENCH_SAMPLES", raising=False)
    app = tmp_path / "gantry_bench" / "api" / "app"
    app.mkdir(parents=True)
    (tmp_path / "gantry" / "samples").mkdir(parents=True)
    assert resolver_for(app / "main.py") == tmp_path / "gantry" / "samples", (
        "the deployed layout resolved somewhere else, which is what made every "
        "download link answer 'not in this checkout'"
    )


def test_an_explicit_setting_wins_over_both(tmp_path, monkeypatch):
    """For any layout the two guesses do not describe."""
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    monkeypatch.setenv("BENCH_SAMPLES", str(elsewhere))
    app = tmp_path / "bench" / "api" / "app"
    app.mkdir(parents=True)
    (tmp_path / "samples").mkdir()
    assert resolver_for(app / "main.py") == elsewhere


def test_the_offered_files_are_named_by_an_allow_list():
    """This route hands out files by name, which is the shape of a traversal bug.

    An allow-list means the URL parameter cannot address anything that is not on
    it, whatever it contains.
    """
    assert set(mainmod.SAMPLE_FILES) == {"two_handed", "one_handed"}
    for key, (filename, description) in mainmod.SAMPLE_FILES.items():
        assert "/" not in filename and ".." not in filename, filename
        assert filename.endswith(".zip")
        assert description


def test_listing_reports_whether_each_file_is_really_there(tmp_path, monkeypatch):
    """So a missing sample is visible before somebody clicks it.

    `available` is computed from the filesystem rather than assumed, which is
    the difference between a UI that can hide a dead link and one that offers a
    download ending in an error page.
    """
    monkeypatch.setattr(mainmod, "SAMPLES", tmp_path)
    (tmp_path / mainmod.SAMPLE_FILES["two_handed"][0]).write_bytes(b"PK\x03\x04zip")

    listing = {s["key"]: s for s in mainmod.samples()["samples"]}
    assert listing["two_handed"]["available"] is True
    assert listing["two_handed"]["bytes"] > 0
    assert listing["one_handed"]["available"] is False
    assert listing["one_handed"]["bytes"] == 0


def test_a_seeded_sample_carries_its_coaching(tmp_path, monkeypatch):
    """The worked examples must show the features the product has.

    They are seeded from a fixture, so every feature added after that fixture
    was written is invisible on the first page a visitor sees unless something
    carries it. Coaching was exactly that: the samples rendered every check and
    verdict correctly and had no advice section at all, while every real
    submission did.
    """
    import json as jsonlib

    from app import samples as samplesmod

    fixture = jsonlib.loads(samplesmod.FIXTURE.read_text())
    for spec in fixture["samples"]:
        coach = spec.get("coach") or {}
        assert coach.get("points"), f"{spec['id']} has no coaching in the fixture"
        assert coach.get("model"), f"{spec['id']} does not say which model wrote it"
        # Grounded, not decorative: the advice names findings the gates actually
        # produced. A point citing a code nothing reported would be the model
        # inventing a measurement, which is the one thing this must not do.
        assert isinstance(coach.get("fixes"), dict)
