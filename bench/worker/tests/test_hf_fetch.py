"""The worker's half of the Hub fetch: refuse before bytes, report either way.

No network anywhere in here. The Hub is a fake module injected into
sys.modules, because what these tests pin is our behaviour around the Hub --
which refusals fire before any download, what the archive looks like, and
that every ending reaches the API as a sentence -- not the Hub itself.
"""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hf  # noqa: E402


class _File:
    def __init__(self, name, size):
        self.rfilename = name
        self.size = size


class _GatedRepoError(Exception):
    pass


class _RepositoryNotFoundError(Exception):
    pass


def fake_hub(monkeypatch, files=None, raises=None, tree=None):
    """Install a fake huggingface_hub that answers from the arguments."""
    hub = types.ModuleType("huggingface_hub")
    utils = types.ModuleType("huggingface_hub.utils")
    utils.GatedRepoError = _GatedRepoError
    utils.RepositoryNotFoundError = _RepositoryNotFoundError

    class _Api:
        def dataset_info(self, repo, files_metadata=False):
            if raises is not None:
                raise raises
            info = types.SimpleNamespace()
            info.siblings = files or []
            return info

    hub.HfApi = _Api

    def snapshot_download(repo_id, repo_type, local_dir):
        root = Path(local_dir)
        for name, size in (tree or {}).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * size)
        return str(root)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)


LEROBOT = [_File("meta/info.json", 100), _File("data/chunk-000/file.parquet", 1000)]


def test_only_a_repo_id_is_a_repo_id():
    assert hf.repo_of({"repo": "lerobot/pusht"}) == "lerobot/pusht"
    for bad in ("", "pusht", "http://evil.example/x", "a/b/c", "a b/c"):
        with pytest.raises(hf.Refusal):
            hf.repo_of({"repo": bad})


def test_a_repo_without_lerobot_meta_is_refused_before_any_bytes(monkeypatch, tmp_path):
    fake_hub(monkeypatch, files=[_File("README.md", 10), _File("train.csv", 10**6)])
    with pytest.raises(hf.Refusal, match="not a LeRobot"):
        hf.preflight("some/tabular", tmp_path)


def test_too_big_is_refused_with_the_numbers(monkeypatch, tmp_path):
    huge = [_File("meta/info.json", 100), _File("videos/all.mp4", hf.MAX_BYTES + 1)]
    fake_hub(monkeypatch, files=huge)
    with pytest.raises(hf.Refusal, match="GB"):
        hf.preflight("some/huge", tmp_path)


def test_gated_and_missing_each_get_their_own_sentence(monkeypatch, tmp_path):
    fake_hub(monkeypatch, raises=_GatedRepoError())
    with pytest.raises(hf.Refusal, match="gated"):
        hf.preflight("some/gated", tmp_path)
    fake_hub(monkeypatch, raises=_RepositoryNotFoundError())
    with pytest.raises(hf.Refusal, match="no public dataset"):
        hf.preflight("some/missing", tmp_path)


def test_the_archive_keeps_the_tree_and_drops_the_cache(monkeypatch, tmp_path):
    fake_hub(monkeypatch, tree={
        "meta/info.json": 20,
        "data/chunk-000/episode.parquet": 50,
        "videos/cam/episode_000000.mp4": 80,
        ".cache/huggingface/download/lock": 5,
    })
    archive = hf.fetch("some/set", tmp_path, lambda *a, **k: None)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert {"meta/info.json", "data/chunk-000/episode.parquet",
            "videos/cam/episode_000000.mp4"} <= {n.replace("\\", "/") for n in names}
    assert not any(".cache" in n for n in names)
    # the downloaded tree is gone: the zip is the artefact, not the copy
    assert not (tmp_path / "hub").exists()


def _calls():
    made = []

    def call(api, path, payload=None):
        made.append((path, payload))
        return {}

    return made, call


def test_a_good_fetch_reports_fetched_with_the_archive(monkeypatch, tmp_path):
    fake_hub(monkeypatch, files=LEROBOT, tree={"meta/info.json": 20, "data/f.parquet": 30})
    made, call = _calls()
    out = hf.run("http://api", call, {"id": "job_1", "params": {"repo": "lerobot/pusht"}},
                 tmp_path, lambda *a, **k: None)
    assert out.startswith("fetched")
    path, payload = made[-1]
    assert path == "/api/jobs/job_1/fetched"
    assert Path(payload["path"]).exists() and payload["repo"] == "lerobot/pusht"


def test_a_refusal_reaches_the_visitor_as_its_own_sentence(monkeypatch, tmp_path):
    fake_hub(monkeypatch, raises=_RepositoryNotFoundError())
    made, call = _calls()
    out = hf.run("http://api", call, {"id": "job_2", "params": {"repo": "no/such"}},
                 tmp_path, lambda *a, **k: None)
    assert out.startswith("refused")
    path, payload = made[-1]
    assert path == "/api/jobs/job_2/fetch-failed"
    assert "no public dataset" in payload["reason"]


def test_our_own_breakage_says_so_rather_than_blaming_the_dataset(monkeypatch, tmp_path):
    fake_hub(monkeypatch, files=LEROBOT)

    def broken(*a, **k):
        raise OSError("disk went away")

    sys.modules["huggingface_hub"].snapshot_download = broken
    made, call = _calls()
    out = hf.run("http://api", call, {"id": "job_3", "params": {"repo": "a/b"}},
                 tmp_path, lambda *a, **k: None)
    assert out.startswith("failed")
    path, payload = made[-1]
    assert path == "/api/jobs/job_3/fetch-failed"
    assert "our side" in payload["reason"] and "disk went away" in payload["reason"]
