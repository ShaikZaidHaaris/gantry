"""Pull a Hugging Face dataset so a visitor does not have to upload one.

Launch day made the case: hundreds of visitors read the worked example, none
of them had a LeRobot archive on the machine they were browsing from, and
every dataset they own lives on the Hub. This closes the distance -- paste the
link, this fetches it, and everything downstream is the same pipeline an
upload feeds.

Refusals before bytes
---------------------
The expensive mistake is discovering a problem after gigabytes have moved, so
everything checkable is checked against the Hub's metadata first: the repo
must exist, must carry a ``meta/info.json`` (a LeRobot export, not four
gigabytes of something else), must fit under the size cap, and must fit on
this disk with room to zip it. Each refusal is a sentence a visitor can act
on, reported through ``fetch-failed`` -- never through the gate vocabulary,
because there is no gate yet and "refused" is reserved for data we actually
read.

The id is validated again here even though the API already did: this module
is what talks to the network, so this module holds the last door. It only
ever speaks to the Hub, about a repo id -- never to a URL a caller supplied.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path

#: Same shape the API enforces: ``owner/name`` in the Hub's own alphabet.
REPO = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})/[A-Za-z0-9._-]{1,96}$")

#: Largest dataset we will pull, in bytes. Bigger than the browser-upload cap
#: on purpose -- the server's bandwidth and disk are ours to budget, a
#: visitor's are not -- but still bounded, because one paste must not be able
#: to fill the disk every other submission shares.
MAX_BYTES = int(os.environ.get("BENCH_HF_MAX_BYTES", 4 * 1024**3))

#: Disk that must remain free after the download AND the zip built from it.
#: Both exist at once for a moment, so the requirement is roughly twice the
#: dataset plus this margin.
FREE_MARGIN = 2 * 1024**3


class Refusal(Exception):
    """A fetch that should not proceed, with a sentence the visitor can act on."""


def repo_of(params: dict) -> str:
    repo = str(params.get("repo") or "").strip()
    if not REPO.match(repo):
        raise Refusal(f"{repo[:120]!r} is not a Hugging Face dataset id")
    return repo


def preflight(repo: str, workdir: Path) -> int:
    """Everything the Hub's metadata can refuse, refused before any bytes move."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as error:  # pragma: no cover - environment, not logic
        raise RuntimeError(f"huggingface_hub is not installed on this worker: {error}")

    try:
        info = HfApi().dataset_info(repo, files_metadata=True)
    except GatedRepoError:
        raise Refusal(
            f"{repo} is gated on Hugging Face, so we cannot pull it. Accept its terms "
            "and download it yourself, then upload the archive here"
        )
    except RepositoryNotFoundError:
        raise Refusal(
            f"there is no public dataset called {repo} on Hugging Face. Check the link "
            "on the dataset's page; private repos are invisible to us"
        )

    files = list(info.siblings or [])
    names = {f.rfilename for f in files}
    if not any(n == "meta/info.json" or n.endswith("/meta/info.json") for n in names):
        raise Refusal(
            f"{repo} has no meta/info.json, so it is not a LeRobot v2 dataset. The "
            "checks read that export; convert with LeRobot and push, or upload raw "
            "egocentric video with a clips.json instead"
        )

    total = sum(f.size or 0 for f in files)
    if total > MAX_BYTES:
        raise Refusal(
            f"{repo} is {total / 1024**3:.1f} GB, over the {MAX_BYTES // 1024**3} GB "
            "fetch limit. Push a subset of the episodes to a new repo, or upload a "
            "trimmed archive: the checks read the same things either way"
        )

    workdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workdir).free
    if total * 2 + FREE_MARGIN > free:
        # Ours, not theirs: the dataset is within policy and the machine is
        # what cannot take it right now.
        raise RuntimeError(
            f"not enough disk for {total / 1024**3:.1f} GB twice over "
            f"({free / 1024**3:.1f} GB free); this is our capacity, not your data"
        )
    return total


def fetch(repo: str, workdir: Path, report) -> Path:
    """Download and zip, returning the archive's path on this disk.

    ZIP_STORED, not deflate: the bulk of a LeRobot export is mp4, which does
    not compress twice, and intake unpacks this again minutes from now.
    """
    from huggingface_hub import snapshot_download

    report("downloading", note=repo)
    tree = Path(
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(workdir / "hub"))
    )

    report("packing", note=repo)
    archive = workdir / "fetched.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(tree.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                zf.write(path, path.relative_to(tree))
    shutil.rmtree(tree, ignore_errors=True)
    return archive


def run(api: str, call, job: dict, workdir: Path, report) -> str:
    """The whole fetch, reported to the API whichever way it ends."""
    try:
        repo = repo_of(job.get("params") or {})
        total = preflight(repo, workdir)
        report("preflight", note=f"{repo}: {total / 1024**2:.0f} MB")
        archive = fetch(repo, workdir, report)
        call(api, f"/api/jobs/{job['id']}/fetched", {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "repo": repo,
        })
        return f"fetched {repo}"
    except Refusal as refusal:
        call(api, f"/api/jobs/{job['id']}/fetch-failed", {"reason": str(refusal)})
        return f"refused: {refusal}"
    except Exception as error:  # noqa: BLE001 - reported, not raised past the job
        call(api, f"/api/jobs/{job['id']}/fetch-failed", {
            "reason": "the fetch broke on our side, not because of the dataset: "
                      f"{type(error).__name__}: {str(error)[:200]}",
        })
        return f"failed: {error}"
