"""Getting the dataset to wherever the worker actually is, and unpacked.

Until this existed, a job handed out a filesystem path and the worker opened it.
That only works because the API and the worker happened to share a disk, and it
quietly made the whole product single-machine: the GPU box, which is the entire
reason the paid gates exist, could not run a worker at all.

So a job now carries a path *and* a URL. A worker on the same disk opens the
path and copies nothing. A worker anywhere else fetches. The local case is an
optimisation, not the mechanism, and neither is the special one.

Unpacking belongs here too
--------------------------
Gates after intake read ``workdir/unpacked``. On one machine that directory was
left behind by intake, so every later gate could assume it. A worker that only
handles the signal check or the robot test has never run intake and has no such
directory -- and it would fail with "the unpacked dataset is missing", which
reads as a bug in the pipeline rather than as a machine that was simply never
given the data. A version of this that fetched the archive and stopped there had
exactly that bug.

So both callers land here, and it is idempotent: present means nothing happens,
absent means fetch and unpack.

Downloading is injected
-----------------------
``download`` is passed in rather than done here, because the caller is the one
that knows the worker token. Two places building the same authenticated request
is how one of them ends up without the header.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

Report = Callable[..., None]
Download = Callable[[str, Path], Path]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing."""


def ensure_dataset(
    job: Mapping[str, Any],
    workdir: Path,
    download: Download,
    report: Report = _quiet,
    *,
    gate: str = "",
) -> Path:
    """The archive on *this* machine, with ``workdir/unpacked`` filled in.

    Returns the archive path, because intake still wants it. The unpacking is
    the part everything after intake depends on, and doing it here is what
    makes a worker location-independent: a box that only runs the signal check
    has never run intake and would otherwise open a directory that only exists
    on the API's host.

    Two versions of this existed at once, one of them defined inside the worker
    and shadowing the other. The shadowing one fetched the archive and stopped,
    so a remote run downloaded ten megabytes and then reported "the unpacked
    dataset is missing" -- which reads as a corrupt upload and was really two
    functions with the same name.
    """
    from gates.intake import unpack

    workdir.mkdir(parents=True, exist_ok=True)
    local = job.get("archive")
    archive = Path(local) if local and Path(local).exists() else None

    if archive is None:
        target = workdir / "dataset.zip"
        if target.exists() and target.stat().st_size == (job.get("archive_bytes") or -1):
            # Already fetched, and the size matches what the API said it sent.
            # A retried job should not re-download gigabytes.
            archive = target
        else:
            url = job.get("archive_url")
            if not url:
                raise FileNotFoundError("the job names no dataset this machine can reach")
            report(
                "fetching the dataset",
                note=f"{(job.get('archive_bytes') or 0) / 1e6:.0f} MB",
            )
            archive = download(url, target)

    # Intake unpacks for itself, and the difference is not duplication: intake
    # is the gate that JUDGES the archive, so a refusal from unpack is its
    # verdict to give -- "this is a v3 export" with the fix attached. Unpacking
    # here first turned that refusal into a RuntimeError, the runner's catch-all
    # called it our machinery breaking, and the visitor got a retry button for
    # a dataset whose problem had a sentence. The raise below stays correct for
    # every LATER gate, where intake has already accepted the archive and a
    # refusal here really is a truncated transfer or a full disk.
    if gate == "g0":
        return archive

    unpacked = workdir / "unpacked"
    if unpacked.exists() and any(unpacked.iterdir()):
        return archive

    # The reporter goes in. `unpack` is no longer just a zip: raw egocentric
    # footage is converted here, which is minutes of hand tracking, and every
    # phase it reports was being dropped on the floor. The phase then sat on
    # "unpacking the dataset" for the whole run, which is the exact failure the
    # timeline exists to prevent: a bar that has not moved is indistinguishable
    # from a worker that has died.
    #
    # This is also where the work happens for *every* gate, not just intake. A
    # box that only runs the signal check unpacks here too, so this is the one
    # place the conversion can be reported from.
    report("unpacking the dataset")
    root, problems = unpack(archive, unpacked, report)
    if root is None:
        # Intake already accepted this archive, so a failure here is ours -- a
        # truncated transfer or a full disk, not a bad upload. Raised rather
        # than returned, so the worker reports it as our failure and not as a
        # judgement on somebody's data.
        detail = problems[0]["summary"] if problems else "no readable dataset inside"
        raise RuntimeError(f"the archive would not unpack on this worker: {detail}")
    return archive


def clean(workdir: Path) -> None:
    """Remove a fetched copy, for a worker that is not holding the original."""
    shutil.rmtree(workdir, ignore_errors=True)
