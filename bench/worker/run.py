"""The worker: claim a job, run its gate, say where it is, report the verdict.

Deliberately dumb. It knows how to talk to the API and which function handles
which gate; everything else is the gate's business. A gate that raises is
reported as ``failed`` -- our machinery broke -- which the product presents
differently from ``refused``, where the user's data genuinely did not pass.
Conflating those two would blame a user for our bug.

Why the heartbeat is a thread
-----------------------------
Gates block. The signal check trains for ten minutes and the robot test runs
closed-loop for hours, and both spend that time inside a single call that
returns once. A worker that beat between steps would go quiet for the whole of
it and be swept as dead by a server that cannot tell "busy" from "gone" -- and
being swept mid-run means an honest result is thrown away and reported as our
failure, which is the expensive direction to be wrong in.

So the beat runs on its own thread and continues regardless of what the gate is
doing. The gate calls ``report(...)`` whenever it moves; the thread sends
whatever the latest position is on its own schedule. Nothing in a gate has to
think about beat intervals, and a gate that never reports still stays alive.

Where the dataset comes from
----------------------------
Fetched over HTTP, into this worker's own scratch. The job also names a local
path, and that is used when the file is actually there -- a worker on the API's
host should not copy a ten gigabyte archive to itself. But the path is an
optimisation, not the mechanism: a worker on a GPU box has no such file and
downloads instead, and neither case is the special one. The version that only
handed out a path worked solely because the API and the worker happened to share
a disk, which is not a design so much as a coincidence that had not broken yet.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import threading
import time
import traceback
from pathlib import Path

import urllib.error
import urllib.request
import json as jsonlib

import coach
import hf
from fetch import Download, ensure_dataset
from gates import intake, report, robot, signal

HANDLERS = {
    "g0": intake.run,
    "g1": report.run,
    "g2": signal.run,
    "g3": robot.run,
}

#: Longest silence between beats. Comfortably inside the server's staleness
#: window. This is the *floor* on how often we speak, not the rate: a gate that
#: has moved is sent sooner than this.
BEAT = 10.0

#: Shortest gap between two sends. A phase change should appear at once --
#: "training" turning into "evaluating" ten seconds late is a bar that lies
#: about where the run is -- but a counter ticking a thousand times a second
#: must not become a thousand requests. So the sender wakes on change and this
#: is what stops it becoming a flood.
GAP = 0.4


#: Presented on every call. Set from --token or BENCH_WORKER_TOKEN, and the
#: server only checks it when it has one configured -- so a laptop talking to
#: its own API needs nothing, and a box reaching across a network needs both
#: ends set.
TOKEN = os.environ.get("BENCH_WORKER_TOKEN", "")


def headers() -> dict[str, str]:
    out = {"Content-Type": "application/json"}
    if TOKEN:
        out["X-Worker-Token"] = TOKEN
    return out


def call(api: str, path: str, payload: dict | None = None) -> dict:
    url = f"{api}{path}"
    data = jsonlib.dumps(payload or {}).encode()
    request = urllib.request.Request(url, data=data, headers=headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return jsonlib.loads(response.read() or b"{}")


def downloader(api: str) -> Download:
    """A token-carrying fetch of one artefact to one path.

    Streamed in chunks rather than read whole: a contributor's upload is
    routinely gigabytes, and a worker that holds one in memory to write it out
    falls over on exactly the submissions worth running.

    Written to a partial file and moved into place at the end, so a worker that
    dies mid-transfer leaves nothing the next run mistakes for a complete
    archive -- which would fail later, elsewhere, looking like corrupt data
    rather than an interrupted download.
    """

    def download(url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        request = urllib.request.Request(f"{api}{url}", headers=headers())
        with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(target)
        return target

    return download


class Progress:
    """Where the gate is, and a thread that keeps telling the server so.

    ``report`` is what a gate is handed. It only writes to memory, so a gate may
    call it as often as it likes -- once per training step is fine -- without
    turning progress into network traffic.
    """

    def __init__(self, api: str, job_id: str, *, beat: float = BEAT, gap: float = GAP):
        self._api = api
        self._job = job_id
        self._beat = beat
        self._gap = gap
        self._lock = threading.Lock()
        self._at: dict | None = None
        self._moved = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def report(
        self,
        phase: str,
        current: int | None = None,
        total: int | None = None,
        note: str = "",
    ) -> None:
        """Say where the gate is. ``total`` may be omitted when it is unknown.

        Omitted rather than guessed: a stage that does not know how much work is
        left gets an indeterminate bar, and inventing a denominator to make the
        bar move is a lie the user cannot check.
        """
        with self._lock:
            self._at = {"phase": phase, "current": current, "total": total, "note": note}
        self._moved.set()

    def _send(self) -> None:
        with self._lock:
            at = dict(self._at) if self._at else None
        try:
            call(self._api, f"/api/jobs/{self._job}/heartbeat", {"progress": at} if at else {})
        except (urllib.error.URLError, OSError):
            # A missed beat is not worth killing a running job over; the next
            # one is ten seconds away and the staleness window is much wider.
            pass

    def _loop(self) -> None:
        # Beat once immediately, so a job that is claimed and then blocks for a
        # long time is alive from the moment it starts rather than ten seconds
        # in.
        self._send()
        while not self._stop.is_set():
            # Wakes on movement, or on the beat interval if nothing moved.
            self._moved.wait(timeout=self._beat)
            self._moved.clear()
            if self._stop.is_set():
                break
            self._send()
            # The floor. Anything reported during this window is not lost --
            # `report` overwrites in place, so the next send carries the latest
            # position rather than a stale one from the start of the gap.
            self._stop.wait(self._gap)

    def __enter__(self) -> "Progress":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._moved.set()  # wake the sender out of its wait so it can exit
        self._thread.join(timeout=2.0)


#: Gates that need the dataset on this machine. Intake fetches it as its first
#: act; the rest read what is already there. The robot test is on the list
#: because a GPU worker running only that gate has never run intake and would
#: otherwise find nothing.
NEEDS_DATASET = ("g0", "g1", "g2", "g3")


def once(api: str, worker: str, gates: list[str], workroot: Path | None = None) -> bool:
    got = call(api, "/api/jobs/claim", {"worker": worker, "gates": gates}).get("job")
    if not got:
        return False

    print(f"[{worker}] {got['gate_key']} for {got['submission_id']}", flush=True)

    # A fetch is not a gate: it produces the dataset the gates will read, has
    # no verdict vocabulary, and reports through its own two endpoints. hf.run
    # catches everything and phrases it for the person waiting, so the generic
    # failed-gate path below never applies to it.
    if got["gate_key"] == "hf":
        hf_work = os.environ.get("BENCH_HF_WORK", "").strip()
        root = workroot if workroot is not None else (Path(hf_work) if hf_work else Path("/tmp"))
        scratch = root / "hf" / got["id"]
        with Progress(api, got["id"]) as progress:
            outcome = hf.run(api, call, got, scratch, progress.report)
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"[{worker}]   -> {outcome}", flush=True)
        return True

    handler = HANDLERS.get(got["gate_key"])
    if handler is None:
        call(api, f"/api/jobs/{got['id']}/finish", {"status": "failed", "error": "no handler"})
        return True

    # The job names a workdir on the *API's* machine. A worker sharing that
    # disk uses it; one on a GPU box cannot, and writing there would either fail
    # or -- worse -- succeed against a same-named directory that means something
    # else entirely. So a remote worker is given its own root and keeps each
    # job's scratch under the job id.
    if workroot is not None:
        workdir = workroot / got["submission_id"]
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(got["workdir"])

    with Progress(api, got["id"]) as progress:
        try:
            archive = (
                ensure_dataset(got, workdir, downloader(api), progress.report,
                               gate=got["gate_key"])
                if got["gate_key"] in NEEDS_DATASET
                else Path(got.get("archive") or workdir / "dataset.zip")
            )
            # What the user bought, handed to the gate. A gate that ran a
            # different number of scenes than was paid for would be charging for
            # one experiment and running another.
            result = handler(archive, workdir, progress.report, got.get("params") or {})
            call(
                api,
                f"/api/jobs/{got['id']}/finish",
                {
                    "status": result["status"],
                    "verdict": {"summary": result["summary"]},
                    "findings": result.get("findings", []),
                    "detected": result.get("detected", {}),
                    "measures": result.get("measures", {}),
                    "abstained": result.get("abstained", []),
                    # The gate's own working: per-scene pairs, p-values, which
                    # arms ran. Not shown in the timeline, but it is what the
                    # verdict page is built from and what a lab asks for when
                    # it wants to check the arithmetic itself.
                    "detail": result.get("detail", {}),
                },
            )
            print(f"[{worker}]   -> {result['status']}: {result['summary']}", flush=True)
            # Advice after the verdict, never instead of it. The gate's result
            # is already stored by the time this runs, and every failure path
            # inside returns a sentence for the log rather than raising. Only
            # on real verdicts: a "failed" is our machinery, and coaching a
            # visitor about our own outage would be advice about nothing.
            if result["status"] in ("passed", "refused", "abstained"):
                print(f"[{worker}]   {coach.maybe_coach(api, got['submission_id'], call)}", flush=True)
        except Exception as error:  # noqa: BLE001 - one job, not the worker
            traceback.print_exc()
            call(
                api,
                f"/api/jobs/{got['id']}/finish",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "verdict": {
                        "summary": "the check could not complete. This is our fault, not your data's"
                    },
                },
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:7910")
    parser.add_argument("--gates", default="g0")
    parser.add_argument("--token", default="", help="shared secret; overrides BENCH_WORKER_TOKEN")
    parser.add_argument("--name", default="", help="how this worker identifies itself")
    parser.add_argument(
        "--work",
        default=None,
        help="where to keep fetched datasets. Set this on a worker that does not share a "
        "disk with the API, the job's own workdir is a path on the API's machine, and "
        "using it here would write into a directory that does not exist.",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    global TOKEN
    TOKEN = args.token or TOKEN

    worker = args.name or f"{socket.gethostname()}/{args.gates}"
    # Every worker also fetches. A fetch needs bandwidth and disk, which every
    # worker has; making it a flag would mean the one deployment without the
    # flag silently strands every pasted link in "fetching" forever.
    gates = args.gates.split(",")
    if "hf" not in gates:
        gates = gates + ["hf"]
    print(f"[{worker}] polling {args.api} for {gates}", flush=True)
    while True:
        try:
            busy = once(args.api, worker, gates, Path(args.work) if args.work else None)
        except urllib.error.URLError as error:
            print(f"[{worker}] api unreachable: {error}", flush=True)
            busy = False
        if args.once:
            return
        time.sleep(0.5 if busy else 2.0)


if __name__ == "__main__":
    main()
