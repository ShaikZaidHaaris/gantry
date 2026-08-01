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
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import traceback
from pathlib import Path

import urllib.error
import urllib.request
import json as jsonlib

from gates import intake, report, signal

HANDLERS = {"g0": intake.run, "g1": report.run, "g2": signal.run}

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


def call(api: str, path: str, payload: dict | None = None) -> dict:
    url = f"{api}{path}"
    data = jsonlib.dumps(payload or {}).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return jsonlib.loads(response.read() or b"{}")


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


def once(api: str, worker: str, gates: list[str]) -> bool:
    got = call(api, "/api/jobs/claim", {"worker": worker, "gates": gates}).get("job")
    if not got:
        return False

    print(f"[{worker}] {got['gate_key']} for {got['submission_id']}", flush=True)
    handler = HANDLERS.get(got["gate_key"])
    if handler is None:
        call(api, f"/api/jobs/{got['id']}/finish", {"status": "failed", "error": "no handler"})
        return True

    with Progress(api, got["id"]) as progress:
        try:
            result = handler(Path(got["archive"]), Path(got["workdir"]), progress.report)
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
                },
            )
            print(f"[{worker}]   -> {result['status']}: {result['summary']}", flush=True)
        except Exception as error:  # noqa: BLE001 - one job, not the worker
            traceback.print_exc()
            call(
                api,
                f"/api/jobs/{got['id']}/finish",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "verdict": {
                        "summary": "the check could not complete — this is our fault, not your data's"
                    },
                },
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:7910")
    parser.add_argument("--gates", default="g0")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    worker = f"{socket.gethostname()}/{args.gates}"
    gates = args.gates.split(",")
    print(f"[{worker}] polling {args.api} for {gates}", flush=True)
    while True:
        try:
            busy = once(args.api, worker, gates)
        except urllib.error.URLError as error:
            print(f"[{worker}] api unreachable: {error}", flush=True)
            busy = False
        if args.once:
            return
        time.sleep(0.5 if busy else 2.0)


if __name__ == "__main__":
    main()
