"""The worker: claim a job, run its gate, report the verdict.

Deliberately dumb. It knows how to talk to the API and which function handles
which gate; everything else is the gate's business. A gate that raises is
reported as ``failed`` -- our machinery broke -- which the product presents
differently from ``refused``, where the user's data genuinely did not pass.
Conflating those two would blame a user for our bug.
"""

from __future__ import annotations

import argparse
import socket
import time
import traceback
from pathlib import Path

import urllib.error
import urllib.request
import json as jsonlib

from gates import intake

HANDLERS = {"g0": intake.run}


def call(api: str, path: str, payload: dict | None = None) -> dict:
    url = f"{api}{path}"
    data = jsonlib.dumps(payload or {}).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return jsonlib.loads(response.read() or b"{}")


def once(api: str, worker: str, gates: list[str]) -> bool:
    got = call(api, "/api/jobs/claim", {"worker": worker, "gates": gates}).get("job")
    if not got:
        return False

    print(f"[{worker}] {got['gate_key']} for {got['submission_id']}", flush=True)
    handler = HANDLERS.get(got["gate_key"])
    if handler is None:
        call(api, f"/api/jobs/{got['id']}/finish", {"status": "failed", "error": "no handler"})
        return True

    try:
        result = handler(Path(got["archive"]), Path(got["workdir"]))
        call(
            api,
            f"/api/jobs/{got['id']}/finish",
            {
                "status": result["status"],
                "verdict": {"summary": result["summary"]},
                "findings": result.get("findings", []),
                "detected": result.get("detected", {}),
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
                "verdict": {"summary": "the check could not complete — this is our fault, not your data's"},
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
