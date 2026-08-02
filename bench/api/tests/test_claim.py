"""One queued job, two workers, and only one of them may have it.

The claim used to be a plain SELECT-then-mutate under a docstring promising a
guarded UPDATE. Two workers polling the same queue -- which is the normal state
of this system the moment a second GPU box is attached, and which also happens
by accident whenever a stale worker is left running -- could both read the same
``queued`` row and both be told they had won it.

The damage is not a job run twice. It is that the loser writes its result onto a
gate the winner already owns: the second ``finish`` overwrites the first, so a
gate can end up ``failed`` carrying "the check could not complete. This is our
fault, not your data's" for a reason that appears nowhere in its own record.
That was found the hard way, from a stale worker on an interpreter missing a
plugin the branch had added.

The fix restricts the UPDATE to rows still ``queued`` and reads ``rowcount`` to
find out whether this caller was the one that changed it. These tests hold the
guard in place, because the failure it prevents is invisible in single-worker
use and every test before this one used a single worker.
"""

from __future__ import annotations

import os
import threading
import tempfile

import pytest

#: A fresh database per test session, set before the app is imported -- ``db.py``
#: reads it at import time to build the engine.
_TMP = tempfile.mkdtemp(prefix="bench-claim-test-")
os.environ["BENCH_DATA"] = _TMP

from app import db as dbmod  # noqa: E402
from app import main as mainmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def queued_submission(client, gates=("g0",)):
    """A submission with real gate rows and one queued job per named gate."""
    with dbmod.SessionLocal() as session:
        key = session.scalars(dbmod.select(dbmod.Benchmark)).first().key
    sub = client.post("/api/submissions", json={"name": "race", "benchmark": key}).json()
    with dbmod.SessionLocal() as session:
        session.add(
            dbmod.DatasetVersion(
                id=dbmod.new_id("dsv"), submission_id=sub["id"], version=1, path="/none", bytes=0
            )
        )
        for spec in mainmod.GATES:
            session.add(
                dbmod.Gate(
                    id=dbmod.new_id("gate"), submission_id=sub["id"], key=spec["key"],
                    status="queued", version=1,
                )
            )
        for gate in gates:
            session.add(
                dbmod.Job(
                    id=dbmod.new_id("job"), submission_id=sub["id"], gate_key=gate,
                    status="queued", version=1,
                )
            )
        session.commit()
    return sub["id"]


def claim(client, worker, gates=("g0",)):
    return client.post(
        "/api/jobs/claim", json={"worker": worker, "gates": list(gates)}
    ).json()["job"]


def test_concurrent_workers_never_both_win_the_same_job(client):
    """The defect itself, which only appears when the two claims *overlap*.

    Two sequential requests never race: the first commits ``running`` and the
    second's SELECT no longer matches, which is why every test written before
    this one passed against the broken code. The bug needs both callers to read
    the row before either writes -- the normal state of two workers polling one
    queue, and what happens by accident whenever a stale worker is left running.

    So the requests are fired from threads held at a barrier and released
    together. This is a race test: it cannot prove the absence of a race, and
    against the unguarded code it fails on most runs rather than all of them.
    Repeating over several fresh jobs is what makes that "most" close enough to
    "always" to be a useful regression test. With the guard it passes every
    time, because the database decides the winner rather than the interleaving.
    """
    rounds, workers = 12, 4
    for _ in range(rounds):
        queued_submission(client)
        ready = threading.Barrier(workers)
        won: list[dict] = []
        lock = threading.Lock()

        def grab(name: str) -> None:
            ready.wait(timeout=5)
            got = claim(client, name)
            if got is not None:
                with lock:
                    won.append(got)

        threads = [threading.Thread(target=grab, args=(f"worker-{i}",)) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        ids = [g["id"] for g in won]
        assert len(ids) == len(set(ids)), (
            f"job {ids} was handed to more than one worker; the loser will overwrite "
            "the winner's result and the gate will report a failure with no cause "
            "anywhere in its own record"
        )


def test_the_endpoint_refuses_a_job_another_worker_already_holds(client):
    """The guard reached through the route, with the row taken out from under it.

    Claiming, then putting the row back to ``queued`` only in this session's
    view, is not something a caller can do -- but flipping the stored row to
    ``running`` between a worker's read and its write is exactly what the other
    worker does. The endpoint must come back empty rather than hand out a job
    it did not win.
    """
    queued_submission(client)
    first = claim(client, "worker-a")
    assert first is not None

    # The row is now `running`. A second poll must not resurrect it, whatever
    # order the workers arrive in.
    assert claim(client, "worker-b") is None
    assert claim(client, "worker-a") is None, "not even to the worker that holds it"

    with dbmod.SessionLocal() as session:
        row = session.get(dbmod.Job, first["id"])
        assert row.status == "running"
        assert row.claimed_by == "worker-a"
        assert row.attempts == 1, "one claim is one attempt"


def test_a_second_worker_still_gets_a_different_queued_job(client):
    """The guard must not serialise the queue into a single-worker system.

    Refusing the contended row is correct; refusing an idle worker any work at
    all would be a fix that costs the throughput the second box was added for.
    """
    queued_submission(client, gates=("g0", "g1"))
    first = claim(client, "worker-a", gates=("g0", "g1"))
    second = claim(client, "worker-b", gates=("g0", "g1"))

    assert first is not None and second is not None
    assert first["id"] != second["id"], "two queued jobs, two workers, two claims"
    assert {first["gate_key"], second["gate_key"]} == {"g0", "g1"}


def test_an_empty_queue_is_not_an_error(client):
    """No work is a normal answer, and must stay distinguishable from losing a race.

    Both return ``{"job": None}`` on purpose: the worker's behaviour is the same
    either way -- poll again -- and inventing a second shape would be a branch
    in every worker for a case it cannot act on differently.
    """
    assert claim(client, "worker-a") is None
