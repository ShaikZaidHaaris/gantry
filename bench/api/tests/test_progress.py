"""The live channel: progress, resume, and what happens when a worker dies.

These are the parts of the product that are hardest to eyeball, because the two
gates that exist today finish in under a second. The gates that make this
machinery matter -- ten minutes for the signal check, hours for the robot test
-- do not exist yet, so the behaviour is pinned here rather than discovered
later on a run somebody paid for.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

#: A fresh database per test session. Set before the app is imported, because
#: ``db.py`` reads it at import time to build the engine.
_TMP = tempfile.mkdtemp(prefix="bench-test-")
os.environ["BENCH_DATA"] = _TMP

from app import db as dbmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def job(client):
    """A submission with a claimed g0 job, without going near a real archive."""
    with dbmod.SessionLocal() as session:
        benchmark = session.scalars(dbmod.select(dbmod.Benchmark)).first()
        key = benchmark.key
    sub = client.post("/api/submissions", json={"name": "t", "benchmark": key}).json()
    with dbmod.SessionLocal() as session:
        session.add(
            dbmod.DatasetVersion(
                id=dbmod.new_id("dsv"), submission_id=sub["id"], version=1, path="/none", bytes=0
            )
        )
        session.add(
            dbmod.Job(id=dbmod.new_id("job"), submission_id=sub["id"], gate_key="g0", status="queued")
        )
        session.commit()
    claimed = client.post("/api/jobs/claim", json={"worker": "w", "gates": ["g0"]}).json()["job"]
    return sub["id"], claimed["id"]


def gate_of(client, sub_id, key="g0"):
    body = client.get(f"/api/submissions/{sub_id}").json()
    return next(g for g in body["gates"] if g["key"] == key)


# -- progress ---------------------------------------------------------------


def test_progress_is_stored_and_visible_while_running(client, job):
    sub_id, job_id = job
    client.post(
        f"/api/jobs/{job_id}/heartbeat",
        json={"progress": {"phase": "training", "current": 1400, "total": 3000, "note": "lora"}},
    )
    gate = gate_of(client, sub_id)
    assert gate["progress"] == {
        "phase": "training",
        "current": 1400,
        "total": 3000,
        "note": "lora",
    }


def test_progress_never_touches_the_event_log(client, job):
    """A 3,000-step run must not put 3,000 rows in a log meant to be read."""
    sub_id, job_id = job
    before = len(client.get(f"/api/submissions/{sub_id}").json()["events"])
    for step in range(50):
        client.post(
            f"/api/jobs/{job_id}/heartbeat",
            json={"progress": {"phase": "training", "current": step, "total": 3000}},
        )
    after = client.get(f"/api/submissions/{sub_id}").json()["events"]
    assert len(after) == before
    # ...and the position is still current, so nothing was lost by not logging.
    assert gate_of(client, sub_id)["progress"]["current"] == 49


def test_a_finished_gate_reports_no_position(client, job):
    """"step 2,999 of 3,000" beside a green tick reads as a run that stopped short."""
    sub_id, job_id = job
    client.post(
        f"/api/jobs/{job_id}/heartbeat",
        json={"progress": {"phase": "evaluating", "current": 49, "total": 50}},
    )
    client.post(f"/api/jobs/{job_id}/finish", json={"status": "passed", "verdict": {"summary": "ok"}})
    assert gate_of(client, sub_id)["progress"] == {}


def test_a_stage_that_cannot_count_says_so(client, job):
    """No total rather than a made-up one: the bar goes indeterminate, not to 0%."""
    sub_id, job_id = job
    client.post(f"/api/jobs/{job_id}/heartbeat", json={"progress": {"phase": "unpacking"}})
    assert gate_of(client, sub_id)["progress"] == {"phase": "unpacking"}


def test_unknown_progress_fields_are_dropped(client, job):
    """The progress channel is not a side door for arbitrary gate state."""
    sub_id, job_id = job
    client.post(
        f"/api/jobs/{job_id}/heartbeat",
        json={"progress": {"phase": "x", "secret": "smuggled", "loss": 0.3}},
    )
    assert gate_of(client, sub_id)["progress"] == {"phase": "x"}


def test_an_empty_note_is_absent_not_blank(client, job):
    """A note of "" is not a note; sending it makes the UI render nothing,
    visibly."""
    sub_id, job_id = job
    client.post(
        f"/api/jobs/{job_id}/heartbeat", json={"progress": {"phase": "unpacking", "note": ""}}
    )
    assert gate_of(client, sub_id)["progress"] == {"phase": "unpacking"}


# -- a worker that dies -----------------------------------------------------


def test_a_dead_worker_becomes_our_failure_not_a_refusal(client, job):
    """A gate stuck on "running" forever looks like patience is the answer."""
    sub_id, job_id = job
    with dbmod.SessionLocal() as session:
        row = session.get(dbmod.Job, job_id)
        row.heartbeat_at = "2020-01-01T00:00:00+00:00"
        session.commit()

    gate = gate_of(client, sub_id)
    assert gate["status"] == "failed"
    assert "our fault" in gate["verdict"]["summary"]
    # Never "refused" — that vocabulary is a judgement on somebody's data, and
    # this was our machine.
    assert client.get(f"/api/submissions/{sub_id}").json()["status"] == "failed"


def test_a_busy_worker_is_not_swept(client, job):
    """Long gates block for hours; the beat runs on its own thread and continues."""
    sub_id, job_id = job
    client.post(f"/api/jobs/{job_id}/heartbeat", json={"progress": {"phase": "training"}})
    assert gate_of(client, sub_id)["status"] == "running"


# -- the stream -------------------------------------------------------------


@pytest.fixture(autouse=True)
def brief_stream(monkeypatch):
    """Make the stream close itself after one idle poll.

    In production it holds the connection for fifteen minutes and the browser
    reconnects; here it has to end so a test can read the body. Bounded by
    turning the two tuning constants down rather than by adding a test-only
    parameter to the endpoint, which would be product surface that exists only
    for tests.
    """
    from app import main as mainmod

    monkeypatch.setattr(mainmod, "POLL", 0.01)
    monkeypatch.setattr(mainmod, "IDLE_LIMIT", 1)


def read_frames(client, sub_id, headers=None):
    """Everything the stream has to say before it goes quiet."""
    response = client.get(f"/api/submissions/{sub_id}/events", headers=headers or {})
    assert response.status_code == 200
    return response.text


def test_frames_carry_an_id_so_a_reconnect_can_resume(client, job):
    """Without `id:`, EventSource's own resume does nothing and a reload during
    a four-hour run silently loses everything that happened while away."""
    sub_id, _ = job
    text = read_frames(client, sub_id)
    assert "id: " in text
    assert "data: " in text


def test_last_event_id_replays_only_what_was_missed(client, job):
    sub_id, _ = job
    everything = read_frames(client, sub_id)
    ids = [int(line.split(": ", 1)[1]) for line in everything.splitlines() if line.startswith("id: ")]
    assert ids, "the fixture should have produced at least one event"

    later = read_frames(client, sub_id, headers={"Last-Event-ID": str(max(ids))})
    replayed = [int(l.split(": ", 1)[1]) for l in later.splitlines() if l.startswith("id: ")]
    assert all(i > max(ids) for i in replayed)


def test_the_running_position_arrives_as_its_own_frame(client, job):
    sub_id, job_id = job
    client.post(
        f"/api/jobs/{job_id}/heartbeat",
        json={"progress": {"phase": "evaluating", "current": 12, "total": 50}},
    )
    text = read_frames(client, sub_id, headers={"Last-Event-ID": "999999"})
    assert "event: progress" in text
    body = [l for l in text.splitlines() if l.startswith("data: ")][0]
    payload = json.loads(body.removeprefix("data: "))
    assert payload == {"gate": "g0", "phase": "evaluating", "current": 12, "total": 50}
    # No id on a progress frame: nobody wants a replay of a progress bar, and
    # advancing the resume cursor on one would skip durable events.
    assert "id: " not in text
