"""The Hugging Face fetch: paste a link, the worker pulls it, gates proceed.

This path exists because launch day proved the upload wall: hundreds of
visitors read the worked example, and not one had a LeRobot archive on the
machine in front of them. Their datasets live on the Hub, so the product
fetches from the Hub.

What these tests hold in place:

  * the repo id is an allow-list, not a parser -- every shape a person pastes
    reduces to ``owner/name`` or is refused, and nothing URL-like ever
    survives to the worker;
  * a fetch job has no gate row, and every endpoint it touches must survive
    that -- the claim, the heartbeat and the generic finish path all used to
    assume a gate exists;
  * registration is the same act as an upload: same versioning, same gates,
    same g0 queue, so nothing downstream can tell how the bytes arrived.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import update

_TMP = tempfile.mkdtemp(prefix="bench-hf-test-")
os.environ["BENCH_DATA"] = _TMP

from app import db as dbmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def draft(client, name="hub-test"):
    return client.post("/api/submissions", json={"name": name}).json()


def hf_job(sub_id):
    with dbmod.SessionLocal() as session:
        return session.scalar(
            dbmod.select(dbmod.Job).where(
                dbmod.Job.submission_id == sub_id, dbmod.Job.gate_key == "hf"
            )
        )


def drain():
    """Fail every leftover queued job, so a claim in this test gets this test's.

    The suite shares one database and the claim hands out the oldest queued
    row -- without this, a flow test claims whatever an earlier test left
    behind, and fails only when the whole file runs.
    """
    with dbmod.SessionLocal() as session:
        session.execute(
            update(dbmod.Job)
            .where(dbmod.Job.status.in_(("queued", "running")))
            .values(status="failed")
        )
        session.commit()


# ------------------------------------------------------------ validation ---


def test_every_shape_a_person_pastes_reduces_to_the_id(client):
    sub = draft(client)
    for pasted in (
        "lerobot/pusht",
        "https://huggingface.co/datasets/lerobot/pusht",
        "http://www.huggingface.co/datasets/lerobot/pusht/",
        "hf.co/datasets/lerobot/pusht",
        "https://huggingface.co/datasets/lerobot/pusht/tree/main",
        "https://huggingface.co/datasets/lerobot/pusht?p=1#files",
    ):
        got = client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": pasted})
        assert got.status_code == 200, (pasted, got.text)
        job = hf_job(sub["id"])
        assert json.loads(job.params_json)["repo"] == "lerobot/pusht", pasted
        # reset for the next shape: one fetch at a time is its own rule
        with dbmod.SessionLocal() as session:
            session.execute(update(dbmod.Job).values(status="failed"))
            session.commit()


def test_what_is_not_a_dataset_id_is_refused(client):
    sub = draft(client)
    for pasted in (
        "",
        "pusht",
        "https://example.com/datasets/lerobot/pusht",
        "lerobot/pusht; rm -rf /",
        "https://huggingface.co/lerobot",
        "../../../etc/passwd",
        "a b/c d",
    ):
        got = client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": pasted})
        assert got.status_code == 422, (pasted, got.status_code, got.text)
    assert hf_job(sub["id"]) is None


def test_one_fetch_at_a_time(client):
    sub = draft(client)
    assert client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "a/b"}).status_code == 200
    second = client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "a/b"})
    assert second.status_code == 409


# ------------------------------------------------------------- the flow ---


def test_enqueue_marks_the_submission_fetching(client):
    sub = draft(client)
    got = client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "lerobot/pusht"}).json()
    assert got["status"] == "fetching"
    kinds = [e["kind"] for e in got["events"]]
    assert "fetch.queued" in kinds


def test_claim_survives_a_job_with_no_gate(client):
    drain()
    sub = draft(client)
    client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "lerobot/pusht"})
    got = client.post("/api/jobs/claim", json={"worker": "t", "gates": ["hf"]}).json()["job"]
    assert got is not None and got["gate_key"] == "hf"
    assert got["params"]["repo"] == "lerobot/pusht"
    # no dataset yet, and the claim says so instead of crashing
    assert got["archive"] is None and got["version"] == 0
    fresh = client.get(f"/api/submissions/{sub['id']}").json()
    assert fresh["status"] == "fetching"  # not "running": there is no gate to run


def test_fetched_registers_a_version_like_an_upload_would(client):
    drain()
    sub = draft(client)
    client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "lerobot/pusht"})
    job = client.post("/api/jobs/claim", json={"worker": "t", "gates": ["hf"]}).json()["job"]

    built = Path(_TMP) / "fetched.zip"
    built.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # an empty but real zip
    done = client.post(
        f"/api/jobs/{job['id']}/fetched",
        json={"path": str(built), "bytes": built.stat().st_size, "repo": "lerobot/pusht"},
    )
    assert done.status_code == 200 and done.json()["version"] == 1

    fresh = client.get(f"/api/submissions/{sub['id']}").json()
    assert fresh["status"] == "queued"
    assert fresh["dataset"]["version"] == 1
    assert [g["status"] for g in fresh["gates"]] == ["queued"] * 4
    kinds = [e["kind"] for e in fresh["events"]]
    assert "dataset.fetched" in kinds and "gate.queued" in kinds
    # the archive was moved into the submission's own folder, not referenced
    # wherever the worker happened to build it
    assert not built.exists()
    stored = Path(dbmod.STORAGE) / sub["id"] / "v1" / "dataset.zip"
    assert stored.exists()

    # and intake is claimable exactly as if this had been an upload
    g0 = client.post("/api/jobs/claim", json={"worker": "t", "gates": ["g0"]}).json()["job"]
    assert g0 is not None and g0["archive"] == str(stored)


def test_fetch_failed_returns_the_submission_to_draft(client):
    drain()
    sub = draft(client)
    client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "a/b"})
    job = client.post("/api/jobs/claim", json={"worker": "t", "gates": ["hf"]}).json()["job"]
    said = "there is no public dataset called a/b on Hugging Face"
    client.post(f"/api/jobs/{job['id']}/fetch-failed", json={"reason": said})

    fresh = client.get(f"/api/submissions/{sub['id']}").json()
    assert fresh["status"] == "draft"
    failure = [e for e in fresh["events"] if e["kind"] == "fetch.failed"]
    assert failure and failure[-1]["reason"] == said
    # and the visitor may immediately try again
    assert client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "a/c"}).status_code == 200


def test_generic_finish_on_a_fetch_job_does_not_crash_or_lie(client):
    drain()
    """The worker's catch-all failure path calls finish; a fetch has no gate."""
    sub = draft(client)
    client.post(f"/api/submissions/{sub['id']}/dataset/hf", json={"repo": "a/b"})
    job = client.post("/api/jobs/claim", json={"worker": "t", "gates": ["hf"]}).json()["job"]
    done = client.post(
        f"/api/jobs/{job['id']}/finish",
        json={"status": "failed", "error": "boom", "verdict": {"summary": "the fetch broke"}},
    )
    assert done.status_code == 200
    fresh = client.get(f"/api/submissions/{sub['id']}").json()
    assert fresh["status"] == "draft"
    assert any(e["kind"] == "fetch.failed" for e in fresh["events"])
