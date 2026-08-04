"""The two doors the advice layer uses, and who may open them.

The advice itself is generated on the worker; these are the API's halves: a
worker-authed read of the record and a worker-authed write of the points. The
property that matters is directional: visitors read advice and never write it,
workers write it and no cookie is involved. A visitor who could write this
field could put words on somebody's report page under our label.
"""

from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="bench-coach-test-")
os.environ["BENCH_DATA"] = _TMP
os.environ["BENCH_TRUST_HEADER"] = "x-bench-edge"
os.environ["BENCH_TRUST_SECRET"] = "test-edge-secret"
os.environ["BENCH_IP_SALT"] = "test-salt"

from app import identity  # noqa: E402
from app import main as mainmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TOKEN = "test-worker-token"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(mainmod, "WORKER_TOKEN", TOKEN)
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def as_ip(client, ip: str, method: str = "get", path: str = "/api/me", **kw):
    headers = {**kw.pop("headers", {}), identity.client_ip_header(): ip}
    if identity.trusts_edge():
        headers[identity.trust_header()] = identity.trust_secret()
    return getattr(client, method)(path, headers=headers, **kw)


def a_submission(client, ip: str = "93.184.216.34") -> str:
    benches = as_ip(client, ip, "get", "/api/benchmarks").json()
    key = (benches if isinstance(benches, list) else benches["benchmarks"])[0]["key"]
    made = as_ip(client, ip, "post", "/api/submissions", json={"name": "c", "benchmark": key})
    assert made.status_code < 300, made.text
    return made.json()["id"]


def test_a_visitor_cannot_write_advice(client):
    """The property the whole file exists for.

    The card is labelled as ours. A visitor who could write it would be
    publishing under that label onto somebody's report.
    """
    sub = a_submission(client)
    r = as_ip(client, "93.184.216.34", "post", f"/api/submissions/{sub}/coach",
              json={"points": ["planted advice"]})
    assert r.status_code == 401


def test_a_visitor_cannot_use_the_worker_read_either(client):
    sub = a_submission(client)
    r = as_ip(client, "8.8.8.8", "post", f"/api/submissions/{sub}/for-worker", json={})
    assert r.status_code == 401, (
        "the worker read skips ownership on purpose; without the token gate it "
        "would let any visitor read any submission by id"
    )


def test_the_worker_writes_and_the_owner_reads_it_back(client):
    sub = a_submission(client)
    w = client.post(f"/api/submissions/{sub}/coach",
                    headers={"X-Worker-Token": TOKEN},
                    json={"points": ["add 30 demos of the lift", "keep both hands in frame"],
                          "model": "gpt-5-nano"})
    assert w.status_code == 200, w.text

    body = as_ip(client, "93.184.216.34", path=f"/api/submissions/{sub}").json()
    assert body["coach"]["points"] == ["add 30 demos of the lift", "keep both hands in frame"]
    assert body["coach"]["model"] == "gpt-5-nano"


def test_four_is_enforced_at_the_door_too(client):
    """The generator slices to four, but this is the door and the generator is
    merely one caller of it."""
    sub = a_submission(client)
    w = client.post(f"/api/submissions/{sub}/coach",
                    headers={"X-Worker-Token": TOKEN},
                    json={"points": ["1", "2", "3", "4", "5", "6"]})
    assert w.status_code == 200
    body = as_ip(client, "93.184.216.34", path=f"/api/submissions/{sub}").json()
    assert len(body["coach"]["points"]) == 4


def test_nothing_is_not_an_update(client):
    sub = a_submission(client)
    w = client.post(f"/api/submissions/{sub}/coach",
                    headers={"X-Worker-Token": TOKEN}, json={"points": ["  ", ""]})
    assert w.status_code == 422


def test_the_worker_read_returns_the_full_record(client):
    sub = a_submission(client)
    r = client.post(f"/api/submissions/{sub}/for-worker",
                    headers={"X-Worker-Token": TOKEN}, json={})
    assert r.status_code == 200
    assert r.json()["id"] == sub
    assert "gates" in r.json()
