"""The visit log: every human-shaped request, written down and kept.

The launch-day question was "who came, and what did they do" and the only
answer lived in journald, which rotates. These tests hold the durable answer
in place, and hold its two restraints: worker machinery is never logged (the
claim loop would bury a month of visitors in a day), and logging failure never
takes a page down -- the log is a record of the product working, not a
component the product needs.
"""

from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="bench-visits-test-")
os.environ["BENCH_DATA"] = _TMP

from app import db as dbmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def visits(path=None):
    with dbmod.SessionLocal() as session:
        rows = session.scalars(
            dbmod.select(dbmod.Visit).order_by(dbmod.Visit.id)
        ).all()
        return [r for r in rows if path is None or r.path.startswith(path)]


def test_a_page_view_is_written_down(client):
    before = len(visits("/api/me"))
    got = client.get("/api/me", headers={
        "referer": "https://www.reddit.com/r/robotics/comments/abc/",
        "user-agent": "Mozilla/5.0 (test)",
    })
    assert got.status_code == 200
    rows = visits("/api/me")
    assert len(rows) == before + 1
    row = rows[-1]
    assert row.method == "GET" and row.status == 200
    assert row.referer.startswith("https://www.reddit.com/")
    assert "Mozilla" in row.ua
    assert row.ts  # when, or the whole table answers nothing


def test_the_visitor_is_attributed_once_they_have_a_cookie(client):
    me = client.get("/api/me").json()
    row = visits("/api/me")[-1]
    # The first-ever request had no cookie to attribute; the next one does.
    second = client.get("/api/me")
    assert second.status_code == 200
    attributed = visits("/api/me")[-1]
    assert attributed.visitor == me["org"]["id"]


def test_worker_machinery_is_not_a_visit(client):
    before = len(visits())
    client.post("/api/jobs/claim", json={"worker": "t", "gates": ["g0"]})
    client.post("/api/jobs/claim", json={"worker": "t", "gates": ["g0"]})
    assert len(visits()) == before


def test_the_query_string_travels_with_the_path(client):
    client.get("/api/compare?benchmark=pick_dual_bottles&rung=solved")
    row = visits("/api/compare")[-1]
    assert "benchmark=pick_dual_bottles" in row.path


def test_an_error_is_still_a_visit(client):
    got = client.get("/api/submissions/sub_does_not_exist")
    assert got.status_code == 404
    row = visits("/api/submissions/sub_does_not_exist")[-1]
    assert row.status == 404
