"""Seeded worked examples, and the line between "everyone may read this" and
"anyone may touch this".

A first-time visitor could previously see nothing at all: submissions are scoped
to whoever uploaded them, so the only way to find out what a verdict looks like
was to upload a dataset, which is the commitment the verdict exists to help
somebody decide about. Two finished runs are now seeded as examples.

That makes them the only rows in the product that are readable by everybody, and
a world-readable row inside an authorization model built entirely on ownership
deserves tests. The design is sound: reads go through :func:`readable`, which
admits ``demo``, while every write keeps comparing ``org_id`` directly against
the caller's, and a sample's org is one no visitor is ever handed. So writes
refuse without having to know samples exist.

That is a good design and it was untested, which means nothing would notice if a
later route reached for ``readable()`` because it was there and convenient. These
pin the boundary rather than the implementation: a sample can be read, cannot be
written by any route, never appears in anybody's list, and never reaches the
leaderboard.
"""

from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="bench-demo-test-")
os.environ["BENCH_DATA"] = _TMP
os.environ["BENCH_TRUST_HEADER"] = "x-bench-edge"
os.environ["BENCH_TRUST_SECRET"] = "test-edge-secret"
os.environ["BENCH_IP_SALT"] = "test-salt"

from app import identity  # noqa: E402
from app import samples as samplesmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def as_ip(client, ip: str, method: str = "get", path: str = "/api/me", **kw):
    headers = {**kw.pop("headers", {}), identity.client_ip_header(): ip}
    if identity.trusts_edge():
        headers[identity.trust_header()] = identity.trust_secret()
    return getattr(client, method)(path, headers=headers, **kw)


def a_sample_id(client) -> str:
    ids = list(samplesmod.ids().values())
    assert ids, "no samples seeded, so none of these tests prove anything"
    return ids[0]


def test_a_stranger_can_read_a_sample(client):
    """The entire reason samples exist.

    Every real submission 404s for everyone but its owner. If that applied here
    too the feature would be invisible, and the test below about writes would
    pass for the wrong reason.
    """
    sid = a_sample_id(client)
    r = as_ip(client, "93.184.216.34", path=f"/api/submissions/{sid}")
    assert r.status_code == 200, "a seeded example is not readable, so nobody can see one"
    assert r.json()["demo"] is True


def test_two_unrelated_visitors_both_see_the_same_sample(client):
    """Readable by everybody, not merely by whoever happened to be first."""
    sid = a_sample_id(client)
    for ip in ("93.184.216.34", "198.51.100.7", "8.8.8.8"):
        assert as_ip(client, ip, path=f"/api/submissions/{sid}").status_code == 200, ip


def test_no_route_lets_a_visitor_change_a_sample(client):
    """The property that makes world-readable safe.

    Enumerated over every writing route rather than asserted about one, because
    the guard is repeated per route rather than centralised: the failure mode is
    a *new* route that reaches for ``readable()`` because it is right there and
    reads like the obvious helper.
    """
    sid = a_sample_id(client)
    writes = [
        ("post", f"/api/submissions/{sid}/listed", {"json": {"listed": True}}),
        ("post", f"/api/submissions/{sid}/meaning", {"json": {"meaning": {}}}),
        ("post", f"/api/submissions/{sid}/gates/g3/start", {"json": {}}),
        ("post", f"/api/submissions/{sid}/gates/g3/retry", {}),
        ("post", f"/api/submissions/{sid}/dataset", {"files": {"file": ("d.zip", b"PK\x03\x04", "application/zip")}}),
    ]
    for method, path, kw in writes:
        r = as_ip(client, "93.184.216.34", method=method, path=path, **kw)
        assert r.status_code >= 400, (
            f"{method.upper()} {path} returned {r.status_code}: a visitor can "
            "modify the shared worked example that everybody else is reading"
        )


def test_a_sample_is_in_nobodys_list(client):
    """Readable by everyone is not the same as owned by everyone.

    If samples appeared in the visitor's own list they would read as something
    that person uploaded, and the count of "your submissions" would be wrong for
    every visitor on their first ever page load.

    The visitor uploads something real first, deliberately. Asserting an absence
    against an empty list proves nothing: it passes whether the filter works or
    the whole endpoint is broken. With one genuine submission present, the
    assertion is that this list is working *and* excludes the sample.
    """
    sid = a_sample_id(client)
    benches = as_ip(client, "93.184.216.34", path="/api/benchmarks").json()
    key = (benches if isinstance(benches, list) else benches["benchmarks"])[0]["key"]
    made = as_ip(
        client,
        "93.184.216.34",
        method="post",
        path="/api/submissions",
        json={"name": "mine", "benchmark": key},
    )
    assert made.status_code < 300, made.text
    mine = made.json()["id"]

    listed = as_ip(client, "93.184.216.34", path="/api/submissions").json()
    ids = [s["id"] for s in (listed if isinstance(listed, list) else listed.get("submissions", []))]

    assert mine in ids, "the visitor's own submission is missing, so this list proves nothing"
    assert sid not in ids, "a seeded example is being counted as this visitor's own work"


def test_a_sample_never_reaches_the_leaderboard(client):
    """The board ranks what people chose to publish.

    A seeded example on it would be us competing in our own benchmark, which is
    the one entry nobody can audit.
    """
    sid = a_sample_id(client)
    board = as_ip(client, "93.184.216.34", path="/api/compare").json()
    rows = board if isinstance(board, list) else board.get("rows", board.get("submissions", []))
    assert all(row.get("id") != sid for row in rows), "a worked example is ranked on the shared board"


def test_a_sample_carries_no_contact_address(client):
    """``as_submission`` promised ``email`` only ever reaches the owning org.

    True until these rows became world-readable, at which point the promise
    needed enforcing at the source rather than trusting the fixture to be empty.
    """
    sid = a_sample_id(client)
    body = as_ip(client, "93.184.216.34", path=f"/api/submissions/{sid}").json()
    assert body.get("email", "") == "", "a sample is publishing an email address to everybody"
