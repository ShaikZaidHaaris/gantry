"""Identity is the access control here, so these are authorization tests.

Every submission is scoped to an org and the org is now derived from the
caller's address. That makes the address-resolution code a permission check
wearing a networking hat: get it wrong and the failure is not a wrong label on
a page, it is one visitor reading another's uploads.

Two failure directions, and the file covers both because they are cured by
opposite mistakes:

  * too trusting -- believing a forwarding header from anyone, which lets a
    caller pick their own identity and read whatever that identity owns;
  * too partitioning -- failing to resolve an address and silently putting
    everybody in one org, which is the quiet version of no isolation at all.

The leaderboard tests are here rather than beside the compare arithmetic for
the same reason: publishing is a permission, and the default has to be off.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="bench-identity-test-")
os.environ["BENCH_DATA"] = _TMP

#: The edge, configured for the whole file. Without it the module is in direct
#: mode, forwarding headers are ignored, and every TestClient request resolves
#: to the same peer -- which puts the suite in one org and makes the isolation
#: tests pass for the wrong reason. Configuring it is what makes them able to
#: fail.
#:
#: This used to have to run before ``app.identity`` was imported, and that
#: ordering is exactly why the module now reads its configuration per call
#: instead: when another test file imported the app first, these tests silently
#: ran with no isolation at all and two of them failed for a reason that had
#: nothing to do with the code under test.
os.environ["BENCH_TRUST_HEADER"] = "x-bench-edge"
os.environ["BENCH_TRUST_SECRET"] = "test-edge-secret"
os.environ["BENCH_IP_SALT"] = "test-salt"

from app import db as dbmod  # noqa: E402
from app import identity  # noqa: E402
from app import main as mainmod  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def as_ip(client, ip: str, method: str = "get", path: str = "/api/me", **kw):
    """A request that arrives as though from ``ip``.

    TestClient's own peer is 'testclient', so the tests drive the *edge* path:
    they present the shared secret and the client-IP header, which is exactly
    what Cloudflare does and exactly what an attacker must not be able to fake.
    """
    headers = {**kw.pop("headers", {}), identity.client_ip_header(): ip}
    if identity.trusts_edge():
        headers[identity.trust_header()] = identity.trust_secret()
    return getattr(client, method)(path, headers=headers, **kw)


# -- resolving the address --------------------------------------------------


def test_a_forwarding_header_is_ignored_without_proof_of_our_edge(monkeypatch):
    """The bypass this whole design exists to prevent.

    An origin reachable directly -- which on AWS it usually is, whatever the DNS
    says -- must not let a caller name themselves. Without the shared secret
    configured, the header is not evidence of anything and is dropped.
    """
    monkeypatch.delenv("BENCH_TRUST_HEADER", raising=False)
    monkeypatch.delenv("BENCH_TRUST_SECRET", raising=False)

    class Req:
        headers = {identity.client_ip_header(): "203.0.113.9", "x-forwarded-for": "203.0.113.9"}
        client = type("C", (), {"host": "10.0.0.5"})()

    assert identity.client_ip(Req()) == "10.0.0.5", (
        "an unproven forwarding header was believed; anyone able to reach the "
        "origin could then choose whose submissions they see"
    )


def test_a_forwarding_header_is_read_when_the_edge_proves_itself(monkeypatch):
    monkeypatch.setenv("BENCH_TRUST_HEADER", "x-bench-edge")
    monkeypatch.setenv("BENCH_TRUST_SECRET", "s3cret")

    class Req:
        headers = {"x-bench-edge": "s3cret", identity.client_ip_header(): "203.0.113.9"}
        client = type("C", (), {"host": "10.0.0.5"})()

    assert identity.client_ip(Req()) == "203.0.113.9"


def test_a_wrong_secret_is_not_a_near_miss(monkeypatch):
    """No partial credit: a bad secret falls all the way back to the peer."""
    monkeypatch.setenv("BENCH_TRUST_HEADER", "x-bench-edge")
    monkeypatch.setenv("BENCH_TRUST_SECRET", "s3cret")

    class Req:
        headers = {"x-bench-edge": "wrong", identity.client_ip_header(): "203.0.113.9"}
        client = type("C", (), {"host": "10.0.0.5"})()

    assert identity.client_ip(Req()) == "10.0.0.5"


def test_private_hops_in_a_forwarded_chain_are_skipped(monkeypatch):
    """A client behind its own NAT may have prepended 10.x.

    Taking the left-most entry blindly would put every such visitor in one org,
    which is the quiet way to lose isolation while looking like you have it.
    """
    monkeypatch.setenv("BENCH_TRUST_HEADER", "x-bench-edge")
    monkeypatch.setenv("BENCH_TRUST_SECRET", "s3cret")

    # A routable address on purpose. The 203.0.113/198.51.100 ranges the rest
    # of this file uses are RFC 5737 documentation blocks, which Python's
    # ``ipaddress`` reports as private -- so they are skipped here exactly like
    # a NAT hop would be. That only matters for the chain path, which is the
    # one that filters; the single-value header path takes whatever the edge
    # states, because there the edge is the authority.
    class Req:
        headers = {
            "x-bench-edge": "s3cret",
            "x-forwarded-for": "10.1.2.3, 93.184.216.34, 172.16.0.1",
        }
        client = type("C", (), {"host": "10.0.0.5"})()

    assert identity.client_ip(Req()) == "93.184.216.34"


def test_the_chain_resolver_believes_whatever_is_left_most(monkeypatch):
    """Documenting a sharp edge, because a deployment choice depends on it.

    Left-most is the correct read behind Cloudflare, which owns the header and
    rewrites it. It is the wrong read behind a proxy that *appends*, which is
    the default for nginx and for Caddy: a client that sends its own
    ``X-Forwarded-For`` gets the real address appended after it, so the value
    the resolver picks is the one the client wrote.

    This test asserts that hazard rather than a fix, because the fix does not
    belong here. A resolver cannot tell a forged hop from a real one by looking
    at the chain; only the proxy knows which entries it added. So the proxy is
    made to overwrite instead of append, and `deploy/Caddyfile.template` does
    exactly that. If this test ever starts failing, that config assumption has
    moved and the Caddyfile needs rereading.
    """
    monkeypatch.setenv("BENCH_TRUST_HEADER", "x-bench-edge")
    monkeypatch.setenv("BENCH_TRUST_SECRET", "s3cret")
    monkeypatch.setenv("BENCH_CLIENT_IP", "cf-connecting-ip")  # absent, so the chain is used

    class Req:
        # What an appending proxy produces when the client prepends a lie:
        # their chosen value first, their real address second.
        headers = {"x-bench-edge": "s3cret", "x-forwarded-for": "8.8.8.8, 93.184.216.34"}
        client = type("C", (), {"host": "10.0.0.5"})()

    assert identity.client_ip(Req()) == "8.8.8.8"


def test_a_proxy_set_single_value_header_beats_a_forged_chain(monkeypatch):
    """The configuration that makes a plain reverse proxy safe.

    Caddy writes the TCP peer into a single-value header on every request,
    overwriting anything that arrived, and ``BENCH_CLIENT_IP`` points at that
    header. The chain is then never consulted, so the previous test's hazard is
    unreachable: a client can still send an ``X-Forwarded-For`` full of lies and
    it changes nothing.

    Worth stating as a test rather than a comment in a Caddyfile, because the
    two files are edited years apart by different people.
    """
    monkeypatch.setenv("BENCH_TRUST_HEADER", "x-bench-edge")
    monkeypatch.setenv("BENCH_TRUST_SECRET", "s3cret")
    monkeypatch.setenv("BENCH_CLIENT_IP", "x-real-ip")

    class Req:
        headers = {
            "x-bench-edge": "s3cret",
            "x-real-ip": "93.184.216.34",  # written by the proxy from {remote_host}
            "x-forwarded-for": "8.8.8.8",  # written by the client, hopefully
        }
        client = type("C", (), {"host": "10.0.0.5"})()

    assert identity.client_ip(Req()) == "93.184.216.34", (
        "the forged chain won over the proxy's own header, so a visitor can "
        "choose whose submissions they read"
    )


def test_a_tunnel_only_origin_trusts_the_header_from_loopback(monkeypatch):
    """The quick-tunnel deployment, which cannot attach a secret.

    ``cloudflared tunnel --url http://127.0.0.1:8090`` has no zone, so there is
    no Transform Rule to add a shared secret with. Without a second way to earn
    trust, that whole deployment silently runs with every visitor in one org --
    which is the configuration this product is actually deployed in today.

    The argument here is different: uvicorn binds to loopback and the connector
    dials it locally, so nothing outside the host can open a socket to the
    origin at all. The header cannot be forged from outside because there is no
    outside that can reach it.
    """
    monkeypatch.delenv("BENCH_TRUST_HEADER", raising=False)
    monkeypatch.delenv("BENCH_TRUST_SECRET", raising=False)
    monkeypatch.setenv("BENCH_TRUST_TUNNEL", "1")

    class Req:
        headers = {identity.client_ip_header(): "203.0.113.9"}
        client = type("C", (), {"host": "127.0.0.1"})()

    assert identity.client_ip(Req()) == "203.0.113.9"
    assert identity.describe()["mode"] == "tunnel"


def test_tunnel_mode_still_refuses_a_header_from_anywhere_but_loopback(monkeypatch):
    """The flag is a claim about the binding, so the binding is checked anyway.

    If the operator sets this and then binds to 0.0.0.0, a request arriving over
    the network is not covered by the argument that justified the flag -- so it
    is not trusted, and the peer is used instead.
    """
    monkeypatch.delenv("BENCH_TRUST_HEADER", raising=False)
    monkeypatch.delenv("BENCH_TRUST_SECRET", raising=False)
    monkeypatch.setenv("BENCH_TRUST_TUNNEL", "1")

    class Req:
        headers = {identity.client_ip_header(): "203.0.113.9"}
        client = type("C", (), {"host": "10.0.0.9"})()

    assert identity.client_ip(Req()) == "10.0.0.9", (
        "a networked peer was trusted on the strength of a flag that only holds "
        "for a loopback connector"
    )


def test_tunnel_mode_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("BENCH_TRUST_HEADER", raising=False)
    monkeypatch.delenv("BENCH_TRUST_SECRET", raising=False)
    monkeypatch.delenv("BENCH_TRUST_TUNNEL", raising=False)

    class Req:
        headers = {identity.client_ip_header(): "203.0.113.9"}
        client = type("C", (), {"host": "127.0.0.1"})()

    assert identity.client_ip(Req()) == "127.0.0.1"
    assert identity.describe()["mode"] == "direct"
    assert identity.describe()["warning"], "direct mode must say so"


def test_the_stored_handle_is_not_the_address():
    """Addresses are personal data; this product only needs to tell people apart."""
    key = identity.org_key("203.0.113.9")
    assert "203.0.113.9" not in key
    assert key == identity.org_key("203.0.113.9"), "must be stable for one address"
    assert key != identity.org_key("203.0.113.10"), "must differ between addresses"
    assert "203.0.113.9" not in identity.label("203.0.113.9"), "never show the address"


# -- one visitor, one set of submissions ------------------------------------


def test_two_addresses_cannot_see_each_others_submissions(client):
    """The property the user asked for, stated as an attempt to violate it."""
    a = as_ip(client, "203.0.113.1", "post", "/api/submissions",
              json={"name": "mine", "benchmark": "pick_dual_bottles"}).json()

    listed_by_b = as_ip(client, "198.51.100.2", path="/api/submissions").status_code
    assert listed_by_b == 200

    rows = as_ip(client, "198.51.100.2", path="/api/submissions").json()["submissions"]
    assert all(r["id"] != a["id"] for r in rows), "B can see A's submission in the list"

    direct = as_ip(client, "198.51.100.2", path=f"/api/submissions/{a['id']}")
    assert direct.status_code == 404, "B fetched A's submission by id"

    stream = as_ip(client, "198.51.100.2", path=f"/api/submissions/{a['id']}/events")
    assert stream.status_code == 404, (
        "B opened A's event stream -- this endpoint took no viewer at all before, "
        "so every submission's log was world-readable"
    )

    assert as_ip(client, "203.0.113.1", path=f"/api/submissions/{a['id']}").status_code == 200


def test_one_address_keeps_its_own_submissions_across_requests(client):
    ip = "203.0.113.77"
    made = as_ip(client, ip, "post", "/api/submissions",
                 json={"name": "keep", "benchmark": "pick_dual_bottles"}).json()
    rows = as_ip(client, ip, path="/api/submissions").json()["submissions"]
    assert any(r["id"] == made["id"] for r in rows)
    assert as_ip(client, ip).json()["org"]["id"] == as_ip(client, ip).json()["org"]["id"]


def test_me_reports_how_identity_was_decided(client):
    """So a proxy misconfiguration is readable rather than inferred from a leak."""
    body = as_ip(client, "203.0.113.5").json()
    assert body["identity"]["mode"] in ("edge", "direct")
    assert "org" in body and body["org"]["id"].startswith("org_")


# -- publishing is a choice -------------------------------------------------


def test_a_new_submission_is_not_on_the_leaderboard(client):
    made = as_ip(client, "203.0.113.20", "post", "/api/submissions",
                 json={"name": "quiet", "benchmark": "pick_dual_bottles"}).json()
    assert made["listed"] is False, "a result must never be published by default"


def test_publishing_needs_a_result_to_publish(client):
    """Otherwise the switch appears to work and the entry never appears."""
    made = as_ip(client, "203.0.113.21", "post", "/api/submissions",
                 json={"name": "unfinished", "benchmark": "pick_dual_bottles"}).json()
    resp = as_ip(client, "203.0.113.21", "post", f"/api/submissions/{made['id']}/listed",
                 json={"listed": True})
    assert resp.status_code == 409
    assert "nothing to rank" in resp.text


def test_nobody_can_publish_somebody_elses_submission(client):
    made = as_ip(client, "203.0.113.22", "post", "/api/submissions",
                 json={"name": "not yours", "benchmark": "pick_dual_bottles"}).json()
    resp = as_ip(client, "198.51.100.99", "post", f"/api/submissions/{made['id']}/listed",
                 json={"listed": True})
    assert resp.status_code == 404


def _rankable(sub_id: str, listed: bool) -> None:
    """Give a submission a finished robot test, so the board would rank it.

    Without this the submission has no passed ``g3`` and ``compare`` skips it
    before the opt-in filter is ever consulted -- which makes an "is it hidden?"
    test pass whether the filter exists or not. Found by deleting the filter and
    watching the test still go green.
    """
    with dbmod.SessionLocal() as session:
        sub = session.get(dbmod.Submission, sub_id)
        sub.listed = listed
        session.add(
            dbmod.DatasetVersion(
                id=dbmod.new_id("dsv"), submission_id=sub_id, version=1, path="/none", bytes=0
            )
        )
        session.add(
            dbmod.Gate(
                id=dbmod.new_id("gate"), submission_id=sub_id, key="g3", status="passed",
                version=1,
                detail_json=json.dumps({
                    "order": ["solved"],
                    "reached": {"scene-1": {"solved": True}, "scene-2": {"solved": False}},
                }),
            )
        )
        session.commit()


def test_an_unlisted_submission_is_invisible_to_others_on_the_board(client):
    """The leaderboard reads across visitors, so its filter is a permission too."""
    made = as_ip(client, "203.0.113.23", "post", "/api/submissions",
                 json={"name": "hidden", "benchmark": "pick_dual_bottles"}).json()
    _rankable(made["id"], listed=False)

    mine = as_ip(client, "203.0.113.23", path="/api/compare?benchmark=pick_dual_bottles").json()
    assert any(e["id"] == made["id"] for e in mine["entries"]), (
        "its owner must see their own unlisted entry, or they cannot judge it "
        "before deciding whether to publish"
    )
    assert next(e for e in mine["entries"] if e["id"] == made["id"])["private"] is True

    theirs = as_ip(client, "198.51.100.50", path="/api/compare?benchmark=pick_dual_bottles")
    assert theirs.status_code == 200
    assert all(e["id"] != made["id"] for e in theirs.json()["entries"]), (
        "an unpublished result appeared on somebody else's leaderboard"
    )


def test_publishing_puts_it_on_everybody_elses_board(client):
    """The other direction, so the filter cannot simply hide everything."""
    made = as_ip(client, "203.0.113.24", "post", "/api/submissions",
                 json={"name": "shared", "benchmark": "pick_dual_bottles"}).json()
    _rankable(made["id"], listed=True)

    theirs = as_ip(client, "198.51.100.51", path="/api/compare?benchmark=pick_dual_bottles").json()
    row = next((e for e in theirs["entries"] if e["id"] == made["id"]), None)
    assert row is not None, "a published result did not reach another visitor's board"
    assert row["mine"] is False
    assert row["name"] == "shared", "the published name is what is shown"
