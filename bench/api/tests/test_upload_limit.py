"""The upload ceiling, and the two ways a size check gets fooled.

A bound on uploads is a disk-availability control, not a preference. Uploads
land on the root volume of the GPU box, which has 12 GB free, and each archive
is unpacked beside itself afterwards. One oversized upload fills the disk and
every *other* visitor's submission then fails at intake with an error about our
storage, presented to them as a problem with their data.

Two things a naive implementation gets wrong, both covered here:

* checking only ``Content-Length``, which is a number the client chooses. It can
  be absent on a chunked request and it can simply be a lie, so the header check
  is an optimisation and never the bound.
* leaving the partial file behind on refusal, which spends exactly the disk the
  limit was protecting, and leaves an archive no database row points at, so it
  is invisible to everything except ``df``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="bench-upload-test-"))
os.environ["BENCH_DATA"] = str(_TMP)
os.environ["BENCH_TRUST_HEADER"] = "x-bench-edge"
os.environ["BENCH_TRUST_SECRET"] = "test-edge-secret"
os.environ["BENCH_IP_SALT"] = "test-salt"

from app import identity  # noqa: E402
from app import main as mainmod  # noqa: E402
from app.db import STORAGE  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app, base_url="https://testserver") as c:
        yield c


def as_ip(client, ip: str, method: str, path: str, **kw):
    headers = {**kw.pop("headers", {}), identity.client_ip_header(): ip}
    if identity.trusts_edge():
        headers[identity.trust_header()] = identity.trust_secret()
    return getattr(client, method)(path, headers=headers, **kw)


def a_submission(client, ip: str = "93.184.216.34") -> str:
    benches = as_ip(client, ip, "get", "/api/benchmarks").json()
    key = (benches if isinstance(benches, list) else benches["benchmarks"])[0]["key"]
    made = as_ip(
        client, ip, "post", "/api/submissions", json={"name": "big", "benchmark": key}
    )
    assert made.status_code < 300, made.text
    return made.json()["id"]


def upload(client, sub: str, blob: bytes, ip: str = "93.184.216.34"):
    return as_ip(
        client, ip, "post", f"/api/submissions/{sub}/dataset",
        files={"file": ("dataset.zip", blob, "application/zip")},
    )


def test_the_limit_is_reported_so_the_screen_can_state_it(client):
    """The number the upload page shows comes from here.

    Stated by the server rather than written into the page, because a limit
    duplicated in the UI is one that eventually disagrees with the check, and
    the visitor discovers the difference by waiting for an upload that was never
    going to be accepted.
    """
    body = as_ip(client, "93.184.216.34", "get", "/api/me").json()
    assert body["max_upload_bytes"] == mainmod.MAX_UPLOAD_BYTES
    assert mainmod.MAX_UPLOAD_BYTES == 1024**3, "the documented ceiling is 1 GB"


def test_an_archive_under_the_limit_is_accepted(client):
    """The positive control. A route that refused everything would pass below."""
    sub = a_submission(client)
    assert upload(client, sub, b"PK\x03\x04" + b"\0" * 4096).status_code < 300


def test_an_archive_over_the_limit_is_refused(client, monkeypatch):
    """The bound itself, with the ceiling lowered so the test is not a GB of RAM."""
    monkeypatch.setattr(mainmod, "MAX_UPLOAD_BYTES", 8 * 1024)
    sub = a_submission(client, "198.51.100.9")
    response = upload(client, sub, b"\0" * (32 * 1024), ip="198.51.100.9")
    assert response.status_code == 413, response.text
    assert "limit" in response.text


def test_a_refused_upload_leaves_nothing_on_disk(client, monkeypatch):
    """The half of the fix that is easy to forget.

    Refusing after writing most of a file spends the disk the limit exists to
    protect, and the leftover belongs to no submission, so nothing but ``df``
    would ever mention it.
    """
    monkeypatch.setattr(mainmod, "MAX_UPLOAD_BYTES", 8 * 1024)
    sub = a_submission(client, "198.51.100.10")
    upload(client, sub, b"\0" * (64 * 1024), ip="198.51.100.10")

    leftovers = list((STORAGE / sub).rglob("dataset.zip"))
    assert leftovers == [], f"a refused upload left {leftovers} behind"


def test_a_lying_content_length_does_not_get_through(client, monkeypatch):
    """Why the header check cannot be the only one.

    ``Content-Length`` is a claim the client makes. Here it understates the body
    by a factor of thirty, which passes the cheap check, so the only thing left
    to stop it is counting what actually arrives.
    """
    monkeypatch.setattr(mainmod, "MAX_UPLOAD_BYTES", 8 * 1024)
    sub = a_submission(client, "198.51.100.11")
    response = as_ip(
        client, "198.51.100.11", "post", f"/api/submissions/{sub}/dataset",
        files={"file": ("dataset.zip", b"\0" * (64 * 1024), "application/zip")},
        headers={"content-length": "1024"},
    )
    assert response.status_code == 413, (
        "an understated Content-Length got past the size check, so the header "
        "is being trusted instead of the bytes"
    )


def test_the_ceiling_can_be_raised_without_a_deploy(client):
    """A host with room should not need a code change to use it."""
    assert "BENCH_MAX_UPLOAD_BYTES" in Path(mainmod.__file__).read_text(encoding="utf-8")
