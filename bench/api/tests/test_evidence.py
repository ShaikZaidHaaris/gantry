"""The evidence bundle is the report page in table form, and nothing more.

Two boundaries pinned here. First, fidelity: every row in the bundle must come
from the submission dict unchanged -- same counts, same values -- because the
bundle's one promise is that it does not compute. Second, reach: the bundle is
readable exactly where the report is readable (a seeded example for everyone,
a stranger's submission for no one), and it never carries the uploader's
contact address, because an export is made to leave.

The builder tests feed ``sample_results.json`` directly: it is the same shape
:func:`app.main.as_submission` produces, which is the property that lets the
worked examples and the live rows share one exporter.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import zipfile

import pytest

_TMP = tempfile.mkdtemp(prefix="bench-evidence-test-")
os.environ["BENCH_DATA"] = _TMP
os.environ["BENCH_TRUST_HEADER"] = "x-bench-edge"
os.environ["BENCH_TRUST_SECRET"] = "test-edge-secret"
os.environ["BENCH_IP_SALT"] = "test-salt"

from app import evidence, identity  # noqa: E402
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


def fixture_sample() -> dict:
    """The first worked example, in the shape ``as_submission`` produces.

    The fixture stores gates and events the way the database rows do, with
    the working serialised into ``*_json`` strings; the builder takes the
    parsed shape the API serves. This adapter is the same parse ``as_gate``
    and ``as_submission`` perform, kept deliberately dumb.
    """
    samples = json.loads(samplesmod.FIXTURE.read_text())["samples"]
    assert samples, "no fixture samples, so none of these tests prove anything"
    raw = samples[0]
    return {
        **{k: v for k, v in raw.items() if k not in {"gates", "events"}},
        "gates": [
            {
                "key": g["key"],
                "status": g["status"],
                "trials": g.get("trials", 0),
                "cost_cents": g.get("cost_cents", 0),
                "started_at": g.get("started_at", ""),
                "finished_at": g.get("finished_at", ""),
                "verdict": json.loads(g.get("verdict_json") or "{}"),
                "findings": json.loads(g.get("findings_json") or "[]"),
                "measures": json.loads(g.get("measures_json") or "{}"),
                "abstained": json.loads(g.get("abstained_json") or "[]"),
                "detail": json.loads(g.get("detail_json") or "{}"),
            }
            for g in raw["gates"]
        ],
        "events": [
            {"ts": e["ts"], "kind": e["kind"], **json.loads(e.get("payload_json") or "{}")}
            for e in raw["events"]
        ],
    }


def gate(sub: dict, key: str) -> dict:
    return next(g for g in sub["gates"] if g["key"] == key)


def parse(entry: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(entry.decode("utf-8"))))


# ---------------------------------------------------------------------------
# The builder: counts and values survive the flattening.


def test_tables_carry_every_row_of_the_source():
    sub = fixture_sample()
    built = evidence.tables(sub)

    assert len(built["gates"]) == len(sub["gates"])
    assert len(built["findings"]) == sum(len(g["findings"]) for g in sub["gates"])
    assert len(built["measures"]) == sum(len(g["measures"]) for g in sub["gates"])
    assert len(built["events"]) == len(sub["events"])

    pairs = gate(sub, "g2")["detail"]["pairs"]
    assert len(built["signal_pairs"]) == len(pairs)

    ladder = gate(sub, "g3")["detail"]["ladder"]
    assert len(built["ladder"]) == sum(len(step["arms"]) for step in ladder)
    assert len(built["ladder_vs_baseline"]) == len(ladder)


def test_values_are_copied_not_computed():
    sub = fixture_sample()
    built = evidence.tables(sub)

    first = gate(sub, "g2")["detail"]["pairs"][0]
    row = built["signal_pairs"][0]
    assert row["episode"] == first["episode"]
    assert row["error_yours"] == first["yours"]
    assert row["error_shuffled"] == first["shuffled"]

    step = gate(sub, "g3")["detail"]["ladder"][0]
    arm_name, arm = next(iter(step["arms"].items()))
    lrow = next(r for r in built["ladder"] if r["rung"] == step["rung"] and r["arm"] == arm_name)
    assert lrow["wins"] == arm.get("wins")
    assert lrow["n"] == arm.get("n")


def test_absent_gate_means_absent_table():
    built = evidence.tables({"gates": [], "events": [], "coach": {}})
    assert built == {}


# ---------------------------------------------------------------------------
# The bundle: a zip whose manifest tells the truth about its contents.


def test_bundle_round_trips_through_the_manifest():
    sub = fixture_sample()
    payload = evidence.bundle(sub)

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))

        assert manifest["magic"] == evidence.MAGIC
        assert manifest["format_version"] == evidence.FORMAT_VERSION
        assert manifest["submission"]["id"] == sub["id"]

        # Every table the manifest declares exists, parses, and has the row
        # count the manifest claims; no undeclared file rides along.
        assert names == {"manifest.json"} | {t["file"] for t in manifest["tables"].values()}
        for name, spec in manifest["tables"].items():
            rows = parse(zf.read(spec["file"]))
            assert len(rows) == spec["rows"]
            assert list(rows[0]) == list(spec["columns"])


def test_booleans_and_nones_survive_csv():
    sub = fixture_sample()
    payload = evidence.bundle(sub)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        rows = parse(zf.read("ladder.csv"))
    measured = {r["measured"] for r in rows}
    assert measured <= {"true", "false"}
    unmeasured = [r for r in rows if r["measured"] == "false"]
    assert unmeasured and all(r["rate"] == "" for r in unmeasured)  # no number where none was taken


# ---------------------------------------------------------------------------
# The endpoint: readable where the report is readable, and no address leaves.


def test_sample_bundle_is_readable_by_anyone(client):
    sample_id = list(samplesmod.ids().values())[0]
    r = as_ip(client, "203.0.113.7", path=f"/api/submissions/{sample_id}/evidence")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["magic"] == evidence.MAGIC
    assert manifest["submission"]["id"] == sample_id


def test_missing_submission_is_a_404(client):
    r = as_ip(client, "203.0.113.7", path="/api/submissions/does-not-exist/evidence")
    assert r.status_code == 404


def test_bundle_never_carries_an_email(client):
    sample_id = list(samplesmod.ids().values())[0]
    r = as_ip(client, "203.0.113.7", path=f"/api/submissions/{sample_id}/evidence")
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("manifest.json"))
    assert "email" not in json.dumps(manifest)
