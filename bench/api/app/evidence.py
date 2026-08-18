"""The evidence bundle: a submission's verdict, flattened into tables.

Why this exists
---------------
A verdict is a summary; the working behind it lives in the gates' ``*_json``
blobs, readable by this product's screens and by nothing else. A contributor
who wants to interrogate their result -- which held-out clips dragged the
signal check down, which rung of the ladder broke, what happened between one
gate and the next -- needs the rows, in a shape a query engine can read
without knowing our schema.

This module exports exactly what the gates recorded, one CSV per kind of
evidence plus a ``manifest.json``, zipped. Any Arrow/DuckDB tool can then
answer questions with auditable SQL instead of re-parsing our JSON; the
intended first reader is Bagel (github.com/Extelligence-ai/bagel), which
sniffs the manifest's ``magic`` and serves each table as a topic.

Nothing here is computed. A bundle that derived new numbers on the way out
would be a second implementation of the gates' arithmetic, and the first time
the two disagreed the export would be quietly lying about the verdict. Ratios,
margins and deltas are one ``SELECT`` away for whoever reads the bundle, and
that query is theirs to audit.

CSV rather than parquet because this API deliberately has no pyarrow: the
tables are small (hundreds of rows), every reader that speaks parquet also
speaks CSV, and a dependency added for a file format would be the largest
package in the venv. The manifest carries a per-column type map so a reader
that wants real types can cast instead of sniff.

The input is the dict :func:`app.main.as_submission` produces with
``deep=True`` -- the same shape ``sample_results.json`` carries -- so the
builder works identically on a live row and on the seeded worked examples,
and a test can feed it the fixture without a database.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any

MAGIC = "GANTRY_EVIDENCE"
FORMAT_VERSION = 1

#: Column types per table, recorded in the manifest so a reader can cast
#: rather than sniff. CSV has no types; this is the bundle's word on what the
#: columns mean. ``float`` columns may be empty where the source recorded
#: nothing (a CI that was not computed, a p-value that does not apply).
SCHEMAS: dict[str, dict[str, str]] = {
    "gates": {
        "gate": "string",
        "name": "string",
        "status": "string",
        "summary": "string",
        "trials": "int",
        "cost_cents": "int",
        "started_at": "timestamp",
        "finished_at": "timestamp",
    },
    "findings": {
        "gate": "string",
        "code": "string",
        "severity": "string",
        "summary": "string",
        "prescription": "string",
        "module": "string",
    },
    "measures": {
        "gate": "string",
        "measure": "string",
        "value": "float",
        "n": "int",
        "ci_lo": "float",
        "ci_hi": "float",
        "units": "string",
        "method": "string",
        "module": "string",
    },
    "abstained": {
        "gate": "string",
        "module": "string",
        "reason": "string",
        "codes": "json",
    },
    "signal_pairs": {
        "episode": "string",
        "error_yours": "float",
        "error_shuffled": "float",
        "better": "bool",
    },
    "ladder": {
        "rung": "string",
        "rung_index": "int",
        "arm": "string",
        "measured": "bool",
        "wins": "int",
        "n": "int",
        "rate": "float",
        "ci_lo": "float",
        "ci_hi": "float",
        "unmeasured": "int",
    },
    "ladder_vs_baseline": {
        "rung": "string",
        "rung_index": "int",
        "measured": "bool",
        "scenes": "int",
        "detail": "json",
    },
    "events": {
        "ts": "timestamp",
        "kind": "string",
        "gate": "string",
        "detail": "json",
    },
    "coach": {
        "kind": "string",
        "code": "string",
        "title": "string",
        "detail": "string",
    },
}


def _gate(sub: dict, key: str) -> dict:
    return next((g for g in sub.get("gates", []) if g.get("key") == key), {})


def _ci(pair: Any) -> tuple[Any, Any]:
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1]
    return None, None


def tables(sub: dict) -> dict[str, list[dict]]:
    """Flatten one submission dict into the bundle's tables.

    Only tables with rows are returned: an absent table means the gate that
    would have produced it has not run, and an empty file would read as "ran
    and found nothing", which is a different claim.
    """
    out: dict[str, list[dict]] = {}

    gates = [
        {
            "gate": g.get("key", ""),
            "name": g.get("name", ""),
            "status": g.get("status", ""),
            "summary": (g.get("verdict") or {}).get("summary", ""),
            "trials": g.get("trials", 0),
            "cost_cents": g.get("cost_cents", 0),
            "started_at": g.get("started_at", ""),
            "finished_at": g.get("finished_at", ""),
        }
        for g in sub.get("gates", [])
    ]
    if gates:
        out["gates"] = gates

    findings = [
        {
            "gate": g.get("key", ""),
            "code": f.get("code", ""),
            "severity": f.get("severity", ""),
            "summary": f.get("summary", ""),
            "prescription": f.get("prescription", ""),
            "module": f.get("module", ""),
        }
        for g in sub.get("gates", [])
        for f in g.get("findings", [])
    ]
    if findings:
        out["findings"] = findings

    measures = []
    for g in sub.get("gates", []):
        for name, m in (g.get("measures") or {}).items():
            lo, hi = _ci(m.get("ci"))
            measures.append(
                {
                    "gate": g.get("key", ""),
                    "measure": name,
                    "value": m.get("value"),
                    "n": m.get("n"),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "units": m.get("units", ""),
                    "method": m.get("method", ""),
                    "module": m.get("module", ""),
                }
            )
    if measures:
        out["measures"] = measures

    abstained = [
        {
            "gate": g.get("key", ""),
            "module": a.get("module", ""),
            "reason": a.get("reason", ""),
            "codes": json.dumps(a.get("codes", [])),
        }
        for g in sub.get("gates", [])
        for a in g.get("abstained", [])
    ]
    if abstained:
        out["abstained"] = abstained

    # G2's working: one row per held-out clip, the fit on your data against the
    # same fit with the actions detached. The margin is deliberately not a
    # column -- it is `error_shuffled - error_yours` in whatever query wants it.
    pairs = [
        {
            "episode": p.get("episode", ""),
            "error_yours": p.get("yours"),
            "error_shuffled": p.get("shuffled"),
            "better": p.get("better"),
        }
        for p in (_gate(sub, "g2").get("detail") or {}).get("pairs", [])
    ]
    if pairs:
        out["signal_pairs"] = pairs

    # G3's working: the ladder, one row per rung per arm, and the per-rung
    # baseline contrast in its own table because its columns are not the arms'.
    g3 = _gate(sub, "g3").get("detail") or {}
    ladder, contrasts = [], []
    for i, step in enumerate(g3.get("ladder", [])):
        rung = step.get("rung", "")
        for arm, r in (step.get("arms") or {}).items():
            lo, hi = _ci(r.get("ci"))
            ladder.append(
                {
                    "rung": rung,
                    "rung_index": i,
                    "arm": arm,
                    "measured": r.get("measured"),
                    "wins": r.get("wins"),
                    "n": r.get("n"),
                    "rate": r.get("rate"),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "unmeasured": r.get("unmeasured"),
                }
            )
        vs = step.get("vs baseline")
        if vs is not None:
            known = {"rung", "measured", "scenes"}
            contrasts.append(
                {
                    "rung": rung,
                    "rung_index": i,
                    "measured": vs.get("measured"),
                    "scenes": vs.get("scenes"),
                    # Whatever else the gate recorded (a p-value when the
                    # contrast was measurable) rides along rather than being
                    # dropped: an export that trims fields it does not
                    # recognise is editing the evidence.
                    "detail": json.dumps({k: v for k, v in vs.items() if k not in known}),
                }
            )
    if ladder:
        out["ladder"] = ladder
    if contrasts:
        out["ladder_vs_baseline"] = contrasts

    events = [
        {
            "ts": e.get("ts", ""),
            "kind": e.get("kind", ""),
            "gate": e.get("gate", ""),
            "detail": json.dumps(
                {k: v for k, v in e.items() if k not in {"ts", "kind", "gate"}}
            ),
        }
        for e in sub.get("events", [])
    ]
    if events:
        out["events"] = events

    # Coaching: ``points`` is a list of {title, detail}; ``fixes`` is a dict
    # keyed by the finding code it addresses, {code: {say, detail}}. Both
    # flatten to one row per piece of advice, with the code kept so a query
    # can join a fix back onto the finding that earned it.
    coach = sub.get("coach") or {}
    advice = [
        {"kind": "point", "code": "", "title": p.get("title", ""), "detail": p.get("detail", "")}
        for p in coach.get("points") or []
    ] + [
        {"kind": "fix", "code": code, "title": f.get("say", ""), "detail": f.get("detail", "")}
        for code, f in (coach.get("fixes") or {}).items()
    ]
    if advice:
        out["coach"] = advice

    return out


def manifest(sub: dict, built: dict[str, list[dict]]) -> dict:
    """The bundle's index: who this is about, what each gate concluded, and
    which tables exist with what columns. The scalar diagnostics G3 records
    beside its ladder (alpha, whether a control arm existed) live here rather
    than in a one-row table, because SQL over a single row is ceremony."""
    g3 = _gate(sub, "g3").get("detail") or {}
    dataset = sub.get("dataset") or {}
    return {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "submission": {
            "id": sub.get("id", ""),
            "name": sub.get("name", ""),
            "status": sub.get("status", ""),
            "current_gate": sub.get("current_gate", ""),
            "created_at": sub.get("created_at", ""),
            "benchmark": sub.get("benchmark"),
            "demo": bool(sub.get("demo")),
        },
        "dataset": {
            "version": dataset.get("version"),
            "bytes": dataset.get("bytes"),
            "detected": dataset.get("detected") or {},
        },
        "gates": [
            {
                "key": g.get("key", ""),
                "name": g.get("name", ""),
                "status": g.get("status", ""),
                "summary": (g.get("verdict") or {}).get("summary", ""),
            }
            for g in sub.get("gates", [])
        ],
        "g3_context": {
            k: g3.get(k)
            for k in ("alpha", "has_control", "has_baseline", "objects_tracked", "order")
            if k in g3
        },
        "coach_model": (sub.get("coach") or {}).get("model", ""),
        "tables": {
            name: {
                "file": f"{name}.csv",
                "rows": len(rows),
                "columns": SCHEMAS[name],
            }
            for name, rows in built.items()
        },
    }


def _csv_bytes(name: str, rows: list[dict]) -> bytes:
    """One table as CSV. Column order comes from the schema, not the dicts, so
    the file's shape is stable whatever order the rows were assembled in.
    ``None`` becomes an empty cell; booleans become ``true``/``false``, which
    both Arrow and DuckDB read back as booleans."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(SCHEMAS[name]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: ("" if v is None else ("true" if v is True else ("false" if v is False else v)))
                for k, v in row.items()
            }
        )
    return buf.getvalue().encode("utf-8")


def bundle(sub: dict) -> bytes:
    """The zip: ``manifest.json`` plus one CSV per non-empty table."""
    built = tables(sub)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest(sub, built), indent=1))
        for name, rows in built.items():
            zf.writestr(f"{name}.csv", _csv_bytes(name, rows))
    return buf.getvalue()
