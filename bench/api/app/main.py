"""The API. Thin, honest, and it never computes a result of its own.

Every number the UI shows was written by a worker into ``gates`` or
``dataset_versions``. The API's job is to hold state, hand out upload slots,
queue work, and stream the event log. When the product later moves to Postgres
and S3, only ``db.py`` and the two storage helpers change.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .identity import (
    COOKIE_MAX_AGE,
    client_ip,
    cookie_name,
    cookie_secure,
    describe,
    label,
    label_for,
    mint,
    org_key,
    token_key,
    verified,
)
from .samples import ids as sample_ids
from .samples import seed as seed_samples

from .db import (
    STORAGE,
    WORKER_TOKEN,
    Benchmark,
    DatasetVersion,
    Event,
    Gate,
    Job,
    Org,
    SessionLocal,
    Submission,
    Visit,
    emit,
    init_db,
    new_id,
    now,
    sweep_stale,
)

app = FastAPI(title="Gantry Bench API", version="1")


#: Paths the visit log ignores. The worker's claim loop polls twice a second
#: and its heartbeats arrive every ten; a month of real visitors would drown in
#: a day of machinery. Static assets stay out too -- the page view is the fact
#: worth keeping, not the forty files it pulled in.
_UNLOGGED = ("/api/jobs", "/assets/", "/favicon", "/healthz")


@app.middleware("http")
async def _log_visit(request: Request, call_next):
    """Every human-shaped request, written down before it is forgotten.

    The journald copy of this information rotated away while we were asking
    "who came yesterday"; this is the durable answer. Logging must never cost
    a visitor anything, so the write happens after the response is built and
    every failure inside is swallowed -- a full disk should break uploads, not
    page views, and a page view is not worth a 500.
    """
    started = time.monotonic()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        try:
            path = request.url.path
            if not any(path.startswith(skip) for skip in _UNLOGGED) and (
                path.startswith("/api/") or "." not in path.rsplit("/", 1)[-1]
            ):
                cookie = request.cookies.get(cookie_name())
                visitor = ""
                with SessionLocal() as session:
                    if verified(cookie):
                        org = session.scalar(
                            select(Org).where(Org.token_hash == token_key(cookie))
                        )
                        visitor = org.id if org else ""
                    session.add(
                        Visit(
                            ip=client_ip(request),
                            visitor=visitor,
                            method=request.method,
                            path=(path + (("?" + request.url.query) if request.url.query else ""))[:300],
                            status=response.status_code if response is not None else 500,
                            ms=int((time.monotonic() - started) * 1000),
                            referer=request.headers.get("referer", "")[:300],
                            ua=request.headers.get("user-agent", "")[:200],
                        )
                    )
                    session.commit()
        except Exception:  # noqa: BLE001 - the log must never take the page down
            pass

#: The gauntlet, in order, with what each costs and how it is described to a
#: user. Held here rather than in the UI so the API, the worker and the page
#: cannot disagree about what the gates are.
#: Every gate is free for now. The prices stay computed and shown -- what a run
#: costs is a real number and hiding it would make the trial-count choice look
#: arbitrary -- but nothing is withheld behind one. ``cost_cents`` here is the
#: charge, and it is zero; ``costing()`` still reports what the GPU time is
#: worth, which is what the budget panel quotes.
#:
#: The gauntlet. ``sized`` marks the gate whose price is a *choice*: the robot
#: test runs as many scenes as you buy, and how many you buy decides what the
#: run can conclude. The others are fixed work at a fixed price, and offering a
#: trial slider for them would be a control over nothing.
GATES = [
    {"key": "g0", "name": "Intake", "question": "Can we read this at all?", "cost_cents": 0, "eta": "seconds", "sized": False},
    {"key": "g1", "name": "Data report", "question": "What is this footage like?", "cost_cents": 0, "eta": "about a minute", "sized": False},
    {"key": "g2", "name": "Signal check", "question": "Is there anything learnable here?", "cost_cents": 0, "eta": "about ten minutes", "sized": False},
    {"key": "g3", "name": "Robot test", "question": "Does the robot actually get better?", "cost_cents": 0, "eta": "a few hours", "sized": True},
]


def worker(x_worker_token: str | None = Header(default=None)) -> None:
    """Guards every route a worker uses.

    Open when no token is configured, which is right for a laptop talking to
    its own API on loopback and wrong the moment the API is reachable from
    another machine. Attaching a GPU box means setting ``BENCH_WORKER_TOKEN``
    on both ends; leaving it unset there would let anything that can route to
    the port claim jobs and download other people's datasets.
    """
    if WORKER_TOKEN and x_worker_token != WORKER_TOKEN:
        raise HTTPException(401, "a worker token is required")


def db() -> Session:
    with SessionLocal() as session:
        yield session


def _settle(session: Session, org: Org, lookup) -> Org:
    """Commit a new org, or take the one that won a race with this request.

    Two requests from one visitor arriving together both find nothing and both
    insert. The unique index means exactly one wins; the loser re-reads rather
    than raising, because the failure a user would otherwise see is a 500 on
    their first ever click.
    """
    session.add(org)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        found = lookup()
        if found is None:
            raise
        return found
    return org


def viewer(request: Request, response: Response, session: Session = Depends(db)):
    """One org per visitor. The single seam every route asks through.

    A visitor is a signed cookie we issued, not an address. The address was both
    too coarse and too fragile: everyone behind one NAT shared a submission list,
    and a reconnected router made somebody a stranger to their own uploads. A
    cookie is carried by the person rather than by the network they happen to be
    on, so it fixes both directions at once.

    Three cases, in order:

      * a valid cookie: that org, and nothing else is consulted.
      * no cookie, but this address already has an org: adopt it and set a
        cookie for it. This is the migration, and it runs once per visitor with
        nothing to do on their part. Without it every org that predates the
        cookie would be stranded with its submissions inside.
      * neither: a new org, keyed by the cookie alone.

    What this is and is not. It is a bearer token, so it partitions visitors and
    does not authenticate them: whoever holds the cookie is that visitor. That
    is the same trust level the address had, which is why it is signed, HttpOnly
    and Secure rather than a bare number a visitor could edit. The remaining
    limit is worth stating plainly: clearing cookies, a private window or a
    second device is a new visitor, and the way back from that is the email
    already collected at upload, not anything in here.
    """
    # `verified` answers whether we issued this; the whole cookie is what gets
    # hashed. Both paths must hash the identical string, and hashing the signed
    # value on the way out while hashing the unsigned one on the way back in is
    # a bug that presents exactly as the cookie never being sent.
    cookie = request.cookies.get(cookie_name())
    if verified(cookie):
        key = token_key(cookie)
        org = session.scalar(select(Org).where(Org.token_hash == key))
        if org is not None:
            return {"org_id": org.id, "org": org}
        # Signed by us but unknown, which is a database restored from before the
        # cookie was issued. Treat it as no cookie and let the visitor be
        # adopted or created below, rather than handing back a 404 they cannot
        # act on.

    fresh = mint()
    key = token_key(fresh)

    # The migration. Only ever reads an org that has no cookie yet, so it cannot
    # take one that already belongs to a browser.
    ip = client_ip(request)
    org = session.scalar(
        select(Org).where(Org.ip_hash == org_key(ip), Org.token_hash.is_(None))
    )
    if org is not None:
        org.token_hash = key
        session.commit()
    else:
        org = _settle(
            session,
            Org(id=new_id("org"), name=label_for(key), token_hash=key),
            lambda: session.scalar(select(Org).where(Org.token_hash == key)),
        )

    response.set_cookie(
        cookie_name(),
        fresh,
        max_age=COOKIE_MAX_AGE,
        httponly=True,       # no script reads it, so a page flaw cannot lift it
        secure=cookie_secure(),  # https only wherever a proxy terminates TLS
        samesite="lax",      # not attached to cross-site requests
        path="/",
    )
    return {"org_id": org.id, "org": org}


def _contact(raw: Any) -> str:
    """An address we could actually write to, or nothing at all.

    Optional by design, so the check refuses rather than corrects: a value that
    cannot be an address is dropped instead of being stored as a half-address
    somebody later tries to send to. Blank is a legitimate answer -- this is a
    way to reach you if a run breaks, not an account.

    Deliberately not a full RFC 5322 grammar. Anything stricter than "one @,
    something either side, a dot after it" rejects real addresses, and the only
    real test of an address is sending to it.
    """
    text = (raw or "").strip()[:200]
    if not text:
        return ""
    local, _, domain = text.partition("@")
    if not local or not domain or "." not in domain or " " in text:
        raise HTTPException(422, f"{text!r} does not look like an email address")
    return text


def latest_version(session: Session, sub_id: str) -> int:
    """The upload a submission is currently about."""
    row = session.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.submission_id == sub_id)
        .order_by(DatasetVersion.version.desc())
    ).first()
    return row.version if row else 1


def gates_for(session: Session, sub_id: str, version: int) -> list[Gate]:
    order = {g["key"]: i for i, g in enumerate(GATES)}
    rows = session.scalars(
        select(Gate).where(Gate.submission_id == sub_id, Gate.version == version)
    ).all()
    return sorted(rows, key=lambda g: order.get(g.key, 99))


def as_gate(row: Gate) -> dict:
    spec = next(g for g in GATES if g["key"] == row.key)
    return {
        "key": row.key,
        "name": spec["name"],
        "question": spec["question"],
        "eta": spec["eta"],
        # What it will cost, or what it did: a bought gate keeps the price it
        # was bought at, so a later change to the cost model cannot rewrite the
        # bill on a run that already happened.
        "cost_cents": row.cost_cents or spec["cost_cents"],
        "sized": spec["sized"],
        #: Whether the product will offer to run this again. Only ever true for
        #: our own failures, never for a refusal.
        "retryable": row.status == "failed",
        #: What was bought, once it has been. Zero until then, and zero forever
        #: on a gate whose price is not a choice.
        "trials": row.trials,
        "status": row.status,
        "verdict": json.loads(row.verdict_json or "{}"),
        "findings": json.loads(row.findings_json or "[]"),
        "measures": json.loads(row.measures_json or "{}"),
        "abstained": json.loads(row.abstained_json or "[]"),
        "detail": json.loads(row.detail_json or "{}"),
        # Only while running. A finished gate's last position is noise, and
        # showing "step 2,999 of 3,000" next to a green tick reads as a run that
        # stopped one short.
        "progress": json.loads(row.progress_json or "{}") if row.status == "running" else {},
        "started_at": row.started_at,
        "finished_at": row.finished_at,
    }


def as_submission(session: Session, sub: Submission, deep: bool = False, owner: bool = True) -> dict:
    bench = session.get(Benchmark, sub.benchmark_id)
    version = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == sub.id).order_by(DatasetVersion.version.desc())
    ).first()
    current = version.version if version else 1
    gates = gates_for(session, sub.id, current)
    out = {
        "id": sub.id,
        "name": sub.name,
        "status": sub.status,
        "current_gate": sub.current_gate,
        #: Published to the shared leaderboard, or not. False for every
        #: submission that predates the flag, which is the safe direction: a
        #: result never becomes public because a column was added.
        "listed": bool(sub.listed),
        #: A seeded example is readable by everyone, so it is the one row
        #: reaching this function that its reader does not own. Blanked here
        #: rather than trusted to be empty in the fixture, because the fixture
        #: is data and this is the invariant.
        "demo": bool(sub.demo),
        #: Whether the reader owns this. The page uses it the way it already
        #: uses `demo`: a stranger reading a published result gets the report,
        #: not the controls, and a control that would only ever answer 404 is
        #: better absent than present and broken.
        "mine": bool(owner) and not sub.demo,
        #: Only ever the owner's to see. Blanked for demos and for strangers
        #: reading a published result: publishing shares the report, and the
        #: contact address is not part of the report.
        "email": "" if (sub.demo or not owner) else (sub.email or ""),
        #: Generated advice, or {} when none was written. Rendered as its own
        #: labelled card; nothing else reads it.
        "coach": json.loads(sub.coach_json or "{}"),
        "created_at": sub.created_at,
        "benchmark": {"key": bench.key, "name": bench.name, "simulator": bench.simulator} if bench else None,
        "gates": [as_gate(g) for g in gates],
        "dataset": {
            "version": version.version,
            "bytes": version.bytes,
            "detected": json.loads(version.detected_json or "{}"),
            "meaning": json.loads(version.meaning_json or "{}"),
        }
        if version
        else None,
    }
    if deep:
        out["events"] = [
            {"ts": e.ts, "kind": e.kind, **json.loads(e.payload_json or "{}")}
            for e in session.scalars(select(Event).where(Event.submission_id == sub.id).order_by(Event.id)).all()
        ]
    return out


# ---------------------------------------------------------------------------


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # After the benchmarks exist, since a sample points at one. Failure here is
    # logged and swallowed: worked examples are a convenience, and an API that
    # will not start because a demo fixture is malformed has traded the product
    # for the brochure.
    try:
        with SessionLocal() as session:
            written = seed_samples(session)
        if written:
            print(f"seeded {written} worked example(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"could not seed worked examples: {exc}")


@app.get("/api/me")
def me(who=Depends(viewer)):
    """Who the server thinks you are, and how it decided.

    ``identity`` is reported rather than kept quiet because the difference
    between "every visitor is separate" and "every visitor is one org" is
    invisible from the outside, and a proxy misconfiguration silently causes
    the second. An operator can read it off this endpoint instead of finding
    out from a user who can see somebody else's uploads.
    """
    org = who["org"]
    return {
        "org": {"id": org.id, "name": org.name},
        "identity": describe(),
        # Reported so the upload screen states the same number the server
        # enforces. A limit written into the copy separately is one that
        # eventually disagrees with the check, and the visitor finds out after
        # waiting for the upload.
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.get("/api/benchmarks")
def benchmarks(session: Session = Depends(db)):
    rows = session.scalars(select(Benchmark)).all()
    return {
        "benchmarks": [
            {
                "key": b.key,
                "name": b.name,
                "task": b.task,
                "embodiment": b.embodiment,
                "simulator": b.simulator,
                "reference": json.loads(b.reference_json or "{}"),
            }
            for b in rows
        ],
        "gates": GATES,
    }


#: Cheapest experiment we will sell. Below this the arithmetic is not close --
#: it is not an experiment, and quoting a price for one would be selling a
#: number rather than an answer.
FLOOR_TRIALS = 20


def costing(bench: Benchmark, trials: int) -> dict:
    """What this many trials costs and how long it takes, from measured rates.

    Both arms are trained and evaluated: the contributor's data and its own
    shuffled control. The baseline is not retrained -- it is the same model
    every time, and charging for it again would be charging twice.
    """
    cost = json.loads(bench.cost_json or "{}")
    rate = cost.get("gpu_cents_per_hour", 0) / 3600.0
    trained = cost.get("arms_trained", 2)
    evaluated = cost.get("arms_evaluated", 2)
    seconds = trained * cost.get("train_seconds", 0) + evaluated * trials * cost.get(
        "seconds_per_trial", 0
    )
    return {
        "seconds": int(seconds),
        "cents": int(round(seconds * rate)),
        "arms_trained": trained,
        "arms_evaluated": evaluated,
        "measured_on": cost.get("measured_on", ""),
    }


@app.get("/api/benchmarks/{key}/plan")
def plan(key: str, trials: int = 50, magnitude: float = 0.08, session: Session = Depends(db)):
    """What a budget can and cannot see, before it is spent.

    The one number nobody in this space quotes. A contributor choosing a trial
    count is choosing what the run is able to conclude, and the honest way to
    present that choice is to say what each option can detect -- not to let them
    pick fifty, get a null, and read it as "the data did not help" when fifty
    trials could never have separated the effect they cared about.

    Every figure here is computed by the pipeline's own sizing, against the
    baseline this benchmark has actually recorded. Nothing on this route
    invents a rate: with no measured baseline the sizing falls back to the
    noisiest case and says so, rather than picking a flattering one.
    """
    from gantry.spine.inference import trials_needed
    from gantry_feedback_power import Budget, plan_for
    from gantry_feedback_power.power import _smallest_detectable

    bench = session.scalar(select(Benchmark).where(Benchmark.key == key))
    if bench is None:
        raise HTTPException(404, "no such benchmark")

    reference = json.loads(bench.reference_json or "{}")
    base = reference.get("baseline") or {}
    baseline = (base.get("wins") / base["n"]) if base.get("n") else None

    trials = max(FLOOR_TRIALS, min(2000, int(trials)))
    verdict = plan_for(Budget(trials=trials, magnitude=magnitude), baseline=baseline)
    alpha = Budget(trials=trials, magnitude=magnitude).corrected_alpha()

    # Sized against the noisiest case when nothing has been recorded, matching
    # what plan_for does, so the two cannot disagree about what was assumed.
    assumed = 0.5 if baseline is None else baseline
    return {
        "benchmark": bench.key,
        "trials": trials,
        "magnitude": magnitude,
        "baseline": {
            "rate": round(assumed, 4),
            "measured": baseline is not None,
            "wins": base.get("wins"),
            "n": base.get("n"),
            "note": reference.get("note", ""),
        },
        "detects": round(_smallest_detectable(assumed, trials, alpha), 4),
        "needed": trials_needed(assumed, magnitude, alpha=alpha),
        "ok": verdict.ok,
        "reasons": [
            {"code": r.code, "summary": r.message, "hint": r.hint or "", "detail": dict(r.detail)}
            for r in verdict.reasons
        ],
        "cost": costing(bench, trials),
    }


def _wilson(wins: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _sign_test(left: int, right: int) -> float:
    """The same exact test the gates use, on the same kind of pairs."""
    from math import comb

    n = left + right
    if n == 0:
        return 1.0
    top = max(left, right)
    return min(1.0, 2 * sum(comb(n, i) for i in range(top, n + 1)) / (2**n))


def letters(order: list[str], indistinct: set[frozenset[str]]) -> dict[str, str]:
    """Compact letter display: two entries share a letter when nothing separates them.

    The point of a leaderboard is not the order -- at these sample sizes the
    order is mostly noise -- it is which gaps are real. A ranked table without
    this reads as a total ordering and invites "we came third", when third and
    second may be the same result twice. Sharing a letter is the table saying so
    without a paragraph.

    Greedy, which is the standard construction: walk the ranking, and put each
    entry in the first group whose members it is indistinguishable from all of.
    """
    groups: list[list[str]] = []
    for name in order:
        for group in groups:
            if all(frozenset((name, other)) in indistinct for other in group):
                group.append(name)
                break
        else:
            groups.append([name])
    out: dict[str, str] = {}
    for index, group in enumerate(groups):
        for name in group:
            out[name] = out.get(name, "") + chr(ord("a") + index)
    return out


@app.get("/api/submissions/{sub_id}/versions")
def versions(sub_id: str, who=Depends(viewer), session: Session = Depends(db)):
    """Every upload of this submission, and what changed between them.

    The product loop is submit, fix, resubmit, see the change, and the last step
    is the one that makes the first three worth doing. Without it a contributor
    who refilms has two reports and no way to tell whether refilming worked.

    What this deliberately does *not* do is declare a winner. Two versions were
    evaluated on their own runs; whether v2 is better than v1 is a comparison
    between two experiments, and the honest home for that is the leaderboard,
    where it is paired scene by scene. Here the job is narrower: say what moved,
    and let the size of the move speak.
    """
    sub = session.get(Submission, sub_id)
    if not readable(sub, who):
        raise HTTPException(404, "no such submission")

    uploads = session.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.submission_id == sub_id)
        .order_by(DatasetVersion.version)
    ).all()

    out = []
    for upload in uploads:
        gates = gates_for(session, sub_id, upload.version)
        findings = [
            f
            for gate in gates
            for f in json.loads(gate.findings_json or "[]")
        ]
        out.append({
            "version": upload.version,
            "bytes": upload.bytes,
            "created_at": upload.created_at,
            "detected": json.loads(upload.detected_json or "{}"),
            "gates": {g.key: g.status for g in gates},
            "findings": findings,
            "verdicts": {
                g.key: json.loads(g.verdict_json or "{}").get("summary", "")
                for g in gates
                if g.status not in ("queued", "running")
            },
        })

    changes = []
    for older, newer in zip(out, out[1:]):
        before = {f["code"] for f in older["findings"]}
        after = {f["code"] for f in newer["findings"]}
        by_code = {f["code"]: f for f in older["findings"] + newer["findings"]}
        changes.append({
            "from": older["version"],
            "to": newer["version"],
            # Fixed, appeared, still there. Named by code, because a finding's
            # wording may improve between runs while the thing it found does not.
            "fixed": [{"code": c, "summary": by_code[c]["summary"]} for c in sorted(before - after)],
            "new": [{"code": c, "summary": by_code[c]["summary"]} for c in sorted(after - before)],
            "remaining": [{"code": c, "summary": by_code[c]["summary"]} for c in sorted(before & after)],
            "clips": [
                (older["detected"] or {}).get("episodes"),
                (newer["detected"] or {}).get("episodes"),
            ],
            "frames": [
                (older["detected"] or {}).get("frames"),
                (newer["detected"] or {}).get("frames"),
            ],
        })

    return {"submission": sub_id, "current": uploads[-1].version if uploads else 0,
            "versions": out, "changes": changes}


@app.get("/api/compare")
def compare(benchmark: str, rung: str = "solved", who=Depends(viewer), session: Session = Depends(db)):
    """Every finished submission on one benchmark, ranked and paired.

    Paired, which is the whole difference between this and a spreadsheet. Two
    submissions at 8% might have agreed on every scene or disagreed on sixteen,
    and only the second is evidence about which is better. So every pair is
    compared on the scenes they *both* ran, and scenes where they agreed are
    excluded from the test while still counting in the rates.

    A submission that ran different scenes contributes only its overlap, and the
    overlap is reported. Comparing across disjoint scene sets would be
    comparing two different experiments and calling it a ranking.
    """
    bench = session.scalar(select(Benchmark).where(Benchmark.key == benchmark))
    if bench is None:
        raise HTTPException(404, "no such benchmark")

    # Yours always, everyone else's only if they published it. Two conditions
    # rather than one, because a leaderboard you cannot see your own unlisted
    # entry on is a leaderboard you cannot check before publishing to -- and
    # the point of opt-in is that you get to look first.
    rows = []
    available: list[str] = []
    for sub in session.scalars(
        select(Submission).where(
            Submission.benchmark_id == bench.id,
            # A seeded example is not a competitor. Ranking it would put a
            # result nobody submitted above results people did, and the compact
            # letter display would then be grouping it with them.
            Submission.demo.is_(False),
            or_(Submission.org_id == who["org_id"], Submission.listed.is_(True)),
        )
    ).all():
        gate = session.scalar(
            select(Gate).where(
                Gate.submission_id == sub.id,
                Gate.key == "g3",
                Gate.status == "passed",
                Gate.version == latest_version(session, sub.id),
            )
        )
        if gate is None:
            continue
        detail = json.loads(gate.detail_json or "{}")
        reached = detail.get("reached") or {}
        # Every rung this submission measured, in the order its own ladder had
        # them -- which is the order they are climbed, not alphabetical.
        for name in detail.get("order") or []:
            if name not in available:
                available.append(name)
        scored = {scene: v.get(rung) for scene, v in reached.items() if v.get(rung) is not None}
        if not scored:
            continue
        wins = sum(1 for v in scored.values() if v)
        mine = sub.org_id == who["org_id"]
        rows.append({
            "id": sub.id,
            # Somebody else's entry is named by them, and that name is the only
            # thing about them this exposes -- no org, no address, no hash. It
            # is on the board because its owner published it, so the name they
            # chose is the name they published.
            "name": sub.name,
            "mine": mine,
            #: Yours and unpublished, so you can see it here and nobody else
            #: can. Drawn differently by the UI rather than hidden, because a
            #: private entry you cannot see is one you cannot decide about.
            "private": mine and not sub.listed,
            "wins": wins,
            "n": len(scored),
            "rate": round(wins / len(scored), 4),
            "ci": _wilson(wins, len(scored)),
            "scenes": scored,
        })

    rows.sort(key=lambda r: (-r["rate"], r["name"]))
    pairs = []
    indistinct: set[frozenset[str]] = set()
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            shared = sorted(set(left["scenes"]) & set(right["scenes"]))
            l_only = sum(1 for s in shared if left["scenes"][s] and not right["scenes"][s])
            r_only = sum(1 for s in shared if right["scenes"][s] and not left["scenes"][s])
            p = _sign_test(l_only, r_only)
            separated = bool(shared) and p <= 0.05
            if not separated:
                indistinct.add(frozenset((left["id"], right["id"])))
            pairs.append({
                "left": left["id"], "right": right["id"],
                "shared_scenes": len(shared), "left_only": l_only, "right_only": r_only,
                "agreed": len(shared) - l_only - r_only,
                "p_value": round(p, 4), "separated": separated,
            })

    group = letters([r["id"] for r in rows], indistinct)
    reference = json.loads(bench.reference_json or "{}")
    return {
        "benchmark": {"key": bench.key, "name": bench.name, "simulator": bench.simulator},
        "rung": rung,
        # Offered rather than assumed. A rung one submission measured and
        # another did not is still worth ranking on; the entries that could not
        # answer simply do not appear, which is visible.
        "rungs": available or [rung],
        "baseline": reference.get("baseline"),
        "entries": [
            {**{k: v for k, v in r.items() if k != "scenes"}, "group": group.get(r["id"], "")}
            for r in rows
        ],
        "pairs": pairs,
    }


@app.get("/api/submissions")
def list_submissions(who=Depends(viewer), session: Session = Depends(db)):
    sweep_stale(session)
    # Samples are excluded here, not filtered in the UI. This list is "your
    # work", and a seeded example sitting in it would be counted, sorted and
    # compared alongside things the visitor actually uploaded.
    rows = session.scalars(
        select(Submission)
        .where(Submission.org_id == who["org_id"], Submission.demo.is_(False))
        .order_by(Submission.created_at.desc())
    ).all()
    return {"submissions": [as_submission(session, s) for s in rows]}


@app.post("/api/submissions")
async def create_submission(body: dict, who=Depends(viewer), session: Session = Depends(db)):
    bench = session.scalar(select(Benchmark).where(Benchmark.key == body.get("benchmark", "pick_dual_bottles")))
    if bench is None:
        raise HTTPException(404, "no such benchmark")
    sub = Submission(
        id=new_id("sub"),
        org_id=who["org_id"],
        name=(body.get("name") or "untitled").strip()[:80],
        benchmark_id=bench.id,
        created_by=who["org_id"],
        email=_contact(body.get("email")),
        status="draft",
    )
    session.add(sub)
    for spec in GATES:
        # Gates are made when a dataset arrives, not when the submission is
        # named -- they belong to an upload, and there is not one yet.
        pass
    emit(session, sub.id, "submission.created", name=sub.name, benchmark=bench.key)
    session.commit()
    return as_submission(session, sub)


def readable(sub: Submission | None, who) -> bool:
    """Whether this visitor may *read* this submission.

    Yours, a seeded example, or a result its owner published. Reading is as far
    as any of it goes: every route that writes keeps comparing orgs directly,
    so those refuse without needing to know demos or publishing exist.

    ``listed`` is here because the leaderboard put it in front of strangers.
    The first published result produced rows every visitor could see and only
    the owner could open: click, 404, on the most public screen in the
    product. A leaderboard whose entries cannot be read is a claim with the
    evidence withheld, and the whole pitch of this product is that the
    evidence is the point. Publishing therefore shares the report page itself;
    what it never shares is the uploader's contact address, which
    :func:`as_submission` blanks for any reader who is not the owner.
    """
    return sub is not None and (
        sub.org_id == who["org_id"] or bool(sub.demo) or bool(sub.listed)
    )


@app.get("/api/submissions/{sub_id}")
def get_submission(sub_id: str, who=Depends(viewer), session: Session = Depends(db)):
    sweep_stale(session)
    sub = session.get(Submission, sub_id)
    if not readable(sub, who):
        raise HTTPException(404, "no such submission")
    return as_submission(session, sub, deep=True, owner=sub.org_id == who["org_id"])


#: Largest archive we accept, in bytes.
#:
#: A ceiling rather than a guess at what anybody needs: the box this runs on has
#: 12 GB free on its root volume, uploads land there, and each one is then
#: unpacked beside itself. Without a bound, one visitor fills the disk and every
#: other submission fails at intake with an error about our storage, which is
#: our fault presented as theirs.
#:
#: Read from the environment so a host with room can raise it without a deploy.
MAX_UPLOAD_BYTES = int(os.environ.get("BENCH_MAX_UPLOAD_BYTES", 1024**3))


def _store_within(source, target: Path, limit: int) -> int:
    """Copy an upload to disk, stopping the moment it exceeds ``limit``.

    Enforced here as well as on the declared length, because Content-Length is
    a claim the client makes: it can be wrong, absent on a chunked request, or
    a deliberate lie, and the header check alone would let all three through.
    Counting what actually arrives is the only bound that holds.

    The partial file is removed on refusal. Leaving it costs the disk space the
    limit exists to protect, and a half-written archive that no row points at is
    invisible to everything except `df`.
    """
    written = 0
    try:
        with target.open("wb") as out:
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        413,
                        f"this archive is larger than the {limit // 1024**3} GB limit. "
                        "Trim the export, or upload a subset of the episodes: the checks "
                        "read the same things either way",
                    )
                out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    return written


@app.post("/api/submissions/{sub_id}/dataset")
async def upload_dataset(
    sub_id: str,
    file: UploadFile,
    request: Request,
    who=Depends(viewer),
    session: Session = Depends(db),
):
    """Store the archive and queue intake.

    Local disk stands in for object storage; the swap to presigned S3 changes
    this function and nothing that calls it.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")

    # Refuse on the declared size first, so an upload that cannot possibly be
    # accepted is turned away before it spends the visitor's bandwidth rather
    # than after. Not trusted: `_store_within` counts the bytes that arrive.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"this archive is larger than the {MAX_UPLOAD_BYTES // 1024**3} GB limit. "
            "Trim the export, or upload a subset of the episodes: the checks read "
            "the same things either way",
        )

    previous = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == sub_id).order_by(DatasetVersion.version.desc())
    ).first()
    version = (previous.version + 1) if previous else 1

    folder = STORAGE / sub_id / f"v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "dataset.zip"
    stored = _store_within(file.file, target, MAX_UPLOAD_BYTES)

    _activate_version(session, sub, target, stored, version, "dataset.uploaded")
    session.commit()
    return as_submission(session, sub, deep=True)


def _activate_version(
    session: Session,
    sub: Submission,
    target: Path,
    stored: int,
    version: int,
    kind: str,
    **extra,
) -> None:
    """A stored archive becomes the submission's newest version, gates queued.

    One function because there are now two ways bytes arrive -- a browser
    upload and a fetch from Hugging Face -- and the moment they became two
    copies of this block they could disagree about what an upload *is*.

    A new version gets its own gates. The previous version keeps its verdicts,
    which is the whole point of a resubmission: "did what I changed help" is
    unanswerable if running v2 overwrites the v1 it would be compared against.
    """
    row = DatasetVersion(
        id=new_id("dsv"), submission_id=sub.id, version=version, path=str(target), bytes=stored
    )
    session.add(row)
    for spec in GATES:
        session.add(
            Gate(id=new_id("gate"), submission_id=sub.id, key=spec["key"], status="queued", version=version)
        )
    sub.status = "queued"
    sub.current_gate = "g0"
    session.add(
        Job(id=new_id("job"), submission_id=sub.id, gate_key="g0", status="queued", version=version)
    )
    emit(session, sub.id, kind, version=version, bytes=stored, **extra)
    emit(session, sub.id, "gate.queued", gate="g0", version=version)


#: What a Hugging Face dataset id looks like: ``owner/name``, each side the
#: characters the Hub itself allows. An allow-list, not a parser: anything this
#: does not match is refused, so the fetch can never be pointed at an arbitrary
#: URL -- the worker only ever speaks to the Hub, about a repo named here.
_HF_REPO = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})/[A-Za-z0-9._-]{1,96}$")


def _hf_repo(raw: Any) -> str:
    """A Hub dataset id, extracted from whatever shape a visitor pasted.

    People paste the whole URL, the URL with /tree/main on the end, or the bare
    id; all three mean the same dataset and all three are accepted. What is
    never accepted is anything that does not reduce to ``owner/name``.
    """
    text = str(raw or "").strip()
    text = re.sub(r"^https?://(www\.)?huggingface\.co/", "", text)
    text = re.sub(r"^hf\.co/", "", text)
    text = re.sub(r"^datasets/", "", text)
    text = text.split("?")[0].split("#")[0].rstrip("/")
    # Anything after the repo id (tree/main, blob/..., resolve/...) is the Hub's
    # own navigation, not part of the name.
    parts = text.split("/")
    if len(parts) > 2:
        text = "/".join(parts[:2])
    if not _HF_REPO.match(text):
        raise HTTPException(
            422,
            f"{str(raw)[:120]!r} does not look like a Hugging Face dataset. Paste the "
            "dataset page's link, like huggingface.co/datasets/lerobot/pusht, or just "
            "the id, like lerobot/pusht",
        )
    return text


@app.post("/api/submissions/{sub_id}/dataset/hf")
async def fetch_from_hub(sub_id: str, body: dict, who=Depends(viewer), session: Session = Depends(db)):
    """Queue a fetch of a Hub dataset instead of receiving an upload.

    The reason this exists is the launch-day funnel: hundreds of visitors read
    the worked example and not one had a LeRobot archive sitting on the machine
    they were browsing from -- because their datasets live on the Hub. So the
    product meets the data where it is: paste the link, the worker pulls it,
    and everything downstream is the same pipeline an upload feeds.

    The download happens on the worker, not here: it owns the disk the dataset
    lands on, it already heartbeats through long work, and an API request that
    streamed gigabytes from the Hub would hold a connection hostage for
    however long that takes.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")
    repo = _hf_repo(body.get("repo"))

    if session.scalar(
        select(Job).where(
            Job.submission_id == sub_id,
            Job.gate_key == "hf",
            Job.status.in_(("queued", "running")),
        )
    ):
        raise HTTPException(409, "a fetch for this submission is already underway")

    session.add(
        Job(
            id=new_id("job"),
            submission_id=sub_id,
            gate_key="hf",
            status="queued",
            version=0,
            params_json=json.dumps({"repo": repo}),
        )
    )
    sub.status = "fetching"
    emit(session, sub_id, "fetch.queued", repo=repo)
    session.commit()
    return as_submission(session, sub, deep=True)


@app.post("/api/jobs/{job_id}/fetched")
def job_fetched(job_id: str, body: dict, session: Session = Depends(db), _=Depends(worker)):
    """A fetch landed: the archive the worker built becomes a real version.

    Worker-authed, like every door a worker uses. The path is trusted the same
    way finish_job's verdict is trusted -- the worker already holds the token
    that could write anything -- but it is still moved into the submission's
    own folder, so storage keeps one layout however the bytes arrived.
    """
    row = session.get(Job, job_id)
    if row is None or row.gate_key != "hf":
        raise HTTPException(404, "no such fetch")
    sub = session.get(Submission, row.submission_id)
    if sub is None:
        raise HTTPException(404, "no such submission")
    source = Path(str(body.get("path") or ""))
    if not source.is_file():
        raise HTTPException(422, f"no archive at {str(source)[:200]!r}")

    previous = session.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.submission_id == sub.id)
        .order_by(DatasetVersion.version.desc())
    ).first()
    version = (previous.version + 1) if previous else 1
    folder = STORAGE / sub.id / f"v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "dataset.zip"
    if source.resolve() != target.resolve():
        shutil.move(str(source), str(target))
    stored = target.stat().st_size

    row.status = "done"
    row.version = version
    _activate_version(
        session, sub, target, stored, version, "dataset.fetched",
        repo=str(body.get("repo") or "")[:200],
    )
    session.commit()
    return {"ok": True, "version": version}


@app.post("/api/jobs/{job_id}/fetch-failed")
def job_fetch_failed(job_id: str, body: dict, session: Session = Depends(db), _=Depends(worker)):
    """A fetch that cannot produce a dataset, explained to the person waiting.

    The reason lands in the event log because the visitor's page is watching
    it; the submission returns to draft so the upload card comes back and they
    can correct the link or upload instead. A fetch failure is never presented
    through the gate vocabulary -- there is no gate yet, and "refused" is a
    judgement reserved for data we actually read.
    """
    row = session.get(Job, job_id)
    if row is None or row.gate_key != "hf":
        raise HTTPException(404, "no such fetch")
    sub = session.get(Submission, row.submission_id)
    reason = str(body.get("reason") or "the fetch could not complete")[:400]
    row.status = "failed"
    row.error = reason
    if sub is not None and sub.status == "fetching":
        sub.status = "draft"
    emit(session, row.submission_id, "fetch.failed", reason=reason)
    session.commit()
    return {"ok": True}


@app.post("/api/submissions/{sub_id}/meaning")
def confirm_meaning(sub_id: str, body: dict, who=Depends(viewer), session: Session = Depends(db)):
    """The one human step: what the channels *mean*.

    Recorded against the dataset version, because the answer is a property of
    that upload and a v2 may legitimately answer differently.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")
    version = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == sub_id).order_by(DatasetVersion.version.desc())
    ).first()
    if version is None:
        raise HTTPException(400, "no dataset uploaded yet")
    version.meaning_json = json.dumps(body or {})
    emit(session, sub_id, "schema.confirmed", **(body or {}))
    session.commit()
    return as_submission(session, sub, deep=True)


def _samples_dir() -> Path:
    """Where the committed sample datasets are, on a laptop or on a host.

    Served from the repository rather than copied into the web bundle: they are
    24 MB, they are already in the tree, and a second copy is one that can drift
    from the one the README's numbers came from.

    Found by looking rather than by counting parent directories, because the two
    layouts do not agree. In a checkout this file is `bench/api/app/main.py`, so
    the repository root is three parents up and `samples/` sits in it. Deployed,
    it is `gantry_bench/api/app/main.py` while the pipeline and its samples are
    in a sibling tree at `gantry/` -- so three parents up is `/home/ubuntu`, and
    the same arithmetic points at a directory that does not exist.

    That mismatch is invisible locally and produced a download link that
    returned "not in this checkout" on the live site. Both candidates are
    checked, and `BENCH_SAMPLES` overrides for any layout neither describes.
    """
    override = os.environ.get("BENCH_SAMPLES", "").strip()
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[3] / "samples",              # a checkout
        here.parents[3] / "gantry" / "samples",   # deployed beside the pipeline
        here.parents[2] / "samples",
    ):
        if candidate.is_dir():
            return candidate
    return here.parents[3] / "samples"


SAMPLES = _samples_dir()

#: Named explicitly rather than globbed. This route hands out files by name from
#: a directory, which is the shape of every path-traversal bug ever written; an
#: allow-list means the parameter cannot address anything that is not on it.
SAMPLE_FILES = {
    "two_handed": (
        "baseline_plus_ego_two_handed.zip",
        "The same 50, plus ego clips where both hands were tracked",
    ),
    "one_handed": (
        "baseline_plus_ego_one_handed.zip",
        "The same 50, plus ego clips where one hand was mostly absent",
    ),
}


@app.get("/api/samples")
def samples():
    """What can be downloaded, what each one is for, and its finished result.

    ``result`` is the id of the seeded worked example for that dataset, or null
    where none was seeded. It is returned rather than hardcoded in the frontend
    so a deployment without the fixture degrades to offering the download alone,
    instead of linking to a page that 404s.
    """
    seeded = sample_ids()
    out = []
    for key, (filename, what) in SAMPLE_FILES.items():
        path = SAMPLES / filename
        out.append({
            "key": key,
            "filename": filename,
            "what": what,
            "bytes": path.stat().st_size if path.exists() else 0,
            "available": path.exists(),
            "result": seeded.get(key),
        })
    return {"samples": out}


@app.get("/api/samples/{key}")
def sample(key: str):
    entry = SAMPLE_FILES.get(key)
    if entry is None:
        raise HTTPException(404, "no such sample")
    path = SAMPLES / entry[0]
    if not path.exists():
        raise HTTPException(
            404,
            f"{entry[0]} is not in this checkout; it lives in samples/ on the "
            "closed-loop branch",
        )
    return FileResponse(path, media_type="application/zip", filename=entry[0])


@app.post("/api/submissions/{sub_id}/listed")
def set_listed(sub_id: str, body: dict, who=Depends(viewer), session: Session = Depends(db)):
    """Publish this result to the shared leaderboard, or take it back down.

    Reversible on purpose, and in both directions. A benchmark that let you
    publish but never withdraw would make the decision unaskable in practice --
    people would decline rather than risk it -- and the honest reading of a
    withdrawn result is that its owner no longer stands behind it, which is
    information the board is better off not showing.

    Publishing shares the report page itself: the name you gave the submission,
    its rungs, the verdict with its evidence, the findings, and any generated
    advice. It never shares your email, your org id, or anything derived from
    them. The page had to come with the row: the first published result put an
    entry on every visitor's leaderboard that none of them could open, and a
    ranked claim whose evidence is withheld is the thing this product exists
    not to be.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")

    listed = bool(body.get("listed"))
    if listed:
        # Only a finished robot test can be ranked, so only a finished robot
        # test can be published. Without this the switch would appear to work
        # and the entry would never show, which reads as the product losing it.
        done = session.scalar(
            select(Gate).where(
                Gate.submission_id == sub.id,
                Gate.key == "g3",
                Gate.status == "passed",
                Gate.version == latest_version(session, sub.id),
            )
        )
        if done is None:
            raise HTTPException(
                409,
                "the robot test has not produced a result for this version yet, so "
                "there is nothing to rank",
            )

    sub.listed = listed
    emit(session, sub.id, "listed.changed", listed=listed)
    session.commit()
    return as_submission(session, sub, deep=True)


#: How often the stream looks for something new. Also the rate progress frames
#: coalesce at: a worker beating ten times a second still produces one frame
#: per tick here, so a chatty gate cannot flood a browser.
POLL = 1.0

#: Polls with nothing to say before the stream closes. EventSource reconnects
#: on its own, so this is a ceiling on how long one connection is held open,
#: not on how long a user can watch.
IDLE_LIMIT = 900


@app.get("/api/submissions/{sub_id}/events")
async def stream_events(
    sub_id: str,
    request: Request,
    after: int = 0,
    who=Depends(viewer),
    session: Session = Depends(db),
):
    """The live channel. Two frame types, on purpose.

    Scoped like every other route, which it was not: this endpoint took no
    viewer at all, so any submission id streamed its whole event log --
    filenames, gate outcomes, timings -- to anyone who asked. It was the one
    hole in an otherwise consistently scoped API, and the least visible,
    because a browser opening its own stream never exercises the gap.

    ``message`` frames are the durable log -- queued, started, finished. Each
    carries an ``id:``, so a browser that loses its connection reconnects with
    ``Last-Event-ID`` and is sent exactly what it missed. Without the id line,
    EventSource's own resume does nothing and a user who reloads during a
    four-hour run silently loses every event that happened while they were
    away. That is the reload most likely to happen.

    ``progress`` frames are where the running gate is up to. They carry no id
    and are not replayable, which is right: nobody wants a replay of a
    progress bar, and the current position already arrives with the record on
    reconnect. They are read off the gate row rather than a queue, so the
    stream coalesces at its own rate no matter how fast a worker reports --
    a training loop beating ten times a second still produces one frame here.
    """
    # Checked once, before the stream opens, and with the same 404-for-not-yours
    # shape the rest of the API uses: a 403 would confirm the id exists, which
    # is itself worth knowing to somebody guessing.
    owned = session.get(Submission, sub_id)
    if not readable(owned, who):
        raise HTTPException(404, "no such submission")
    # Released NOW, not when the stream ends. The dependency session stays
    # open for the whole response, and for a stream that is up to fifteen
    # minutes -- so every open report page held one pooled connection doing
    # nothing, and the fifteenth concurrent visitor found the pool empty and
    # was answered 500. The loop below opens its own short session per tick.
    session.close()

    async def gen():
        last = int(request.headers.get("last-event-id") or after or 0)
        seen: str | None = None
        idle = 0
        while idle < IDLE_LIMIT and not await request.is_disconnected():
            frames: list[str] = []
            with SessionLocal() as session:
                sweep_stale(session)
                rows = session.scalars(
                    select(Event).where(Event.submission_id == sub_id, Event.id > last).order_by(Event.id)
                ).all()
                for row in rows:
                    last = row.id
                    payload = {"id": row.id, "ts": row.ts, "kind": row.kind, **json.loads(row.payload_json or "{}")}
                    frames.append(f"id: {row.id}\ndata: {json.dumps(payload)}\n\n")

                running = session.scalars(
                    select(Gate).where(Gate.submission_id == sub_id, Gate.status == "running")
                ).first()
                current = (
                    json.dumps({"gate": running.key, **json.loads(running.progress_json or "{}")})
                    if running
                    else None
                )
                # Only on change. An unchanged bar re-sent every second is a
                # heartbeat wearing a progress bar's clothes, and it makes a
                # stalled stage look busy.
                if current != seen:
                    seen = current
                    if current:
                        frames.append(f"event: progress\ndata: {current}\n\n")

            for frame in frames:
                yield frame
            if frames:
                idle = 0
            else:
                idle += 1
                yield ": keep-alive\n\n"
            await asyncio.sleep(POLL)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# worker-facing: claim, heartbeat, finish
# ---------------------------------------------------------------------------


@app.post("/api/submissions/{sub_id}/gates/{key}/start")
def start_gate(sub_id: str, key: str, body: dict | None = None, who=Depends(viewer), session: Session = Depends(db)):
    """Buy a gate. The only way a paid gate ever runs.

    Free gates queue themselves when the one before them passes. Paid ones do
    not, and this asymmetry is the product: a contributor is never billed for
    work they did not ask for, and the decision is made at the gate rather than
    guessed at upload time.

    Guarded by what has already happened, not by what the caller says. Starting
    the robot test on a submission whose data report has not run would produce a
    real bill for a real day of GPU on footage nobody has looked at.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")
    spec = next((g for g in GATES if g["key"] == key), None)
    if spec is None:
        raise HTTPException(404, "no such gate")

    order = [g["key"] for g in GATES]
    version = latest_version(session, sub_id)
    gates = {g.key: g for g in gates_for(session, sub_id, version)}
    gate = gates.get(key)
    if gate is None:
        raise HTTPException(404, "this submission has no such gate")
    if gate.status != "queued":
        raise HTTPException(409, f"the {spec['name'].lower()} has already run ({gate.status})")

    # An abstention is not a refusal, and this is the one place the difference
    # had been lost. "Not separated" is a statement about how much footage there
    # was, not about the footage itself, so a contributor whose signal check
    # could not conclude may still choose to buy the run that can. Blocking them
    # treats "we could not tell" as "no", which is precisely the misreading
    # signal.py's own docstring says loses a customer who was right.
    #
    # `refused` and `failed` still block, and for different reasons: the first
    # is a judgement on the data that must not be re-rolled, the second is our
    # machinery, which should be retried rather than spent around.
    before = order[: order.index(key)]
    ready = ("passed", "abstained")
    unfinished = [k for k in before if gates.get(k) and gates[k].status not in ready]
    if unfinished:
        names = ", ".join(next(g["name"] for g in GATES if g["key"] == k).lower() for k in unfinished)
        raise HTTPException(409, f"the {names} has not passed yet")

    if session.scalar(
        select(Job).where(Job.submission_id == sub_id, Job.gate_key == key, Job.status.in_(("queued", "running")))
    ):
        raise HTTPException(409, "that gate is already queued")

    # What is being bought. For a sized gate the trial count decides what the
    # run can conclude, so it is validated here rather than taken on trust: an
    # experiment too small to answer its own question should not be sellable,
    # and the price is recomputed from the count rather than accepted from the
    # caller, who would otherwise be quoting their own bill.
    body = body or {}
    if spec["sized"]:
        trials = int(body.get("trials") or 0)
        if trials < FLOOR_TRIALS:
            raise HTTPException(
                400,
                f"{trials or 'no'} scenes is below the floor of {FLOOR_TRIALS}; at that "
                "size no effect can be separated from noise, whatever the data is like",
            )
        bench = session.get(Benchmark, sub.benchmark_id)
        gate.trials = trials
        gate.cost_cents = costing(bench, trials)["cents"] if bench else spec["cost_cents"]
    else:
        gate.trials = 0
        gate.cost_cents = int(body.get("cost_cents", spec["cost_cents"]))

    session.add(Job(id=new_id("job"), submission_id=sub_id, gate_key=key, status="queued"))
    sub.status = "running"
    emit(
        session,
        sub_id,
        "gate.queued",
        gate=key,
        bought=True,
        cost_cents=gate.cost_cents,
        **({"trials": gate.trials} if gate.trials else {}),
    )
    session.commit()
    return as_submission(session, sub, deep=True)


#: Times a gate may be retried before the product stops offering it. A gate
#: that has broken three times is broken in a way another attempt will not fix,
#: and a button that keeps being offered invites somebody to keep pressing it.
MAX_ATTEMPTS = 3


@app.post("/api/submissions/{sub_id}/gates/{key}/retry")
def retry_gate(sub_id: str, key: str, who=Depends(viewer), session: Session = Depends(db)):
    """Run a gate again after *our* machinery broke.

    Only ``failed``, and that restriction is the point of the endpoint rather
    than a detail of it.

    ``failed`` means our worker died, our disk filled, our runner could not find
    its trainer. Nothing about the contributor's data was judged, so running it
    again is finishing work already paid for.

    ``refused`` means the data was judged and did not pass. Offering a retry
    there would be offering to re-roll until the answer is liked, which is the
    one thing a benchmark cannot let you do -- and it would be worse here than
    in most places, because the gates that refuse are the cheap ones a person
    could afford to spin all afternoon.

    The bought size is kept. A retry is the same experiment, not a new one, and
    a gate that quietly came back at a different trial count would make the
    verdict describe a run nobody ordered.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")
    version = latest_version(session, sub_id)
    gate = session.scalar(
        select(Gate).where(Gate.submission_id == sub_id, Gate.key == key, Gate.version == version)
    )
    if gate is None:
        raise HTTPException(404, "this submission has no such gate")
    if gate.status != "failed":
        raise HTTPException(
            409,
            f"only a gate that failed on our side can be run again; this one is {gate.status!r}"
            + (
                ". A refusal is a judgement on the data, and re-rolling it until the answer "
                "changes is not something a benchmark can offer"
                if gate.status == "refused"
                else ""
            ),
        )

    attempts = session.scalars(
        select(Job).where(Job.submission_id == sub_id, Job.gate_key == key)
    ).all()
    if len(attempts) >= MAX_ATTEMPTS:
        raise HTTPException(
            409,
            f"this gate has been attempted {len(attempts)} times; something is wrong that "
            "another run will not fix",
        )

    gate.status = "queued"
    gate.verdict_json = "{}"
    gate.findings_json = "[]"
    gate.detail_json = "{}"
    gate.progress_json = "{}"
    gate.started_at = gate.finished_at = ""
    # A new job rather than reviving the old one, so the failed attempt stays in
    # the record. A run whose history is edited to look clean is not auditable.
    session.add(
        Job(id=new_id("job"), submission_id=sub_id, gate_key=key, status="queued", version=version)
    )
    sub.status = "running"
    emit(session, sub_id, "gate.retried", gate=key, attempt=len(attempts) + 1)
    session.commit()
    return as_submission(session, sub, deep=True)


@app.post("/api/submissions/{sub_id}/for-worker")
def submission_for_worker(sub_id: str, body: dict, session: Session = Depends(db), _=Depends(worker)):
    """The full record, for the worker that is coaching it.

    Worker-authed, not viewer-scoped: the worker has no cookie and already
    holds the dataset itself, so the record is not an escalation. POST because
    the worker's transport only speaks POST, and a second verb there for one
    read would be ceremony.
    """
    sub = session.get(Submission, sub_id)
    if sub is None:
        raise HTTPException(404, "no such submission")
    return as_submission(session, sub, deep=True)


@app.post("/api/submissions/{sub_id}/coach")
def store_coach(sub_id: str, body: dict, session: Session = Depends(db), _=Depends(worker)):
    """Store generated advice. Worker-authed: visitors read this, never write it.

    At most four points, enforced here as well as at generation, because this
    is the door and the generator is merely one caller of it.
    """
    sub = session.get(Submission, sub_id)
    if sub is None:
        raise HTTPException(404, "no such submission")
    points = []
    for p in body.get("points") or []:
        if isinstance(p, dict):
            title = str(p.get("title", "")).strip()[:120]
            detail = str(p.get("detail", "")).strip()[:600]
        else:
            title, detail = str(p).strip()[:120], ""
        if title:
            points.append({"title": title, "detail": detail})
        if len(points) == 3:
            break
    # Per-finding one-liners, keyed by finding code. Same posture as points:
    # caps are enforced at this door because the generator is one caller of it,
    # and a wall of text stored here becomes a wall of text on the page.
    raw = body.get("fixes") or {}
    fixes = {}
    if isinstance(raw, dict):
        for code, entry in list(raw.items())[:16]:
            if isinstance(entry, dict):
                say = str(entry.get("say", "")).strip()[:160]
                detail = str(entry.get("detail", "") or entry.get("do", "")).strip()[:400]
            else:
                say, detail = "", str(entry).strip()[:400]
            if say or detail:
                fixes[str(code)[:64]] = {"say": say, "detail": detail}
    if not points and not fixes:
        raise HTTPException(422, "nothing to store")
    sub.coach_json = json.dumps(
        {"points": points, "fixes": fixes, "model": str(body.get("model") or "")}
    )
    emit(session, sub_id, "feedback.written", points=len(points))
    session.commit()
    return {"stored": len(points), "fixes": len(fixes)}


@app.post("/api/jobs/claim")
def claim_job(body: dict, session: Session = Depends(db), _=Depends(worker)):
    """One job, claimed atomically.

    SQLite has no SKIP LOCKED, so the claim is a guarded UPDATE and the worker
    re-reads to confirm it won. On Postgres this becomes SELECT ... FOR UPDATE
    SKIP LOCKED and nothing else changes.
    """
    worker = body.get("worker", "worker")
    kinds = body.get("gates") or ["g0"]
    row = session.scalars(
        select(Job).where(Job.status == "queued", Job.gate_key.in_(kinds)).order_by(Job.created_at)
    ).first()
    if row is None:
        return {"job": None}
    # The guard this function's docstring has always described. A plain
    # SELECT-then-mutate lets two workers read the same queued row and both
    # return it, and the damage is not a duplicated run: the loser crashes or
    # finishes second and writes its result onto a gate the winner already owns,
    # so a gate reads `failed` for a reason that is nowhere in its own record.
    # Restricting the UPDATE to rows still `queued` means exactly one caller can
    # change it; rowcount is how we learn whether we were that caller.
    claimed = session.execute(
        update(Job)
        .where(Job.id == row.id, Job.status == "queued")
        .values(status="running", claimed_by=worker, attempts=Job.attempts + 1, heartbeat_at=now())
    )
    if claimed.rowcount != 1:
        # Someone else took it between the select and the update. Not an error,
        # and not worth retrying inside one request -- the worker polls again.
        session.rollback()
        return {"job": None}
    session.refresh(row)
    # A fetch job has no gate row -- the version it will produce does not exist
    # yet -- so everything gate-shaped below is conditional on there being one.
    gate = session.scalar(
        select(Gate).where(
            Gate.submission_id == row.submission_id, Gate.key == row.gate_key, Gate.version == row.version
        )
    )
    if gate is not None:
        gate.status = "running"
        gate.started_at = now()
        emit(session, row.submission_id, "gate.started", gate=row.gate_key)
    else:
        emit(session, row.submission_id, "fetch.started", job=row.id)
    sub = session.get(Submission, row.submission_id)
    sub.status = "running" if gate is not None else sub.status
    sub.current_gate = row.gate_key if gate is not None else sub.current_gate
    version = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == row.submission_id).order_by(DatasetVersion.version.desc())
    ).first()
    # What the buyer chose, or what the fetch was asked to pull. Carried on
    # the job so the run is the size it was sold as -- "no difference found"
    # means nothing without the trial count beside it -- and so a fetch knows
    # its repo without a second request.
    params = json.loads(row.params_json or "{}")
    if gate is not None and gate.trials:
        params["trials"] = gate.trials
    session.commit()
    return {
        "job": {
            "id": row.id,
            "submission_id": row.submission_id,
            "gate_key": row.gate_key,
            # Both, deliberately. A worker sharing this disk opens the path and
            # copies nothing; one on another machine fetches the URL. Same job,
            # same code path, and the fast case stays fast.
            "archive": version.path if version else None,
            "archive_url": f"/api/jobs/{row.id}/archive" if version else None,
            "archive_bytes": version.bytes if version else 0,
            "params": params,
            "version": version.version if version else 0,
            "workdir": str(STORAGE / row.submission_id / f"v{version.version}") if version else None,
        }
    }


@app.get("/api/jobs/{job_id}/archive")
def job_archive(job_id: str, session: Session = Depends(db), _=Depends(worker)):
    """The dataset, for the worker that claimed this job.

    The reason a worker can live somewhere else. Until this existed the job
    handed out a filesystem path, which only worked because the API and the
    worker shared a disk -- so the GPU box, which is the entire point of the
    paid gates, could not run one.

    Scoped to the job rather than to the submission: a worker is given exactly
    the artefact for the work it claimed, and holds no standing access to
    anything. In production this becomes a presigned URL to object storage and
    the worker code does not change.
    """
    row = session.get(Job, job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    version = session.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.submission_id == row.submission_id)
        .order_by(DatasetVersion.version.desc())
    ).first()
    if version is None or not Path(version.path).exists():
        raise HTTPException(404, "no dataset stored for that job")
    # Released before the send, which can take minutes for a multi-gigabyte
    # archive on a slow link. The connection would otherwise sit idle in a
    # transaction for the whole transfer, which is the same pool leak the
    # event stream had, on the route that holds it longest.
    path, name = version.path, Path(version.path).name
    session.close()
    return FileResponse(path, filename=name)


@app.post("/api/jobs/{job_id}/finish")
def finish_job(job_id: str, body: dict, session: Session = Depends(db), _=Depends(worker)):
    row = session.get(Job, job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    status = body.get("status", "failed")
    # By the job's version, not the submission's latest. A worker finishing a
    # long v1 run after v2 was uploaded would otherwise write its verdict onto
    # v2 -- a result about footage it never saw.
    gate = session.scalar(
        select(Gate).where(
            Gate.submission_id == row.submission_id, Gate.key == row.gate_key, Gate.version == row.version
        )
    )
    if gate is None:
        # A job with no gate is a fetch that came through the generic error
        # path. Record the failure on the job and the log, and put the
        # submission back where the visitor can act on it.
        row.status = "failed"
        row.error = body.get("error", "")[:400]
        sub = session.get(Submission, row.submission_id)
        if sub is not None and sub.status == "fetching":
            sub.status = "draft"
        emit(session, row.submission_id, "fetch.failed",
             reason=(body.get("verdict") or {}).get("summary", "the fetch could not complete"))
        session.commit()
        return {"ok": True}
    gate.status = status
    gate.finished_at = now()
    gate.verdict_json = json.dumps(body.get("verdict") or {})
    gate.findings_json = json.dumps(body.get("findings") or [])
    gate.measures_json = json.dumps(body.get("measures") or {})
    gate.abstained_json = json.dumps(body.get("abstained") or [])
    gate.detail_json = json.dumps(body.get("detail") or {})
    row.status = "done" if status in ("passed", "refused", "abstained") else "failed"
    row.error = body.get("error", "")

    if body.get("detected"):
        version = session.scalars(
            select(DatasetVersion).where(DatasetVersion.submission_id == row.submission_id).order_by(DatasetVersion.version.desc())
        ).first()
        if version is not None:
            version.detected_json = json.dumps(body["detected"])

    sub = session.get(Submission, row.submission_id)
    # A refusal or our own failure ends the run; a pass parks the submission
    # until the user buys the next gate.
    sub.status = {"passed": "awaiting_user", "refused": "refused", "abstained": "abstained"}.get(status, "failed")

    # A free gate runs on its own. The contributor gets a real report on their
    # filming before being asked to decide anything or spend anything, which is
    # the whole reason the cheap gates come first.
    #
    # It stops at the first gate that is a *decision*, and `sized` is what marks
    # one: the robot test runs as many scenes as you ask for, and how many you
    # ask for decides what the run can conclude. `cost_cents` used to be the
    # test, which was the same thing while the gates were priced and became
    # nothing at all the moment every price went to zero. On a host with
    # BENCH_RUNNER set that left a passing signal check starting hours of GPU
    # with no human anywhere in the loop, on a public URL, billed to nobody.
    #
    # Price is a policy that has changed once and will change again. Whether a
    # gate has a knob on it is a fact about the gate.
    if status == "passed":
        order = [g["key"] for g in GATES]
        following = order[order.index(row.gate_key) + 1 :]
        for key in following:
            spec = next(g for g in GATES if g["key"] == key)
            if spec["cost_cents"] > 0 or spec["sized"]:
                break
            session.add(
                Job(id=new_id("job"), submission_id=row.submission_id, gate_key=key,
                    status="queued", version=row.version)
            )
            emit(session, row.submission_id, "gate.queued", gate=key)
            sub.status = "running"
            break

    emit(
        session,
        row.submission_id,
        "gate.finished",
        gate=row.gate_key,
        status=status,
        summary=(body.get("verdict") or {}).get("summary", ""),
    )
    session.commit()
    return {"ok": True}


#: What a worker may say about where it is. Anything else is dropped rather
#: than stored, so a gate cannot smuggle arbitrary state through the progress
#: channel and have the UI grow a special case for it.
#:
#: ``total`` is allowed to be absent. A stage that genuinely does not know how
#: much work is left says so and gets an indeterminate bar; inventing a
#: denominator to make the bar move is the one thing this must not do.
PROGRESS_FIELDS = ("phase", "current", "total", "note")


@app.post("/api/jobs/{job_id}/heartbeat")
def heartbeat(job_id: str, body: dict, session: Session = Depends(db), _=Depends(worker)):
    """Liveness, and optionally where the gate is up to.

    Progress overwrites in place and never appends to the event log. A
    three-thousand-step training run would otherwise put three thousand rows
    into a log whose entire value is that a person can read it.
    """
    row = session.get(Job, job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    row.heartbeat_at = now()
    progress = body.get("progress")
    if progress:
        gate = session.scalar(
            select(Gate).where(Gate.submission_id == row.submission_id, Gate.key == row.gate_key)
        )
        if gate is not None:
            # Empty is absent. A note of "" is not a note, and passing it
            # through means the UI renders a blank element and has to learn to
            # check for one.
            gate.progress_json = json.dumps(
                {
                    k: progress[k]
                    for k in PROGRESS_FIELDS
                    if progress.get(k) is not None and progress[k] != ""
                }
            )
    session.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if WEB_DIST.exists():  # pragma: no cover - only in a built deployment
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """The SPA, or a real file sitting beside it at the root of the bundle.

        Mounting ``/assets`` alone is not enough, and the way it fails is
        peculiarly hard to see. Vite emits hashed bundles into ``dist/assets``
        but copies everything in ``public/`` to the *root* of ``dist`` -- so
        ``/hero-rig.jpg`` matched nothing, fell through to this handler, and was
        answered with ``index.html`` carrying HTTP 200 and ``text/html``. The
        browser asked for an image, received a web page, and drew nothing. No
        404 anywhere, no console error worth noticing, and it works perfectly in
        development because there Vite serves ``public/`` itself.

        So: if the path names a file that actually exists inside the bundle,
        return that file; otherwise fall back to the app, which is what makes
        client-side routes like /submissions/abc work on a hard reload.
        """
        if full_path:
            candidate = (WEB_DIST / full_path).resolve()
            # Confined to the bundle. `resolve()` collapses any `..` before the
            # check, so a crafted path cannot climb out of it and read the
            # database or the env file next door.
            try:
                inside = candidate.is_relative_to(WEB_DIST.resolve())
            except AttributeError:  # pragma: no cover - Python < 3.9
                inside = str(candidate).startswith(str(WEB_DIST.resolve()))
            if inside and candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
