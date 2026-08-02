"""The API. Thin, honest, and it never computes a result of its own.

Every number the UI shows was written by a worker into ``gates`` or
``dataset_versions``. The API's job is to hold state, hand out upload slots,
queue work, and stream the event log. When the product later moves to Postgres
and S3, only ``db.py`` and the two storage helpers change.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db import (
    STORAGE,
    WORKER_TOKEN,
    Benchmark,
    DatasetVersion,
    Event,
    Gate,
    Job,
    Membership,
    Org,
    SessionLocal,
    Submission,
    User,
    emit,
    init_db,
    new_id,
    now,
    sweep_stale,
)

app = FastAPI(title="Gantry Bench API", version="1")

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


def viewer(session: Session = Depends(db), x_demo_user: str | None = Header(default=None)):
    """Development auth: one seeded org, header override for a second identity.

    Deliberately a single seam. Real auth replaces this function and nothing
    else -- every route already asks "who is this and which org" through it.
    """
    email = x_demo_user or "demo@lab.example"
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(id=new_id("usr"), email=email)
        org = Org(id=new_id("org"), name=email.split("@")[1].split(".")[0].title() + " Lab")
        session.add_all([user, org, Membership(user_id=user.id, org_id=org.id, role="owner")])
        session.commit()
    org_id = session.scalar(select(Membership.org_id).where(Membership.user_id == user.id))
    return {"user": user, "org_id": org_id}


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


def as_submission(session: Session, sub: Submission, deep: bool = False) -> dict:
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


@app.get("/api/me")
def me(who=Depends(viewer), session: Session = Depends(db)):
    org = session.get(Org, who["org_id"])
    return {"email": who["user"].email, "org": {"id": org.id, "name": org.name}}


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
    if sub is None or sub.org_id != who["org_id"]:
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

    rows = []
    available: list[str] = []
    for sub in session.scalars(
        select(Submission).where(Submission.org_id == who["org_id"], Submission.benchmark_id == bench.id)
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
        rows.append({
            "id": sub.id,
            "name": sub.name,
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
    rows = session.scalars(
        select(Submission).where(Submission.org_id == who["org_id"]).order_by(Submission.created_at.desc())
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
        created_by=who["user"].id,
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


@app.get("/api/submissions/{sub_id}")
def get_submission(sub_id: str, who=Depends(viewer), session: Session = Depends(db)):
    sweep_stale(session)
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")
    return as_submission(session, sub, deep=True)


@app.post("/api/submissions/{sub_id}/dataset")
async def upload_dataset(sub_id: str, file: UploadFile, who=Depends(viewer), session: Session = Depends(db)):
    """Store the archive and queue intake.

    Local disk stands in for object storage; the swap to presigned S3 changes
    this function and nothing that calls it.
    """
    sub = session.get(Submission, sub_id)
    if sub is None or sub.org_id != who["org_id"]:
        raise HTTPException(404, "no such submission")

    previous = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == sub_id).order_by(DatasetVersion.version.desc())
    ).first()
    version = (previous.version + 1) if previous else 1

    folder = STORAGE / sub_id / f"v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "dataset.zip"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    row = DatasetVersion(
        id=new_id("dsv"), submission_id=sub_id, version=version, path=str(target), bytes=target.stat().st_size
    )
    session.add(row)

    # A new upload gets its own gates. The previous version keeps its verdicts,
    # which is the whole point of a resubmission: "did what I changed help" is
    # unanswerable if running v2 overwrites the v1 it would be compared against.
    for spec in GATES:
        session.add(
            Gate(id=new_id("gate"), submission_id=sub_id, key=spec["key"], status="queued", version=version)
        )
    sub.status = "queued"
    sub.current_gate = "g0"
    session.add(
        Job(id=new_id("job"), submission_id=sub_id, gate_key="g0", status="queued", version=version)
    )
    emit(session, sub_id, "dataset.uploaded", version=version, bytes=row.bytes)
    emit(session, sub_id, "gate.queued", gate="g0", version=version)
    session.commit()
    return as_submission(session, sub, deep=True)


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


#: How often the stream looks for something new. Also the rate progress frames
#: coalesce at: a worker beating ten times a second still produces one frame
#: per tick here, so a chatty gate cannot flood a browser.
POLL = 1.0

#: Polls with nothing to say before the stream closes. EventSource reconnects
#: on its own, so this is a ceiling on how long one connection is held open,
#: not on how long a user can watch.
IDLE_LIMIT = 900


@app.get("/api/submissions/{sub_id}/events")
async def stream_events(sub_id: str, request: Request, after: int = 0):
    """The live channel. Two frame types, on purpose.

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

    before = order[: order.index(key)]
    unfinished = [k for k in before if gates.get(k) and gates[k].status not in ("passed",)]
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
    gate = session.scalar(
        select(Gate).where(
            Gate.submission_id == row.submission_id, Gate.key == row.gate_key, Gate.version == row.version
        )
    )
    gate.status = "running"
    gate.started_at = now()
    sub = session.get(Submission, row.submission_id)
    sub.status = "running"
    sub.current_gate = row.gate_key
    version = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == row.submission_id).order_by(DatasetVersion.version.desc())
    ).first()
    emit(session, row.submission_id, "gate.started", gate=row.gate_key)
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
            # What the buyer chose. Carried to the gate so the run is the size
            # it was sold as, and so the verdict can say which number produced
            # it -- "no difference found" means nothing without the trial count
            # beside it.
            "params": {"trials": gate.trials} if gate.trials else {},
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
    return FileResponse(version.path, filename=Path(version.path).name)


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
    # the whole reason the cheap gates come first. A gate that costs money is
    # never started without the user buying it.
    if status == "passed":
        order = [g["key"] for g in GATES]
        following = order[order.index(row.gate_key) + 1 :]
        for key in following:
            spec = next(g for g in GATES if g["key"] == key)
            if spec["cost_cents"] > 0:
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
        return FileResponse(WEB_DIST / "index.html")
