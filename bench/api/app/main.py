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
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    STORAGE,
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
#: The gauntlet. ``sized`` marks the gate whose price is a *choice*: the robot
#: test runs as many scenes as you buy, and how many you buy decides what the
#: run can conclude. The others are fixed work at a fixed price, and offering a
#: trial slider for them would be a control over nothing.
GATES = [
    {"key": "g0", "name": "Intake", "question": "Can we read this at all?", "cost_cents": 0, "eta": "seconds", "sized": False},
    {"key": "g1", "name": "Data report", "question": "What is this footage like?", "cost_cents": 0, "eta": "about a minute", "sized": False},
    {"key": "g2", "name": "Signal check", "question": "Is there anything learnable here?", "cost_cents": 100, "eta": "about ten minutes", "sized": False},
    {"key": "g3", "name": "Robot test", "question": "Does the robot actually get better?", "cost_cents": 3400, "eta": "a few hours", "sized": True},
]


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


def as_gate(row: Gate) -> dict:
    spec = next(g for g in GATES if g["key"] == row.key)
    return {
        "key": row.key,
        "name": spec["name"],
        "question": spec["question"],
        "eta": spec["eta"],
        "cost_cents": spec["cost_cents"],
        "sized": spec["sized"],
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
    gates = session.scalars(
        select(Gate).where(Gate.submission_id == sub.id)
    ).all()
    order = {g["key"]: i for i, g in enumerate(GATES)}
    gates = sorted(gates, key=lambda g: order.get(g.key, 99))
    version = session.scalars(
        select(DatasetVersion).where(DatasetVersion.submission_id == sub.id).order_by(DatasetVersion.version.desc())
    ).first()
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
        session.add(Gate(id=new_id("gate"), submission_id=sub.id, key=spec["key"], status="queued"))
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

    gate = session.scalar(select(Gate).where(Gate.submission_id == sub_id, Gate.key == "g0"))
    gate.status = "queued"
    sub.status = "queued"
    sub.current_gate = "g0"
    session.add(Job(id=new_id("job"), submission_id=sub_id, gate_key="g0", status="queued"))
    emit(session, sub_id, "dataset.uploaded", version=version, bytes=row.bytes)
    emit(session, sub_id, "gate.queued", gate="g0")
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
    gates = {g.key: g for g in session.scalars(select(Gate).where(Gate.submission_id == sub_id)).all()}
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

    gate.cost_cents = int((body or {}).get("cost_cents", spec["cost_cents"]))
    session.add(Job(id=new_id("job"), submission_id=sub_id, gate_key=key, status="queued"))
    sub.status = "running"
    emit(session, sub_id, "gate.queued", gate=key, bought=True, cost_cents=gate.cost_cents)
    session.commit()
    return as_submission(session, sub, deep=True)


@app.post("/api/jobs/claim")
def claim_job(body: dict, session: Session = Depends(db)):
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
    row.status = "running"
    row.claimed_by = worker
    row.attempts += 1
    row.heartbeat_at = now()
    gate = session.scalar(select(Gate).where(Gate.submission_id == row.submission_id, Gate.key == row.gate_key))
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
            "archive": version.path if version else None,
            "workdir": str(STORAGE / row.submission_id / f"v{version.version}") if version else None,
        }
    }


@app.post("/api/jobs/{job_id}/finish")
def finish_job(job_id: str, body: dict, session: Session = Depends(db)):
    row = session.get(Job, job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    status = body.get("status", "failed")
    gate = session.scalar(select(Gate).where(Gate.submission_id == row.submission_id, Gate.key == row.gate_key))
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
            session.add(Job(id=new_id("job"), submission_id=row.submission_id, gate_key=key, status="queued"))
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
def heartbeat(job_id: str, body: dict, session: Session = Depends(db)):
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
