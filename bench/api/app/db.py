"""The product's state, all of it, in one schema.

SQLite for development, Postgres in production -- the engine URL is the only
difference, so nothing here uses a feature one has and the other lacks. All
JSON lives in TEXT columns serialised by the app: portable, inspectable, and
boring.

Two tables carry the product's honesty guarantees:

* ``gates`` -- one row per gate per submission, with a verdict from the same
  closed vocabulary the UI pills use: queued / running / passed / refused /
  abstained / failed. ``failed`` means *our* machinery broke and is never
  presented as a fact about the user's data.
* ``events`` -- append-only per submission. The live timeline and the activity
  history are two readers of this one log, so they cannot disagree.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

BENCH_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BENCH_DATA", BENCH_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = DATA_DIR / "storage"
STORAGE.mkdir(exist_ok=True)

#: Shared secret a worker presents to claim jobs and fetch datasets. Unset
#: means open, which is right for a laptop talking to its own API and wrong for
#: anything reachable from another machine -- so the moment a GPU box is
#: attached, this is the thing to set on both ends.
WORKER_TOKEN = os.environ.get("BENCH_WORKER_TOKEN", "")

DB_URL = os.environ.get("BENCH_DB", f"sqlite:///{DATA_DIR}/bench.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

if DB_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(conn, _):  # pragma: no cover - driver hook
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


class Base(DeclarativeBase):
    pass


class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    #: Which visitor this org belongs to, as a salted hash of their address --
    #: never the address itself. Unique, so two requests from one visitor can
    #: never race into two orgs and split their own submissions in half.
    #: Nullable because orgs created before IP identity have no visitor.
    ip_hash: Mapped[str | None] = mapped_column(String, unique=True, nullable=True, index=True)
    created_at: Mapped[str] = mapped_column(String, default=now)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[str] = mapped_column(String, default=now)


class Membership(Base):
    __tablename__ = "memberships"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    org_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String, default="member")


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    key: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str] = mapped_column(String)
    task: Mapped[str] = mapped_column(String)
    embodiment: Mapped[str] = mapped_column(String)
    simulator: Mapped[str] = mapped_column(String)
    #: Reference numbers a user calibrates hope against, JSON.
    reference_json: Mapped[str] = mapped_column(Text, default="{}")
    #: What an hour of this benchmark actually costs us, measured from our own
    #: runs rather than quoted from a price list. Kept per benchmark because a
    #: task with a 1,500-step limit costs four times what a 400-step one does,
    #: and quoting one price for both would be wrong in whichever direction the
    #: customer noticed.
    cost_json: Mapped[str] = mapped_column(Text, default="{}")


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String)
    benchmark_id: Mapped[str] = mapped_column(String)
    #: draft -> queued -> running -> a terminal gate verdict for the whole
    #: submission (intake_complete for build step 1).
    status: Mapped[str] = mapped_column(String, default="draft")
    current_gate: Mapped[str] = mapped_column(String, default="")
    created_by: Mapped[str] = mapped_column(String)
    #: Where to reach whoever uploaded this, if we need to.
    #:
    #: Optional, and stored in the clear because an address you cannot read is
    #: an address you cannot write to -- unlike the visitor's IP next door,
    #: which is hashed precisely because nothing here ever needs to recover it.
    #: It exists for one purpose: a run that breaks hours in, on our side, with
    #: nobody watching. It is also the only way back to a submission after the
    #: uploader's address changes, since identity here is the address.
    email: Mapped[str] = mapped_column(String, default="")
    #: On the shared leaderboard, or not. Off until its owner turns it on.
    listed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[str] = mapped_column(String, default=now)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    submission_id: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    path: Mapped[str] = mapped_column(String)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    #: What intake *detected*, JSON -- episodes, frames, fps, channels.
    detected_json: Mapped[str] = mapped_column(Text, default="{}")
    #: What the user *confirmed* the channels mean, JSON. The one human step.
    meaning_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[str] = mapped_column(String, default=now)


class Gate(Base):
    __tablename__ = "gates"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    submission_id: Mapped[str] = mapped_column(String, index=True)
    #: Which upload this gate judged. The product loop is submit, fix, resubmit,
    #: see the change -- and that last step is only possible if v1's results
    #: survive v2 being run. Gates used to be per submission, so a second upload
    #: silently overwrote the results it was supposed to be compared against.
    version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    key: Mapped[str] = mapped_column(String)  # g0 g1 g2 g3
    status: Mapped[str] = mapped_column(String, default="queued")
    #: Headline result, JSON: one line + the facts the row shows collapsed.
    verdict_json: Mapped[str] = mapped_column(Text, default="{}")
    findings_json: Mapped[str] = mapped_column(Text, default="[]")
    #: Per-gate numbers the UI charts rather than lists. Kept apart from
    #: findings because a measurement is not a complaint, and a gate that
    #: measures a lot while flagging nothing is a good outcome, not an empty one.
    measures_json: Mapped[str] = mapped_column(Text, default="{}")
    #: Modules that declined, with their reason. Never dropped: a report that
    #: silently omits what it could not judge reads as a clean bill of health.
    abstained_json: Mapped[str] = mapped_column(Text, default="[]")
    #: The gate's own working, for anything that wants to re-derive the verdict:
    #: per-scene outcomes, discordant counts, p-values, which arms ran. Kept
    #: apart from ``measures`` because a lab checking our arithmetic needs the
    #: rows, not the summary.
    detail_json: Mapped[str] = mapped_column(Text, default="{}")
    #: Where the running gate is up to, overwritten in place.
    #:
    #: Deliberately *not* in the event log. Progress is volatile and high rate --
    #: a three-thousand-step training run would append three thousand rows to a
    #: log whose whole value is that a person can read it. The log keeps the
    #: narrative (queued, started, finished); this keeps the current position,
    #: and the stream polls it. It also means a reload during a four-hour run
    #: shows where the run actually is rather than an empty bar.
    progress_json: Mapped[str] = mapped_column(Text, default="{}")
    started_at: Mapped[str] = mapped_column(String, default="")
    finished_at: Mapped[str] = mapped_column(String, default="")
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    #: Scenes bought, for a gate whose price is a choice. Zero on the rest.
    #:
    #: Kept on the gate rather than the submission because a resubmission may
    #: legitimately buy a different number, and a verdict is meaningless without
    #: it: "no difference found" says nothing until you know how many scenes
    #: were run, since the trial count is what decides which differences the run
    #: could ever have seen.
    trials: Mapped[int] = mapped_column(Integer, default=0)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    submission_id: Mapped[str] = mapped_column(String, index=True)
    #: The upload this job is for, so a worker finishing late cannot write its
    #: verdict onto a newer one.
    version: Mapped[int] = mapped_column(Integer, default=1)
    gate_key: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="queued", index=True)
    claimed_by: Mapped[str] = mapped_column(String, default="")
    heartbeat_at: Mapped[str] = mapped_column(String, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String, default=now)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(String, index=True)
    ts: Mapped[str] = mapped_column(String, default=now)
    kind: Mapped[str] = mapped_column(String)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")


def emit(session: Session, submission_id: str, kind: str, **payload) -> None:
    """One log, two readers (live timeline, activity feed)."""
    session.add(Event(submission_id=submission_id, kind=kind, payload_json=json.dumps(payload)))


#: Columns added after the first schema shipped. Real migrations land with
#: Postgres; until then this keeps a developer's existing demo database
#: working instead of asking them to delete it.
#: ``column: (sql_type, default)``.
#:
#: The type used to be hardcoded ``TEXT`` for everything here, and that was a
#: real bug rather than an untidiness. A boolean added this way is stored as the
#: *string* ``'0'``, and ``bool('0')`` is True in Python -- so on any database
#: that predated the column, every submission read back as published to the
#: shared leaderboard. The opposite of what opt-in means, on the one flag where
#: getting it wrong exposes somebody's results.
#:
#: It survived the tests because tests build a fresh database, where
#: ``create_all`` makes the column ``BOOLEAN`` and this path never runs. Only a
#: migrated database -- which is to say every real one -- saw it. ``trials`` and
#: the two ``version`` columns had the same defect and are fixed with it.
LATER = {
    "gates": {
        "measures_json": ("TEXT", "'{}'"),
        "abstained_json": ("TEXT", "'[]'"),
        "progress_json": ("TEXT", "'{}'"),
        "detail_json": ("TEXT", "'{}'"),
        "trials": ("INTEGER", "0"),
        "version": ("INTEGER", "1"),
    },
    "jobs": {"version": ("INTEGER", "1")},
    "benchmarks": {"cost_json": ("TEXT", "'{}'")},
    # Which visitor an org belongs to. NULL on every org that existed before
    # this, which is right: they belong to nobody in particular and stay
    # reachable only by whoever already had their submissions.
    "orgs": {"ip_hash": ("TEXT", "NULL")},
    # Opt-in to the shared leaderboard, and the default is the whole point. A
    # result stands next to other people's because its owner said so, never
    # because they did not find the switch.
    # `email` defaults to empty rather than NULL so every read is a string and
    # no caller has to think about which absence it got.
    "submissions": {"listed": ("INTEGER", "0"), "email": ("TEXT", "''")},
}

#: Columns an earlier version of :data:`LATER` created as TEXT, whose values are
#: therefore strings like ``'0'`` on any database migrated before the fix above.
#: Rewritten in place once, because changing a column's type in SQLite means
#: rebuilding the table and the values are what actually matter.
REPAIR = {
    "submissions": ["listed"],
    "gates": ["trials", "version"],
    "jobs": ["version"],
}

#: Seconds without a heartbeat before a running job is presumed dead. Workers
#: beat on a background thread every :data:`~worker.run.BEAT` seconds, which
#: continues through a gate's longest blocking call, so this only trips when the
#: worker process is genuinely gone.
STALE_AFTER = 90.0


def sweep_stale(session: Session, *, after: float = STALE_AFTER) -> int:
    """Fail jobs whose worker stopped beating, and say it was our fault.

    A gate stuck on "running" forever is the worst of the failure modes: it
    looks like patience is the answer, so nobody investigates, and the user
    eventually concludes their data broke something. Marked ``failed`` rather
    than ``refused`` -- the vocabulary already keeps our machinery breaking
    separate from a judgement on somebody's data, and this is squarely the
    former.

    Swept lazily on read here. A real deployment runs it on a timer; doing it
    on read means a submission nobody is looking at stays stale a while longer,
    which is harmless and much less machinery.
    """
    from datetime import datetime, timezone

    stale = 0
    cutoff = datetime.now(timezone.utc).timestamp() - after
    for job in session.scalars(select(Job).where(Job.status == "running")).all():
        try:
            beat = datetime.fromisoformat(job.heartbeat_at).timestamp()
        except (TypeError, ValueError):
            continue
        if beat > cutoff:
            continue
        job.status = "failed"
        job.error = f"the worker stopped responding after {int(after)}s"
        gate = session.scalar(
            select(Gate).where(
                Gate.submission_id == job.submission_id,
                Gate.key == job.gate_key,
                Gate.version == job.version,
            )
        )
        if gate is not None:
            gate.status = "failed"
            gate.finished_at = now()
            gate.verdict_json = json.dumps(
                {"summary": "the machine running this check stopped responding. This is our fault, not your data's"}
            )
        submission = session.get(Submission, job.submission_id)
        if submission is not None:
            submission.status = "failed"
        emit(session, job.submission_id, "gate.finished", gate=job.gate_key, status="failed",
             summary="the worker stopped responding")
        stale += 1
    if stale:
        session.commit()
    return stale


def init_db() -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table, columns in LATER.items():
            have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            for column, (sql_type, default) in columns.items():
                if column not in have:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {column} {sql_type} DEFAULT {default}"
                    )
        # Rebuild any numeric column the earlier TEXT-for-everything migration
        # created, rather than merely rewriting its values.
        #
        # An UPDATE cannot fix this, which is worth stating because it is the
        # obvious thing to reach for and it silently does nothing: SQLite applies
        # the column's *affinity* on write, so `CAST(listed AS INTEGER)` yields 0
        # and storing it into a TEXT column turns it straight back into '0'.
        #
        # A text flag is then wrong in both directions at once. `bool('0')` is
        # True in Python, so an unpublished submission reads as published; and
        # `'1' = 1` is false in SQL, so a genuinely published one is filtered off
        # the shared board. One is a privacy failure and the other loses the
        # feature -- from the same column.
        #
        # So: add a correctly typed column, copy through CAST, drop the old, and
        # rename. Needs SQLite 3.35 for DROP/RENAME COLUMN; anything older skips
        # the repair rather than refusing to start, and only a database migrated
        # by the broken version is affected at all.
        for table, columns in REPAIR.items():
            info = list(conn.exec_driver_sql(f"PRAGMA table_info({table})"))
            declared = {row[1]: (row[2] or "").upper() for row in info}
            for column in columns:
                if declared.get(column) != "TEXT":
                    continue  # already correctly typed, or the table has no such column
                try:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {column}__fixed INTEGER DEFAULT 0"
                    )
                    conn.exec_driver_sql(
                        f"UPDATE {table} SET {column}__fixed = "
                        f"CAST(COALESCE({column}, 0) AS INTEGER)"
                    )
                    conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN {column}")
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} RENAME COLUMN {column}__fixed TO {column}"
                    )
                except Exception:  # pragma: no cover - old SQLite, or a half-done repair
                    # A wrong value is bad; a product that will not start is
                    # worse, and this is recoverable by hand.
                    pass
    with SessionLocal() as s:
        if not s.query(Benchmark).count():
            s.add(
                Benchmark(
                    id=new_id("bm"),
                    key="pick_dual_bottles",
                    name="Pick two bottles (dual-arm)",
                    task="pick_dual_bottles",
                    embodiment="aloha-agilex",
                    simulator="RoboTwin 2.0",
                    reference_json=json.dumps(
                        {
                            "baseline": {"wins": 12, "n": 100},
                            "expert": 0.893,
                            "note": "baseline trained on the simulator's own 50 demonstrations",
                        }
                    ),
                    # Measured on an L40S (g6e.2xlarge, $2.24/hr on demand):
                    # a 3,000-step LoRA fine-tune took 1h45m, and closed-loop
                    # scenes at the 400-step cap averaged 2 minutes each. Two
                    # arms are trained and evaluated -- the contributor's data
                    # and its own shuffled control -- and the baseline is not
                    # retrained, because it is the same model every time and
                    # charging for it again would be charging twice.
                    cost_json=json.dumps(
                        {
                            "gpu_cents_per_hour": 224,
                            "train_seconds": 6300,
                            "seconds_per_trial": 120,
                            "arms_trained": 2,
                            "arms_evaluated": 2,
                            "probe_seconds": 600,
                            "measured_on": "L40S g6e.2xlarge, pick_dual_bottles, 3000-step LoRA",
                        }
                    ),
                )
            )
            s.commit()
