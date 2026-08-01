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
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

BENCH_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BENCH_DATA", BENCH_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = DATA_DIR / "storage"
STORAGE.mkdir(exist_ok=True)

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
    started_at: Mapped[str] = mapped_column(String, default="")
    finished_at: Mapped[str] = mapped_column(String, default="")
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    submission_id: Mapped[str] = mapped_column(String, index=True)
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
LATER = {"gates": {"measures_json": "'{}'", "abstained_json": "'[]'"}}


def init_db() -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table, columns in LATER.items():
            have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            for column, default in columns.items():
                if column not in have:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {column} TEXT DEFAULT {default}"
                    )
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
                )
            )
            s.commit()
