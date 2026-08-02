"""The schema as a *migrated* database has it, not as a fresh one does.

Every other test file starts from an empty directory, where ``create_all``
builds the tables from the models and the ``LATER`` migration never runs. That
is the one configuration no deployment is ever in: a real database predates the
columns added to it, so the migration path is the only path that matters, and
it was the only path nothing covered.

What that hid: ``LATER`` added every column as ``TEXT``, so ``submissions.listed``
was stored as the string ``'0'`` -- and ``bool('0')`` is True. On a migrated
database every submission read back as **published to the shared leaderboard**,
which is the exact inverse of opt-in, on the one flag where being wrong exposes
somebody's results to strangers. Fresh-database tests all passed.

So this file builds the old schema by hand, migrates it, and reads the values
back the way the application does.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

#: A database of its own, created before the app is imported so ``db.py`` binds
#: its engine to this file rather than to another test's.
_TMP = Path(tempfile.mkdtemp(prefix="bench-migration-test-"))
os.environ["BENCH_DATA"] = str(_TMP)

from app import db as dbmod  # noqa: E402


def old_schema(path: Path) -> None:
    """The tables as they were before any of ``LATER`` existed."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE submissions (
            id TEXT PRIMARY KEY, org_id TEXT, name TEXT, benchmark_id TEXT,
            status TEXT, current_gate TEXT, created_by TEXT, created_at TEXT
        );
        CREATE TABLE gates (
            id TEXT PRIMARY KEY, submission_id TEXT, key TEXT, status TEXT,
            verdict_json TEXT, findings_json TEXT, cost_cents INTEGER,
            started_at TEXT, finished_at TEXT
        );
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, submission_id TEXT, gate_key TEXT, status TEXT,
            claimed_by TEXT, attempts INTEGER, heartbeat_at TEXT, created_at TEXT
        );
        CREATE TABLE orgs (id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
        """
    )
    conn.execute(
        "INSERT INTO submissions (id, org_id, name, status) VALUES ('sub_old', 'org_1', 'before', 'draft')"
    )
    conn.execute("INSERT INTO gates (id, submission_id, key, status) VALUES ('g_old', 'sub_old', 'g3', 'passed')")
    conn.commit()
    conn.close()


@pytest.fixture()
def migrated(tmp_path):
    """An old database, brought forward by the real migration code."""
    path = tmp_path / "old.db"
    old_schema(path)

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        for table, columns in dbmod.LATER.items():
            have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            if not have:
                continue
            for column, (sql_type, default) in columns.items():
                if column not in have:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {column} {sql_type} DEFAULT {default}"
                    )
        for table, columns in dbmod.REPAIR.items():
            existing = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            for column in columns:
                if column in existing:
                    conn.exec_driver_sql(
                        f"UPDATE {table} SET {column} = CAST({column} AS INTEGER) "
                        f"WHERE typeof({column}) = 'text'"
                    )
    engine.dispose()
    return path


def read(path: Path, sql: str):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()
    finally:
        conn.close()


def test_a_migrated_submission_is_not_published(migrated):
    """The defect, stated as the thing it caused.

    ``bool('0')`` is True, so a boolean stored as text reads as set. On the
    leaderboard flag that means every pre-existing submission silently becomes
    visible to every other visitor.
    """
    value, kind = read(migrated, "SELECT listed, typeof(listed) FROM submissions WHERE id='sub_old'")
    assert kind == "integer", f"listed came back as {kind}, so it is a string like '0'"
    assert bool(value) is False, (
        "a submission that predates the leaderboard flag reads as published; "
        "everyone's earlier results would appear on the shared board at once"
    )


def test_counts_survive_the_migration_as_numbers(migrated):
    """``trials`` and ``version`` had the same defect, with quieter symptoms.

    A trial count of ``'0'`` is truthy, so the job dispatcher forwards
    ``{"trials": "0"}`` as though a number had been bought; a version of ``'1'``
    is compared against an integer version everywhere else.
    """
    trials, trials_kind = read(migrated, "SELECT trials, typeof(trials) FROM gates WHERE id='g_old'")
    version, version_kind = read(migrated, "SELECT version, typeof(version) FROM gates WHERE id='g_old'")
    assert trials_kind == "integer" and trials == 0
    assert version_kind == "integer" and version == 1
    assert not trials, "'0' would be truthy and read as a purchased trial count"


def test_the_migration_is_idempotent(migrated):
    """It runs on every start, so running it twice must change nothing."""
    before = read(migrated, "SELECT listed, email FROM submissions WHERE id='sub_old'")

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{migrated}")
    with engine.begin() as conn:
        for table, columns in dbmod.REPAIR.items():
            for column in columns:
                conn.exec_driver_sql(
                    f"UPDATE {table} SET {column} = CAST({column} AS INTEGER) "
                    f"WHERE typeof({column}) = 'text'"
                )
    engine.dispose()

    assert read(migrated, "SELECT listed, email FROM submissions WHERE id='sub_old'") == before


def test_every_later_entry_declares_a_type():
    """The rule the bug broke, pinned so the next column cannot repeat it.

    A new entry added as a bare string would go back to TEXT-for-everything,
    and the next boolean would be published-by-default all over again.
    """
    for table, columns in dbmod.LATER.items():
        for column, spec in columns.items():
            assert isinstance(spec, tuple) and len(spec) == 2, (
                f"{table}.{column} must be (sql_type, default); a bare default "
                "reintroduces the TEXT-for-everything bug"
            )
            sql_type, _ = spec
            assert sql_type in {"TEXT", "INTEGER", "REAL"}, f"{table}.{column}: {sql_type}"
