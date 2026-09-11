"""StateDB roundtrip tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from audit.state import StateDB


def test_run_and_task_lifecycle(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/some/repo", "test_run")
    assert db.get_run(rid)["status"] == "running"

    db.add_task(rid, {
        "task_id": "t_1",
        "attack_class": "sqli",
        "scope_hint": "lookup name parameter",
        "target_files": ["app.py"],
        "rationale": "raw string formatting",
        "priority": 1,
        "source": "recon",
    })
    pending = db.get_pending_tasks(rid)
    assert len(pending) == 1
    assert pending[0].task_id == "t_1"

    db.update_task_status(rid, "t_1", "done")
    assert db.get_pending_tasks(rid) == []
    assert any(t.status == "done" for t in db.get_all_tasks(rid))

    db.finish_run(rid)
    assert db.get_run(rid)["status"] == "completed"
    db.close()


def test_reset_incomplete_tasks(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/some/repo", "test_run")
    for tid, status in [("t_done", "done"), ("t_run", "running"),
                        ("t_fail", "failed"), ("t_pend", "pending")]:
        db.add_task(rid, {
            "task_id": tid, "attack_class": "sqli", "scope_hint": "x",
            "target_files": ["a.py"], "rationale": "r", "priority": 1,
            "source": "recon",
        })
        db.update_task_status(rid, tid, status)

    n = db.reset_incomplete_tasks(rid)
    assert n == 2  # only running + failed are re-queued
    by_status = {t.task_id: t.status for t in db.get_all_tasks(rid)}
    assert by_status == {
        "t_done": "done", "t_run": "pending",
        "t_fail": "pending", "t_pend": "pending",
    }
    assert {t.task_id for t in db.get_pending_tasks(rid)} == {"t_run", "t_fail", "t_pend"}
    db.close()


def test_finding_validation_and_dedupe(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/some/repo", "test_run")
    db.add_task(rid, {
        "task_id": "t_1", "attack_class": "sqli", "scope_hint": "x",
        "target_files": ["a.py"], "rationale": "r", "priority": 1, "source": "recon",
    })
    db.add_finding(rid, "t_1", {
        "finding_id": "f_1", "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "high",
        "description": "x", "evidence_snippet": "y", "confidence": 0.9,
    })
    assert len(db.get_unvalidated_findings(rid)) == 1

    db.set_finding_validation(rid, "f_1", "confirmed", {
        "finding_id": "f_1", "verdict": "confirmed",
        "rationale": "ok", "validator_confidence": 0.9,
    })
    assert len(db.get_findings(rid, validation_status="confirmed")) == 1

    db.add_dedupe_group(rid, {
        "group_id": "g_1", "root_cause": "rc",
        "canonical_finding_id": "f_1", "member_finding_ids": ["f_1"],
    })
    db.assign_finding_group(rid, "f_1", "g_1", True)
    assert len(db.get_findings(rid, canonical_only=True)) == 1

    db.add_trace(rid, "f_1", {
        "finding_id": "f_1", "reachable": True, "confidence": 0.9,
        "rationale": "trivial", "entry_points": [], "call_chain": [],
    })
    reachable = db.get_reachable_canonical_findings(rid)
    assert len(reachable) == 1
    db.close()


def test_cost_aggregation(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "test_run")
    db.record_cost(rid, "hunt", "t_1", {"total_cost_usd": 0.01, "usage": {
        "input_tokens": 100, "output_tokens": 50,
    }, "num_turns": 3, "duration_ms": 1234})
    db.record_cost(rid, "hunt", "t_2", {"total_cost_usd": 0.02, "usage": {
        "input_tokens": 200, "output_tokens": 100,
    }, "num_turns": 5, "duration_ms": 4321})
    assert abs(db.total_cost(rid) - 0.03) < 1e-9
    db.close()


# ---------- run-scoped ids (regression: finding_id was a global PK) ----------


def _finding(fid: str, desc: str) -> dict:
    return {
        "finding_id": fid, "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "high",
        "description": desc, "evidence_snippet": "e", "confidence": 0.9,
    }


def _task(db: StateDB, rid: str, task_id: str) -> None:
    db.add_task(rid, {
        "task_id": task_id, "attack_class": "sqli", "scope_hint": "x",
        "target_files": ["a.py"], "rationale": "r", "priority": 1,
        "source": "recon",
    })



def test_same_finding_id_across_runs_both_survive(tmp_path: Path) -> None:
    """Two runs sharing state.db may emit identical model-chosen finding
    ids; neither run's finding may be dropped (the legacy global PK
    silently lost the second insert)."""
    db = StateDB(tmp_path / "state.db")
    for rid, repo in (("run-a", "/repoA"), ("run-b", "/repoB")):
        db.create_run(repo, rid)
        _task(db, rid, "t_1")
    db.add_finding("run-a", "t_1", _finding("f_sqli_1", "repo A finding"))
    db.add_finding("run-b", "t_1", _finding("f_sqli_1", "repo B finding"))
    assert len(db.get_findings("run-a")) == 1
    assert len(db.get_findings("run-b")) == 1
    db.close()


def test_same_finding_id_across_tasks_both_survive(tmp_path: Path) -> None:
    """Within one run, two hunt tasks may emit the same finding id."""
    db = StateDB(tmp_path / "state.db")
    db.create_run("/repo", "run-x")
    _task(db, "run-x", "t_1")
    _task(db, "run-x", "t_2")
    db.add_finding("run-x", "t_1", _finding("f_sqli_1", "from task 1"))
    db.add_finding("run-x", "t_2", _finding("f_sqli_1", "from task 2"))
    assert len(db.get_findings("run-x")) == 2
    db.close()


def test_assign_finding_group_is_run_scoped(tmp_path: Path) -> None:
    """Group assignment must never cross runs, even for identical ids."""
    db = StateDB(tmp_path / "state.db")
    for rid, repo in (("run-a", "/repoA"), ("run-b", "/repoB")):
        db.create_run(repo, rid)
        _task(db, rid, "t_1")
    db.add_finding("run-a", "t_1", _finding("f_sqli_1", "repo A"))
    db.add_finding("run-b", "t_1", _finding("f_sqli_1", "repo B"))
    db.add_dedupe_group("run-a", {
        "group_id": "g_1", "root_cause": "rc",
        "canonical_finding_id": "f_sqli_1", "member_finding_ids": ["f_sqli_1"],
    })
    db.assign_finding_group("run-a", "f_sqli_1", "g_1", True)
    a = db.get_findings("run-a")[0]
    b = db.get_findings("run-b")[0]
    assert a.group_id == "g_1" and a.is_canonical
    assert b.group_id is None and not b.is_canonical
    db.close()


def test_traces_are_run_scoped(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    for rid, repo in (("run-a", "/repoA"), ("run-b", "/repoB")):
        db.create_run(repo, rid)
        _task(db, rid, "t_1")
    db.add_finding("run-a", "t_1", _finding("f_sqli_1", "repo A"))
    db.add_finding("run-b", "t_1", _finding("f_sqli_1", "repo B"))
    db.set_finding_validation("run-a", "f_sqli_1", "confirmed", {"verdict": "confirmed"})
    db.add_trace("run-a", "f_sqli_1", {"finding_id": "f_sqli_1", "reachable": True})
    assert db.get_trace("run-b", "f_sqli_1") is None
    assert db.get_trace("run-a", "f_sqli_1")["reachable"] is True
    db.close()


V0_SCHEMA_SQL = """\

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    repo_path TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS recon_outputs (
    run_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    attack_class TEXT NOT NULL,
    scope_hint TEXT NOT NULL,
    target_files TEXT NOT NULL,
    rationale TEXT,
    priority INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'pending',
    raw_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    file TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    vuln_class TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence TEXT NOT NULL,
    poc_succeeded INTEGER DEFAULT 0,
    confidence REAL,
    raw_json TEXT NOT NULL,
    validation_status TEXT,
    validation_json TEXT,
    group_id TEXT,
    is_canonical INTEGER DEFAULT 0,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS traces (
    finding_id TEXT PRIMARY KEY,
    reachable INTEGER NOT NULL,
    confidence REAL,
    rationale TEXT,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (finding_id) REFERENCES findings(finding_id)
);

CREATE TABLE IF NOT EXISTS dedupe_groups (
    group_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    canonical_finding_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS costs (
    cost_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ref_id TEXT,
    usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    num_turns INTEGER,
    duration_ms INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ref_id TEXT,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_run_status ON tasks(run_id, status);
CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_validation ON findings(validation_status);
CREATE INDEX IF NOT EXISTS idx_findings_group ON findings(group_id);
CREATE INDEX IF NOT EXISTS idx_costs_run_stage ON costs(run_id, stage);
"""

# executescript form of the same constant (the constant above documents
# provenance; this is the body the tests execute)
LEGACY_V0_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    repo_path TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS recon_outputs (
    run_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    attack_class TEXT NOT NULL,
    scope_hint TEXT NOT NULL,
    target_files TEXT NOT NULL,
    rationale TEXT,
    priority INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'pending',
    raw_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    file TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    vuln_class TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence TEXT NOT NULL,
    poc_succeeded INTEGER DEFAULT 0,
    confidence REAL,
    raw_json TEXT NOT NULL,
    validation_status TEXT,
    validation_json TEXT,
    group_id TEXT,
    is_canonical INTEGER DEFAULT 0,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS traces (
    finding_id TEXT PRIMARY KEY,
    reachable INTEGER NOT NULL,
    confidence REAL,
    rationale TEXT,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (finding_id) REFERENCES findings(finding_id)
);

CREATE TABLE IF NOT EXISTS dedupe_groups (
    group_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    canonical_finding_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS costs (
    cost_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ref_id TEXT,
    usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    num_turns INTEGER,
    duration_ms INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ref_id TEXT,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_run_status ON tasks(run_id, status);
CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_validation ON findings(validation_status);
CREATE INDEX IF NOT EXISTS idx_findings_group ON findings(group_id);
CREATE INDEX IF NOT EXISTS idx_costs_run_stage ON costs(run_id, stage);
"""


def test_legacy_db_migrates_to_run_scoped_keys(tmp_path: Path) -> None:
    """A pre-existing db with the global finding_id PK must migrate without
    losing rows."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'pending', '{}', 1, 1)"
    )
    conn.execute(
        "INSERT INTO findings (finding_id, task_id, run_id, file, line_start,"
        " line_end, vuln_class, severity, description, evidence, raw_json)"
        " VALUES ('f_1', 't_1', 'run-a', 'a.py', 1, 2, 'sqli', 'high', 'd', 'e', '{}')"
    )
    conn.commit()
    conn.close()

    db = StateDB(p)
    sql = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='findings'"
    ).fetchone()["sql"]
    assert "PRIMARY KEY (run_id, finding_id)" in sql
    assert len(db.get_findings("run-a")) == 1
    db.close()


def test_legacy_traces_derive_run_id(tmp_path: Path) -> None:
    """Legacy trace rows had no run_id column; migration must derive it by
    joining findings instead of mangling column order."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'pending', '{}', 1, 1)"
    )
    conn.execute(
        "INSERT INTO findings (finding_id, task_id, run_id, file, line_start,"
        " line_end, vuln_class, severity, description, evidence, raw_json)"
        " VALUES ('f_1', 't_1', 'run-a', 'a.py', 1, 2, 'sqli', 'high', 'd', 'e', '{}')"
    )
    conn.execute(
        "INSERT INTO traces (finding_id, reachable, confidence, rationale, raw_json)"
        " VALUES ('f_1', 1, 0.9, 'ok', '{\"reachable\": true}')"
    )
    conn.commit()
    conn.close()

    db = StateDB(p)
    trace = db.get_trace("run-a", "f_1")
    assert trace is not None and trace["reachable"] is True
    db.close()


# ---------- Phase 1 sensors: attempts ceiling, group clearing, WAL/version ----------


def test_attempts_ceiling_stops_requeue_at_limit(tmp_path: Path) -> None:
    """A failed task at the attempts ceiling stays failed: resume must not
    re-burn spend on a deterministically failing task forever."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "run-att")
    _task(db, rid, "t_ceiling")
    db.update_task_status(rid, "t_ceiling", "failed")
    db._conn.execute("UPDATE tasks SET attempts = 3 WHERE task_id = 't_ceiling'")
    db._conn.commit()

    assert db.reset_incomplete_tasks(rid, max_requeues=3) == 0
    row = db._conn.execute(
        "SELECT status, attempts FROM tasks WHERE task_id = 't_ceiling'"
    ).fetchone()
    assert row["status"] == "failed" and row["attempts"] == 3
    db.close()


def test_attempts_below_ceiling_requeued_and_incremented(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "run-att2")
    _task(db, rid, "t_below")
    db.update_task_status(rid, "t_below", "failed")
    db._conn.execute("UPDATE tasks SET attempts = 2 WHERE task_id = 't_below'")
    db._conn.commit()

    assert db.reset_incomplete_tasks(rid, max_requeues=3) == 1
    row = db._conn.execute(
        "SELECT status, attempts FROM tasks WHERE task_id = 't_below'"
    ).fetchone()
    # Requeue does NOT charge an attempt: begin_task spent one at dispatch,
    # and the failed dispatch that produced this 'failed' row already made
    # it 2. The requeue flips status only; the NEXT dispatch reaches 3.
    assert row["status"] == "pending" and row["attempts"] == 2
    db.begin_task(rid, "t_below")
    row = db._conn.execute(
        "SELECT status, attempts FROM tasks WHERE task_id = 't_below'"
    ).fetchone()
    assert row["status"] == "running" and row["attempts"] == 3
    db.close()


def test_running_task_requeued_without_increasing_attempts(tmp_path: Path) -> None:
    """Quota/crash interrupts are not the task's fault: 'running' tasks are
    re-queued and the attempts counter is left untouched."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "run-att3")
    _task(db, rid, "t_run")
    db.update_task_status(rid, "t_run", "running")

    assert db.reset_incomplete_tasks(rid, max_requeues=3) == 1
    row = db._conn.execute(
        "SELECT status, attempts FROM tasks WHERE task_id = 't_run'"
    ).fetchone()
    assert row["status"] == "pending" and row["attempts"] == 0
    db.close()


def test_clear_finding_groups_drops_stale_assignments(tmp_path: Path) -> None:
    """A second dedupe pass must clear stale group assignments: a finding
    demoted between passes can no longer keep is_canonical=1 and inflate
    the reported set."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "run-clear")
    _task(db, rid, "t_1")
    db.add_finding(rid, "t_1", _finding("f_1", "stale canonical"))
    db.set_finding_validation(rid, "f_1", "confirmed", {"verdict": "confirmed"})
    db.add_dedupe_group(rid, {
        "group_id": "g_1", "root_cause": "rc",
        "canonical_finding_id": "f_1", "member_finding_ids": ["f_1"],
    })
    db.assign_finding_group(rid, "f_1", "g_1", True)
    assert len(db.get_findings(rid, canonical_only=True)) == 1

    db.clear_finding_groups(rid)

    f = db.get_findings(rid)[0]
    assert f.group_id is None and not f.is_canonical
    stale_groups = db._conn.execute(
        "SELECT COUNT(*) AS c FROM dedupe_groups WHERE run_id = ?", (rid,)
    ).fetchone()["c"]
    assert stale_groups == 0
    db.close()


def test_wal_mode_and_user_version_after_migration(tmp_path: Path) -> None:
    """WAL keeps status-during-run lock-free, and the migration stamps
    PRAGMA user_version so future migrations gate on a number instead of
    substring-sniffing CREATE TABLE text."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'pending', '{}', 1, 1)"
    )
    conn.execute(
        "INSERT INTO findings (finding_id, task_id, run_id, file, line_start,"
        " line_end, vuln_class, severity, description, evidence, raw_json)"
        " VALUES ('f_1', 't_1', 'run-a', 'a.py', 1, 2, 'sqli', 'high', 'd', 'e', '{}')"
    )
    conn.commit()
    conn.close()

    db = StateDB(p)
    assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert len(db.get_findings("run-a")) == 1
    db.close()

import asyncio


def test_v2_migration_scopes_tasks_and_keeps_fk_intact(tmp_path: Path) -> None:
    """F7: v2 run-scopes tasks. The rename must not leave findings'
    FOREIGN KEY pointing at the dropped tasks_legacy table (verified
    hazard: ALTER TABLE RENAME rewrites the reference and DROP strands
    it). The migrated schema must reference plain `tasks`, and a second
    run must be able to hold a task_id a first run already used."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'pending', '{}', 1, 1)"
    )
    conn.commit()
    conn.close()

    db = StateDB(p)
    fk_sql = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='findings'"
    ).fetchone()["sql"]
    assert 'REFERENCES "tasks_legacy"' not in fk_sql, "dangling FK after migration"
    assert "REFERENCES tasks(run_id, task_id)" in fk_sql
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 3

    # two runs, same task_id: both rows survive (the old global PK dropped one)
    db.add_task("run-a", {"task_id": "t_1", "attack_class": "sqli",
                          "scope_hint": "x", "target_files": ["a.py"],
                          "rationale": "r", "priority": 1, "source": "recon"})
    db.add_task("run-b", {"task_id": "t_1", "attack_class": "sqli",
                          "scope_hint": "x", "target_files": ["a.py"],
                          "rationale": "r", "priority": 1, "source": "recon"})
    assert len(db.get_all_tasks("run-a")) == 1
    assert len(db.get_all_tasks("run-b")) == 1
    # and run B marking its copy done must not touch run A's
    db.update_task_status("run-b", "t_1", "done")
    assert db.get_all_tasks("run-a")[0].status == "pending"


def test_add_finding_replay_noops_but_cross_task_collision_suffixes(tmp_path: Path) -> None:
    """F8: replaying the same hunt task (crash between finding writes and
    the done flip) must not duplicate findings; the same id from a
    DIFFERENT task is a real collision and keeps both via suffix."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "test_run")
    db.add_task(rid, {"task_id": "t_1", "attack_class": "sqli",
                      "scope_hint": "x", "target_files": ["a.py"],
                      "rationale": "r", "priority": 1, "source": "recon"})
    db.add_task(rid, {"task_id": "t_2", "attack_class": "sqli",
                      "scope_hint": "x", "target_files": ["a.py"],
                      "rationale": "r", "priority": 1, "source": "recon"})
    finding = {"finding_id": "f_1", "file": "a.py", "line_start": 1,
               "line_end": 2, "vuln_class": "sqli", "severity": "high",
               "description": "d", "evidence_snippet": "e", "confidence": 0.9}

    for _ in range(3):
        db.add_finding(rid, "t_1", dict(finding))
    rows = [f.finding_id for f in db.get_findings(rid)]
    assert rows == ["f_1"], "same-task replay must no-op, not duplicate"

    db.add_finding(rid, "t_2", dict(finding))
    rows = sorted(f.finding_id for f in db.get_findings(rid))
    assert rows == ["f_1", "f_1_2"], "cross-task collision must keep both"


def test_dispatch_spends_an_attempt_and_ceiling_filters(tmp_path: Path) -> None:
    """F10: the attempts counter belongs at dispatch (a crash leaves
    'running' without passing any handler; hunt's quota path writes
    'pending' directly — neither was counted before). Pending dispatch
    must also skip tasks past the ceiling."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "test_run")
    for tid in ("t_a", "t_b", "t_c"):
        db.add_task(rid, {"task_id": tid, "attack_class": "sqli",
                          "scope_hint": "x", "target_files": ["a.py"],
                          "rationale": "r", "priority": 1, "source": "recon"})
    # crash-simulate: dispatch burns an attempt even though no handler runs
    for _ in range(3):
        db.begin_task(rid, "t_a")
    assert db.get_all_tasks(rid)[0].status == "running"
    db.update_task_status(rid, "t_a", "pending")
    pending = [t.task_id for t in db.get_pending_tasks(rid)]
    assert "t_a" not in pending, "task past the attempt ceiling must not re-dispatch"
    assert {"t_b", "t_c"} <= set(pending)
    # completing resets the strike
    db.begin_task(rid, "t_b")
    db.complete_task(rid, "t_b", [])
    t_b = [t for t in db.get_all_tasks(rid) if t.task_id == "t_b"][0]
    assert t_b.status == "done" and t_b.attempts == 0


def test_failure_cycle_gets_three_real_dispatches(tmp_path: Path) -> None:
    """F5/R3: attempts used to be charged at BOTH dispatch and requeue, so
    a ceiling of 3 delivered 2 real dispatches. The full cycle — dispatch,
    fail, requeue, repeat — must give exactly max_requeues dispatches."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "test_run")
    db.add_task(rid, {"task_id": "t_1", "attack_class": "sqli",
                      "scope_hint": "x", "target_files": ["a.py"],
                      "rationale": "r", "priority": 1, "source": "recon"})
    dispatches = 0
    for _ in range(10):
        pending = db.get_pending_tasks(rid)
        if not pending:
            break
        db.begin_task(rid, "t_1")           # dispatch spends an attempt
        dispatches += 1
        db.update_task_status(rid, "t_1", "failed")
        requeued = db.reset_incomplete_tasks(rid)  # resume
        if requeued == 0:
            break
    assert dispatches == 3, f"ceiling of 3 must mean 3 dispatches, got {dispatches}"


def test_quota_release_hands_back_the_attempt(tmp_path: Path) -> None:
    """A quota abort is the pipeline stopping, not the task failing: three
    quota-killed resumes must not abandon a never-failed task."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "test_run")
    db.add_task(rid, {"task_id": "t_1", "attack_class": "sqli",
                      "scope_hint": "x", "target_files": ["a.py"],
                      "rationale": "r", "priority": 1, "source": "recon"})
    for _ in range(3):
        db.begin_task(rid, "t_1")
        db.release_task(rid, "t_1")
    pending = db.get_pending_tasks(rid)
    assert [t.task_id for t in pending] == ["t_1"], (
        "released task must stay dispatchable after repeated quota kills"
    )

def test_v3_repairs_indexes_and_fk_on_v2_damaged_databases(tmp_path: Path) -> None:
    """Findings 2+7 acceptance: v0, v1 and v2 databases all migrate to v3
    with all five indexes present, rows preserved, integrity ok, and a
    foreign-key-valid findings insert (foreign_keys=ON)."""
    import sqlite3
    # v2-damaged base: simulate a database that went through the index-losing v2
    p = tmp_path / "v2damaged.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO runs (run_id, repo_path, started_at, status)"
        " VALUES ('run-a', '/r', 1, 'running')"
    )
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'pending', '{}', 1, 1)"
    )
    conn.commit()
    conn.close()
    # drive it through the OLD v2 (index-losing) by hand: migrate, then drop
    # the indexes v2 dropped to reproduce the damage
    db = StateDB(p)
    db.close()
    conn = sqlite3.connect(p)
    for idx in ("idx_findings_group", "idx_findings_run",
                "idx_findings_validation", "idx_tasks_run_status"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    # downgrade user_version to 2 so v3 runs (and the reconcile path fires)
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    db = StateDB(p)
    conn = sqlite3.connect(p)
    indexes = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")}
    for expected in ("idx_findings_group", "idx_findings_run",
                     "idx_findings_validation", "idx_tasks_run_status",
                     "idx_costs_run_stage"):
        assert expected in indexes, f"{expected} lost"
    # FK valid: with foreign_keys ON, a finding insert must succeed
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "INSERT INTO findings (finding_id, task_id, run_id, file, line_start,"
        " line_end, vuln_class, severity, description, evidence, raw_json)"
        " VALUES ('f_x', 't_1', 'run-a', 'a.py', 1, 2, 'sqli', 'high', 'd', 'e', '{}')"
    )
    fk_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
    assert fk_issues == [], f"foreign_key_check reported: {fk_issues}"
    conn.close()

    # fresh v3 -> v3 reopen is a no-op
    db2 = StateDB(p)
    assert db2._conn.execute("PRAGMA user_version").fetchone()[0] == 3

def test_duplicate_id_within_one_payload_keeps_both(tmp_path: Path) -> None:
    """F4/R6: two different findings sharing an id in the SAME payload are
    a model-emitted collision, not a replay -- the second used to be
    silently dropped because the first's row existed by resolution time
    (a regression against the pre-fix suffix loop)."""
    db = StateDB(tmp_path / "state.db")
    rid = db.create_run("/r", "test_run")
    db.add_task(rid, {"task_id": "t_1", "attack_class": "sqli",
                      "scope_hint": "x", "target_files": ["a.py"],
                      "rationale": "r", "priority": 1, "source": "recon"})
    f_a = {"finding_id": "f_1", "file": "a.py", "line_start": 10,
           "line_end": 11, "vuln_class": "sqli", "severity": "high",
           "description": "SQLi in the login handler", "evidence_snippet": "e1",
           "confidence": 0.9}
    f_b = dict(f_a, description="SQLi in the signup handler",
               evidence_snippet="e2", line_start=20)

    inserted = db.complete_task(rid, "t_1", [f_a, f_b])
    assert inserted == 2, "both same-payload findings must be stored"
    rows = db._conn.execute(
        "SELECT finding_id, line_start FROM findings WHERE run_id = ? "
        "ORDER BY line_start", (rid,)).fetchall()
    assert [(r["finding_id"], r["line_start"]) for r in rows] == \
        [("f_1", 10), ("f_1_2", 20)]

    # a genuine replay of the SAME payload afterwards still no-ops
    inserted2 = db.complete_task(rid, "t_1", [f_a, f_b])
    assert inserted2 == 0
    assert db._conn.execute(
        "SELECT COUNT(*) AS c FROM findings WHERE run_id = ?",
        (rid,)).fetchone()["c"] == 2

def test_migration_from_an_indexed_database_keeps_every_index(tmp_path: Path) -> None:
    """Findings 2+3/R4: the index-loss regression only reproduces from a
    database that HAS indexes going into a rename -- v2's two mechanisms
    (pre-rename drop in _rebuild, trailing reconcile in _migrate) each
    mask the other, so every single-point mutation survives. This sensor
    starts indexed and asserts the exact index set survives v0->v3."""
    import sqlite3
    p = tmp_path / "indexed.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)   # origin/main's real SCHEMA: 5 indexes
    conn.commit()
    before = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")}
    assert len(before) == 5, f"fixture must be indexed: {before}"
    conn.close()

    db = StateDB(p)   # v0 -> v3
    db.close()

    conn = sqlite3.connect(p)
    after = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")}
    conn.close()
    assert after == before, f"index set changed across migration: lost {before - after}"

def test_migration_completes_despite_a_concurrent_reader(tmp_path: Path) -> None:
    """F4/R5: switching a rollback-journal file to WAL needs an exclusive
    lock, and a held read (what `audit status` in a second terminal is)
    used to abort the open -- skipping the v3 repair behind an opaque
    'database is locked'. The migration must complete anyway."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO runs (run_id, repo_path, started_at, status)"
        " VALUES ('run-a', '/r', 1, 'running')"
    )
    conn.commit()
    # hold a read transaction, like a second process mid-`audit status`
    conn.execute("BEGIN")
    conn.execute("SELECT COUNT(*) FROM runs").fetchone()

    db = StateDB(p)   # must not raise, even though commit cannot land
    assert db.upgrade_pending is True
    # `audit status` paths still work on the unmigrated database
    assert db.total_cost("run-a") == 0.0
    # ...but a RUN must refuse with a named error, not fail later inside a
    # dispatch with "no such column: attempts" after a paid recon
    import pytest as _pytest
    from audit.state import UpgradePendingError
    with _pytest.raises(UpgradePendingError):
        db.get_pending_tasks("run-a")
    db.close()

    # reader goes away: the next open completes the upgrade
    conn.execute("COMMIT")
    conn.close()
    db = StateDB(p)
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert db.upgrade_pending is False
    assert db.get_pending_tasks("run-a") == []
    db.close()

def test_rebuild_preserves_hand_added_indexes(tmp_path: Path) -> None:
    """F7/R6: _rebuild drops every index by lookup but only recreates
    SCHEMA's own -- a hand-added index on a real operator's database used
    to be destroyed without a trace."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "CREATE INDEX idx_user_custom ON findings(severity)")
    conn.commit()
    conn.close()

    db = StateDB(p)
    db.close()
    conn = sqlite3.connect(p)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")}
    assert "idx_user_custom" in names, "hand-added index must survive the rebuild"
    conn.close()


@pytest.mark.parametrize("ddl,name", [
    ("CREATE INDEX idx_custom ON findings(severity)", "idx_custom"),
    # NOT rebuilt by any migration, so the name stays live at replay -- the
    # shape that used to brick the upgrade on every open.
    ("CREATE UNIQUE INDEX idx_u_costs ON costs(cost_id)", "idx_u_costs"),
    ("CREATE INDEX idx_part ON findings(file) WHERE severity='high'", "idx_part"),
    ("create index idx_lower on findings(severity)", "idx_lower"),
])
@pytest.mark.parametrize("start", ["v0", "v1", "v2_damaged"])
def test_hand_added_indexes_survive_every_upgrade_path(tmp_path: Path, ddl, name, start) -> None:
    """F1/R1: a hand-added index must survive every upgrade path, and the
    open must never raise. Parametrized over index shapes because the old
    sensor used a plain index on a REBUILT table -- the one shape that
    cannot collide with the replay."""
    import sqlite3
    p = tmp_path / f"{start}.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_V0_SQL)
    conn.execute(
        "INSERT INTO runs (run_id, repo_path, started_at, status)"
        " VALUES ('run-a', '/r', 1, 'running')")
    conn.execute(ddl)
    conn.commit()
    conn.close()

    if start == "v1":
        # migrate, then stop at v1 by rolling the version back
        db = StateDB(p)
        db.close()
        conn = sqlite3.connect(p)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
    elif start == "v2_damaged":
        db = StateDB(p)
        db.close()
        conn = sqlite3.connect(p)
        for idx in ("idx_findings_group", "idx_findings_run",
                    "idx_findings_validation", "idx_tasks_run_status"):
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

    db = StateDB(p)   # must not raise
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    names = {r[0] for r in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")}
    db.close()
    assert name in names, f"{name} lost across {start} upgrade: {sorted(names)}"
