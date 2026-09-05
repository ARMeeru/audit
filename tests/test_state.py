"""StateDB roundtrip tests."""

from __future__ import annotations

from pathlib import Path

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

    db.update_task_status("t_1", "done")
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
        db.update_task_status(tid, status)

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


LEGACY_FINDINGS_SQL = """
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, source TEXT NOT NULL,
    attack_class TEXT NOT NULL, scope_hint TEXT NOT NULL, target_files TEXT NOT NULL,
    rationale TEXT, priority INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'pending', raw_json TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE traces (
    finding_id TEXT PRIMARY KEY,
    reachable INTEGER NOT NULL,
    confidence REAL,
    rationale TEXT,
    raw_json TEXT NOT NULL
);
CREATE TABLE findings (
    finding_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL, run_id TEXT NOT NULL, file TEXT NOT NULL,
    line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
    vuln_class TEXT NOT NULL, severity TEXT NOT NULL,
    description TEXT NOT NULL, evidence TEXT NOT NULL,
    poc_succeeded INTEGER DEFAULT 0, confidence REAL, raw_json TEXT NOT NULL,
    validation_status TEXT, validation_json TEXT, group_id TEXT,
    is_canonical INTEGER DEFAULT 0
);
"""


def test_legacy_db_migrates_to_run_scoped_keys(tmp_path: Path) -> None:
    """A pre-existing db with the global finding_id PK must migrate without
    losing rows."""
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript(LEGACY_FINDINGS_SQL)
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'done', '{}', 1, 1)"
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
    conn.executescript(LEGACY_FINDINGS_SQL)
    conn.execute(
        "INSERT INTO tasks (task_id, run_id, source, attack_class, scope_hint,"
        " target_files, rationale, priority, status, raw_json, created_at, updated_at)"
        " VALUES ('t_1', 'run-a', 'recon', 'sqli', 'x', '[]', '', 3, 'done', '{}', 1, 1)"
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
