"""SQLite-backed run state. JSONL artifacts in results/ are the source of
truth for raw agent output; this DB is the queryable index used for
orchestration, resume, and reporting."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
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
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    attack_class TEXT NOT NULL,
    scope_hint TEXT NOT NULL,
    target_files TEXT NOT NULL,
    rationale TEXT,
    priority INTEGER NOT NULL DEFAULT 3,
    attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    raw_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    PRIMARY KEY (run_id, task_id)
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT NOT NULL,
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
    PRIMARY KEY (run_id, finding_id),
    FOREIGN KEY (run_id, task_id) REFERENCES tasks(run_id, task_id)
);

CREATE TABLE IF NOT EXISTS traces (
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    reachable INTEGER NOT NULL,
    confidence REAL,
    rationale TEXT,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (run_id, finding_id)
);

CREATE TABLE IF NOT EXISTS dedupe_groups (
    group_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    canonical_finding_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (run_id, group_id),
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


@dataclass
class Task:
    task_id: str
    run_id: str
    source: str
    attack_class: str
    scope_hint: str
    target_files: list[str]
    rationale: str
    priority: int
    status: str
    raw_json: dict
    attempts: int = 0


@dataclass
class Finding:
    finding_id: str
    task_id: str
    run_id: str
    file: str
    line_start: int
    line_end: int
    vuln_class: str
    severity: str
    description: str
    evidence: str
    poc_succeeded: bool
    confidence: float | None
    raw_json: dict
    validation_status: str | None
    validation_json: dict | None
    group_id: str | None
    is_canonical: bool


class StateDB:
    MIGRATION_VERSION = 3

    def __init__(self, db_path: Path):
        self.path = db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        # WAL: readers (`audit status` during a run) no longer trip the
        # writer's lock, and a crash between checkpoint and commit is
        # recovered from the WAL instead of corrupting the write.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        self._migrate()
        self._conn.commit()

    def _create_schema(self) -> None:
        # executescript() would COMMIT any open transaction, so the schema
        # runs statement-by-statement and migrations stay atomic.
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= self.MIGRATION_VERSION:
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._migrate_v1()
            self._migrate_v2()
            self._migrate_v3()
            # Reconcile indexes: v2 shipped without the pre-rename drop and
            # destroyed four of five on every database it touched. CREATE
            # INDEX IF NOT EXISTS is a no-op for those still present.
            self._create_schema()
            self._conn.execute(f"PRAGMA user_version = {self.MIGRATION_VERSION}")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _migrate_v1(self) -> None:
        """v0 -> v1: run-scoped identity keys, tasks.attempts column.

        finding_id and group_id were globally unique keys while their values
        are model-emitted and only unique per run, so two runs sharing this
        db silently dropped each other's rows. Columns are mapped by name;
        legacy traces had no run_id column, so theirs is derived by joining
        findings (migrated first, whose legacy rows carry it)."""
        legacy = (
            ("findings", "PRIMARY KEY (run_id, finding_id)", None),
            ("traces", "PRIMARY KEY (run_id, finding_id)",
             # legacy traces lack run_id: derive it from findings, whose
             # legacy rows are migrated first and hold globally-unique ids
             "SELECT f.run_id AS run_id, t.finding_id AS finding_id, "
             "t.reachable AS reachable, t.confidence AS confidence, "
             "t.rationale AS rationale, t.raw_json AS raw_json "
             "FROM {table}_legacy t JOIN findings f ON f.finding_id = t.finding_id"),
            ("dedupe_groups", "PRIMARY KEY (run_id, group_id)", None),
        )
        indexes = {
            "findings": ("idx_findings_run", "idx_findings_validation", "idx_findings_group"),
        }
        for table, marker, custom_select in legacy:
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if row is None or marker in row["sql"]:
                continue
            for idx in indexes.get(table, ()):
                self._conn.execute(f"DROP INDEX IF EXISTS {idx}")
            self._conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            self._create_schema()
            new_cols = [r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if custom_select:
                self._conn.execute(
                    f"INSERT INTO {table} ({', '.join(new_cols)}) "
                    f"{custom_select.format(table=table)}"
                )
            else:
                legacy_cols = [
                    r["name"] for r in self._conn.execute(f"PRAGMA table_info({table}_legacy)").fetchall()
                ]
                self._conn.execute(
                    f"INSERT OR IGNORE INTO {table} ({', '.join(new_cols)}) "
                    f"SELECT {', '.join(legacy_cols)} FROM {table}_legacy"
                )
            self._conn.execute(f"DROP TABLE {table}_legacy")
        task_cols = [r["name"] for r in self._conn.execute("PRAGMA table_info(tasks)").fetchall()]
        if "attempts" not in task_cols:
            self._conn.execute("ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")

    def _rebuild(self, tables: tuple[str, ...]) -> None:
        """Rename -> recreate from SCHEMA -> copy by name -> drop, inside
        the caller's transaction.

        Indexes are dropped FIRST and by lookup, not from a hardcoded list:
        an index follows its table on rename and keeps its name, so
        CREATE INDEX IF NOT EXISTS silently no-ops and DROP TABLE takes the
        index with it. v2 lost four of five indexes exactly this way."""
        for table in tables:
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = ? AND name NOT LIKE 'sqlite_%'", (table,)
            ).fetchall():
                self._conn.execute(f"DROP INDEX IF EXISTS {row['name']}")
        for table in tables:
            self._conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
        self._create_schema()
        for table in tables:
            new_cols = [r["name"] for r in self._conn.execute(
                f"PRAGMA table_info({table})").fetchall()]
            legacy_cols = [r["name"] for r in self._conn.execute(
                f"PRAGMA table_info({table}_legacy)").fetchall()]
            # map by name; new-only columns (e.g. tasks.attempts) take
            # their SCHEMA defaults
            shared = [c for c in new_cols if c in legacy_cols]
            before = self._conn.execute(
                f"SELECT COUNT(*) FROM {table}_legacy").fetchone()[0]
            self._conn.execute(
                f"INSERT INTO {table} ({', '.join(shared)}) "
                f"SELECT {', '.join(shared)} FROM {table}_legacy"
            )
            after = self._conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if after != before:
                log.warning("migration: %s carried %d of %d rows",
                            table, after, before)
            self._conn.execute(f"DROP TABLE {table}_legacy")

    def _migrate_v2(self) -> None:
        """v1 -> v2: tasks joined the run-scoped key family.

        task_id stayed a global PRIMARY KEY in v1 while its values are
        model-emitted and only unique per run — exactly the argument v1's
        docstring makes for findings — so two runs sharing this database
        dropped each other's tasks (INSERT OR IGNORE), and update_task_status
        hit whichever run's row came first.

        FK hazard (verified against a real database): renaming tasks to
        tasks_legacy rewrites findings' FOREIGN KEY to reference
        tasks_legacy, and DROP leaves it dangling there permanently. v1 got
        away with the same pattern only because traces — the FK holder it
        renamed — was itself rebuilt in the same transaction. v2 renames and
        rebuilds all three tables in one transaction, so no dangling
        reference ever survives. Runs after v1, so traces carry run_id."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
        ).fetchone()
        if row is None or "PRIMARY KEY (run_id, task_id)" in row["sql"]:
            return
        self._rebuild(("tasks", "findings", "traces"))

    def _migrate_v3(self) -> None:
        """v2 -> v3: findings' FK matches tasks' composite primary key.

        v2 gave tasks PRIMARY KEY (run_id, task_id) but left findings
        declaring FOREIGN KEY (task_id) REFERENCES tasks(task_id). A
        single-column FK needs a unique parent index, so SQLite rejects it
        as a foreign key mismatch: PRAGMA foreign_key_check errors and,
        with foreign_keys=ON, every finding insert fails. Rebuilding
        findings alone is safe -- the FK is its own, so no other table's
        SQL is rewritten."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'findings'"
        ).fetchone()
        if row is None or "REFERENCES tasks(run_id, task_id)" in row["sql"]:
            return
        self._rebuild(("findings",))

    # ---------- runs ----------

    def create_run(self, repo_path: str, run_id: str | None = None) -> str:
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO runs (run_id, repo_path, started_at, status) VALUES (?, ?, ?, ?)",
            (run_id, repo_path, time.time(), "running"),
        )
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, time.time(), run_id),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def list_runs(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC"
            ).fetchall()
        )

    # ---------- recon ----------

    def save_recon_output(self, run_id: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO recon_outputs (run_id, raw_json) VALUES (?, ?)",
            (run_id, json.dumps(payload)),
        )
        self._conn.commit()

    def get_recon_output(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT raw_json FROM recon_outputs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row["raw_json"]) if row else None

    # ---------- tasks ----------

    def add_task(self, run_id: str, task: dict) -> None:
        now = time.time()
        self._conn.execute(
            """INSERT OR IGNORE INTO tasks
            (task_id, run_id, source, attack_class, scope_hint, target_files,
             rationale, priority, status, raw_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (
                task["task_id"],
                run_id,
                task.get("source", "recon"),
                task["attack_class"],
                task["scope_hint"],
                json.dumps(task["target_files"]),
                task.get("rationale", ""),
                int(task.get("priority", 3)),
                json.dumps(task),
                now,
                now,
            ),
        )
        self._conn.commit()

    def get_pending_tasks(self, run_id: str, max_attempts: int = 3) -> list[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE run_id = ? AND status = 'pending' "
            "AND attempts < ? ORDER BY priority, created_at",
            (run_id, max_attempts),
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_all_tasks(self, run_id: str) -> list[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_task_status(self, run_id: str, task_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE run_id = ? AND task_id = ?",
            (status, time.time(), run_id, task_id),
        )
        self._conn.commit()

    def begin_task(self, run_id: str, task_id: str) -> None:
        """Flip a task to 'running' and spend one attempt, atomically.

        The attempts counter belongs at the point the attempt is spent
        (dispatch), not where the task is re-queued: a crash leaves status
        'running' without passing through any handler, and hunt's quota
        path writes 'pending' directly — neither increments a requeue-side
        counter, so a deterministically hanging task re-burned full spend
        on every resume."""
        self._conn.execute(
            "UPDATE tasks SET status = 'running', attempts = attempts + 1, "
            "updated_at = ? WHERE run_id = ? AND task_id = ?",
            (time.time(), run_id, task_id),
        )
        self._conn.commit()

    def complete_task(self, run_id: str, task_id: str, findings: list[dict]) -> int:
        """Persist a hunt task's findings and flip it to done in one
        transaction, resetting the attempt strike. A crash between the
        finding writes and the status flip used to leave the task 'running';
        the resume re-dispatch then re-inserted every finding (the findings
        and the done flip are now indivisible). Same-task replays no-op.
        Returns the number of NEW findings inserted."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            inserted = 0
            for finding in findings:
                fid, insert = self._resolve_finding_id(run_id, task_id,
                                                       finding["finding_id"])
                if not insert:
                    continue
                if fid != finding["finding_id"]:
                    finding = dict(finding, finding_id=fid)
                self._conn.execute(
                    """INSERT INTO findings
                    (finding_id, task_id, run_id, file, line_start, line_end,
                     vuln_class, severity, description, evidence, poc_succeeded,
                     confidence, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fid, task_id, run_id,
                        finding["file"], finding["line_start"], finding["line_end"],
                        finding["vuln_class"], finding["severity"],
                        finding["description"], finding["evidence_snippet"],
                        1 if (finding.get("poc") or {}).get("succeeded") else 0,
                        finding.get("confidence"),
                        json.dumps(finding),
                    ),
                )
                inserted += 1
            self._conn.execute(
                "UPDATE tasks SET status = 'done', attempts = 0, updated_at = ? "
                "WHERE run_id = ? AND task_id = ?",
                (time.time(), run_id, task_id),
            )
            self._conn.commit()
            return inserted
        except Exception:
            self._conn.rollback()
            raise

    def max_stage_cost(self, run_id: str, stage: str) -> float | None:
        """Most expensive single completed task in this stage of this run.
        Used as the per-task in-flight estimate for budget checks: an upper
        bound, not a mean — a mean under-shoots exactly when a stage runs
        long, which is when the cap matters."""
        row = self._conn.execute(
            "SELECT MAX(usd) AS m FROM costs WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        ).fetchone()
        return float(row["m"]) if row and row["m"] is not None else None

    def max_stage_cost_any_run(self, stage: str) -> float | None:
        """Same as max_stage_cost, across every run sharing this database:
        a first hunt stage has no rows for ITS run, which is the case a
        hard default was silently covering."""
        row = self._conn.execute(
            "SELECT MAX(usd) AS m FROM costs WHERE stage = ?",
            (stage,),
        ).fetchone()
        return float(row["m"]) if row and row["m"] is not None else None

    def release_task(self, run_id: str, task_id: str) -> None:
        """Return a task to 'pending' WITHOUT keeping the attempt charge: a
        quota abort is the pipeline stopping, not the task failing. Three
        quota-killed resumes must not abandon a task that never genuinely
        failed."""
        self._conn.execute(
            "UPDATE tasks SET status = 'pending', "
            "attempts = MAX(0, attempts - 1), updated_at = ? "
            "WHERE run_id = ? AND task_id = ?",
            (time.time(), run_id, task_id),
        )
        self._conn.commit()

    def count_abandoned_tasks(self, run_id: str) -> int:
        """Failed tasks past the requeue ceiling — work silently given up
        on. surfaced so an operator can see the abandonment, not just the
        green completion."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM tasks WHERE run_id = ? AND status = 'failed' "
            "AND attempts >= 3",
            (run_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def reset_incomplete_tasks(self, run_id: str, max_requeues: int = 3) -> int:
        """Re-queue interrupted and failed tasks so a resumed run re-attempts
        them. 'running' tasks (quota/crash interrupts — not the task's fault)
        are always re-queued. 'failed' tasks are re-queued only under a retry
        ceiling: each re-queue increments attempts, and a task that has been
        re-queued max_requeues times stays failed, so a deterministically
        failing task stops re-burning spend on every resume. Returns the
        number of tasks reset."""
        now = time.time()
        cur_run = self._conn.execute(
            "UPDATE tasks SET status = 'pending', updated_at = ? "
            "WHERE run_id = ? AND status = 'running'",
            (now, run_id),
        )
        cur_fail = self._conn.execute(
            # No increment here: begin_task spends the attempt at dispatch,
            # and get_pending_tasks filters on the ceiling. Incrementing in
            # both places charged two attempts per failure cycle, so a task
            # got 2 dispatches under a ceiling of 3.
            "UPDATE tasks SET status = 'pending', updated_at = ? "
            "WHERE run_id = ? AND status = 'failed' AND attempts < ?",
            (now, run_id, max_requeues),
        )
        self._conn.commit()
        return cur_run.rowcount + cur_fail.rowcount

    @staticmethod
    def _row_to_task(r: sqlite3.Row) -> Task:
        return Task(
            task_id=r["task_id"],
            run_id=r["run_id"],
            source=r["source"],
            attack_class=r["attack_class"],
            scope_hint=r["scope_hint"],
            target_files=json.loads(r["target_files"]),
            rationale=r["rationale"] or "",
            priority=r["priority"],
            status=r["status"],
            raw_json=json.loads(r["raw_json"]),
            attempts=r["attempts"] if "attempts" in r.keys() else 0,
        )

    # ---------- findings ----------

    def _resolve_finding_id(
        self, run_id: str, task_id: str, base: str
    ) -> tuple[str, bool]:
        """Resolve a model-emitted finding id for (run_id, task_id).

        Returns (finding_id, insert_needed). A row already held by the SAME
        task means this is a replay of an interrupted dispatch (crash
        between the finding writes and the done flip): no insert. The same
        id from a DIFFERENT task is a genuine model-emitted collision and
        gets a suffix — model ids are only unique per task by prompt
        convention."""
        existing = self._conn.execute(
            "SELECT task_id FROM findings WHERE run_id = ? AND finding_id = ?",
            (run_id, base),
        ).fetchone()
        if existing is not None and existing["task_id"] == task_id:
            return base, False
        fid = base
        n = 1
        while self._conn.execute(
            "SELECT 1 FROM findings WHERE run_id = ? AND finding_id = ?",
            (run_id, fid),
        ).fetchone():
            # Two tasks may emit the same id; keep both by suffixing and
            # keep raw_json consistent so downstream stages reference the
            # rewritten id.
            n += 1
            fid = f"{base}_{n}"
        return fid, True

    def add_finding(self, run_id: str, task_id: str, finding: dict) -> str:
        """Insert one finding. Returns the (possibly suffixed) finding_id."""
        poc = finding.get("poc") or {}
        fid, insert = self._resolve_finding_id(run_id, task_id,
                                               finding["finding_id"])
        if not insert:
            return fid
        if fid != finding["finding_id"]:
            finding = dict(finding, finding_id=fid)
        self._conn.execute(
            """INSERT INTO findings
            (finding_id, task_id, run_id, file, line_start, line_end,
             vuln_class, severity, description, evidence, poc_succeeded,
             confidence, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fid,
                task_id,
                run_id,
                finding["file"],
                finding["line_start"],
                finding["line_end"],
                finding["vuln_class"],
                finding["severity"],
                finding["description"],
                finding["evidence_snippet"],
                1 if poc.get("succeeded") else 0,
                finding.get("confidence"),
                json.dumps(finding),
            ),
        )
        self._conn.commit()

    def get_findings(self, run_id: str, *, validation_status: str | None = None,
                     canonical_only: bool = False) -> list[Finding]:
        sql = "SELECT * FROM findings WHERE run_id = ?"
        args: list[Any] = [run_id]
        if validation_status is not None:
            sql += " AND validation_status = ?"
            args.append(validation_status)
        if canonical_only:
            sql += " AND is_canonical = 1"
        rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_finding(r) for r in rows]

    def get_unvalidated_findings(self, run_id: str) -> list[Finding]:
        rows = self._conn.execute(
            "SELECT * FROM findings WHERE run_id = ? AND validation_status IS NULL",
            (run_id,),
        ).fetchall()
        return [self._row_to_finding(r) for r in rows]

    def set_finding_validation(self, run_id: str, finding_id: str, status: str, payload: dict) -> None:
        self._conn.execute(
            "UPDATE findings SET validation_status = ?, validation_json = ? "
            "WHERE run_id = ? AND finding_id = ?",
            (status, json.dumps(payload), run_id, finding_id),
        )
        self._conn.commit()

    def assign_finding_group(
        self, run_id: str, finding_id: str, group_id: str, is_canonical: bool
    ) -> None:
        self._conn.execute(
            "UPDATE findings SET group_id = ?, is_canonical = ? "
            "WHERE run_id = ? AND finding_id = ?",
            (group_id, 1 if is_canonical else 0, run_id, finding_id),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_finding(r: sqlite3.Row) -> Finding:
        return Finding(
            finding_id=r["finding_id"],
            task_id=r["task_id"],
            run_id=r["run_id"],
            file=r["file"],
            line_start=r["line_start"],
            line_end=r["line_end"],
            vuln_class=r["vuln_class"],
            severity=r["severity"],
            description=r["description"],
            evidence=r["evidence"],
            poc_succeeded=bool(r["poc_succeeded"]),
            confidence=r["confidence"],
            raw_json=json.loads(r["raw_json"]),
            validation_status=r["validation_status"],
            validation_json=json.loads(r["validation_json"]) if r["validation_json"] else None,
            group_id=r["group_id"],
            is_canonical=bool(r["is_canonical"]),
        )

    # ---------- traces ----------

    def add_trace(self, run_id: str, finding_id: str, payload: dict) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO traces
            (run_id, finding_id, reachable, confidence, rationale, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                finding_id,
                1 if payload.get("reachable") else 0,
                payload.get("confidence"),
                payload.get("rationale", ""),
                json.dumps(payload),
            ),
        )
        self._conn.commit()

    def get_trace(self, run_id: str, finding_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT raw_json FROM traces WHERE run_id = ? AND finding_id = ?",
            (run_id, finding_id),
        ).fetchone()
        return json.loads(row["raw_json"]) if row else None

    def get_reachable_canonical_findings(self, run_id: str) -> list[tuple[Finding, dict]]:
        out: list[tuple[Finding, dict]] = []
        for f in self.get_findings(run_id, validation_status="confirmed", canonical_only=True):
            tr = self.get_trace(run_id, f.finding_id)
            if tr and tr.get("reachable"):
                out.append((f, tr))
        return out

    # ---------- dedupe ----------

    def clear_finding_groups(self, run_id: str) -> None:
        """Drop all group assignments for a run before re-applying a fresh
        dedupe pass. Without this, a finding demoted between passes keeps
        is_canonical=1 and inflates the reported set."""
        self._conn.execute(
            "UPDATE findings SET group_id = NULL, is_canonical = 0 WHERE run_id = ?",
            (run_id,),
        )
        self._conn.execute("DELETE FROM dedupe_groups WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def latest_artifact_path(self, run_id: str, stage: str, kind: str) -> str | None:
        """Path of the most recent artifact row matching (stage, kind), or
        None. Used for content markers like the dedupe confirmed-set hash."""
        row = self._conn.execute(
            "SELECT path FROM artifacts WHERE run_id = ? AND stage = ? AND kind = ? "
            "ORDER BY artifact_id DESC LIMIT 1",
            (run_id, stage, kind),
        ).fetchone()
        return row["path"] if row else None

    def untraced_canonical_ids(self, run_id: str) -> list[str]:
        """Confirmed canonical findings with no trace row. Trace failures
        are retryable (no verdict persisted), so "every canonical has a
        trace" is no longer an invariant — the report must say which
        canonicals it could not assess instead of silently omitting them."""
        rows = self._conn.execute(
            """SELECT f.finding_id FROM findings f
            WHERE f.run_id = ? AND f.validation_status = 'confirmed'
              AND f.is_canonical = 1
              AND NOT EXISTS (SELECT 1 FROM traces t
                              WHERE t.run_id = f.run_id AND t.finding_id = f.finding_id)""",
            (run_id,),
        ).fetchall()
        return [r["finding_id"] for r in rows]

    def count_dedupe_groups(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM dedupe_groups WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row["c"]) if row else 0

    def apply_dedupe_groups(
        self, run_id: str, prepared: list[tuple[dict, list[str], str]]
    ) -> int:
        """Atomically replace the run's grouping: clear stale assignments,
        insert the new groups, assign members. One transaction, so a crash
        mid-apply cannot leave a run with no canonicals, and a finding the
        second pass omits cannot keep is_canonical=1 (the stale-canonical
        defect). Each tuple is (group_dict, validated_member_ids, canonical_id).
        Returns the number of groups applied."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "UPDATE findings SET group_id = NULL, is_canonical = 0 WHERE run_id = ?",
                (run_id,),
            )
            self._conn.execute("DELETE FROM dedupe_groups WHERE run_id = ?", (run_id,))
            for group, member_ids, canonical in prepared:
                self._conn.execute(
                    """INSERT OR REPLACE INTO dedupe_groups
                    (group_id, run_id, root_cause, canonical_finding_id, raw_json)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        group["group_id"],
                        run_id,
                        group["root_cause"],
                        canonical,
                        json.dumps(group),
                    ),
                )
                for fid in member_ids:
                    self._conn.execute(
                        "UPDATE findings SET group_id = ?, is_canonical = ? "
                        "WHERE run_id = ? AND finding_id = ?",
                        (group["group_id"], 1 if fid == canonical else 0, run_id, fid),
                    )
            self._conn.commit()
            return len(prepared)
        except Exception:
            self._conn.rollback()
            raise

    def add_dedupe_group(self, run_id: str, group: dict) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO dedupe_groups
            (group_id, run_id, root_cause, canonical_finding_id, raw_json)
            VALUES (?, ?, ?, ?, ?)""",
            (
                group["group_id"],
                run_id,
                group["root_cause"],
                group["canonical_finding_id"],
                json.dumps(group),
            ),
        )
        self._conn.commit()

    # ---------- costs ----------

    def record_cost(
        self,
        run_id: str,
        stage: str,
        ref_id: str | None,
        result_msg: dict,
    ) -> None:
        usage = result_msg.get("usage") or {}
        self._conn.execute(
            """INSERT INTO costs
            (run_id, stage, ref_id, usd, input_tokens, output_tokens,
             cache_read_tokens, cache_creation_tokens, num_turns, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                stage,
                ref_id,
                result_msg.get("total_cost_usd"),
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("cache_read_input_tokens"),
                usage.get("cache_creation_input_tokens"),
                result_msg.get("num_turns"),
                result_msg.get("duration_ms"),
                time.time(),
            ),
        )
        self._conn.commit()

    def total_cost(self, run_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) AS total FROM costs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return float(row["total"]) if row else 0.0

    # ---------- artifacts ----------

    def count_artifacts(self, run_id: str, stage: str) -> int:
        """Agent invocations already consumed for a stage. The artifacts table
        is the source of truth for expansion-loop bounds: derived counts cannot
        desync from the work the way persisted counters can."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM artifacts WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        ).fetchone()
        return int(row["c"]) if row else 0

    def stage_summary(self, run_id: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT stage, status, COUNT(*) AS c FROM tasks WHERE run_id = ? "
            "GROUP BY stage, status ORDER BY stage",
            (run_id,),
        ).fetchall())

    def stage_costs(self, run_id: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT stage, SUM(usd) AS usd FROM costs WHERE run_id = ? "
            "GROUP BY stage ORDER BY stage",
            (run_id,),
        ).fetchall())

    def add_artifact(
        self, run_id: str, stage: str, ref_id: str | None, kind: str, path: str
    ) -> None:
        self._conn.execute(
            """INSERT INTO artifacts
            (run_id, stage, ref_id, kind, path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, stage, ref_id, kind, path, time.time()),
        )
        self._conn.commit()

    # ---------- context manager ----------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@contextmanager
def open_db(path: Path) -> Iterator[StateDB]:
    db = StateDB(path)
    try:
        yield db
    finally:
        db.close()
