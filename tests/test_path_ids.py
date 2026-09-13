"""Path-component sensors: every operator- or model-supplied string that becomes
a filesystem component must be rejected when it can escape its directory.

Two primitives, one root cause:

  * `--run-id` is operator-controlled and reaches results/ and work/ through
    `RESULTS / run_id` plus `mkdir(parents=True)`. Nothing upstream validates
    it, so this one was reachable by anyone who can pass a flag.
  * `task_id` reaches `WORK / run_id / "hunt" / task_id` and the artifact
    filename `<task_id>.jsonl`. Correcting an earlier claim of mine: this is NOT
    reachable through the pipeline, because every producer of tasks (`recon`,
    `gapfill`, `feedback`) `$ref`s hunt_task.schema.json, which pins task_id to
    `^[a-z0-9_-]{1,64}$` before the DB ever sees it — pinned by
    test_schema_already_blocks_a_traversing_task_id below. What was missing is
    the storage-side backstop: `db.add_task` and the path builders trusted the
    value, so anything that did not come through an agent schema (a tampered or
    hand-edited DB, a future importer) got an escape. That is the gap these
    tests close.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import audit.runner as runner_mod
from audit.config import load_config
from audit.stages._common import SCHEMAS, StageContext
from audit.state import StateDB

# Values that must never become a path component.
ESCAPES = [
    "..",
    "../escape",
    "../../../../tmp/owned",
    "a/b",
    "a\\b",
    "/abs/path",
    "~/.ssh",
    "a\x00b",
    "a\nb",
    # `$` in an anchored `match` also matches before a trailing newline, so
    # these two were accepted and became real directory names on disk.
    "run_x\n",
    "..\n",
    "a\r",
    "",
    "x" * 129,
    "é",
]

# The shapes the harness itself generates, including the longest id the schemas
# permit: finding.schema.json allows `f_` plus 64 characters, and
# _resolve_finding_id appends `_2` on a collision, so the ceiling has to clear
# 66. It used to be 64, which made a schema-legal id raise inside the runner.
GOOD = ["run_ab12cd34", "r", "f_1", "task-01", "a.b_c-d", "x" * 64,
        "f_" + "a" * 64, "f_" + "a" * 60 + "_2"]


def _ctx(run_id: str, repo: Path) -> StageContext:
    return StageContext(run_id=run_id, repo_path=repo, config=load_config())


def test_create_run_rejects_a_traversing_run_id(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    with pytest.raises(ValueError):
        db.create_run(str(tmp_path / "repo"), "../escape")


# `match="unsafe"` on purpose. A bare `pytest.raises(ValueError)` was satisfied
# for the NUL cases by CPython's own `os.mkdir: embedded null character in path`,
# so two of these entries passed identically at origin/main and could never have
# detected the defect they were written for. Matching the harness's own message
# makes every entry red against the unfixed code for the right reason.
@pytest.mark.parametrize("bad", ESCAPES)
def test_results_dir_rejects_a_traversing_run_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _ctx(bad, tmp_path).results_dir("report")


@pytest.mark.parametrize("bad", ESCAPES)
def test_hunt_work_dir_rejects_a_traversing_task_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _ctx("run_ok", tmp_path).work_dir("hunt", bad)


@pytest.mark.parametrize("bad", ESCAPES)
def test_artifact_path_rejects_a_traversing_name(tmp_path: Path, bad: str) -> None:
    fn = getattr(runner_mod, "_artifact_path", None)
    assert fn is not None, (
        "audit.runner._artifact_path is missing: the artifact filename is built "
        "inline from a model-supplied name"
    )
    with pytest.raises(ValueError, match="unsafe"):
        fn(tmp_path, bad)


def test_create_run_rejects_a_case_insensitive_collision(tmp_path: Path) -> None:
    """macOS and Windows fold case, so two ids differing only in case share one
    directory while SQLite keeps two rows: writing through one and reading
    through the other silently crosses runs."""
    db = StateDB(tmp_path / "state.db")
    db.create_run(str(tmp_path / "repo"), "run_ab12cd34")
    with pytest.raises(ValueError, match="case-insensitive"):
        db.create_run(str(tmp_path / "repo"), "RUN_AB12CD34")


@pytest.mark.parametrize("good", GOOD)
def test_generated_ids_still_work(tmp_path: Path, good: str) -> None:
    """The anti-overblocking direction. These are the shapes the harness itself
    generates (`run_` plus hex, schema-constrained `f_` ids, recon task ids)."""
    assert _ctx(good, tmp_path).results_dir("report").is_dir()
    assert _ctx(good, tmp_path).work_dir("hunt", "task-01").is_dir()
    fn = getattr(runner_mod, "_artifact_path", None)
    assert fn is not None
    assert fn(tmp_path, good).name == f"{good}.jsonl"


def test_generated_run_ids_are_accepted_by_the_state_db(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.create_run(str(tmp_path / "repo"), "run_ab12cd34")
    assert db.get_run("run_ab12cd34") is not None


def test_schema_already_blocks_a_traversing_task_id_on_the_agent_path() -> None:
    """The artifact behind the corrected severity of the task_id case: the
    agent boundary rejects it before storage, so what is left is a backstop
    rather than a live traversal. Pinned as a test so a future schema edit that
    loosens the pattern fails here instead of silently reopening the path."""
    from audit.json_utils import validate_schema

    task = {
        "task_id": "../../../../tmp/owned",
        "attack_class": "sql_injection",
        "scope_hint": "HTTP handler reads name and passes it to a raw query",
        "target_files": ["app.py"],
        "rationale": "tainted input reaches a raw SQL sink",
        "priority": 3,
        "source": "recon",
    }
    errors = validate_schema(task, SCHEMAS / "hunt_task.schema.json")
    assert any("task_id" in e for e in errors), (
        "hunt_task.schema.json no longer rejects a traversing task_id, so the "
        f"storage-side backstop is the only defence left: {errors[:2]}"
    )


def test_schema_legal_long_finding_id_survives_the_artifact_path() -> None:
    """The ceiling must clear the schemas it validates against. finding_id is
    `^f_[a-z0-9_-]{1,64}$`, so a legal id reaches 66 characters, and a
    collision-suffixed one more; a 64-character ceiling turned a schema-valid
    finding into a ValueError inside the runner, which validate/trace did not
    catch, which failed the whole run after the exploration spend was sunk."""
    from audit.json_utils import validate_schema
    from audit.paths import safe_component

    longest = "f_" + "a" * 64
    hunt_output = {
        "task_id": "t_core_auth_1",
        "gaps_observed": [],
        "findings": [{
            "finding_id": longest,
            "file": "app.py",
            "line_start": 1,
            "line_end": 2,
            "vuln_class": "sql_injection",
            "severity": "high",
            "description": "tainted input reaches a raw SQL sink",
            "evidence_snippet": "cursor.execute(q)",
            "confidence": 0.8,
        }],
    }
    assert validate_schema(
        hunt_output, SCHEMAS / "finding.schema.json"
    ) == [], "the fixture is no longer schema-legal, so this proves nothing"
    assert safe_component(longest, kind="artifact name") == longest
    assert safe_component(longest + "_2", kind="artifact name") == longest + "_2"


def test_safe_component_is_the_single_chokepoint() -> None:
    """Every path component routes through one validator, so a future call site
    has somewhere obvious to go and the audit has one place to look."""
    from audit.paths import safe_component

    assert safe_component("run_ok") == "run_ok"
    with pytest.raises(ValueError, match="run_id"):
        safe_component("../escape", kind="run_id")
