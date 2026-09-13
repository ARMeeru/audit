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
    "",
    "x" * 65,
    "é",
]

GOOD = ["run_ab12cd34", "r", "f_1", "task-01", "a.b_c-d", "x" * 64]


def _ctx(run_id: str, repo: Path) -> StageContext:
    return StageContext(run_id=run_id, repo_path=repo, config=load_config())


def test_create_run_rejects_a_traversing_run_id(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    with pytest.raises(ValueError):
        db.create_run(str(tmp_path / "repo"), "../escape")


@pytest.mark.parametrize("bad", ESCAPES)
def test_results_dir_rejects_a_traversing_run_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        _ctx(bad, tmp_path).results_dir("report")


@pytest.mark.parametrize("bad", ESCAPES)
def test_hunt_work_dir_rejects_a_traversing_task_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        _ctx("run_ok", tmp_path).work_dir("hunt", bad)


@pytest.mark.parametrize("bad", ESCAPES)
def test_artifact_path_rejects_a_traversing_name(tmp_path: Path, bad: str) -> None:
    fn = getattr(runner_mod, "_artifact_path", None)
    assert fn is not None, (
        "audit.runner._artifact_path is missing: the artifact filename is built "
        "inline from a model-supplied name"
    )
    with pytest.raises(ValueError):
        fn(tmp_path, bad)


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


def test_safe_component_is_the_single_chokepoint() -> None:
    """Every path component routes through one validator, so a future call site
    has somewhere obvious to go and the audit has one place to look."""
    from audit.paths import safe_component

    assert safe_component("run_ok") == "run_ok"
    with pytest.raises(ValueError, match="run_id"):
        safe_component("../escape", kind="run_id")
