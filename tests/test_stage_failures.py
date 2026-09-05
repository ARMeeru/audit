"""Stage failure-path tests: cost accounting and trace retry semantics.

These run the real stage code with a stubbed `run_agent`, so no API calls
are made."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import audit.stages._common as common_mod
import audit.stages.hunt as hunt_mod
import audit.stages.trace as trace_mod
from audit.config import load_config
from audit.runner import AgentRunError
from audit.state import StateDB
from audit.stages._common import StageContext


@pytest.fixture()
def stage_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    ctx = StageContext(run_id="poc", repo_path=tmp_path / "repo", config=load_config())
    return db, ctx


def _add_task(db: StateDB, task_id: str = "t_1") -> None:
    db.add_task("poc", {
        "task_id": task_id, "attack_class": "sqli", "scope_hint": "x",
        "target_files": ["a.py"], "rationale": "r", "priority": 1,
        "source": "recon",
    })


def _add_confirmed_canonical_finding(db: StateDB, finding_id: str = "f_1") -> None:
    _add_task(db)
    db.add_finding("poc", "t_1", {
        "finding_id": finding_id, "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "high", "description": "d",
        "evidence_snippet": "e", "confidence": 0.9,
    })
    db.set_finding_validation("poc", finding_id, "confirmed", {"verdict": "confirmed"})
    db.assign_finding_group("poc", finding_id, "g_1", True)
    db.add_dedupe_group("poc", {
        "group_id": "g_1", "root_cause": "rc",
        "canonical_finding_id": finding_id, "member_finding_ids": [finding_id],
    })


def test_failed_hunt_attempt_records_cost(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hunt task that fails after real API spend (retries + repair turns)
    must contribute that spend to the run's cost ledger."""
    db, ctx = stage_env
    _add_task(db)

    def failing_agent(**kwargs):
        e = AgentRunError("[hunt/t_1] schema validation failed after retries")
        e.result_msg = {"total_cost_usd": 0.05, "usage": {"input_tokens": 10}}
        raise e

    monkeypatch.setattr(hunt_mod, "run_agent", failing_agent)
    asyncio.run(hunt_mod.run_hunt(ctx, db))
    assert db.total_cost("poc") == pytest.approx(0.05)


def test_failed_trace_attempt_is_retryable_on_resume(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed tracer must NOT persist an unreachable verdict: resume
    skips findings that already have a trace row, so persisting one would
    permanently hide the finding from every future report."""
    db, ctx = stage_env
    _add_confirmed_canonical_finding(db)

    def failing_agent(**kwargs):
        e = AgentRunError("[trace/f_1] schema validation failed after retries")
        e.result_msg = {"total_cost_usd": 0.02, "usage": {"input_tokens": 10}}
        raise e

    monkeypatch.setattr(trace_mod, "run_agent", failing_agent)
    asyncio.run(trace_mod.run_trace(ctx, db))
    # no verdict persisted -> --resume re-attempts the trace
    assert db.get_trace("poc", "f_1") is None
    # and the failed attempt's spend is still on the ledger
    assert db.total_cost("poc") == pytest.approx(0.02)


def test_dedupe_ignores_member_ids_from_other_runs(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Group assignment is scoped to the run: a hallucinated or foreign
    member id must not flip canonical flags on rows it does not own."""
    db, ctx = stage_env
    _add_confirmed_canonical_finding(db, "f_1")
    # a foreign run happens to hold a row with an id the dedupe agent emits
    db.create_run("/other", "run-other")
    _add_task(db, "t_1")
    db.add_finding("run-other", "t_1", {
        "finding_id": "f_foreign", "file": "x.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "low", "description": "other run",
        "evidence_snippet": "e", "confidence": 0.9,
    })

    async def ok_agent(**kwargs):
        class R:
            payload = {"groups": [{
                "group_id": "g_1", "root_cause": "rc" * 10,
                "member_finding_ids": ["f_1", "f_foreign"],
                "canonical_finding_id": "f_1",
            }]}
            raw_result_message = {"total_cost_usd": 0.0, "usage": {}}
            artifact_path = Path("/tmp/x.jsonl")
            cost_usd = 0.0
        return R()

    import audit.stages.dedupe as dedupe_mod
    monkeypatch.setattr(dedupe_mod, "run_agent", ok_agent)
    asyncio.run(dedupe_mod.run_dedupe(ctx, db))
    foreign = [f for f in db.get_findings("run-other")][0]
    assert foreign.group_id is None and not foreign.is_canonical
    mine = db.get_findings("poc")[0]
    assert mine.group_id == "g_1" and mine.is_canonical
