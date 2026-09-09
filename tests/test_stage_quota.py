"""Quota-handling sensors for the concurrent and closing stages.

Stubs run_agent per stage module - no API calls. Each test asserts the
CORRECT behavior: quota death records the attempt's spend, stops dispatching
siblings (drain), never persists a permanent verdict, and aborts the stage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import audit.stages.validate as validate_mod
import audit.stages.trace as trace_mod
import audit.stages.report as report_mod
from audit.config import load_config
from audit.runner import QuotaExhaustedError
from audit.state import StateDB
from audit.stages._common import StageContext


@pytest.fixture()
def stage_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import audit.stages._common as common_mod
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    ctx = StageContext(run_id="q", repo_path=tmp_path / "repo", config=load_config())
    return db, ctx, tmp_path


def _seed_task(db: StateDB, task_id: str = "t_1") -> None:
    db.add_task("q", {
        "task_id": task_id, "attack_class": "sqli", "scope_hint": "x",
        "target_files": ["a.py"], "rationale": "r", "priority": 1,
        "source": "recon",
    })


def _quota(spend: float = 0.02) -> QuotaExhaustedError:
    e = QuotaExhaustedError("[stage/x] quota_exhausted: session limit")
    e.result_msg = {"total_cost_usd": spend, "usage": {"input_tokens": 10}}
    return e


def test_validate_quota_records_cost_drains_and_aborts(stage_env, monkeypatch):
    """Quota on the first validation: spend is ledgered, siblings are not
    dispatched (drain), the exception aborts the stage, and NO finding is
    graded (it stays unvalidated for resume)."""
    db, ctx, tmp = stage_env
    _seed_task(db)
    for fid in ("f_1", "f_2", "f_3"):
        db.add_finding("q", "t_1", {
            "finding_id": fid, "file": "a.py", "line_start": 1, "line_end": 2,
            "vuln_class": "sqli", "severity": "high", "description": "d",
            "evidence_snippet": "e", "confidence": 0.9,
        })

    calls = []

    async def quota_agent(**kwargs):
        calls.append("called")
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None, "stage must pass on_attempt for spend ledgering"
        on_attempt({"total_cost_usd": 0.03, "usage": {"input_tokens": 10}})
        raise _quota(0.03)

    monkeypatch.setattr(validate_mod, "run_agent", quota_agent)
    with pytest.raises(QuotaExhaustedError):
        asyncio.run(validate_mod.run_validate(ctx, db))

    assert len(calls) == 1, "drain: siblings must not be dispatched after quota abort"
    assert db.total_cost("q") == pytest.approx(0.03)
    assert all(f.validation_status is None for f in db.get_findings("q"))


def test_trace_quota_records_cost_and_stays_retryable(stage_env, monkeypatch):
    """A quota-killed tracer persists nothing: resume re-attempts the trace,
    and the attempt's spend is still ledgered."""
    db, ctx, tmp = stage_env
    _seed_task(db)
    db.add_finding("q", "t_1", {
        "finding_id": "f_1", "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "high", "description": "d",
        "evidence_snippet": "e", "confidence": 0.9,
    })
    db.set_finding_validation("q", "f_1", "confirmed", {"verdict": "confirmed"})
    db.assign_finding_group("q", "f_1", "g_1", True)
    db.add_dedupe_group("q", {
        "group_id": "g_1", "root_cause": "rc",
        "canonical_finding_id": "f_1", "member_finding_ids": ["f_1"],
    })

    calls = []

    async def quota_agent(**kwargs):
        calls.append("called")
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None, "stage must pass on_attempt for spend ledgering"
        on_attempt({"total_cost_usd": 0.05, "usage": {"input_tokens": 10}})
        raise _quota(0.05)

    monkeypatch.setattr(trace_mod, "run_agent", quota_agent)
    with pytest.raises(QuotaExhaustedError):
        asyncio.run(trace_mod.run_trace(ctx, db))

    assert len(calls) == 1
    assert db.total_cost("q") == pytest.approx(0.05)
    # nothing permanent was persisted for the killed trace
    assert db.get_trace("q", "f_1") is None


def test_report_quota_emits_fallback_report(stage_env, tmp_path: Path, monkeypatch):
    """A quota-killed report agent must still produce the deterministic
    fallback report from state.db (lesson from an earlier live campaign, in test form)."""
    db, ctx, tmp = stage_env
    _seed_task(db)
    db.add_finding("q", "t_1", {
        "finding_id": "f_1", "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "high", "description": "desc",
        "evidence_snippet": "ev", "confidence": 0.9,
    })
    db.set_finding_validation("q", "f_1", "confirmed", {"verdict": "confirmed"})
    db.assign_finding_group("q", "f_1", "g_1", True)
    db.add_dedupe_group("q", {
        "group_id": "g_1", "root_cause": "rc",
        "canonical_finding_id": "f_1", "member_finding_ids": ["f_1"],
    })
    db.add_trace("q", "f_1", {"finding_id": "f_1", "reachable": True,
                              "entry_points": [], "call_chain": []})

    async def quota_agent(**kwargs):
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None, "stage must pass on_attempt for spend ledgering"
        on_attempt({"total_cost_usd": 0.01, "usage": {"input_tokens": 10}})
        raise _quota(0.01)

    monkeypatch.setattr(report_mod, "run_agent", quota_agent)
    out = asyncio.run(report_mod.run_report(ctx, db))

    assert out.exists()
    import json
    payload = json.loads(out.read_text())
    assert payload["summary"]["total"] >= 1
    assert payload["findings"], "fallback report must carry the reachable finding"
