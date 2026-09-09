"""Stage failure-path tests: cost accounting and trace retry semantics.

These run the real stage code with a stubbed `run_agent`, so no API calls
are made. The stubs invoke the on_attempt kwarg the stage passes in, which
is the stage side of the spend-ledger contract; the runner side (per-attempt
firing inside run_agent) is covered in test_runner_cost.py."""

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
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None, "stage must pass on_attempt for spend ledgering"
        on_attempt({"total_cost_usd": 0.05, "usage": {"input_tokens": 10}})
        e = AgentRunError("[hunt/t_1] schema validation failed after retries")
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
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None, "stage must pass on_attempt for spend ledgering"
        on_attempt({"total_cost_usd": 0.02, "usage": {"input_tokens": 10}})
        e = AgentRunError("[trace/f_1] schema validation failed after retries")
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

def test_dedupe_canonical_falls_back_to_first_member(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4: a hallucinated canonical_finding_id (not among the members) used
    to make `fid == canonical` false for every member, leaving zero canonical
    findings and an empty report. Correct behavior: fall back to the first
    surviving member."""
    db, ctx = stage_env
    _add_confirmed_canonical_finding(db, "f_1")
    _add_task(db, "t_2")
    db.add_finding("poc", "t_2", {
        "finding_id": "f_2", "file": "b.py", "line_start": 3, "line_end": 4,
        "vuln_class": "xss", "severity": "medium", "description": "d2",
        "evidence_snippet": "e2", "confidence": 0.8,
    })
    db.set_finding_validation("poc", "f_2", "confirmed", {"verdict": "confirmed"})

    async def lying_agent(**kwargs):
        class R:
            payload = {"groups": [{
                "group_id": "g_9", "root_cause": "rc" * 10,
                "member_finding_ids": ["f_1", "f_2"],
                "canonical_finding_id": "f_zzz",   # hallucinated
            }]}
            raw_result_message = {"total_cost_usd": 0.0, "usage": {}}
            artifact_path = Path("/tmp/x.jsonl")
            cost_usd = 0.0
        return R()

    import audit.stages.dedupe as dedupe_mod
    monkeypatch.setattr(dedupe_mod, "run_agent", lying_agent)
    asyncio.run(dedupe_mod.run_dedupe(ctx, db))
    canonicals = db.get_findings("poc", canonical_only=True)
    assert [f.finding_id for f in canonicals] == ["f_1"], (
        "bogus canonical must fall back to the first member, not empty the set"
    )


def test_second_dedupe_pass_demotes_omitted_findings(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F9: clear_finding_groups existed but was never wired in, so a finding
    the second pass omits kept is_canonical=1 and inflated the report. The
    re-apply must be atomic clear-then-apply."""
    db, ctx = stage_env
    _add_confirmed_canonical_finding(db, "f_1")
    _add_task(db, "t_2")
    db.add_finding("poc", "t_2", {
        "finding_id": "f_2", "file": "b.py", "line_start": 3, "line_end": 4,
        "vuln_class": "xss", "severity": "medium", "description": "d2",
        "evidence_snippet": "e2", "confidence": 0.8,
    })
    db.set_finding_validation("poc", "f_2", "confirmed", {"verdict": "confirmed"})

    payloads = [
        # pass 1: two groups, both canonical
        {"groups": [
            {"group_id": "g_1", "root_cause": "rc" * 10,
             "member_finding_ids": ["f_1"], "canonical_finding_id": "f_1"},
            {"group_id": "g_2", "root_cause": "rc" * 11,
             "member_finding_ids": ["f_2"], "canonical_finding_id": "f_2"},
        ]},
        # pass 2: feedback hunted up f_3, validation confirmed it, and the
        # regrouping absorbs f_1 while OMITTING f_2 (same confirmed set plus
        # f_3 — the hash changes, so the pass actually runs)
        {"groups": [
            {"group_id": "g_3", "root_cause": "rc" * 12,
             "member_finding_ids": ["f_1", "f_3"], "canonical_finding_id": "f_1"},
        ]},
    ]
    calls = {"n": 0}

    async def scripted_agent(**kwargs):
        class R:
            payload = payloads[calls["n"]]
            raw_result_message = {"total_cost_usd": 0.0, "usage": {}}
            artifact_path = Path("/tmp/x.jsonl")
            cost_usd = 0.0
        calls["n"] += 1
        return R()

    import audit.stages.dedupe as dedupe_mod
    monkeypatch.setattr(dedupe_mod, "run_agent", scripted_agent)
    asyncio.run(dedupe_mod.run_dedupe(ctx, db))
    assert sorted(f.finding_id for f in db.get_findings("poc", canonical_only=True)) \
        == ["f_1", "f_2"]

    # feedback's hunt + validate confirm a new finding between the passes
    _add_task(db, "t_3")
    db.add_finding("poc", "t_3", {
        "finding_id": "f_3", "file": "c.py", "line_start": 5, "line_end": 6,
        "vuln_class": "sqli", "severity": "low", "description": "d3",
        "evidence_snippet": "e3", "confidence": 0.7,
    })
    db.set_finding_validation("poc", "f_3", "confirmed", {"verdict": "confirmed"})
    asyncio.run(dedupe_mod.run_dedupe(ctx, db))
    canonicals = [f.finding_id for f in db.get_findings("poc", canonical_only=True)]
    assert canonicals == ["f_1"], (
        "a finding omitted by the second pass must lose is_canonical"
    )


def test_dedupe_skips_agent_when_confirmed_set_unchanged(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F19: --finalize is designed for repeat use; a re-invocation over an
    unchanged confirmed set must not re-pay the dedupe agent call."""
    db, ctx = stage_env
    _add_confirmed_canonical_finding(db, "f_1")

    calls = {"n": 0}

    async def counting_agent(**kwargs):
        calls["n"] += 1
        class R:
            payload = {"groups": [{
                "group_id": "g_1", "root_cause": "rc" * 10,
                "member_finding_ids": ["f_1"], "canonical_finding_id": "f_1",
            }]}
            raw_result_message = {"total_cost_usd": 0.0, "usage": {}}
            artifact_path = Path("/tmp/x.jsonl")
            cost_usd = 0.0
        return R()

    import audit.stages.dedupe as dedupe_mod
    monkeypatch.setattr(dedupe_mod, "run_agent", counting_agent)
    asyncio.run(dedupe_mod.run_dedupe(ctx, db))
    asyncio.run(dedupe_mod.run_dedupe(ctx, db))
    assert calls["n"] == 1, "unchanged confirmed set must skip the agent call"

def test_failed_validation_leaves_finding_unvalidated(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: a validator that fails after real attempts used to persist
    'needs_more_info' — a terminal verdict that get_unvalidated_findings
    filters out, so resume never retried and the finding never reached
    dedupe/trace/report. Correct behavior: no verdict persisted, the
    finding stays in the unvalidated pile for resume."""
    db, ctx = stage_env
    _add_task(db)
    db.add_finding("poc", "t_1", {
        "finding_id": "f_1", "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "critical", "description": "d",
        "evidence_snippet": "e", "confidence": 0.9,
    })

    def failing_agent(**kwargs):
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None
        on_attempt({"total_cost_usd": 0.04, "usage": {"input_tokens": 10}})
        raise AgentRunError("[validate/f_1] schema validation failed after retries")

    import audit.stages.validate as validate_mod
    monkeypatch.setattr(validate_mod, "run_agent", failing_agent)
    asyncio.run(validate_mod.run_validate(ctx, db))

    unvalidated = db.get_unvalidated_findings("poc")
    assert [f.finding_id for f in unvalidated] == ["f_1"], (
        "a transient validation failure must leave the finding retryable"
    )
    assert db.get_findings("poc")[0].validation_status is None

def test_failed_tracer_is_named_in_the_report(
    stage_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3: a deterministically failing tracer used to make the finding
    vanish — no trace row, no warning, exit 0, report written without it.
    Correct behavior: the report names untraced canonicals explicitly."""
    db, ctx = stage_env
    _add_confirmed_canonical_finding(db, "f_1")

    def failing_agent(**kwargs):
        on_attempt = kwargs.get("on_attempt")
        assert on_attempt is not None
        on_attempt({"total_cost_usd": 0.02, "usage": {"input_tokens": 10}})
        raise AgentRunError("[trace/f_1] schema validation failed after retries")

    import audit.stages.trace as trace_mod
    monkeypatch.setattr(trace_mod, "run_agent", failing_agent)
    asyncio.run(trace_mod.run_trace(ctx, db))

    import json
    import audit.stages.report as report_mod
    out = asyncio.run(report_mod.run_report(ctx, db))
    payload = json.loads(out.read_text())
    assert payload["untraced_findings"] == ["f_1"], (
        "the report must name canonicals it could not assess"
    )
    assert payload["degraded"] is True
