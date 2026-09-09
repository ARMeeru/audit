"""F13 sensor: transient-retry spend must reach the ledger.

run_agent's retry loop is where intermediate attempts burn API spend. The
stage call sites never see those attempts, so the ledger recording has to
happen inside the loop via the on_attempt callback. These tests patch
_run_agent_once (not run_agent) so the retry loop itself is executed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import audit.runner as runner_mod
from audit.runner import AgentResult, TransientAgentError, run_agent


def _result(payload: dict, cost: float) -> AgentResult:
    return AgentResult(
        payload=payload, cost_usd=cost, input_tokens=1, output_tokens=1,
        cache_read_tokens=None, cache_creation_tokens=None, num_turns=1,
        duration_ms=1, session_id=None,
        artifact_path=Path("/tmp/unused.jsonl"), repair_used=False,
        raw_result_message={"total_cost_usd": cost},
    )


def _fail_transient(cost: float) -> TransientAgentError:
    e = TransientAgentError("[stage/x] transient: 529")
    e.result_msg = {"total_cost_usd": cost, "usage": {"input_tokens": 10}}
    return e


def test_on_attempt_fires_per_retry_attempt(tmp_path: Path, monkeypatch):
    """Three attempts (fail, fail, success): on_attempt must fire three
    times, so the ledger sees the full spend, not just the final attempt."""
    seen: list[float] = []
    attempts = {"n": 0}

    async def fake_once(**kwargs):
        attempts["n"] += 1
        on_attempt = kwargs["on_attempt"]
        if attempts["n"] == 1:
            on_attempt({"total_cost_usd": 1.0})
            raise _fail_transient(1.0)
        if attempts["n"] == 2:
            on_attempt({"total_cost_usd": 2.0})
            raise _fail_transient(2.0)
        on_attempt({"total_cost_usd": 4.0})
        return _result({"ok": True}, 4.0)

    monkeypatch.setattr(runner_mod, "_run_agent_once", fake_once)
    out = asyncio.run(run_agent(
        stage="hunt", prompt_file=tmp_path / "p.md",
        user_input={}, schema_file=tmp_path / "s.json",
        allowed_tools=[], model="m", cwd=tmp_path,
        artifact_dir=tmp_path, artifact_name="a",
        transient_retries=3, transient_base_delay=0.0,
        on_attempt=lambda msg: seen.append(msg["total_cost_usd"]),
    ))

    assert out.payload == {"ok": True}
    assert seen == [1.0, 2.0, 4.0], (
        "every attempt's spend must be recorded, not only the surviving one"
    )


def test_on_attempt_fires_when_all_retries_exhausted(tmp_path: Path, monkeypatch):
    """All attempts fail: on_attempt fired per attempt, then the final
    transient error propagates for the stage to handle."""
    seen: list[float] = []
    attempts = {"n": 0}

    async def fake_once(**kwargs):
        attempts["n"] += 1
        kwargs["on_attempt"]({"total_cost_usd": float(attempts["n"])})
        raise _fail_transient(float(attempts["n"]))

    monkeypatch.setattr(runner_mod, "_run_agent_once", fake_once)
    with pytest.raises(TransientAgentError):
        asyncio.run(run_agent(
            stage="hunt", prompt_file=tmp_path / "p.md",
            user_input={}, schema_file=tmp_path / "s.json",
            allowed_tools=[], model="m", cwd=tmp_path,
            artifact_dir=tmp_path, artifact_name="a",
            transient_retries=2, transient_base_delay=0.0,
            on_attempt=lambda msg: seen.append(msg["total_cost_usd"]),
        ))

    assert seen == [1.0, 2.0, 3.0]

def test_weekly_limit_classifies_as_quota_not_transient():
    """A weekly-limit response is terminal quota, not a transient blip:
    classifying it transient burned 25 minutes of backoff across 136
    attempts on a live run instead of aborting into a resumable state."""
    from audit.runner import _classify_api_error, QuotaExhaustedError
    label, exc_cls = _classify_api_error(
        "You've hit your weekly limit · resets 1pm (Asia/Dhaka)")
    assert label == "quota_exhausted"
    assert exc_cls is QuotaExhaustedError
