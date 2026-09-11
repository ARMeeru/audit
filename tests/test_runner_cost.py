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


class _FakeClient:
    """Minimal ClaudeSDKClient stand-in: __aenter__/query/__aexit__ only.
    The real _drain is patched out, so receive_response is never called."""

    def __init__(self, options):
        self.options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, text):
        self.last_query = text


def _run_repair_scenario(monkeypatch, tmp_path: Path, drain_sequence, **run_kwargs):
    """Drive the REAL _run_agent_once with a patched _drain that hands back
    cumulative session totals (as the SDK does) across a repair turn."""
    import audit.runner as runner_mod

    prompt = tmp_path / "p.md"
    prompt.write_text("prompt")
    schema = tmp_path / "s.schema.json"
    schema.write_text('{"type":"object","required":["ok"],'
                      '"properties":{"ok":{"type":"boolean"}},'
                      '"additionalProperties":false}')

    calls = {"n": 0}

    async def fake_drain(client, art):
        idx = min(calls["n"], len(drain_sequence) - 1)
        text, result_msg = drain_sequence[idx]
        calls["n"] += 1
        return text, result_msg

    monkeypatch.setattr(runner_mod, "_drain", fake_drain)
    monkeypatch.setattr(runner_mod, "ClaudeSDKClient", _FakeClient)

    seen: list[float] = []

    def on_attempt(msg):  # sync: stages pass plain lambdas
        seen.append(msg.get("total_cost_usd"))

    out = asyncio.run(runner_mod.run_agent(
        stage="hunt", prompt_file=prompt,
        user_input={}, schema_file=schema,
        allowed_tools=[], model="m", cwd=tmp_path,
        artifact_dir=tmp_path, artifact_name="a",
        repair_attempts=3,
        on_attempt=on_attempt, **run_kwargs))
    return out, seen


def test_repair_turns_record_one_session_row_not_three(tmp_path: Path, monkeypatch):
    """F1/R1 regression: total_cost_usd is a cumulative session total, and
    the old per-drain firing summed 1.0 + 1.8 + 2.4 for a session costing
    2.40. One session = one ledger row carrying the final total."""
    from audit.json_utils import extract_json  # noqa: F401  (exercises import path)

    drain_sequence = [
        ('{"nope": 1}', {"total_cost_usd": 1.0, "usage": {"input_tokens": 100}}),
        ('{"nope": 1}', {"total_cost_usd": 1.8, "usage": {"input_tokens": 180}}),
        ('{"ok": true}', {"total_cost_usd": 2.4, "usage": {"input_tokens": 240}}),
    ]
    out, seen = _run_repair_scenario(monkeypatch, tmp_path, drain_sequence)

    assert out.payload == {"ok": True}
    assert seen == [2.4], (
        f"one session must record one row with the session's final total, "
        f"got {seen}"
    )


def test_retries_still_sum_across_sessions(tmp_path: Path, monkeypatch):
    """Cross-session retries are fresh SDK sessions: each contributes its
    own final total, so the ledger sums the real spend."""
    from audit.runner import TransientAgentError

    sessions = [
        # session 1: API error (drain completes, classify -> transient)
        [('api error: 529 overloaded',
          {"total_cost_usd": 1.0, "usage": {}, "is_error": True})],
        # session 2: succeeds
        [('{"ok": true}', {"total_cost_usd": 3.0, "usage": {}})],
    ]

    import audit.runner as runner_mod

    prompt = tmp_path / "p.md"
    prompt.write_text("prompt")
    schema = tmp_path / "s.schema.json"
    schema.write_text('{"type":"object","required":["ok"],'
                      '"properties":{"ok":{"type":"boolean"}},'
                      '"additionalProperties":false}')
    calls = {"n": 0}

    async def fake_drain(client, art):
        idx = min(calls["n"], len(sessions) - 1)
        result = sessions[idx]
        calls["n"] += 1
        return result[0]

    # _drain must return different content per session: wrap with session index
    async def fake_drain_by_session(client, art):
        seq = sessions[min(calls["n"], len(sessions) - 1)]
        # within a session there is exactly one drain in this scenario
        text, msg = seq[0] if isinstance(seq[0], tuple) else seq[0]
        calls["n"] += 1
        return text, msg

    monkeypatch.setattr(runner_mod, "_drain", fake_drain_by_session)
    monkeypatch.setattr(runner_mod, "ClaudeSDKClient", _FakeClient)

    seen: list[float] = []

    def on_attempt(msg):  # sync: stages pass plain lambdas
        seen.append(msg.get("total_cost_usd"))

    out = asyncio.run(runner_mod.run_agent(
        stage="hunt", prompt_file=prompt,
        user_input={}, schema_file=schema,
        allowed_tools=[], model="m", cwd=tmp_path,
        artifact_dir=tmp_path, artifact_name="a",
        transient_retries=3, transient_base_delay=0.0,
        on_attempt=on_attempt))

    assert out.payload == {"ok": True}
    assert seen == [1.0, 3.0], "per-session rows must sum across retries"

def test_quota_classification_covers_limit_wordings_and_status():
    """F9/R8: exact-phrase markers missed 'your usage limit vs', '5-hour
    limit' and 'Opus limit' rewordings -- each reproduces the incident of
    136 futile backoff attempts. Status 429 is quota regardless of prose;
    the approaching-limit warning stays transient."""
    from audit.runner import _classify_api_error, QuotaExhaustedError, TransientAgentError

    terminal = [
        "You've hit your weekly limit · resets 1pm (Asia/Dhaka)",
        "You've hit your session limit · resets 5:10am (UTC)",
        "Claude usage limit reached. Your limit will reset at 3pm",
        "You've hit your usage limit · resets 3pm (UTC)",
        "You've hit your 5-hour limit · resets 9pm",
        "You've hit your Opus limit for this week",
    ]
    for text in terminal:
        label, exc_cls = _classify_api_error(text)
        assert (label, exc_cls) == ("quota_exhausted", QuotaExhaustedError), text

    # status 429: quota whatever the prose says
    label, exc_cls = _classify_api_error("totally unrecognized prose", 429)
    assert (label, exc_cls) == ("quota_exhausted", QuotaExhaustedError)

    # the upgrade warning is a throttle notice, not a block: transient
    label, exc_cls = _classify_api_error(
        "Approaching your usage limit; upgrade for more")
    assert exc_cls is TransientAgentError
