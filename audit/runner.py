"""Run one agent: open a ClaudeSDKClient session, send a JSON input,
parse + schema-validate the final JSON output, and persist a JSONL
artifact of every message exchanged.

Always uses ClaudeSDKClient (not query()) so that a schema-validation
failure can be followed up with a repair turn inside the same session.

API-error handling: the Claude CLI surfaces 529 Overloaded and
subscription-quota-exhausted errors as `ResultMessage(is_error=True)` with
the error text in place of a real assistant response. We detect this
BEFORE schema validation, classify the error, and either retry with
exponential backoff (transient) or raise QuotaExhaustedError (terminal).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from audit.json_utils import extract_json, validate_schema
from audit.paths import (
    GUARDED_TOOLS,
    REPO_ROOT,
    _TOOL_INPUTS,
    bash_guard_hit,
    canonical_path,
    safe_component,
    structured_path_hit,
)

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    payload: dict
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_creation_tokens: int | None
    num_turns: int | None
    duration_ms: int | None
    session_id: str | None
    artifact_path: Path
    repair_used: bool
    raw_result_message: dict = field(default_factory=dict)


class AgentRunError(RuntimeError):
    """Schema validation failed after repair attempts (model produced
    parseable output that didn't match the schema)."""


class TransientAgentError(RuntimeError):
    """API returned a transient error (529 Overloaded, generic 5xx).
    The agent call should be retried with backoff."""


class QuotaExhaustedError(RuntimeError):
    """The Claude subscription has run out of quota. Don't retry — abort
    the pipeline and let the user wait for the reset window."""


_QUOTA_MARKERS = (
    "out of extra usage",
    "usage limit reached",
    # Subscription session/usage caps that reset on a timer, e.g.
    # "You've hit your session limit · resets 5:10am (UTC)". The reset is
    # often hours out, so backoff-retrying is futile — treat it as terminal
    # and let the caller abort into a resumable state.
    "session limit",
    "weekly limit",
    "your plan has no remaining",
)

# Subscription limit wording changes ("usage limit reached" vs "your usage
# limit ·" vs "5-hour limit"); an exact-phrase allowlist needed a new entry
# for every rewording and one miss cost 136 attempts of backoff on a live
# run. Match the limit-plus-reset SHAPE instead: "hit your ... limit".
# "Approaching your usage limit; upgrade for more" is a warning, not a
# block, and does not match -- keep that case in the sensor.
_LIMIT_SHAPE_RE = re.compile(r"hit your\b[^.]{0,32}\blimit")

_TRANSIENT_MARKERS = (
    "api error: 529",
    "overloaded",
    "api error: 503",
    "api error: 502",
    "api error: 504",
    "api error: 500",
    "rate_limit",
    "temporarily unavailable",
    "service unavailable",
)


def _classify_api_error(
    text: str, status: int | None = None
) -> tuple[str, type[RuntimeError]]:
    """Return (label, exception_class) for an is_error response.

    Status first: a 429 is a usage limit whatever the prose says (the SDK
    has exposed api_error_status since CLI v2.1.110). The text fallback
    matches exact markers and the limit-plus-reset shape."""
    if status == 429:
        return "quota_exhausted", QuotaExhaustedError
    t = (text or "").lower()
    if any(m in t for m in _QUOTA_MARKERS) or _LIMIT_SHAPE_RE.search(t):
        return "quota_exhausted", QuotaExhaustedError
    if any(m in t for m in _TRANSIENT_MARKERS):
        return "transient", TransientAgentError
    # Default to transient — better to retry once than abort on classification miss.
    return "unknown_api_error", TransientAgentError


# ---------- confinement ----------
#
# Two layers, and they are not equivalent.
#
# The sandbox (SDK `sandbox` settings) is the boundary. Commands run inside the
# OS sandbox, which allows writes under the agent's own working directory and the
# directories it was granted, and refuses them elsewhere, so a hunter that has
# been talked into rewriting state.db cannot: the syscall fails.
#
# The tool guard below is a FILTER. For structured tools it resolves the path and
# compares containment, which is exact. For Bash there is no structure to
# resolve, so it is a string match, and a path built at runtime
# (`p=$(printf %s <b64>|base64 -d); sqlite3 "$p" ...`) walks past it. It is the
# only layer in the two geometries where no sandbox can separate harness from
# target: a self-audit, and a platform where the sandbox cannot start. Defence in
# depth, never the boundary, and it must not be described as one.

_SANDBOX_SETTINGS: dict[str, Any] = {
    "enabled": True,
    # Bash is auto-approved once sandboxed: the sandbox, not an interactive
    # prompt, is what makes running a PoC safe.
    "autoAllowBashIfSandboxed": True,
    # The escape hatch stays shut. Left open, the model can pass
    # dangerouslyDisableSandbox and reach the harness tree again, which makes
    # everything above decorative.
    "allowUnsandboxedCommands": False,
}

# Which tool_input fields carry a path, and which tools the matcher must name,
# both come from audit.paths so the vocabulary cannot drift: a tool the guard
# inspects but the matcher omits is a check that can never fire, which is what
# the Read, Grep and Glob entries were before this was centralised.
_GUARD_MATCHER = "|".join(sorted(GUARDED_TOOLS))


def _make_tool_guard(cwd: Path | None = None, workspace: list[Path] | None = None):
    """Build the PreToolUse callback that refuses harness-file tool calls.

    PreToolUse rather than `can_use_tool`: the SDK documents that a whole-tool
    `allowed_tools` entry plus `permission_mode="acceptEdits"` shadows the
    permission callback entirely, so `can_use_tool` would never fire for the
    tools that matter here. PreToolUse fires for every call.

    `cwd` is the session's working directory and `workspace` the directories a
    bare pattern resolves against. Built without them, the guard assumes the
    harness root, which fails closed.
    """
    session_cwd = Path(cwd) if cwd is not None else REPO_ROOT
    workspace_dirs = [Path(p) for p in (workspace or [session_cwd])]

    async def _guard(input: dict, tool_use_id: str | None, context: dict) -> dict:
        tool_name = input.get("tool_name", "")
        tool_input = input.get("tool_input") or {}
        if tool_name == "Bash":
            hit = bash_guard_hit(str(tool_input.get("command", "")), cwd=session_cwd)
        elif tool_name in _TOOL_INPUTS:
            hit = structured_path_hit(
                tool_name, tool_input, cwd=session_cwd, workspace=workspace_dirs
            )
        else:
            hit = None
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Blocked: this call reaches for {hit}. The harness's own state "
                    "is off limits: findings are recorded by the pipeline, not by an "
                    "agent, and a direct write would forge a result. Use Read, Grep "
                    "or Glob to inspect the target, and run PoCs inside your scratch "
                    "directory."
                ),
            }
        }

    return _guard


def _artifact_path(artifact_dir: Path, artifact_name: str) -> Path:
    """Where an agent's JSONL transcript goes.

    `artifact_name` is a hunt task id or a finding id: the first comes from
    recon's output, so it is derived from the target repo, and it used to be
    interpolated into the filename unchecked."""
    return artifact_dir / f"{safe_component(artifact_name, kind='artifact name')}.jsonl"


def _build_options(
    *,
    system_prompt: str,
    allowed_tools: list[str],
    model: str,
    max_turns: int,
    cwd: Path,
    add_dirs: list[Path] | None,
    permission_mode: str,
    sandbox: bool = True,
    network_allow: list[str] | None = None,
    strict_mcp_config: bool = True,
) -> ClaudeAgentOptions:
    """Assemble the SDK options, confinement included.

    Split out of _run_agent_once so the confinement can be asserted without an
    API call (and without pytest-asyncio, which the suite does not install).

    `network_allow` is the sandbox's egress allowlist, normally just the host of
    the operator's `--target-url`. Measured, not assumed: with the sandbox on and
    no network config, every outbound connection is refused, loopback included,
    with `deny network-outbound <host>:443` in the transcript. A static run
    therefore has no egress at all, which is a property worth having; a
    live-target run needs its target named here or its reproduce step cannot
    reach it.
    """
    cwd = Path(cwd)
    dirs = [Path(p) for p in (add_dirs or [])]
    # The agent was handed a directory that encloses the harness checkout, which
    # is the only overlap that matters: the sandbox's write scope is the working
    # directory PLUS the added directories, so a directory containing the
    # checkout confers access to state.db, results/ and .env. A harness-chosen
    # cwd that merely sits inside the checkout (hunt's scratch dir is
    # REPO_ROOT/work/<run>/hunt/<task>) is not this case and must not be
    # reported as one, or the warning fires on every task of every run and stops
    # meaning anything.
    granted = (cwd, *dirs)
    if any(
        canonical_path(p) == REPO_ROOT or REPO_ROOT.is_relative_to(canonical_path(p))
        for p in granted
    ):
        log.warning(
            "[confinement] self-audit: the target was granted the harness checkout "
            "itself or a directory containing it, so the OS sandbox cannot separate "
            "them. state.db and results/ sit inside it and are then protected only "
            "by the tool filter, which obfuscation can walk past."
        )
    sandbox_settings: dict[str, Any] | None = None
    if sandbox:
        # A fresh dict per dispatch: one shared mutable literal would be written
        # through by any SDK version that normalizes sandbox settings in place,
        # reconfiguring confinement for every agent already in flight.
        sandbox_settings = dict(_SANDBOX_SETTINGS)
        if network_allow:
            sandbox_settings["network"] = {"allowedDomains": list(network_allow)}
    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        # `tools` is the set that EXISTS among the CLI's built-in tools;
        # `allowed_tools` only pre-approves. Setting both to the configured list
        # means a stage configured with no Bash does not have Bash at all, rather
        # than merely failing to pre-approve it and falling through to whatever
        # permission_mode allows. Probed rather than assumed: a session declaring
        # tools=["Read"] reports BASH=absent.
        #
        # It does NOT cover MCP tools. A session with a two-name built-in list
        # still carried the operator's own `mcp__claude_ai_Google_Drive__*`
        # servers, because setting_sources=[] disables settings files but not the
        # MCP configuration the CLI loads separately. Those clients run inside
        # this process, so they sit outside the sandbox, and an MCP server with
        # write access is an exfiltration route the sandbox cannot see.
        # `--strict-mcp-config` plus `--setting-sources=` is what closes it. An
        # empty `mcp_servers` would read as if it did, but the SDK only emits
        # `--mcp-config` for a non-empty mapping, so it is dead weight.
        #
        # Caveat worth knowing (README): the CLI refuses to start with
        # `--strict-mcp-config` when an enterprise MCP config is present, which
        # is why this is a config key rather than a constant.
        tools=list(allowed_tools),
        allowed_tools=list(allowed_tools),
        strict_mcp_config=strict_mcp_config,
        model=model,
        max_turns=max_turns,
        cwd=str(cwd),
        add_dirs=[str(p) for p in dirs],
        permission_mode=permission_mode,
        setting_sources=[],
        sandbox=sandbox_settings,
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher=_GUARD_MATCHER,
                    hooks=[_make_tool_guard(cwd, [cwd, *dirs])],
                )
            ]
        },
    )


async def run_agent(
    *,
    stage: str,
    prompt_file: Path,
    user_input: dict,
    schema_file: Path,
    allowed_tools: list[str],
    model: str,
    cwd: Path,
    add_dirs: list[Path] | None = None,
    max_turns: int = 25,
    permission_mode: str = "acceptEdits",
    sandbox: bool = True,
    network_allow: list[str] | None = None,
    strict_mcp_config: bool = True,
    artifact_dir: Path,
    artifact_name: str,
    repair_attempts: int = 1,
    transient_retries: int = 3,
    transient_base_delay: float = 30.0,
    on_attempt: "Callable[[dict], None] | None" = None,
) -> AgentResult:
    """Run one agent, retrying transient API errors with exponential backoff.

    Raises `QuotaExhaustedError` if the subscription is out of quota
    (caller should abort the run). Raises `TransientAgentError` if all
    backoff retries are exhausted. Raises `AgentRunError` if the model
    produced parseable output that doesn't match the schema even after
    repair turns.

    `sandbox` is threaded from the stage config (default true). It is the
    boundary that keeps an agent's Bash out of the harness's own state; the
    PreToolUse filter installed alongside it is defence in depth, not a
    substitute.

    `on_attempt` is invoked ONCE per SDK session with that session's
    final result-message dict, whether the session succeeded or raised.
    Each transient retry opens a new session, so the retry loop's calls
    sum correctly across attempts.

    Do NOT call it per repair turn: ResultMessage.total_cost_usd is a
    RUNNING SESSION TOTAL, so summing per-turn values recorded
    1.00 + 1.80 + 2.40 for a session that cost 2.40.
    """
    last_exc: RuntimeError | None = None
    for attempt in range(transient_retries + 1):
        try:
            return await _run_agent_once(
                stage=stage,
                prompt_file=prompt_file,
                user_input=user_input,
                schema_file=schema_file,
                allowed_tools=allowed_tools,
                model=model,
                cwd=cwd,
                add_dirs=add_dirs,
                max_turns=max_turns,
                permission_mode=permission_mode,
                sandbox=sandbox,
                network_allow=network_allow,
                strict_mcp_config=strict_mcp_config,
                artifact_dir=artifact_dir,
                artifact_name=artifact_name,
                repair_attempts=repair_attempts,
                on_attempt=on_attempt,
            )
        except QuotaExhaustedError:
            raise
        except TransientAgentError as e:
            last_exc = e
            if attempt >= transient_retries:
                break
            delay = min(transient_base_delay * (2 ** attempt), 240.0)
            log.warning(
                "[%s/%s] transient API error (attempt %d/%d): %s — retrying in %.0fs",
                stage, artifact_name, attempt + 1, transient_retries + 1,
                str(e)[:160], delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _run_agent_once(
    *,
    stage: str,
    prompt_file: Path,
    user_input: dict,
    schema_file: Path,
    allowed_tools: list[str],
    model: str,
    cwd: Path,
    add_dirs: list[Path] | None,
    max_turns: int,
    permission_mode: str,
    sandbox: bool,
    network_allow: list[str] | None,
    strict_mcp_config: bool,
    artifact_dir: Path,
    artifact_name: str,
    repair_attempts: int,
    on_attempt: Callable[[dict], None] | None = None,
) -> AgentResult:
    """Single attempt. Raises TransientAgentError / QuotaExhaustedError
    before schema validation if the API returned is_error=True."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_path(artifact_dir, artifact_name)
    cwd.mkdir(parents=True, exist_ok=True)

    system_prompt = prompt_file.read_text()
    # Append the literal schema body so the model never has to guess
    # field names — this drastically reduces schema-validation failures
    # on the first attempt and frees up the repair budget for real
    # ambiguities.
    schema_text = schema_file.read_text()
    system_prompt += (
        "\n\n# Output schema\n\n"
        "Your output MUST validate against this JSON Schema. "
        "Pay attention to nested objects, required fields, and "
        "`additionalProperties: false`.\n\n"
        f"```json\n{schema_text}\n```\n"
    )
    options = _build_options(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        model=model,
        max_turns=max_turns,
        cwd=cwd,
        add_dirs=add_dirs,
        permission_mode=permission_mode,
        sandbox=sandbox,
        network_allow=network_allow,
        strict_mcp_config=strict_mcp_config,
    )

    initial_prompt = json.dumps(user_input, ensure_ascii=False)

    last_text = ""
    last_result_msg: dict[str, Any] = {}
    repair_used = False

    with artifact_path.open("w") as art:
        _write_artifact(art, {"kind": "meta", "stage": stage, "model": model, "started_at": time.time()})
        _write_artifact(art, {"kind": "user", "text": initial_prompt[:50000]})

        try:
            sdk_ctx = ClaudeSDKClient(options=options)
            client = await sdk_ctx.__aenter__()
        except Exception as e:
            if "timeout" in str(e).lower() or "initialize" in str(e).lower():
                raise TransientAgentError(
                    f"[{stage}/{artifact_name}] SDK initialize timeout: {e}"
                ) from e
            raise

        try:
            await client.query(initial_prompt)
            last_text, last_result_msg = await _drain(client, art)

            # Before schema validation: was this a real model response, or
            # did the CLI surface an API error as the assistant text?
            if last_result_msg.get("is_error"):
                label, exc_cls = _classify_api_error(
                    last_text, last_result_msg.get("api_error_status"))
                _write_artifact(art, {"kind": "api_error", "classification": label,
                                      "text": last_text[:1000]})
                e = exc_cls(
                    f"[{stage}/{artifact_name}] {label}: "
                    f"{(last_text or '').strip()[:300]}"
                )
                # The attempt still spent API usage; carry the result
                # message so stages can record it against the run's costs.
                e.result_msg = last_result_msg
                raise e

            attempts = 0
            errors = _validate(last_text, schema_file)
            while errors and attempts < repair_attempts:
                attempts += 1
                repair_used = True
                repair_prompt = _build_repair_prompt(last_text, errors, schema_file)
                _write_artifact(art, {"kind": "repair_request", "text": repair_prompt[:50000]})
                await client.query(repair_prompt)
                last_text, last_result_msg = await _drain(client, art)
                # An API error on the repair turn is also retry-worthy.
                if last_result_msg.get("is_error"):
                    label, exc_cls = _classify_api_error(
                        last_text, last_result_msg.get("api_error_status"))
                    _write_artifact(art, {"kind": "api_error_on_repair",
                                          "classification": label,
                                          "text": last_text[:1000]})
                    e = exc_cls(
                        f"[{stage}/{artifact_name}] {label} on repair turn: "
                        f"{(last_text or '').strip()[:300]}"
                    )
                    e.result_msg = last_result_msg
                    raise e
                errors = _validate(last_text, schema_file)

            if errors:
                _write_artifact(art, {"kind": "schema_errors", "errors": errors})
                e = AgentRunError(
                    f"[{stage}/{artifact_name}] schema validation failed after "
                    f"{repair_attempts} repair attempts: {errors[:5]}"
                )
                e.result_msg = last_result_msg
                raise e

            payload = extract_json(last_text)
            _write_artifact(art, {"kind": "final_payload", "payload": payload})
        finally:
            # One row per SDK session. total_cost_usd is a running session
            # total, so recording per repair turn would sum 1.00 + 1.80 +
            # 2.40 for a session that cost 2.40. Each retry is a new
            # session, so run_agent's retry loop still accumulates
            # correctly across attempts. An initialize failure never
            # drains, so last_result_msg stays empty and nothing is
            # recorded -- no spend happened.
            if on_attempt is not None and last_result_msg:
                on_attempt(last_result_msg)
            await sdk_ctx.__aexit__(None, None, None)

    usage = last_result_msg.get("usage") or {}
    return AgentResult(
        payload=payload,
        cost_usd=last_result_msg.get("total_cost_usd"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        num_turns=last_result_msg.get("num_turns"),
        duration_ms=last_result_msg.get("duration_ms"),
        session_id=last_result_msg.get("session_id"),
        artifact_path=artifact_path,
        repair_used=repair_used,
        raw_result_message=last_result_msg,
    )


async def _drain(client: ClaudeSDKClient, art) -> tuple[str, dict[str, Any]]:
    """Consume the response stream, write each message to the JSONL
    artifact, and return (concatenated assistant text from last
    assistant message, result_message_dict)."""
    text_chunks: list[str] = []
    result_msg: dict[str, Any] = {}
    last_assistant_text: list[str] = []

    async for msg in client.receive_response():
        _write_artifact(art, _serialize_message(msg))
        if isinstance(msg, AssistantMessage):
            last_assistant_text = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    last_assistant_text.append(block.text)
            text_chunks.append("".join(last_assistant_text))
        elif isinstance(msg, ResultMessage):
            result_msg = _result_to_dict(msg)

    final_text = "".join(last_assistant_text) if last_assistant_text else (
        text_chunks[-1] if text_chunks else ""
    )
    return final_text, result_msg


def _validate(text: str, schema_file: Path) -> list[str]:
    try:
        payload = extract_json(text)
    except ValueError as e:
        return [f"json_extract: {e}"]
    return validate_schema(payload, schema_file)


def _build_repair_prompt(prev_output: str, errors: list[str], schema_file: Path) -> str:
    err_block = "\n".join(f"- {e}" for e in errors[:20])
    return (
        "Your previous output failed schema validation against "
        f"`{schema_file.name}`. Errors:\n"
        f"{err_block}\n\n"
        "Re-emit the same response, fixing ONLY these errors. Output a "
        "single JSON object — no prose, no markdown fence."
    )


def _write_artifact(fp, obj: Any) -> None:
    fp.write(json.dumps(obj, default=_json_fallback, ensure_ascii=False) + "\n")
    fp.flush()


def _json_fallback(o: Any) -> Any:
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _serialize_message(msg: Any) -> dict[str, Any]:
    if isinstance(msg, AssistantMessage):
        return {
            "kind": "assistant",
            "model": msg.model,
            "usage": msg.usage,
            "content": [_serialize_block(b) for b in msg.content],
        }
    if isinstance(msg, ResultMessage):
        return {"kind": "result", **_result_to_dict(msg)}
    if dataclasses.is_dataclass(msg):
        return {"kind": type(msg).__name__, **dataclasses.asdict(msg)}
    return {"kind": type(msg).__name__, "repr": repr(msg)}


def _serialize_block(b: Any) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ThinkingBlock):
        return {"type": "thinking", "thinking": b.thinking}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.content,
            "is_error": b.is_error,
        }
    if dataclasses.is_dataclass(b):
        return dataclasses.asdict(b)
    return {"type": type(b).__name__, "repr": repr(b)}


def _result_to_dict(msg: ResultMessage) -> dict[str, Any]:
    return {
        "subtype": msg.subtype,
        "is_error": msg.is_error,
        "api_error_status": getattr(msg, "api_error_status", None),
        "duration_ms": msg.duration_ms,
        "duration_api_ms": msg.duration_api_ms,
        "num_turns": msg.num_turns,
        "session_id": msg.session_id,
        "stop_reason": msg.stop_reason,
        "total_cost_usd": msg.total_cost_usd,
        "usage": msg.usage,
        "result": msg.result,
        "model_usage": msg.model_usage,
    }
