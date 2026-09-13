"""Confinement sensors: an agent's Bash tool must not be able to reach the
harness's own mutable state (state.db, results/) or its credentials.

Every test here asserts the CORRECT behaviour. Against the pre-fix tree they
fail because no confinement exists at all: `allowed_tools` merely pre-approves a
tool (the SDK's removal knob is `tools`), no sandbox settings are passed, and no
PreToolUse hook is installed.

Two layers are under test and they are NOT equivalent:

  * the sandbox (SDK `sandbox` settings) is the boundary: commands run under the
    OS sandbox, so a write outside the agent's own working directory fails.
  * the tool guard (a PreToolUse hook) is a FILTER over the command string. It
    stops the direct and the obvious, and it is what stands in the way when the
    sandbox cannot start (or when the audited repo IS the harness). Obfuscation
    defeats it. It must never be described, here or in the docs, as the boundary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import audit.runner as runner_mod
from audit.config import load_config

# Derived from the module location rather than imported from the code under
# test, so a broken implementation cannot move the target and stay green.
HARNESS_ROOT = Path(runner_mod.__file__).resolve().parent.parent
STATE_DB = HARNESS_ROOT / "state.db"
RESULTS = HARNESS_ROOT / "results"


def _guard():
    factory = getattr(runner_mod, "_make_tool_guard", None)
    assert factory is not None, (
        "audit.runner._make_tool_guard is missing: no PreToolUse hook exists, so "
        "a Bash tool call can write the harness's own state.db"
    )
    return factory()


def _decide(guard, tool_name: str, tool_input: dict) -> dict:
    event = {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": "tu_1",
    }
    return asyncio.run(guard(event, None, {"signal": None}))


def _denied(out: dict) -> bool:
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def _reason(out: dict) -> str:
    return out.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


# ---------- the guard denies the harness's own state ----------


@pytest.mark.parametrize(
    "command",
    [
        f"sqlite3 {STATE_DB} \"update findings set severity='info'\"",
        f"rm -f {STATE_DB}-wal {STATE_DB}-shm",
        f"cp /tmp/forged.json {STATE_DB}",
        f"python3 -c \"import sqlite3; sqlite3.connect('{STATE_DB}')\"",
        f"ls -la {RESULTS}/run_ab12/report",
        f"echo '{{}}' > {RESULTS}/run_ab12/report/report.json",
        f"sed -i s/confirmed/rejected/ {RESULTS}/run_ab12/validate/f_1.jsonl",
        "cat ~/.claude/.credentials.json",
        "cp ~/.claude/.credentials.json /tmp/creds.json",
        f"cat {HARNESS_ROOT}/.env",
        f"cd {HARNESS_ROOT} && sqlite3 state.db 'delete from findings'",
    ],
)
def test_guard_denies_harness_state_references(command: str) -> None:
    out = _decide(_guard(), "Bash", {"command": command})
    assert _denied(out), f"guard allowed a harness-state command: {command}"


@pytest.mark.parametrize(
    "tool_name",
    ["Write", "Edit", "Read"],
)
def test_guard_denies_structured_paths_into_harness_state(tool_name: str) -> None:
    """Write/Edit/Read carry a real path in tool_input, so those are matched by
    containment rather than by string search."""
    guard = _guard()
    assert _denied(_decide(guard, tool_name,
                           {"file_path": str(RESULTS / "r" / "report" / "report.json")}))
    assert _denied(_decide(guard, tool_name, {"file_path": str(STATE_DB)}))


def test_denial_message_says_why() -> None:
    """A bare denial makes the model retry the same call. The reason is what
    turns it into a redirection."""
    out = _decide(_guard(), "Bash", {"command": f"sqlite3 {STATE_DB} 'select 1'"})
    reason = _reason(out)
    assert reason, "denial carried no reason"
    assert "harness" in reason.lower() or "state.db" in reason


# ---------- the guard does not break the product ----------


@pytest.mark.parametrize(
    "tool_name,tool_input",
    [
        ("Bash", {"command": "grep -rn 'subprocess' internal/ | head -50"}),
        ("Bash", {"command": "cd /tmp/audit-scratch/poc_1 && gcc -o poc poc.c && ./poc"}),
        ("Bash", {"command": "python3 -c 'import requests; print(requests.get(\"http://127.0.0.1:8000/health\").status_code)'"}),
        ("Bash", {"command": "git log --oneline -20"}),
        ("Bash", {"command": "find . -name '*.go' | wc -l"}),
        ("Bash", {"command": "sqlite3 ./app.db '.tables'"}),
        ("Bash", {"command": "cat /etc/hosts"}),
        ("Bash", {"command": "go test ./..."}),
    ],
)
def test_guard_allows_legitimate_agent_work(tool_name: str, tool_input: dict) -> None:
    """A guard that denies everything is not a fix, it is a breakage. Hunt must
    still compile and run a PoC in its scratch dir, and a target repo that ships
    its own database must still be readable under its own name."""
    out = _decide(_guard(), tool_name, tool_input)
    assert not _denied(out), f"guard broke legitimate work: {tool_input}"


def test_guard_allows_structured_writes_into_the_target(tmp_path: Path) -> None:
    out = _decide(_guard(), "Write", {"file_path": str(tmp_path / "poc.c")})
    assert not _denied(out)


def test_guard_over_blocks_a_target_file_named_state_db() -> None:
    """The documented cost of matching the harness DB by basename: a target repo
    with its own file called state.db is denied too. The denial is fail-closed
    on purpose, and the reason redirects the agent to Read/Grep, so the run
    continues rather than dying. Recorded here so the trade-off is explicit."""
    out = _decide(_guard(), "Bash", {"command": "sqlite3 ./state.db '.tables'"})
    assert _denied(out)
    assert "state.db" in _reason(out)


# ---------- the SDK options carry the confinement ----------


def _options(**overrides):
    build = getattr(runner_mod, "_build_options", None)
    assert build is not None, (
        "audit.runner._build_options is missing: the options object is built "
        "inline, so nothing can assert what confinement it carries"
    )
    kwargs = dict(
        system_prompt="sys",
        allowed_tools=["Read", "Grep", "Glob"],
        model="claude-sonnet-4-6",
        max_turns=25,
        cwd=Path("/tmp/target"),
        add_dirs=[Path("/tmp/target")],
        permission_mode="acceptEdits",
    )
    kwargs.update(overrides)
    return build(**kwargs)


def test_options_restrict_the_base_tool_set() -> None:
    """`allowed_tools` only pre-approves; `tools` removes. Without it a stage
    configured with no Bash (validate, gapfill, feedback) can still call Bash."""
    opts = _options()
    assert list(opts.tools) == ["Read", "Grep", "Glob"]
    assert list(opts.allowed_tools) == ["Read", "Grep", "Glob"]


def test_options_enable_the_bash_sandbox() -> None:
    opts = _options()
    assert opts.sandbox is not None, "no sandbox settings passed"
    assert opts.sandbox["enabled"] is True
    assert opts.sandbox["autoAllowBashIfSandboxed"] is True
    assert opts.sandbox["allowUnsandboxedCommands"] is False, (
        "with unsandboxed commands allowed the model can step out of the "
        "sandbox via dangerouslyDisableSandbox and reach the harness tree"
    )


def test_options_install_the_pre_tool_use_guard() -> None:
    opts = _options()
    assert opts.hooks, "no hooks installed: nothing gates a Bash call"
    matchers = opts.hooks["PreToolUse"]
    assert matchers, "PreToolUse registered with no matcher"
    assert any("Bash" in (m.matcher or "") for m in matchers), (
        "guard not attached to Bash"
    )
    assert any(m.hooks for m in matchers), "matcher carries no callback"


def test_options_leave_read_only_calls_out_of_the_hook_path() -> None:
    """The guard only pays a callback round trip on Bash-family tools: a hunt
    wave is 50 concurrent agents and their Read/Grep calls are the majority."""
    opts = _options()
    matcher = opts.hooks["PreToolUse"][0].matcher or ""
    assert "Bash" in matcher
    assert "Read" not in matcher


def test_sandbox_can_be_switched_off_by_config() -> None:
    """The flag is a real switch: on a platform where the sandbox cannot start,
    an operator turns it off and keeps the hook, which is documented as a
    filter rather than a boundary."""
    opts = _options(sandbox=False)
    assert opts.sandbox is None or opts.sandbox.get("enabled") is False


def test_self_audit_warns_that_the_boundary_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the target repo IS the harness, cwd/add_dirs grant the agent its own
    harness root, so no path-based boundary can separate the two and only the
    filter stands between a hunter and state.db. That has to be visible in the
    log, not silently assumed away."""
    with caplog.at_level("WARNING"):
        _options(cwd=HARNESS_ROOT, add_dirs=[HARNESS_ROOT])
    assert any("self-audit" in r.message.lower() for r in caplog.records), (
        "a self-audit run must warn that confinement cannot separate the "
        "harness from the target"
    )


def test_normal_run_does_not_warn_about_self_audit(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Control for the test above: the warning is about the target being the
    harness, so an ordinary target must stay quiet."""
    with caplog.at_level("WARNING"):
        _options(cwd=tmp_path, add_dirs=[tmp_path])
    assert not [r for r in caplog.records if "self-audit" in r.message.lower()]


# ---------- config cannot express an unconfined stage ----------


def _write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "stages.yaml"
    p.write_text(body)
    return p


def test_config_rejects_unknown_tool_names(tmp_path: Path) -> None:
    """A typo used to be forwarded verbatim into allowed_tools, so a stage meant
    to have no Bash could be granted something else entirely."""
    p = _write_cfg(tmp_path, """
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read, Bashful]
""")
    with pytest.raises(ValueError, match="Bashful"):
        load_config(p)


def test_config_rejects_bypass_permissions(tmp_path: Path) -> None:
    """`# never bypassPermissions` was a comment. It is an invariant now."""
    p = _write_cfg(tmp_path, """
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
    permission_mode: bypassPermissions
""")
    with pytest.raises(ValueError, match="bypassPermissions"):
        load_config(p)


def test_config_rejects_unknown_permission_mode(tmp_path: Path) -> None:
    p = _write_cfg(tmp_path, """
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
    permission_mode: yolo
""")
    with pytest.raises(ValueError, match="yolo"):
        load_config(p)


def test_shipped_config_requests_the_sandbox() -> None:
    cfg = load_config()
    for name in ("recon", "hunt", "validate", "gapfill", "dedupe", "trace",
                 "feedback", "report"):
        assert cfg.get(name).sandbox is True, f"{name}: confinement not requested"


def test_config_sandbox_flag_is_readable(tmp_path: Path) -> None:
    p = _write_cfg(tmp_path, """
defaults:
  sandbox: false
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
  trace:
    model: m
    concurrency: 1
    tools: [Read]
    sandbox: true
""")
    cfg = load_config(p)
    assert cfg.get("hunt").sandbox is False
    assert cfg.get("trace").sandbox is True, "per-stage override ignored"
