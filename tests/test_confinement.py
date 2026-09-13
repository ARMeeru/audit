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
    ["Write", "Edit", "Read", "Grep"],
)
def test_guard_denies_structured_paths_into_harness_state(tool_name: str) -> None:
    """Write/Edit/Read/Grep carry a real path in tool_input, so those are
    matched by containment rather than by string search. Read matters most: it
    is granted to every stage, and it reads file CONTENTS, so it is the route
    that can pull the subscription credential into a finding's evidence."""
    guard = _guard()
    assert _denied(_decide(guard, tool_name,
                           {"file_path": str(RESULTS / "r" / "report" / "report.json")}))
    assert _denied(_decide(guard, tool_name, {"file_path": str(STATE_DB)}))
    assert _denied(_decide(guard, tool_name,
                           {"file_path": str(Path.home() / ".claude" / ".credentials.json")}))


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


def test_matcher_covers_every_tool_the_guard_inspects() -> None:
    """The wiring, not just the callback. A matcher that omits Read means the
    CLI never invokes the callback for a Read, so the check exists in the code
    and can never fire: narrowing the matcher to "Bash" used to leave all 290
    tests green while silently un-hooking every path-bearing Read call."""
    opts = _options()
    matcher = opts.hooks["PreToolUse"][0].matcher or ""
    for tool in ("Bash", "Write", "Edit", "Read", "Grep", "Glob"):
        assert tool in matcher, (
            f"the guard inspects {tool} input but the matcher never fires for it"
        )


def test_guard_does_not_match_a_grep_pattern() -> None:
    """A hunter grepping the TARGET for the string "state.db" is doing its job.
    Only path-bearing keys are inspected, so a pattern is not a denial."""
    out = _decide(_guard(), "Grep", {"pattern": "state\\.db", "path": "/tmp/target"})
    assert not _denied(out), "a grep pattern was treated as a path"


def test_write_guard_protects_prompts_and_schemas_but_reads_still_work() -> None:
    """Prompts become the next stages' system prompts and schemas decide what
    counts as valid, so a mid-run rewrite redirects the pipeline's judgement.
    They must stay READABLE, though: auditing this repository means reading
    them."""
    guard = _guard()
    prompts = HARNESS_ROOT / "prompts" / "08-report.md"
    schemas = HARNESS_ROOT / "schemas" / "report.schema.json"
    for tool in ("Write", "Edit"):
        assert _denied(_decide(guard, tool, {"file_path": str(prompts)})), tool
        assert _denied(_decide(guard, tool, {"file_path": str(schemas)})), tool
    assert not _denied(_decide(guard, "Read", {"file_path": str(prompts)}))
    assert not _denied(_decide(guard, "Read", {"file_path": str(schemas)}))


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


@pytest.mark.parametrize("geometry", ["parent", "descendant"])
def test_self_audit_warning_covers_containing_and_contained_directories(
    caplog: pytest.LogCaptureFixture, geometry: str
) -> None:
    """The warning used to fire only on exact equality, so `--repo <the parent
    of the checkout>` handed the agent a directory containing state.db with no
    warning at all. The sandbox's write scope is the working directory PLUS
    added directories, so a containing directory is exactly as unconfined as the
    checkout itself."""
    target = HARNESS_ROOT.parent if geometry == "parent" else HARNESS_ROOT / "audit"
    with caplog.at_level("WARNING"):
        _options(cwd=Path("/tmp/elsewhere"), add_dirs=[target])
    assert any("self-audit" in r.message.lower() for r in caplog.records), (
        f"no warning for {geometry} directory {target}"
    )


def test_normal_run_does_not_warn_about_self_audit(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Control for the test above: the warning is about the target overlapping
    the harness, so an ordinary target must stay quiet."""
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


@pytest.mark.parametrize("value", ["null", "0", "[]", "{}", "'false'", "'off'"])
def test_config_refuses_a_non_boolean_sandbox(tmp_path: Path, value: str) -> None:
    """The one consumed key whose silent misparse removes the boundary instead
    of tightening it. bool() coerces `sandbox:` (null), 0, [] and {} to False
    and a quoted "false" to True, so both a typo and an attempt to disable it
    did the wrong thing without saying so."""
    p = _write_cfg(tmp_path, f"""
defaults:
  sandbox: {value}
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
""")
    with pytest.raises(ValueError, match="sandbox must be"):
        load_config(p)


def test_config_accepts_an_unquoted_false(tmp_path: Path) -> None:
    """Control for the test above: the honest way to disable it keeps working."""
    p = _write_cfg(tmp_path, """
defaults:
  sandbox: false
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
""")
    assert load_config(p).get("hunt").sandbox is False


def test_config_accepts_the_sdk_auto_permission_mode(tmp_path: Path) -> None:
    """The pinned SDK's PermissionMode includes "auto". Refusing it would print
    "unknown permission_mode" at an operator who had used a real one."""
    p = _write_cfg(tmp_path, """
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
    permission_mode: auto
""")
    assert load_config(p).get("hunt").permission_mode == "auto"
