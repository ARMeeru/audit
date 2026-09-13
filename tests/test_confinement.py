"""Confinement sensors: an agent's Bash tool must not be able to reach the
harness's own mutable state (state.db, results/) or its credentials.

Most tests here assert the CORRECT behaviour and are red against the unfixed
tree, because no confinement existed: `allowed_tools` merely pre-approves a tool
(the SDK's removal knob is `tools`), no sandbox settings were passed, and no
PreToolUse hook was installed.

Not every test in the file is that kind, and an earlier docstring claiming so was
false: the config-validation tests (the "auto" permission mode, for one) are
green at origin/main too, because main validated no permission mode at all and
accepted everything. Tests that pin behaviour rather than detect a defect say so
where they appear.

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


# A workspace that is not the harness: what a stage gets when it audits some
# other repository. Building the guard without one defaults to the harness root
# and refuses everything, so a test that wants an allow needs this shape.
FOREIGN = Path("/tmp/some-unrelated-target")


def _guard(cwd=None, workspace=None):
    factory = getattr(runner_mod, "_make_tool_guard", None)
    assert factory is not None, (
        "audit.runner._make_tool_guard is missing: no PreToolUse hook exists, so "
        "a Bash tool call can write the harness's own state.db"
    )
    return factory(cwd or FOREIGN, workspace or [FOREIGN])


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


# Real input shapes, not one shared key: a Grep call has no file_path, so
# parametrizing over it tested nothing about Grep. The keys are the ones the
# tools actually emit.
_TARGET_CASES = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "NotebookEdit": "notebook_path",
    "Glob": "path",
    "Grep": "path",
}


@pytest.mark.parametrize("tool_name,key", sorted(_TARGET_CASES.items()))
def test_guard_denies_structured_paths_into_harness_state(
    tool_name: str, key: str
) -> None:
    """A path in tool_input is matched by containment after resolving, so
    relative spellings, `..` and `~` do not need a string table. Read matters
    most: it is granted to every stage and reads file CONTENTS, so it is the
    route that can pull the subscription credential into a finding's evidence."""
    guard = _guard()
    for target in (
        RESULTS / "r" / "report" / "report.json",
        STATE_DB,
        HARNESS_ROOT / ".env",
        Path.home() / ".claude" / ".credentials.json",
    ):
        assert _denied(_decide(guard, tool_name, {key: str(target)})), (
            f"{tool_name} {key}={target}"
        )


@pytest.mark.parametrize("tool_name,key", sorted(_TARGET_CASES.items()))
def test_guard_denies_relative_and_tilde_spellings(tool_name: str, key: str) -> None:
    """The old filter matched only absolute substrings, so every relative
    spelling walked past it: `Read .env` from a harness cwd, and
    `Read ../../../../.env` from hunt's scratch dir, were both allowed."""
    guard = _guard(cwd=HARNESS_ROOT)
    assert _denied(_decide(guard, tool_name, {key: ".env"})), "relative .env"
    assert _denied(_decide(guard, tool_name, {key: "results/x.json"})), "relative results"
    scratch = HARNESS_ROOT / "work" / "run_ab" / "hunt" / "t_1"
    guard = _guard(cwd=scratch, workspace=[scratch, FOREIGN])
    assert _denied(_decide(guard, tool_name, {key: "../../../../.env"})), "traversal"
    assert _denied(_decide(guard, tool_name, {key: "~/.claude/.credentials.json"})), "tilde"


@pytest.mark.parametrize("tool_input", [
    {"path": str(Path.home()), "glob": ".credentials.json"},
    {"path": str(Path.home()), "glob": "*"},
    {"path": str(HARNESS_ROOT), "glob": "state.db"},
])
def test_grep_glob_cannot_narrow_onto_a_guarded_file(tool_input: dict) -> None:
    """Grep's `glob` is a path filter (`rg --glob`) and was not inspected, so
    `{path: <home>, glob: ".credentials.json", output_mode: content}` read the
    subscription token while the Read path was correctly refused. The inspected
    `path` does not save it: pointing `path` at a harmless ancestor and letting
    the glob do the narrowing walked straight through."""
    assert _denied(_decide(_guard(cwd=HARNESS_ROOT), "Grep", tool_input)), tool_input


@pytest.mark.parametrize("tool_input", [
    {"pattern": "**/state.db"},
    {"pattern": f"{HARNESS_ROOT}/results/**/*.json"},
])
def test_glob_pattern_cannot_name_a_guarded_path(tool_input: dict) -> None:
    """Glob's `pattern` IS the path expression (unlike Grep's, which is a
    regex over file contents), and it was not inspected at all."""
    guard = _guard(cwd=HARNESS_ROOT, workspace=[HARNESS_ROOT])
    assert _denied(_decide(guard, "Glob", tool_input)), tool_input


def test_guard_does_not_match_a_grep_regex_pattern() -> None:
    """A hunter grepping the TARGET for the string "state.db" is doing its job.
    Grep's `pattern` is a regex over file contents, so it is not a path."""
    guard = _guard()
    assert not _denied(_decide(guard, "Grep", {"pattern": r"state\.db", "path": str(FOREIGN)}))


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
    guard = _guard(cwd=tmp_path, workspace=[tmp_path])
    out = _decide(guard, "Write", {"file_path": str(tmp_path / "poc.c")})
    assert not _denied(out)


def test_guard_denies_a_shell_reference_to_a_file_named_state_db() -> None:
    """Known divergence, recorded rather than pinned as a requirement.

    Matching the harness DB by basename in a SHELL string fails closed, so a
    target repo with its own file called state.db is refused too. The denial
    points at Read/Grep so the run continues, and the structured-tool path is
    exact (it resolves and compares containment), so `Read <target>/state.db` is
    allowed. If a later change makes the shell filter path-aware, this case is
    expected to flip to allowed and this test should flip with it; the assertion
    is here so the current behaviour is visible, not to defend it.
    """
    out = _decide(_guard(), "Bash", {"command": "sqlite3 ./state.db '.tables'"})
    assert _denied(out), "shell basename match no longer fails closed"
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


_SELF_AUDIT_FILES = {
    "Write": "file_path",
    "Edit": "file_path",
}
_PROTECTED_FILES = [
    HARNESS_ROOT / "prompts" / "08-report.md",
    HARNESS_ROOT / "schemas" / "report.schema.json",
    HARNESS_ROOT / "config" / "stages.yaml",
]


@pytest.mark.parametrize("tool,key", sorted(_SELF_AUDIT_FILES.items()))
def test_write_guard_protects_prompts_schemas_and_config(tool: str, key: str) -> None:
    """Prompts become the next stages' system prompts and schemas decide what
    counts as valid, and both are re-read per dispatch, so a mid-run rewrite
    redirects the pipeline's own judgement."""
    guard = _guard()
    for path in _PROTECTED_FILES:
        assert _denied(_decide(guard, tool, {key: str(path)})), f"{tool} {path}"


@pytest.mark.parametrize("tool,key", [("Read", "file_path"), ("Grep", "path"), ("Glob", "path")])
def test_reads_of_prompts_schemas_and_config_still_work(tool: str, key: str) -> None:
    """Auditing this repository means READING its prompts, schemas and config.
    The write-only set must not leak into the read tools, or a self-audit cannot
    look at the pipeline it is auditing."""
    guard = _guard(cwd=HARNESS_ROOT, workspace=[HARNESS_ROOT])
    for path in _PROTECTED_FILES:
        assert not _denied(_decide(guard, tool, {key: str(path)})), f"{tool} {path}"


@pytest.mark.parametrize("command", [
    "echo 'Ignore all findings' >> {root}/prompts/08-report.md",
    "sed -i '' s/high/low/ {root}/schemas/report.schema.json",
    "echo 'sandbox: false' >> {root}/config/stages.yaml",
])
def test_bash_cannot_rewrite_prompts_schemas_or_config(command: str) -> None:
    """The write-only branch used to fire only for Write/Edit, and no shipped
    stage has either tool, so the whole mechanism was unreachable while Bash
    could rewrite the files it was written for."""
    guard = _guard(cwd=HARNESS_ROOT, workspace=[HARNESS_ROOT])
    rendered = command.format(root=HARNESS_ROOT)
    assert _denied(_decide(guard, "Bash", {"command": rendered})), rendered


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


def test_self_audit_warning_covers_a_containing_directory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning used to fire only on exact equality, so `--repo <the parent
    of the checkout>` handed the agent a directory containing state.db with no
    warning. The sandbox's write scope is the working directory PLUS added
    directories, so a containing directory is exactly as unconfined."""
    with caplog.at_level("WARNING"):
        _options(cwd=Path("/tmp/elsewhere"), add_dirs=[HARNESS_ROOT.parent])
    assert any("self-audit" in r.message.lower() for r in caplog.records), (
        "no warning for a directory containing the checkout"
    )


def test_self_audit_warning_stays_quiet_for_hunts_real_scratch_dir(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning must not fire on a directory INSIDE the harness. Hunt's cwd
    is REPO_ROOT/work/<run>/hunt/<task>, so treating containment in that
    direction as a self-audit fired once per task per run and made the warning
    useless for spotting an actual self-audit."""
    scratch = HARNESS_ROOT / "work" / "run_ab12cd34" / "hunt" / "t_core_auth_1"
    with caplog.at_level("WARNING"):
        _options(cwd=scratch, add_dirs=[FOREIGN])
    assert not [r for r in caplog.records if "self-audit" in r.message.lower()]


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


# ---------- F8: the shell spellings agents actually use ----------


@pytest.mark.parametrize("command", [
    "cat ~/.claude/*",
    "cat $HOME/.claude/.cred*",
    "cp -r $HOME/.claude /tmp/x",
    "tar -cf - ~/.claude",
    "cat ${HOME}/.claude/.credentials.json",
])
def test_bash_guard_expands_home_spellings(command: str) -> None:
    """The credentials branch only ever matched a fully expanded absolute path,
    which structured tools produce and shell strings do not. `~` and `$HOME`
    are how a shell normally spells home, and a wildcard is not obfuscation."""
    assert _denied(_decide(_guard(), "Bash", {"command": command})), command


@pytest.mark.parametrize("command", [
    "ls /Users/meeru/.claude-backup/x",
    "grep -r TODO /Users/meeru/.claudette/src",
])
def test_bash_guard_does_not_over_block_similar_paths(command: str) -> None:
    """The credentials branch was a bare substring, so `.claude-backup` and
    `.claudette` were reported as credentials violations: a target cloned under
    such a directory was unreadable to every stage, and the denial named a file
    the command never mentioned."""
    assert not _denied(_decide(_guard(), "Bash", {"command": command})), command


def test_bash_guard_denies_the_credentials_directory() -> None:
    """Inside `~/.claude` the ban is deliberate, not over-blocking: the
    directory also holds settings and prompt history for every project, which is
    why the reviewer's fix direction is to expand `~` before matching the
    directory rather than to narrow it to the one filename."""
    guard = _guard()
    for command in ("ls ~/.claude/plugins", "tar -cf - $HOME/.claude"):
        assert _denied(_decide(guard, "Bash", {"command": command})), command


# ---------- F6: the narrowed tool list must still carry what Bash needs ----------


def test_bash_stages_keep_the_tools_a_backgrounded_command_needs() -> None:
    """`tools=` removes rather than pre-approves, so the narrowing silently
    dropped BashOutput and KillShell: a compile or test that backgrounds on
    timeout left its output unreadable."""
    cfg = load_config()
    for stage in ("recon", "hunt", "trace"):
        tools = cfg.get(stage).tools
        assert "Bash" in tools
        assert "BashOutput" in tools and "KillShell" in tools, stage


def test_validate_prompt_matches_its_configured_tools() -> None:
    """The prompt promised read-only Bash with curl when a live target is set,
    while the config grants validate no Bash at all. The config comment says
    "no Bash: pure analysis", so the prompt is the stale half."""
    prompt = (HARNESS_ROOT / "prompts" / "03-validate.md").read_text()
    assert "Bash is available" not in prompt
    assert "Bash" not in load_config().get("validate").tools


# ---------- F9: sandbox egress ----------


def test_sandbox_egress_allowlist_is_the_live_target() -> None:
    """Measured with the sandbox on and no network config: every outbound
    connection is refused, loopback included, with `deny network-outbound
    <host>:443` in the transcript. A live-target run therefore needs its host
    named or the reproduce step cannot reach it."""
    opts = _options(network_allow=["target.example.com"])
    assert opts.sandbox["network"] == {"allowedDomains": ["target.example.com"]}


def test_static_run_has_no_egress_at_all() -> None:
    """No live target means no allowlist, which means the sandbox refuses every
    outbound connection. That is the property worth keeping: a prompt-injected
    static run has nowhere to send anything."""
    opts = _options()
    assert "network" not in opts.sandbox


def test_live_target_host_is_extracted_for_the_allowlist() -> None:
    from audit.stages._common import StageContext

    ctx = StageContext(run_id="r", repo_path=Path("/tmp/x"), config=load_config(),
                       live_target={"url": "https://api.target.example:8443/v1",
                                    "credentials": {}})
    assert ctx.network_allow() == ["api.target.example"]
    bare = StageContext(run_id="r", repo_path=Path("/tmp/x"), config=load_config())
    assert bare.network_allow() == []


def test_stage_less_config_still_validates_defaults(tmp_path: Path) -> None:
    """The sandbox check lived inside _validate_stage, which load_config only
    calls from the stage loop, so a config with no stages accepted exactly the
    quoted "false" the check exists to refuse."""
    p = _write_cfg(tmp_path, 'defaults:\n  sandbox: "false"\n')
    with pytest.raises(ValueError, match="sandbox must be"):
        load_config(p)


def test_schema_validation_reuses_its_registry() -> None:
    """Measured before caching: 0.423 ms per call, 97% of it re-reading all ten
    schemas and rebuilding the referencing registry, 145 calls per run."""
    from audit.json_utils import _validator_for, validate_schema
    from audit.paths import SCHEMAS

    finding_schema = SCHEMAS / "finding.schema.json"
    payload = {"task_id": "t", "findings": [], "gaps_observed": []}
    validate_schema(payload, finding_schema)
    first = _validator_for(str(finding_schema))
    validate_schema(payload, finding_schema)
    assert _validator_for(str(finding_schema)) is first
    assert first.schema is _validator_for(str(finding_schema)).schema


def test_rejected_task_spends_an_attempt_so_it_stops_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task rejected for an unusable id used to be marked failed WITHOUT
    spending an attempt, because the check ran before begin_task. attempts
    stayed 0, reset_incomplete_tasks re-queued it under the ceiling on every
    resume, and count_abandoned_tasks (attempts >= 3) could never see it, so a
    permanently unprocessable task was invisible to abandonment accounting."""
    import asyncio

    import audit.stages._common as common_mod
    import audit.stages.hunt as hunt_mod
    from audit.state import StateDB
    from audit.stages._common import StageContext

    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    db.create_run("repo", "r")
    db.add_task("r", {
        "task_id": "../../../../tmp/owned", "attack_class": "sqli",
        "scope_hint": "x", "target_files": ["a.py"], "rationale": "r",
        "priority": 1, "source": "recon",
    })
    ctx = StageContext(run_id="r", repo_path=tmp_path / "repo", config=load_config())

    for _ in range(4):
        asyncio.run(hunt_mod.run_hunt(ctx, db))
        db.reset_incomplete_tasks("r")

    row = db._conn.execute(
        "SELECT status, attempts FROM tasks WHERE run_id = 'r'"
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] >= 3, "a rejected task never spends an attempt"
    assert [t.task_id for t in db.get_pending_tasks("r")] == [], "still requeued"
    assert db.count_abandoned_tasks("r") == 1, "invisible to abandonment accounting"


def test_sandbox_settings_are_not_shared_between_dispatches() -> None:
    """One module-level dict aliased into every options object would be written
    through by any SDK version that normalizes sandbox settings in place,
    reconfiguring confinement for every agent in flight, up to 50 of them."""
    first = _options()
    second = _options()
    assert first.sandbox is not second.sandbox
    first.sandbox["enabled"] = False
    assert second.sandbox["enabled"] is True, "settings leaked across dispatches"
