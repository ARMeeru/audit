"""Confinement sensors: an agent must not be able to reach the harness's own
mutable state (state.db, results/, .env), its credentials, or the files the
pipeline judges itself by.

Two mechanisms and one deliberate non-mechanism:

  * the OS sandbox (SDK `sandbox` settings) is the boundary. Measured: writes
    outside the working directory plus `--add-dir` fail, and with no network
    config every outbound connection is refused.
  * `structured_path_hit` is exact by construction: it resolves the path a tool
    is about to use and compares by filesystem identity, so case-folded
    spellings, hardlinks, `..`, relative spellings and `~` need no string table.
  * `bash_guard_hit` is a speed bump with a written scope. Its non-scope is
    documented in its own docstring rather than tested, because three review
    rounds of expanding it produced eight known bypasses and a hook that could
    block every concurrent agent for 41 seconds. What is tested here is what it
    claims: home spellings, the guarded names, and a path boundary so ordinary
    names in a target are not refused.
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
# other repository.
FOREIGN = Path("/tmp/some-unrelated-target")
HOME = Path.home()


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


# ---------- structured tools: exact containment ----------

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
    """Read matters most: it is granted to every stage and reads file CONTENTS,
    so it is the route that can pull a credential into a finding's evidence."""
    guard = _guard()
    targets = [
        RESULTS / "r" / "report" / "report.json",
        STATE_DB,
        STATE_DB.with_name("state.db-wal"),
        STATE_DB.with_name("state.db-shm"),
        HARNESS_ROOT / ".env",
        HOME / ".claude" / ".credentials.json",
        HOME / ".claude" / "settings.json",
        HOME / ".claude.json",
        HOME / ".claude.json.backup",
    ]
    if tool_name in ("Write", "Edit", "NotebookEdit"):
        # Only the write tools are refused the files a self-audit has to read.
        targets.append(HARNESS_ROOT / "config" / "stages.yaml")
    for target in targets:
        assert _denied(_decide(guard, tool_name, {key: str(target)})), (
            f"{tool_name} {key}={target}"
        )


@pytest.mark.parametrize("tool_name,key", sorted(_TARGET_CASES.items()))
def test_guard_denies_relative_case_folded_and_tilde_spellings(
    tool_name: str, key: str
) -> None:
    """Resolving the path is what makes these exact. Case is the one that got
    through an earlier version: APFS folds it, `resolve()` does not, so
    `<repo>/STATE.DB` was a different Path for the same file."""
    guard = _guard(cwd=HARNESS_ROOT)
    assert _denied(_decide(guard, tool_name, {key: ".env"})), "relative .env"
    assert _denied(_decide(guard, tool_name, {key: "results/x.json"})), "relative results"
    assert _denied(_decide(guard, tool_name, {key: str(STATE_DB).upper()})), "cased"
    assert _denied(
        _decide(guard, tool_name, {key: "~/.claude/.credentials.json"})
    ), "tilde"
    # A cased spelling of a path that does not exist yet: identity cannot see it
    # (nothing to stat) so the case-folded prefix is what catches it.
    assert _denied(
        _decide(guard, tool_name, {key: str(RESULTS).upper() + "/new.json"})
    ), "cased, non-existent"
    scratch = HARNESS_ROOT / "work" / "run_ab" / "hunt" / "t_1"
    guard = _guard(cwd=scratch, workspace=[scratch, FOREIGN])
    assert _denied(_decide(guard, tool_name, {key: "../../../../.env"})), "traversal"


def test_guard_denies_a_hardlink_to_the_state_database(tmp_path: Path) -> None:
    """A hardlink has no path relationship to its target, so a prefix comparison
    cannot see it: `st_dev`/`st_ino` identity can."""
    link = tmp_path / "hl_state"
    try:
        link.hardlink_to(STATE_DB)
    except OSError:
        pytest.skip("hardlinks unavailable on this filesystem")
    assert _denied(_decide(_guard(), "Read", {"file_path": str(link)}))


def test_guard_allows_a_targets_own_state_db_and_env(tmp_path: Path) -> None:
    """The other direction, and the escape hatch the denial message points at: a
    target repo that ships its own `state.db` or `.env` is target data. The check
    compares identity, so it does not fire on a different file with a name that
    matches."""
    (tmp_path / "state.db").write_text("not the harness db")
    (tmp_path / ".env").write_text("TARGET=1")
    guard = _guard(cwd=tmp_path, workspace=[tmp_path])
    for name in ("state.db", "state.db-wal", ".env"):
        assert not _denied(
            _decide(guard, "Read", {"file_path": str(tmp_path / name)})
        ), name


def test_reads_of_prompts_schemas_and_config_still_work() -> None:
    """Auditing this repository means reading its prompts, schemas and config;
    only writes to them are refused."""
    guard = _guard(cwd=HARNESS_ROOT, workspace=[HARNESS_ROOT])
    for path in (
        HARNESS_ROOT / "prompts" / "08-report.md",
        HARNESS_ROOT / "schemas" / "report.schema.json",
        HARNESS_ROOT / "config" / "stages.yaml",
    ):
        assert not _denied(_decide(guard, "Read", {"file_path": str(path)})), path


@pytest.mark.parametrize(
    "tool,key",
    [("Write", "file_path"), ("Edit", "file_path"), ("NotebookEdit", "notebook_path")],
)
def test_write_tools_cannot_rewrite_prompts_schemas_or_the_harness_source(
    tool: str, key: str
) -> None:
    """A rewrite of `audit/*.py` forges the report the operator reads next, and a
    rewrite of a prompt lands in the next stage's system prompt in this run."""
    guard = _guard()
    for path in (
        HARNESS_ROOT / "prompts" / "08-report.md",
        HARNESS_ROOT / "schemas" / "report.schema.json",
        HARNESS_ROOT / "config" / "stages.yaml",
        HARNESS_ROOT / "audit" / "paths.py",
        HARNESS_ROOT / "scripts" / "mutation-check.py",
        HARNESS_ROOT / ".git" / "hooks" / "post-commit",
        HARNESS_ROOT / "pyproject.toml",
    ):
        assert _denied(_decide(guard, tool, {key: str(path)})), f"{tool} {path}"


def test_work_is_the_one_writable_hole() -> None:
    """Hunt compiles and runs PoCs in its scratch dir, which lives under
    `work/`, so that subtree is writable while the checkout around it is not."""
    scratch = HARNESS_ROOT / "work" / "run_ab" / "hunt" / "t_1"
    guard = _guard(cwd=scratch, workspace=[scratch, FOREIGN])
    assert not _denied(_decide(guard, "Write", {"file_path": str(scratch / "poc.c")}))
    assert _denied(
        _decide(guard, "Write", {"file_path": str(HARNESS_ROOT / "audit" / "x.py")})
    )


# ---------- structured tools: the search rules ----------


def test_grep_without_a_glob_cannot_read_a_guarded_file() -> None:
    """Reading is not sandboxed, and ripgrep passes `--hidden` with only VCS
    directories excluded, so a search rooted above a guarded file reads it:
    `Grep {path: <home>, pattern: "sk-ant", output_mode: "content"}` read
    ~/.claude/.credentials.json while the equivalent Read was refused. validate,
    gapfill, feedback, dedupe and report have Grep and no Bash, so this rule is
    their whole defence."""
    guard = _guard()
    for root in (str(HOME), "/", str(HARNESS_ROOT)):
        assert _denied(
            _decide(guard, "Grep",
                    {"path": root, "pattern": "sk-ant", "output_mode": "content"})
        ), root


def test_grep_and_glob_from_a_target_are_allowed() -> None:
    """The anti-over-block direction, and a regression this pins: an earlier
    version matched the PATTERN against guarded filenames with no location test,
    so `*.json` and `.c*` were refused in every target repo."""
    guard = _guard(cwd=FOREIGN, workspace=[FOREIGN])
    for call in (
        {"path": str(FOREIGN), "pattern": "sk-ant"},
        {"path": str(FOREIGN), "glob": "*.json", "pattern": "x"},
        {"glob": "**/report.json", "pattern": "severity"},
        {"path": str(FOREIGN), "pattern": "*.json"},
        {"pattern": "**/*.json"},
        {"pattern": ".c*"},
    ):
        tool = "Glob" if "pattern" in call else "Grep"
        assert not _denied(_decide(guard, tool, call)), call


def test_pattern_prefix_is_judged_against_the_calls_own_root() -> None:
    """ripgrep and Glob match a relative pattern against the call's root, not the
    session cwd, so `glob: "results/*/report.json"` with `path` set to the
    checkout reached the guarded results tree from hunt's scratch dir."""
    scratch = HARNESS_ROOT / "work" / "r" / "hunt" / "t_1"
    # A workspace that does NOT contain the checkout: the call's own `path` is
    # what points at it, which is the case this rule exists for.
    guard = _guard(cwd=scratch, workspace=[scratch, FOREIGN])
    assert _denied(_decide(guard, "Grep",
                           {"path": str(HARNESS_ROOT), "glob": "results/*/report.json",
                            "pattern": "severity"}))
    assert _denied(_decide(guard, "Glob",
                           {"path": str(HARNESS_ROOT), "pattern": "results/**/*.json"}))
    assert _denied(_decide(guard, "Grep",
                           {"path": str(HARNESS_ROOT), "glob": "prompts/*.md"}))


# ---------- the shell filter, to the scope it claims ----------


@pytest.mark.parametrize("command", [
    "cat .env",
    f"cat {HARNESS_ROOT}/.env",
    f"sqlite3 {STATE_DB} 'select 1'",
    "sqlite3 ./state.db '.tables'",
    "rm -f state.db-wal",
    "cat ~/.claude/.credentials.json",
    "cp $HOME/.claude/.credentials.json /tmp/x",
    "cat ~/.claude.json",
    "cat ${HOME}/.claude/settings.json",
    f"echo x >> {HARNESS_ROOT}/audit/paths.py",
    "tar -cf - ~/.claude",
])
def test_bash_guard_refuses_the_guarded_names(command: str) -> None:
    assert _denied(_decide(_guard(), "Bash", {"command": command})), command


def test_bash_guard_does_not_refuse_similar_or_ordinary_names() -> None:
    """A bare substring with no path boundary reported `.claude-backup` and
    `.claudette` as credential violations. Matching `state.db` by basename is the
    one over-block the docstring owns; everything else must stay usable."""
    guard = _guard()
    for command in (
        f"ls {HOME}.claude-backup/x",
        f"grep -r TODO {HOME}.claudette/src",
        "cat /etc/hosts",
        "ls -la /var/log",
        "git log --oneline -20",
        "gcc -o poc poc.c && ./poc",
        "sqlite3 ./app.db '.tables'",
        "cat .env.example",
    ):
        assert not _denied(_decide(guard, "Bash", {"command": command})), command


def test_bash_guard_denial_redirects() -> None:
    """A bare denial makes the model retry the same call; the reason is what turns
    it into a redirection."""
    out = _decide(_guard(), "Bash", {"command": f"sqlite3 {STATE_DB} 'select 1'"})
    reason = _reason(out)
    assert reason and ("harness" in reason.lower() or "state.db" in reason)


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
        cwd=FOREIGN,
        add_dirs=[FOREIGN],
        permission_mode="acceptEdits",
    )
    kwargs.update(overrides)
    return build(**kwargs)


def test_options_restrict_the_base_tool_set() -> None:
    """`allowed_tools` only pre-approves; `tools` removes. Probed, not assumed: a
    session declaring tools=["Read"] reports BASH=absent."""
    assert list(_options().tools) == ["Read", "Grep", "Glob"]


def test_options_enable_the_sandbox_and_suppress_mcp_servers() -> None:
    opts = _options()
    assert opts.sandbox["enabled"] is True
    assert opts.sandbox["allowUnsandboxedCommands"] is False
    assert opts.setting_sources == []
    # MCP clients run in this process, outside the sandbox: a session declaring
    # two built-in tools still carried the operator's Google Drive servers.
    assert opts.strict_mcp_config is True


def test_options_install_the_guard_for_every_inspected_tool() -> None:
    opts = _options()
    alternatives = set(opts.hooks["PreToolUse"][0].matcher.split("|"))
    for tool in ("Bash", "Write", "Edit", "Read", "Grep", "Glob", "NotebookEdit"):
        assert tool in alternatives, f"the matcher never fires for {tool}"


def test_options_sandbox_settings_are_not_shared_between_dispatches() -> None:
    first, second = _options(), _options()
    assert first.sandbox is not second.sandbox
    first.sandbox["enabled"] = False
    assert second.sandbox["enabled"] is True


def test_sandbox_egress_allowlist_is_the_live_target() -> None:
    opts = _options(network_allow=["target.example.com"])
    assert opts.sandbox["network"] == {"allowedDomains": ["target.example.com"]}


def test_static_run_has_no_egress_at_all() -> None:
    assert "network" not in _options().sandbox


def test_live_target_host_is_extracted_for_the_allowlist() -> None:
    from audit.stages._common import StageContext

    ctx = StageContext(run_id="r", repo_path=Path("/tmp/x"), config=load_config(),
                       live_target={"url": "https://api.target.example:8443/v1",
                                    "credentials": {}})
    assert ctx.network_allow() == ["api.target.example"]
    bare = StageContext(run_id="r", repo_path=Path("/tmp/x"), config=load_config())
    assert bare.network_allow() == []


@pytest.mark.parametrize(
    "url", ["http://[::1:8888", "http://user:pa]ss@h/", "http://h[x]/"]
)
def test_a_malformed_live_target_does_not_escape_from_a_stage(url: str) -> None:
    """`network_allow()` parses at the CLI now, but it is still evaluated inside
    every stage's argument list, so it must not be the place a parse error
    surfaces: this used to raise inside recon, escape validate entirely, and burn
    a hunt attempt per task until the retry ceiling abandoned the set."""
    from audit.stages._common import StageContext

    ctx = StageContext(run_id="r", repo_path=Path("/tmp/x"), config=load_config(),
                       live_target={"url": url})
    assert ctx.network_allow() == []


class _CaptureOptions:
    def __init__(self) -> None:
        self.options = None

    async def __call__(self, **kwargs):
        self.options = runner_mod._build_options(
            system_prompt="s",
            allowed_tools=kwargs["allowed_tools"],
            model=kwargs["model"],
            max_turns=kwargs["max_turns"],
            cwd=kwargs["cwd"],
            add_dirs=kwargs["add_dirs"],
            permission_mode=kwargs["permission_mode"],
            sandbox=kwargs["sandbox"],
            network_allow=kwargs["network_allow"],
            strict_mcp_config=kwargs["strict_mcp_config"],
        )
        raise RuntimeError("captured")


def test_stage_call_sites_deliver_the_confinement_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The value has to travel, not just be computed: removing `network_allow=`
    from every stage used to leave the suite green, because the two tests that
    looked like coverage tested opposite ends of the wire."""
    import audit.stages._common as common_mod
    import audit.stages.trace as trace_mod
    from audit.state import StateDB
    from audit.stages._common import StageContext

    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    db.create_run("repo", "r")
    db.add_task("r", {"task_id": "t_1", "attack_class": "sqli", "scope_hint": "x",
                      "target_files": ["a.py"], "rationale": "r", "priority": 1,
                      "source": "recon"})
    db.add_finding("r", "t_1", {
        "finding_id": "f_1", "file": "a.py", "line_start": 1, "line_end": 2,
        "vuln_class": "sqli", "severity": "high", "description": "d",
        "evidence_snippet": "e", "confidence": 0.9})
    db.set_finding_validation("r", "f_1", "confirmed", {"verdict": "confirmed"})
    db.assign_finding_group("r", "f_1", "g_1", True)
    db.add_dedupe_group("r", {"group_id": "g_1", "root_cause": "rc",
                              "canonical_finding_id": "f_1",
                              "member_finding_ids": ["f_1"]})
    ctx = StageContext(run_id="r", repo_path=tmp_path / "repo", config=load_config(),
                       live_target={"url": "https://live.example.com:8443/x",
                                    "credentials": {}})
    capture = _CaptureOptions()
    monkeypatch.setattr(trace_mod, "run_agent", capture)

    with pytest.raises(RuntimeError, match="captured"):
        asyncio.run(trace_mod.run_trace(ctx, db))

    assert capture.options is not None, "the stage never reached the runner"
    assert capture.options.sandbox["network"] == {"allowedDomains": ["live.example.com"]}
    assert capture.options.sandbox["enabled"] is True
    assert capture.options.permission_mode == "acceptEdits"
    assert capture.options.strict_mcp_config is True
    assert list(capture.options.tools) == load_config().get("trace").tools


# ---------- the self-audit warning ----------


def test_self_audit_warning_covers_a_containing_directory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`--repo <the parent of the checkout>` hands the agent a directory that
    contains state.db, results/ and .env, so the sandbox cannot separate them."""
    with caplog.at_level("WARNING"):
        _options(cwd=Path("/tmp/elsewhere"), add_dirs=[HARNESS_ROOT.parent])
    assert any("self-audit" in r.message.lower() for r in caplog.records)


def test_self_audit_warning_stays_quiet_for_hunts_real_scratch_dir(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hunt's cwd is REPO_ROOT/work/<run>/hunt/<task>, so treating containment in
    that direction as a self-audit fired once per task per run and made the
    warning useless for spotting an actual self-audit."""
    scratch = HARNESS_ROOT / "work" / "run_ab12cd34" / "hunt" / "t_core_auth_1"
    with caplog.at_level("WARNING"):
        _options(cwd=scratch, add_dirs=[FOREIGN])
    assert not [r for r in caplog.records if "self-audit" in r.message.lower()]


@pytest.mark.skipif(
    not Path("/System/Volumes/Data").exists(),
    reason="macOS firmlink volume; the same path is a plain symlink elsewhere",
)
def test_firmlink_spelling_of_the_checkout_also_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    firmlink = Path("/System/Volumes/Data" + str(HARNESS_ROOT))
    with caplog.at_level("WARNING"):
        _options(cwd=Path("/tmp/elsewhere"), add_dirs=[firmlink])
    assert any("self-audit" in r.message.lower() for r in caplog.records)


# ---------- config cannot express an unconfined or inert stage ----------


def _write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "stages.yaml"
    p.write_text(body)
    return p


def test_config_rejects_unknown_tool_names(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("body", [
    "stages:\n  hunt:\n    model: m\n    concurrency: 1\n    tools:\n",
    "stages:\n  hunt:\n    model: m\n    concurrency: 1\n    tools: []\n",
    "stages:\n  hunt:\n    model: m\n    concurrency: 1\n",
])
def test_config_refuses_a_stage_with_no_tools(tmp_path: Path, body: str) -> None:
    """A stage with no tools produces nothing while the run completes clean, so
    the loud failure is the point. A bare `tools:` key is None in YAML, and an
    empty list disables every built-in tool."""
    with pytest.raises(ValueError, match="tools"):
        load_config(_write_cfg(tmp_path, body))


def test_config_falls_back_to_default_tools(tmp_path: Path) -> None:
    """`tools` was the one key that skipped `defaults`."""
    p = _write_cfg(tmp_path, """
defaults:
  tools: [Read, Grep]
stages:
  hunt:
    model: m
    concurrency: 1
""")
    assert load_config(p).get("hunt").tools == ["Read", "Grep"]


@pytest.mark.parametrize("key", ["sandbox", "strict_mcp_config"])
@pytest.mark.parametrize("value", ["null", "0", "[]", "{}", "'false'"])
def test_config_refuses_a_non_boolean_switch(
    tmp_path: Path, key: str, value: str
) -> None:
    p = _write_cfg(tmp_path, f"""
defaults:
  {key}: {value}
stages:
  hunt:
    model: m
    concurrency: 1
    tools: [Read]
""")
    with pytest.raises(ValueError, match=key):
        load_config(p)


def test_config_validates_defaults_with_no_stages(tmp_path: Path) -> None:
    """The check used to live inside the stage loop, so a config with no stages
    accepted exactly the quoted "false" it exists to refuse."""
    p = _write_cfg(tmp_path, 'defaults:\n  sandbox: "false"\n')
    with pytest.raises(ValueError, match="sandbox"):
        load_config(p)


def test_shipped_config_requests_both_switches() -> None:
    cfg = load_config()
    for name in ("recon", "hunt", "validate", "gapfill", "dedupe", "trace",
                 "feedback", "report"):
        sc = cfg.get(name)
        assert sc.sandbox is True, f"{name}: sandbox not requested"
        assert sc.strict_mcp_config is True, f"{name}: MCP servers not suppressed"
        assert sc.tools, f"{name}: no tools"


def test_rejected_task_spends_an_attempt_so_it_stops_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task rejected for an unusable id used to be marked failed WITHOUT
    spending an attempt, because the check ran before begin_task. attempts stayed
    0, reset_incomplete_tasks re-queued it on every resume, and
    count_abandoned_tasks (attempts >= 3) could never see it."""
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


def test_schema_validation_reuses_its_registry() -> None:
    """Measured before caching: 0.423 ms per call, 97% of it re-reading all ten
    schemas and rebuilding the referencing registry, with about 145 calls a run."""
    from audit.json_utils import _validator_for, validate_schema
    from audit.paths import SCHEMAS

    schema = SCHEMAS / "finding.schema.json"
    payload = {"task_id": "t", "findings": [], "gaps_observed": []}
    validate_schema(payload, schema)
    first = _validator_for(str(schema))
    validate_schema(payload, schema)
    assert _validator_for(str(schema)) is first
