"""Mutation check: does every sensor actually fire?

Run it with the project venv: `.venv/bin/python scripts/mutation-check.py`.

It breaks one thing at a time, runs the sensors that are supposed to police that
thing, and checks they go red while their controls stay green. The tree is
restored from an in-memory copy after each row and compared against HEAD at the
end, so a crash mid-run cannot leave the tree mutated.

For each mutation: break exactly one thing, run the sensors that police it, and
check they go red while their controls stay green. Files are restored from an
in-memory copy and the tree is compared against HEAD at the end.

The rows marked (F13) are the ones an adversarial review showed were invisible
to the previous harness: they mutated the guard's PRESENCE or its VERDICT, never
the key map, the matcher, or a stage call site, so 313 tests stayed green while
the code they were supposed to police was removed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Rows run against a throwaway copy. Mutating the live tree meant an interrupt
# between the write and the restore left the checkout broken, and it made the
# script unsafe to run beside anything else. The copy is made once; each row
# mutates and restores inside it.
COPY = Path(tempfile.mkdtemp(prefix="audit-mutation-")) / "repo"
_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", "results", "work", ".delta",
)
PY = str(REPO / ".venv/bin/python")
CONF = "tests/test_confinement.py"
PATHS = "tests/test_path_ids.py"
CLI = "tests/test_cli.py"
AUTH = "tests/test_auth.py"
FAIL = "tests/test_stage_failures.py"

MUTATIONS = [
    # ---- confinement: layers and wiring ----
    ("tools base-set removed", "audit/runner.py",
     "        tools=list(allowed_tools),\n", "",
     ["tests/test_confinement.py::test_options_restrict_the_base_tool_set"], []),
    ("sandbox settings removed", "audit/runner.py",
     "        sandbox=sandbox_settings,\n", "        sandbox=None,\n",
     ["tests/test_confinement.py::test_options_enable_the_sandbox_and_suppress_mcp_servers",
      "tests/test_confinement.py::test_sandbox_egress_allowlist_is_the_live_target"], []),
    ("MCP suppression removed", "audit/runner.py",
     "        strict_mcp_config=strict_mcp_config,",
     "        strict_mcp_config=False,",
     ["tests/test_confinement.py::test_options_enable_the_sandbox_and_suppress_mcp_servers"], []),
    ("settings files not suppressed", "audit/runner.py",
     "        setting_sources=[],", "        setting_sources=None,",
     ["tests/test_confinement.py::test_options_enable_the_sandbox_and_suppress_mcp_servers"], []),
    ("hook not installed", "audit/runner.py",
     '        hooks={\n            "PreToolUse": [\n                HookMatcher(\n'
     '                    matcher=_GUARD_MATCHER,\n'
     '                    hooks=[_make_tool_guard(cwd, [cwd, *dirs])],\n'
     '                )\n            ]\n        },\n', "",
     ["tests/test_confinement.py::test_options_install_the_guard_for_every_inspected_tool"], []),
    ("matcher narrowed to Bash", "audit/paths.py",
     'GUARDED_TOOLS = frozenset(_TOOL_INPUTS) | {_BASH_TOOL}',
     'GUARDED_TOOLS = frozenset({_BASH_TOOL})',
     ["tests/test_confinement.py::test_options_install_the_guard_for_every_inspected_tool"], []),
    ("structured check neutered", "audit/runner.py",
     "        elif tool_name in _TOOL_INPUTS:", "        elif False:",
     ["tests/test_confinement.py::test_guard_denies_structured_paths_into_harness_state"], []),
    ("bash filter neutered", "audit/runner.py",
     '            hit = bash_guard_hit(str(tool_input.get("command", "")), cwd=session_cwd)\n',
     "            hit = None\n",
     ["tests/test_confinement.py::test_bash_guard_refuses_the_guarded_names"], []),
    ("write roots not applied to write tools", "audit/paths.py",
     "    if tool_name in WRITE_TOOLS or tool_name == _BASH_TOOL:",
     "    if tool_name == _BASH_TOOL:",
     ["tests/test_confinement.py::test_write_tools_cannot_rewrite_prompts_schemas_or_the_harness_source"], []),
    ("carve-out removed", "audit/paths.py",
     "    return any(_inside(path, hole) for hole in _writable_exceptions())",
     "    return False",
     ["tests/test_confinement.py::test_work_is_the_one_writable_hole"], []),
    ("carve-out too wide", "audit/paths.py",
     "    return any(_inside(path, hole) for hole in _writable_exceptions())",
     "    return True",
     ["tests/test_confinement.py::test_write_tools_cannot_rewrite_prompts_schemas_or_the_harness_source"], []),
    ("Grep enclosure rule removed", "audit/paths.py",
     '    if tool_name == "Grep":', "    if False:",
     ["tests/test_confinement.py::test_grep_without_a_glob_cannot_read_a_guarded_file"], []),
    ("pattern prefix not resolved from the call root", "audit/paths.py",
     "    call_roots = [\n        _resolve(str(tool_input[k]), workspace[0])\n"
     '        for k in ("path",)\n        if tool_input.get(k)\n    ] or [canonical_path(p) for p in workspace]',
     "    call_roots = [canonical_path(p) for p in workspace]",
     ["tests/test_confinement.py::test_pattern_prefix_is_judged_against_the_calls_own_root"], []),
    ("WAL sidecar rule removed", "audit/paths.py",
     "        if _is_harness_db(path):", "        if False:",
     ["tests/test_confinement.py::test_guard_denies_structured_paths_into_harness_state"], []),
    ("identity comparison removed", "audit/paths.py",
     "    if _same_file(child, root):", "    if False:",
     ["tests/test_confinement.py::test_guard_denies_a_hardlink_to_the_state_database"], []),
    ("case folding removed", "audit/paths.py",
     '    child_text = str(child).casefold().rstrip("/")\n'
     '    root_text = str(root).casefold().rstrip("/")',
     '    child_text = str(child).rstrip("/")\n    root_text = str(root).rstrip("/")',
     ["tests/test_confinement.py::test_guard_denies_relative_case_folded_and_tilde_spellings"], []),
    ("credentials entries dropped", "audit/paths.py",
     '        "the Claude state directory": home / ".claude",', "",
     ["tests/test_confinement.py::test_guard_denies_structured_paths_into_harness_state"], []),
    ("home expansion removed", "audit/paths.py",
     "    folded = _expand_shell_home(command).casefold()", "    folded = command.casefold()",
     ["tests/test_confinement.py::test_bash_guard_refuses_the_guarded_names"], []),
    ("credential directory matched by bare name", "audit/paths.py",
     "        if entry.name in _DIR_ENTRY_NAMES:", "        if False:",
     ["tests/test_confinement.py::test_bash_guard_does_not_refuse_similar_or_ordinary_names"], []),
    ("self-audit warning removed", "audit/runner.py",
     '        log.warning(\n            "[confinement] self-audit:',
     '        log.debug(\n            "[confinement] self-audit:',
     ["tests/test_confinement.py::test_self_audit_warning_covers_a_containing_directory"],
     ["tests/test_confinement.py::test_self_audit_warning_stays_quiet_for_hunts_real_scratch_dir"]),
    ("warning fires on any harness-contained cwd", "audit/runner.py",
     "        canonical_path(p) == REPO_ROOT or REPO_ROOT.is_relative_to(canonical_path(p))",
     "        canonical_path(p) == REPO_ROOT or canonical_path(p).is_relative_to(REPO_ROOT)",
     ["tests/test_confinement.py::test_self_audit_warning_stays_quiet_for_hunts_real_scratch_dir"], []),
    ("sandbox settings dict aliased", "audit/runner.py",
     "        sandbox_settings = dict(_SANDBOX_SETTINGS)",
     "        sandbox_settings = _SANDBOX_SETTINGS",
     ["tests/test_confinement.py::test_options_sandbox_settings_are_not_shared_between_dispatches"], []),
    ("egress allowlist dropped", "audit/runner.py",
     '            sandbox_settings["network"] = {"allowedDomains": list(network_allow)}',
     "            pass",
     ["tests/test_confinement.py::test_sandbox_egress_allowlist_is_the_live_target"], []),
    # ---- path components ----
    ("component regex not anchored", "audit/paths.py",
     "    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):",
     "    if not isinstance(value, str) or not _COMPONENT_RE.match(value):",
     ["tests/test_path_ids.py::test_results_dir_rejects_a_traversing_run_id"], []),
    ("id ceiling lowered below the schema", "audit/paths.py",
     'r"[A-Za-z0-9._-]{1,128}"', 'r"[A-Za-z0-9._-]{1,64}"',
     ["tests/test_path_ids.py::test_schema_legal_long_finding_id_survives_the_artifact_path"], []),
    ("work_dir stops validating", "audit/stages/_common.py",
     '        ref_component = "default" if ref is None else safe_component(\n'
     '            ref, kind="work-dir reference"\n        )\n',
     '        ref_component = ref or "default"\n',
     ["tests/test_path_ids.py::test_hunt_work_dir_rejects_a_traversing_task_id"], []),
    ("results_dir stops validating", "audit/stages/_common.py",
     '        d = RESULTS / safe_component(self.run_id, kind="run_id") / stage',
     "        d = RESULTS / self.run_id / stage",
     ["tests/test_path_ids.py::test_results_dir_rejects_a_traversing_run_id"], []),
    ("create_run stops validating", "audit/state.py",
     '        run_id = safe_component(run_id or new_run_id(), kind="run_id")',
     "        run_id = run_id or new_run_id()",
     ["tests/test_path_ids.py::test_create_run_rejects_a_traversing_run_id"], []),
    ("artifact path stops validating", "audit/runner.py",
     '    return artifact_dir / f"{safe_component(artifact_name, kind=\'artifact name\')}.jsonl"',
     '    return artifact_dir / f"{artifact_name}.jsonl"',
     ["tests/test_path_ids.py::test_artifact_path_rejects_a_traversing_name"], []),
    ("run-id resolver uncased", "audit/state.py",
     '            "SELECT run_id FROM runs WHERE lower(run_id) = lower(?)", (run_id,)',
     '            "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)',
     ["tests/test_path_ids.py::test_resolve_run_id_folds_case_for_every_command"], []),
    ("hunt identifier guard removed", "audit/stages/hunt.py",
     '            try:\n                safe_component(task.task_id, kind="task_id")\n'
     "            except UnsafeIdentifier as bad_id:\n",
     "            try:\n                pass\n            except UnsafeIdentifier as bad_id:\n",
     ["tests/test_confinement.py::test_rejected_task_spends_an_attempt_so_it_stops_requeueing"], []),
    ("trace identifier handler narrowed", "audit/stages/trace.py",
     "            except UnsafeIdentifier as e:", "            except ZeroDivisionError as e:",
     ["tests/test_stage_failures.py::test_trace_fails_one_finding_on_an_unusable_identifier"], []),
    ("trace handler widened to ValueError", "audit/stages/trace.py",
     "            except UnsafeIdentifier as e:", "            except ValueError as e:",
     ["tests/test_stage_failures.py::test_trace_does_not_swallow_a_schema_error"], []),
    # ---- config ----
    ("config validation skipped", "audit/config.py",
     "        _validate_stage(name, spec, defaults)\n", "",
     ["tests/test_confinement.py::test_config_rejects_unknown_tool_names"], []),
    ("empty tools accepted", "audit/config.py",
     "    if not tools:", "    if False:",
     ["tests/test_confinement.py::test_config_refuses_a_stage_with_no_tools"], []),
    ("defaults tools fallback removed", "audit/config.py",
     '    tools = spec.get("tools", defaults.get("tools"))',
     '    tools = spec.get("tools") or []',
     ["tests/test_confinement.py::test_config_falls_back_to_default_tools"],
     # still raises under this mutation, so it is a control, not a sensor
     ["tests/test_confinement.py::test_config_refuses_a_stage_with_no_tools"]),
    ("boolean switch check removed", "audit/config.py",
     "    if key in block and not isinstance(block[key], bool):", "    if False:",
     ["tests/test_confinement.py::test_config_refuses_a_non_boolean_switch"], []),
    ("defaults validation skipped", "audit/config.py",
     "    _validate_defaults(defaults)\n", "",
     ["tests/test_confinement.py::test_config_validates_defaults_with_no_stages"], []),
    # ---- CLI ----
    ("target-url scheme not checked", "audit/cli.py",
     '    if parsed.scheme not in ("http", "https"):', "    if False:",
     ["tests/test_cli.py::test_target_url_is_refused_at_the_flag"], []),
    ("target-url host not checked", "audit/cli.py",
     "    if not parsed.hostname:", "    if False:",
     ["tests/test_cli.py::test_target_url_is_refused_at_the_flag"], []),
    # ---- markdown ----
    ("cwe left raw", "audit/cli.py",
     "+ (f\" ({_md_inline(f['cwe'])})\" if f.get(\"cwe\") else \"\"))",
     "+ (f\" ({f['cwe']})\" if f.get(\"cwe\") else \"\"))",
     ["tests/test_cli.py::test_cwe_cannot_inject_markdown"], []),
    ("header left raw", "audit/cli.py",
     "lines.append(f\"# Vulnerability report — {_md_code(report['run_id'])}\")",
     "lines.append(f\"# Vulnerability report — {report['run_id']}\")",
     ["tests/test_cli.py::test_header_fields_cannot_inject_markdown"], []),
    ("summary keys left raw", "audit/cli.py",
     'counts = ", ".join(f"{_md_inline(k)}: {_md_inline(v)}" for k, v in by.items())',
     'counts = ", ".join(f"{k}: {v}" for k, v in by.items())',
     ["tests/test_cli.py::test_summary_counts_cannot_inject_markdown"], []),
    ("unrenderable stub removed", "audit/cli.py",
     "    if broken or not isinstance(report.get(\"findings\"), list):", "    if False:",
     ["tests/test_cli.py::test_unrenderable_report_says_what_is_missing",
      "tests/test_cli.py::test_wrong_typed_field_renders_a_stub"], []),
    ("degraded not surfaced", "audit/cli.py",
     '    if report.get("degraded"):', "    if False:",
     ["tests/test_cli.py::test_markdown_report_marks_a_degraded_report"],
     ["tests/test_cli.py::test_clean_report_has_no_status_furniture"]),
    ("untraced not surfaced", "audit/cli.py",
     '    untraced = report.get("untraced_findings") or []', "    untraced = []",
     ["tests/test_cli.py::test_markdown_report_lists_untraced_canonicals"], []),
    ("render-time validation warning removed", "audit/cli.py",
     '    errors = validate_schema(report, SCHEMAS / "report.schema.json")\n    if errors:',
     "    errors = []\n    if errors:",
     ["tests/test_cli.py::test_markdown_render_warns_on_a_schema_invalid_payload"], []),
    # ---- auth ----
    ("base-url validator neutered", "audit/auth.py",
     "def _base_url_rejection_reason(url: str) -> str:",
     'def _base_url_rejection_reason(url: str) -> str:\n    return ""',
     ["tests/test_auth.py::test_backslash_base_url_reason_names_the_character"], []),
    ("whitespace normalisation removed", "audit/auth.py",
     "    if original != raw:", "    if False:",
     ["tests/test_auth.py::test_edge_whitespace_is_normalised_into_the_env"], []),
    ("broken ports not refused", "audit/auth.py",
     '    except ValueError:\n        return f"its port is not a valid port number ({parts.netloc!r})"',
     '    except ValueError:\n        return ""',
     ["tests/test_auth.py::test_base_url_with_a_broken_port_is_rejected"], []),
    ("schema cache removed", "audit/json_utils.py",
     "@functools.lru_cache(maxsize=None)\ndef _validator_for", "def _validator_for",
     ["tests/test_confinement.py::test_schema_validation_reuses_its_registry"], []),
]


def _prepare_copy() -> None:
    """One clean copy of the tracked tree, without the venv or run artifacts."""
    shutil.copytree(REPO, COPY, ignore=_IGNORE, dirs_exist_ok=True)


class HarnessError(RuntimeError):
    pass


def run(tests: list[str]) -> set[str]:
    if not tests:
        return set()
    # -B and no __pycache__ in the copy: a mutation writes the source, pytest
    # imports it, and the restore can land inside the same mtime tick, in which
    # case a cached .pyc compiled from the MUTATED source is reused and the
    # sensor looks green. That produced two non-reproducible verdicts before.
    for cache in COPY.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    # The venv's editable install maps `audit` straight back to the live
    # checkout through a meta_path finder, so PYTHONPATH alone would import HEAD
    # and prove nothing. Drop the finder and put the copy first.
    bootstrap = (
        "import sys; "
        "sys.meta_path = [f for f in sys.meta_path "
        "if type(f).__name__ != '_EditableFinder']; "
        f"sys.path.insert(0, {str(COPY)!r}); "
        "import audit; "
        f"assert audit.__file__.startswith({str(COPY)!r}), ("
        "'imported the live checkout, not the copy: ' + audit.__file__); "
        "import pytest, sys as _s; "
        f"_s.exit(pytest.main(['-q', '--no-header', '-p', 'no:cacheprovider', *{tests!r}]))"
    )
    proc = subprocess.run(
        [PY, "-B", "-c", bootstrap],
        cwd=COPY, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if proc.returncode not in (0, 1):
        # A mutation that breaks the file (syntax or collection error) used to
        # arrive here as an empty failed set, which reads as "sensor stayed
        # green" and hides a harness bug. It happened once: an anchor matched the
        # wrong function and left a dangling ` if writes else ...`.
        raise HarnessError(
            f"pytest exit {proc.returncode} (mutation likely broke the file)\n"
            + (proc.stdout[-600:] or proc.stderr[-600:])
        )
    failed = set()
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            node = line.split(" ", 1)[1].split(" - ")[0]
            failed.add(node.split("::")[-1].split("[")[0])
    return failed


def _fingerprint() -> str:
    """A hash of the live sources, so the harness can prove it changed nothing."""
    import hashlib

    digest = hashlib.sha256()
    for sub in ("audit", "tests", "config", "prompts", "schemas", "scripts"):
        for path in sorted((REPO / sub).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(str(path).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    problems = 0
    start_fingerprint = _fingerprint()
    print(f"{'mutation':46} {'red':>4} {'green':>6}  verdict")
    _prepare_copy()
    print(f"copy: {COPY}\n")
    for name, rel, old, new, expect_red, expect_green in MUTATIONS:
        path = COPY / rel
        original = path.read_text()
        if old not in original:
            print(f"{name:46} {'-':>4} {'-':>6}  SKIP: anchor not found in {rel}")
            problems += 1
            continue
        path.write_text(original.replace(old, new, 1))
        try:
            failed = run(expect_red + expect_green)
        except HarnessError as e:
            print(f"{name:46} {'-':>4} {'-':>6}  HARNESS ERROR: {e}")
            problems += 1
            continue
        finally:
            path.write_text(original)
        red_ids = {t.split("::", 1)[1] for t in expect_red}
        green_ids = {t.split("::", 1)[1] for t in expect_green}
        if not expect_red and not expect_green:
            verdict = "n/a (documented, no sensor claims it)"
        else:
            red_ok = all(t in failed for t in red_ids)
            green_ok = not any(t in failed for t in green_ids)
            verdict = "ok" if (red_ok and green_ok) else (
                "FAIL (sensor stayed green)" if not red_ok else "FAIL (control went red)"
            )
            if verdict != "ok":
                problems += 1
        print(f"{name:46} {len(expect_red):>4} {len(expect_green):>6}  {verdict}")

    print()
    print(
        "live tree untouched by this run"
        if _fingerprint() == start_fingerprint
        else "LIVE TREE CHANGED: the harness must never mutate the checkout"
    )
    if _fingerprint() != start_fingerprint:
        problems += 1
    shutil.rmtree(COPY.parent, ignore_errors=True)
    print(f"rows run: {len(MUTATIONS)}")
    print("ALL MUTATIONS BEHAVED" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
