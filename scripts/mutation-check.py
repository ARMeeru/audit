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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
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
     [f"{CONF}::test_options_restrict_the_base_tool_set"], []),
    ("sandbox settings removed", "audit/runner.py",
     "        sandbox=sandbox_settings,\n", "        sandbox=None,\n",
     [f"{CONF}::test_options_enable_the_bash_sandbox",
      f"{CONF}::test_sandbox_egress_allowlist_is_the_live_target"], []),
    ("PreToolUse hook not installed", "audit/runner.py",
     '        hooks={\n            "PreToolUse": [\n                HookMatcher(\n'
     '                    matcher=_GUARD_MATCHER,\n                    hooks=[_make_tool_guard(cwd, [cwd, *dirs])],\n'
     '                )\n            ]\n        },\n', "",
     [f"{CONF}::test_options_install_the_pre_tool_use_guard"], []),
    ("guard always allows", "audit/runner.py",
     "        elif tool_name in _TOOL_INPUTS:\n"
     "            hit = structured_path_hit(\n"
     "                tool_name, tool_input, cwd=session_cwd, workspace=workspace_dirs\n"
     "            )\n", "        elif False:\n            hit = None\n",
     [f"{CONF}::test_guard_denies_structured_paths_into_harness_state",
      f"{CONF}::test_guard_denies_relative_and_tilde_spellings",
      f"{CONF}::test_grep_glob_cannot_narrow_onto_a_guarded_file",
      f"{CONF}::test_glob_pattern_cannot_name_a_guarded_path"],
     [f"{CONF}::test_guard_allows_legitimate_agent_work"]),
    ("bash guard neutered", "audit/runner.py",
     '            hit = bash_guard_hit(str(tool_input.get("command", "")), cwd=session_cwd)\n',
     "            hit = None\n",
     [f"{CONF}::test_guard_denies_harness_state_references",
      f"{CONF}::test_bash_guard_expands_home_spellings",
      f"{CONF}::test_bash_cannot_rewrite_prompts_schemas_or_config"],
     [f"{CONF}::test_guard_allows_legitimate_agent_work"]),
    # (F13) the key map: dropping a key left 313 tests green
    ("Read key dropped from the map", "audit/paths.py",
     '    "Read": {"targets": ("file_path",)},', '    "Read": {"targets": ()},',
     [f"{CONF}::test_guard_denies_structured_paths_into_harness_state"],
     [f"{CONF}::test_guard_allows_legitimate_agent_work"]),
    ("Grep path key dropped from the map", "audit/paths.py",
     '    "Grep": {"targets": ("path",), "patterns": ("glob",)},',
     '    "Grep": {"targets": (), "patterns": ("glob",)},',
     [f"{CONF}::test_grep_glob_cannot_narrow_onto_a_guarded_file"], []),
    ("NotebookEdit key dropped from the map", "audit/paths.py",
     '    "NotebookEdit": {"targets": ("notebook_path",)},',
     '    "NotebookEdit": {"targets": ()},',
     [f"{CONF}::test_guard_denies_structured_paths_into_harness_state"], []),
    ("write-only set dropped for write tools", "audit/paths.py",
     '    return {**guarded_paths(), **write_only_guards()} if writes else guarded_paths()',
     "    return guarded_paths()",
     [f"{CONF}::test_write_guard_protects_prompts_schemas_and_config"], []),
    ("write-only set dropped for bash", "audit/paths.py",
     "    return {**guarded_paths(), **write_only_guards()}\n",
     "    return guarded_paths()\n",
     [f"{CONF}::test_bash_cannot_rewrite_prompts_schemas_or_config"], []),
    # (F13) the matcher: omitting a tool left the suite green
    ("matcher narrowed to Bash only", "audit/paths.py",
     'GUARDED_TOOLS = frozenset(_TOOL_INPUTS) | {_BASH_TOOL}',
     'GUARDED_TOOLS = frozenset({_BASH_TOOL})',
     [f"{CONF}::test_matcher_covers_every_tool_the_guard_inspects"], []),
    # ---- path components ----
    ("component regex not anchored", "audit/paths.py",
     "    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):",
     "    if not isinstance(value, str) or not _COMPONENT_RE.match(value):",
     [f"{PATHS}::test_results_dir_rejects_a_traversing_run_id",
      f"{PATHS}::test_artifact_path_rejects_a_traversing_name"], []),
    ("traversal rejection removed", "audit/paths.py",
     '    if value in (".", ".."):\n        raise UnsafeIdentifier(f"unsafe {kind} {value!r}: path traversal")\n',
     "",
     [f"{PATHS}::test_results_dir_rejects_a_traversing_run_id"], []),
    ("id ceiling lowered below the schema", "audit/paths.py",
     'r"[A-Za-z0-9._-]{1,128}"', 'r"[A-Za-z0-9._-]{1,64}"',
     [f"{PATHS}::test_schema_legal_long_finding_id_survives_the_artifact_path"], []),
    ("work_dir stops validating", "audit/stages/_common.py",
     '        ref_component = "default" if ref is None else safe_component(\n'
     '            ref, kind="work-dir reference"\n        )\n',
     '        ref_component = ref or "default"\n',
     [f"{PATHS}::test_hunt_work_dir_rejects_a_traversing_task_id"], []),
    ("results_dir stops validating", "audit/stages/_common.py",
     '        d = RESULTS / safe_component(self.run_id, kind="run_id") / stage',
     "        d = RESULTS / self.run_id / stage",
     [f"{PATHS}::test_results_dir_rejects_a_traversing_run_id"], []),
    ("create_run stops validating", "audit/state.py",
     '        run_id = safe_component(run_id or new_run_id(), kind="run_id")',
     "        run_id = run_id or new_run_id()",
     [f"{PATHS}::test_create_run_rejects_a_traversing_run_id"], []),
    ("artifact path stops validating", "audit/runner.py",
     '    return artifact_dir / f"{safe_component(artifact_name, kind=\'artifact name\')}.jsonl"',
     '    return artifact_dir / f"{artifact_name}.jsonl"',
     [f"{PATHS}::test_artifact_path_rejects_a_traversing_name"], []),
    ("case-insensitive collision check removed", "audit/state.py",
     "        clash = self._conn.execute(",
     "        clash = None and self._conn.execute(",
     [f"{PATHS}::test_create_run_rejects_a_case_insensitive_collision"], []),
    # (F13) the stage call sites
    ("hunt identifier guard removed", "audit/stages/hunt.py",
     "            try:\n                safe_component(task.task_id, kind=\"task_id\")\n"
     "            except UnsafeIdentifier as bad_id:\n",
     "            try:\n                pass\n            except UnsafeIdentifier as bad_id:\n",
     [f"{CONF}::test_rejected_task_spends_an_attempt_so_it_stops_requeueing"], []),
    ("trace identifier handler narrowed", "audit/stages/trace.py",
     "            except UnsafeIdentifier as e:",
     "            except ZeroDivisionError as e:",
     [f"{FAIL}::test_trace_fails_one_finding_on_an_unusable_identifier"], []),
    ("validate identifier handler narrowed", "audit/stages/validate.py",
     "            except UnsafeIdentifier as e:",
     "            except ZeroDivisionError as e:",
     [f"{FAIL}::test_validate_fails_one_finding_on_an_unusable_identifier"], []),
    # ---- config ----
    ("config validation skipped", "audit/config.py",
     "        _validate_stage(name, spec, defaults)\n", "",
     [f"{CONF}::test_config_rejects_unknown_tool_names"], []),
    ("sandbox type check removed", "audit/config.py",
     '    if "sandbox" in block and not isinstance(block["sandbox"], bool):',
     "    if False:",
     [f"{CONF}::test_config_refuses_a_non_boolean_sandbox",
      f"{CONF}::test_stage_less_config_still_validates_defaults"], []),
    ("defaults validation skipped", "audit/config.py",
     "    _validate_defaults(defaults)\n", "",
     [f"{CONF}::test_stage_less_config_still_validates_defaults"], []),
    ("BashOutput dropped from the tools", "config/stages.yaml",
     "    tools: [Read, Grep, Glob, Bash, BashOutput, KillShell]\n"
     "    max_turns: 60                # recon can be long",
     "    tools: [Read, Grep, Glob, Bash]\n"
     "    max_turns: 60                # recon can be long",
     [f"{CONF}::test_bash_stages_keep_the_tools_a_backgrounded_command_needs"], []),
    # ---- self-audit warning ----
    ("self-audit warning removed", "audit/runner.py",
     '        log.warning(\n            "[confinement] self-audit:',
     '        log.debug(\n            "[confinement] self-audit:',
     [f"{CONF}::test_self_audit_warns_that_the_boundary_is_unavailable",
      f"{CONF}::test_self_audit_warning_covers_a_containing_directory"],
     [f"{CONF}::test_normal_run_does_not_warn_about_self_audit"]),
    ("warning back to exact equality", "audit/runner.py",
     "        canonical_path(p) == REPO_ROOT or REPO_ROOT.is_relative_to(canonical_path(p))",
     "        canonical_path(p) == REPO_ROOT",
     [f"{CONF}::test_self_audit_warning_covers_a_containing_directory"],
     [f"{CONF}::test_self_audit_warning_stays_quiet_for_hunts_real_scratch_dir"]),
    ("warning fires on any harness-contained cwd", "audit/runner.py",
     "        canonical_path(p) == REPO_ROOT or REPO_ROOT.is_relative_to(canonical_path(p))",
     "        canonical_path(p) == REPO_ROOT or canonical_path(p).is_relative_to(REPO_ROOT)",
     [f"{CONF}::test_self_audit_warning_stays_quiet_for_hunts_real_scratch_dir"], []),
    ("sandbox settings dict aliased", "audit/runner.py",
     "        sandbox_settings = dict(_SANDBOX_SETTINGS)",
     "        sandbox_settings = _SANDBOX_SETTINGS",
     [f"{CONF}::test_sandbox_settings_are_not_shared_between_dispatches"], []),
    # ---- markdown ----
    ("cwe left raw", "audit/cli.py",
     "+ (f\" ({_md_inline(f['cwe'])})\" if f.get(\"cwe\") else \"\"))",
     "+ (f\" ({f['cwe']})\" if f.get(\"cwe\") else \"\"))",
     [f"{CLI}::test_cwe_cannot_inject_markdown",
      f"{CLI}::test_no_field_can_inject_structure"], []),
    ("call-chain line left raw", "audit/cli.py",
     'where = _md_code("{}:{}".format(frame["file"], frame["line"]))',
     'where = "{}:{}".format(frame["file"], frame["line"])',
     [f"{CLI}::test_trace_frame_line_cannot_inject_markdown"], []),
    ("location left raw", "audit/cli.py",
     '            + _md_code(f"{f[\'file\']}:{f[\'line_start\']}-{f[\'line_end\']}")',
     '            + f"{f[\'file\']}:{f[\'line_start\']}-{f[\'line_end\']}"',
     [f"{CLI}::test_finding_line_numbers_cannot_inject_markdown"], []),
    ("header left raw", "audit/cli.py",
     'lines.append(f"# Vulnerability report — {_md_code(report[\'run_id\'])}")',
     'lines.append(f"# Vulnerability report — {report[\'run_id\']}")',
     [f"{CLI}::test_header_fields_cannot_inject_markdown"], []),
    ("summary keys left raw", "audit/cli.py",
     'counts = ", ".join(f"{_md_inline(k)}: {_md_inline(v)}" for k, v in by.items())',
     'counts = ", ".join(f"{k}: {v}" for k, v in by.items())',
     [f"{CLI}::test_summary_counts_cannot_inject_markdown"], []),
    ("tilde back in the escape class removed", "audit/cli.py",
     '    return re.sub(r"([\\\\`*_{}\\[\\]()#+\\-.!|<>~])", r"\\\\\\1", text)',
     '    return re.sub(r"([\\\\`*_{}\\[\\]()#+\\-.!|<>])", r"\\\\\\1", text)',
     [f"{CLI}::test_tilde_fence_cannot_swallow_later_findings"], []),
    ("line terminators no longer folded", "audit/cli.py",
     '    return re.sub(r"[\\r\\n\\v\\f\\x85\\u2028\\u2029]+", " ", str(value))',
     "    return str(value)",
     [f"{CLI}::test_inline_escaper_neutralizes_line_structure"], []),
    ("leading whitespace no longer stripped", "audit/cli.py",
     '    text = _fold_line_terminators(s).lstrip(" \\t")',
     "    text = _fold_line_terminators(s)",
     [f"{CLI}::test_inline_escaper_strips_leading_whitespace",
      f"{CLI}::test_indented_description_cannot_open_a_code_block"], []),
    ("code spans go back through the prose escaper", "audit/cli.py",
     '    text = _fold_line_terminators(v)\n    longest = max(',
     '    text = _md_inline(v)\n    longest = max(',
     # only the backslash sensor: escaping also keeps a span closed, so the
     # backtick sensor is insensitive to this one and is covered below.
     [f"{CLI}::test_code_span_fields_render_without_backslashes"], []),
    ("code span delimiter not padded", "audit/cli.py",
     'delimiter = "`" * (longest + 1)', 'delimiter = "`"',
     [f"{CLI}::test_a_backtick_in_a_field_cannot_end_its_code_span"], []),
    ("missing-key guard removed", "audit/cli.py",
     '    required = ("run_id", "target", "summary", "findings")',
     "    required = ()",
     [f"{CLI}::test_unrenderable_report_says_what_is_missing"], []),
    ("degraded not surfaced", "audit/cli.py",
     '    if report.get("degraded"):', "    if False:",
     [f"{CLI}::test_markdown_report_marks_a_degraded_report"],
     [f"{CLI}::test_clean_report_has_no_status_furniture"]),
    ("untraced not surfaced", "audit/cli.py",
     '    untraced = report.get("untraced_findings") or []', "    untraced = []",
     [f"{CLI}::test_markdown_report_lists_untraced_canonicals"], []),
    ("render-time validation warning removed", "audit/cli.py",
     '    errors = validate_schema(report, SCHEMAS / "report.schema.json")\n    if errors:',
     "    errors = []\n    if errors:",
     [f"{CLI}::test_markdown_render_warns_on_a_schema_invalid_payload"], []),
    # ---- auth ----
    ("base-url validator neutered", "audit/auth.py",
     '    original = url or ""\n    raw = original.strip()',
     '    return ""\n    original = url or ""\n    raw = original.strip()',
     [f"{AUTH}::test_backslash_base_url_reason_names_the_character",
      f"{AUTH}::test_base_url_with_edge_whitespace_is_rejected"], []),
    ("edge whitespace judged stripped", "audit/auth.py",
     "    if original != raw:", "    if False:",
     [f"{AUTH}::test_base_url_with_edge_whitespace_is_rejected"], []),
    ("broken ports no longer refused", "audit/auth.py",
     '    except ValueError:\n        return f"its port is not a valid port number ({parts.netloc!r})"',
     '    except ValueError:\n        return ""',
     [f"{AUTH}::test_base_url_with_a_broken_port_is_rejected"], []),
    # ---- schema cache ----
    ("schema cache removed", "audit/json_utils.py",
     "@functools.lru_cache(maxsize=None)\ndef _validator_for",
     "def _validator_for",
     [f"{CONF}::test_schema_validation_reuses_its_registry"], []),
]


def run(tests: list[str]) -> set[str]:
    if not tests:
        return set()
    # -B and no existing __pycache__: a mutation writes the source, pytest
    # imports it, and the restore can land inside the same mtime tick, in which
    # case a cached .pyc compiled from the MUTATED source is reused and the
    # sensor looks green. That produced two non-reproducible verdicts before.
    for cache in REPO.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    proc = subprocess.run(
        [PY, "-B", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         *tests],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if proc.returncode not in (0, 1):
        # A mutation that breaks the file (syntax or collection error) used to
        # arrive here as an empty failed set, which reads as "sensor stayed
        # green" and hides a harness bug. It happened once: an anchor matched the
        # wrong function and left a dangling ` if writes else ...`.
        print(f"  !! pytest exit {proc.returncode}: mutation likely broke the file")
        print("  " + (proc.stdout[-600:] or proc.stderr[-600:]).replace("\n", "\n  "))
    failed = set()
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            node = line.split(" ", 1)[1].split(" - ")[0]
            failed.add(node.split("::")[-1].split("[")[0])
    return failed


def main() -> int:
    problems = 0
    print(f"{'mutation':46} {'red':>4} {'green':>6}  verdict")
    for name, rel, old, new, expect_red, expect_green in MUTATIONS:
        path = REPO / rel
        original = path.read_text()
        if old not in original:
            print(f"{name:46} {'-':>4} {'-':>6}  SKIP: anchor not found in {rel}")
            problems += 1
            continue
        path.write_text(original.replace(old, new, 1))
        try:
            failed = run(expect_red + expect_green)
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

    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO).returncode
    print()
    print("tree clean vs HEAD" if dirty == 0 else "TREE DIRTY (fixes uncommitted?)")
    print("ALL MUTATIONS BEHAVED" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
