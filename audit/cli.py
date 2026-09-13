"""Click-based CLI: auth-check, run, status, report."""

from __future__ import annotations

import asyncio
import json
import re
import logging
import os
import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from audit.auth import AuthError, configure_auth


_FALSY_ENV = {"", "0", "false", "no", "off"}


def _allow_api_key_from_env_or_flag(flag: bool) -> bool:
    """A user may opt into api_key mode via --allow-api-key OR via
    AUDIT_ALLOW_API_KEY in the env. Either is sufficient. Env parsing is
    case-insensitive and treats the usual negatives (no/off/0/false) as
    opt-out, so `AUDIT_ALLOW_API_KEY=no` never silently enables metered
    billing."""
    if flag:
        return True
    return os.environ.get("AUDIT_ALLOW_API_KEY", "").strip().lower() not in _FALSY_ENV


def _require_run_id(run_id: str) -> str:
    """A run id names directories under results/ and work/, so reject anything
    that is not a plain path component before it is used at all. The state layer
    validates too; this exists so the operator gets a usage error naming the
    flag instead of a ValueError traceback."""
    try:
        return safe_component(run_id, kind="--run-id")
    except ValueError as e:
        raise click.BadParameter(str(e)) from None
from audit.config import load_config
from audit.json_utils import validate_schema
from audit.orchestrator import CostExceeded, run_pipeline
from audit.paths import REPO_ROOT, RESULTS as RESULTS_ROOT, STATE_DB, safe_component
from audit.state import StateDB
from audit.stages._common import SCHEMAS

DB_PATH = STATE_DB

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True,
                              show_path=False, markup=False)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """audit — Cloudflare-style 8-stage vulnerability discovery agent."""
    ctx.ensure_object(dict)
    _setup_logging(verbose)


@main.command("auth-check")
@click.option("--allow-api-key", is_flag=True, default=False,
              help="Honor ANTHROPIC_API_KEY for metered Anthropic billing "
                   "(also via AUDIT_ALLOW_API_KEY=1).")
def auth_check(allow_api_key: bool) -> None:
    """Verify Claude Code auth is configured correctly."""
    allow = _allow_api_key_from_env_or_flag(allow_api_key)
    try:
        status = configure_auth(allow_api_key=allow)
    except AuthError as e:
        console.print(f"[red]auth error:[/red] {e}")
        sys.exit(2)
    if status.auth_mode == "oauth_token":
        console.print("[green]OK[/green] using CLAUDE_CODE_OAUTH_TOKEN")
    elif status.auth_mode == "api_key":
        if status.gateway_base_url:
            console.print(
                f"[green]OK[/green] using ANTHROPIC_API_KEY against "
                f"{status.gateway_base_url} (metered API billing)"
            )
        console.print(
            "[green]OK[/green] using ANTHROPIC_API_KEY (metered Anthropic API billing)"
        )
    elif status.auth_mode == "keychain_login":
        console.print(
            f"[green]OK[/green] using stored login from {status.credentials_file}"
        )
    elif status.auth_mode == "macos_keychain_login":
        console.print(
            "[green]OK[/green] using macOS Keychain-backed Claude Code login"
        )
    elif status.auth_mode == "gateway":
        console.print(
            f"[green]OK[/green] using LLM gateway at {status.gateway_base_url} "
            "(ANTHROPIC_AUTH_TOKEN)"
        )
        if status.gateway_model:
            console.print(f"          ANTHROPIC_MODEL={status.gateway_model}")
    if status.api_key_scrubbed:
        console.print("[yellow]scrubbed[/yellow] ANTHROPIC_API_KEY removed from env "
                      "(it would have outranked the active auth mode)")
    if status.auth_token_scrubbed:
        console.print("[yellow]scrubbed[/yellow] ANTHROPIC_AUTH_TOKEN removed from env "
                      "(no gateway base URL set — leaving it would outrank subscription)")
    console.print(f"claude CLI: {status.claude_cli_path} ({status.claude_cli_version})")


@main.command("run")
@click.option("--repo", "repo", required=True, type=click.Path(exists=True, file_okay=False),
              help="Path to the target source-code repo.")
@click.option("--run-id", default=None, help="Run identifier (default: random).")
@click.option("--resume", is_flag=True, help="Resume an existing run-id.")
@click.option("--max-cost-usd", default=None, type=float,
              help="Abort if cumulative cost crosses this threshold.")
@click.option("--max-concurrency", default=None, type=int,
              help="Cap every stage's concurrency to this (cost containment).")
@click.option("--max-recon-tasks", default=None, type=int,
              help="Cap the number of initial Hunt tasks Recon may emit.")
@click.option("--target-url", default=None,
              help="Optional: URL of a live deployment the agents can hit "
                   "to confirm findings (e.g. http://server.local:8888).")
@click.option("--target-creds", "target_creds", multiple=True,
              metavar="KEY=VALUE",
              help="Credentials for the live target. Repeat the flag for "
                   "each KEY=VALUE pair (e.g. --target-creds email=admin@x "
                   "--target-creds password=...).")
@click.option("--scope-notes", "scope_notes_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Optional: path to a text file with target-specific scope "
                   "rules / exclusions; passed verbatim to every stage.")
@click.option("--config", "config_path", default=None, type=click.Path(),
              help="Override config/stages.yaml.")
@click.option("--finalize", "finalize", is_flag=True, default=False,
              help="Skip exploration (no hunt/gapfill/feedback): validate any "
                   "remaining findings, dedupe, trace, and write the report "
                   "from current state. Combine with --resume.")
@click.option("--finalize-cost-usd", "finalize_cost_usd", default=None, type=float,
              help="Optional flat cap on spend within ONE finalize invocation "
                   "(per-invocation, not cumulative - a tripped cap leaves the "
                   "run resumable).")
@click.option("--allow-api-key", is_flag=True, default=False,
              help="Honor ANTHROPIC_API_KEY for metered Anthropic billing "
                   "(also via AUDIT_ALLOW_API_KEY=1).")
def run(repo: str, run_id: str | None, resume: bool, max_cost_usd: float | None,
        max_concurrency: int | None, max_recon_tasks: int | None,
        target_url: str | None, target_creds: tuple[str, ...],
        scope_notes_path: str | None,
        config_path: str | None,
        finalize: bool,
        finalize_cost_usd: float | None,
        allow_api_key: bool) -> None:
    """Run the full 8-stage pipeline against a target repo."""
    allow = _allow_api_key_from_env_or_flag(allow_api_key)
    try:
        configure_auth(allow_api_key=allow)
    except AuthError as e:
        console.print(f"[red]auth error:[/red] {e}")
        sys.exit(2)

    config = load_config(Path(config_path)) if config_path else load_config()
    if max_concurrency is not None:
        config.cap_concurrency(max_concurrency)
        console.print(f"[cyan]capped concurrency to {max_concurrency} across all stages[/cyan]")

    # Live-target plumbing — agents will receive {"url": ..., "credentials": {...}}
    # in their user_input when set.
    live_target: dict | None = None
    if target_url:
        creds: dict[str, str] = {}
        for kv in target_creds:
            if "=" not in kv:
                console.print(f"[red]invalid --target-creds {kv!r} — expected KEY=VALUE[/red]")
                sys.exit(2)
            k, _, v = kv.partition("=")
            creds[k.strip()] = v.strip()
        live_target = {"url": target_url, "credentials": creds}
        console.print(f"[cyan]live target:[/cyan] {target_url} (creds: {sorted(creds)})")
    elif target_creds:
        console.print("[yellow]--target-creds without --target-url is ignored[/yellow]")

    scope_notes: str | None = None
    if scope_notes_path:
        scope_notes = Path(scope_notes_path).read_text()
        console.print(f"[cyan]scope notes loaded:[/cyan] {scope_notes_path} ({len(scope_notes)} chars)")

    run_id = _require_run_id(run_id or f"run_{uuid.uuid4().hex[:8]}")
    repo_path = Path(repo).resolve()

    db = StateDB(DB_PATH)
    try:
        report = asyncio.run(run_pipeline(
            repo_path=repo_path,
            run_id=run_id,
            db=db,
            config=config,
            max_cost_usd=max_cost_usd,
            finalize=finalize,
            finalize_cost_usd=finalize_cost_usd,
            resume=resume,
            max_recon_tasks=max_recon_tasks,
            live_target=live_target,
            scope_notes=scope_notes,
        ))
        run_row = db.get_run(run_id)
        if run_row is not None and run_row["status"] == "partial":
            console.print(
                f"[yellow]partial[/yellow] run_id={run_id} report={report} — "
                "closed with gaps (untraced canonicals or fallback report); "
                "--resume re-attempts the missing pieces"
            )
            sys.exit(4)
        console.print(f"[green]done[/green] run_id={run_id} report={report}")
    except CostExceeded as e:
        console.print(f"[yellow]aborted[/yellow] {e}")
        sys.exit(3)
    except Exception as e:
        console.print(f"[red]failed[/red] {type(e).__name__}: {e}")
        raise
    finally:
        db.close()


@main.command("status")
@click.option("--run-id", default=None)
def status(run_id: str | None) -> None:
    """Show pipeline status: tasks, findings, traces, cost."""
    db = StateDB(DB_PATH)
    try:
        if run_id is None:
            _show_runs_table(db)
            return
        run_id = _require_run_id(run_id)
        run = db.get_run(run_id)
        if run is None:
            console.print(f"[red]unknown run_id {run_id!r}[/red]")
            sys.exit(1)
        _show_run_detail(db, run_id)
    finally:
        db.close()


@main.command("report")
@click.option("--run-id", required=True)
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json")
def report(run_id: str, fmt: str) -> None:
    """Print (or generate) the final report."""
    db = StateDB(DB_PATH)
    try:
        report_path = RESULTS_ROOT / _require_run_id(run_id) / "report" / "report.json"
        if not report_path.exists():
            console.print(f"[red]no report at {report_path}[/red]")
            sys.exit(1)
        payload = json.loads(report_path.read_text())
        if fmt == "json":
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(_render_markdown_report(payload))
    finally:
        db.close()


def _show_runs_table(db: StateDB) -> None:
    runs = db.list_runs()
    t = Table(title="runs", show_lines=False)
    t.add_column("run_id")
    t.add_column("repo")
    t.add_column("status")
    t.add_column("cost ($)")
    for r in runs:
        t.add_row(r["run_id"], r["repo_path"], r["status"],
                  f"{db.total_cost(r['run_id']):.4f}")
    console.print(t)


def _show_run_detail(db: StateDB, run_id: str) -> None:
    tasks = db.get_all_tasks(run_id)
    findings = db.get_findings(run_id)
    confirmed = [f for f in findings if f.validation_status == "confirmed"]
    canonical = [f for f in confirmed if f.is_canonical]
    reachable = db.get_reachable_canonical_findings(run_id)

    t = Table(title=f"run {run_id}", show_lines=False)
    t.add_column("metric"); t.add_column("count")
    t.add_row("tasks (total)", str(len(tasks)))
    t.add_row("tasks (pending)", str(sum(1 for x in tasks if x.status == "pending")))
    t.add_row("tasks (done)", str(sum(1 for x in tasks if x.status == "done")))
    t.add_row("tasks (failed)", str(sum(1 for x in tasks if x.status == "failed")))
    t.add_row("findings (raw)", str(len(findings)))
    t.add_row("findings (confirmed)", str(len(confirmed)))
    t.add_row("findings (canonical)", str(len(canonical)))
    t.add_row("findings (reachable)", str(len(reachable)))
    t.add_row("total cost ($)", f"{db.total_cost(run_id):.4f}")
    console.print(t)

    per_stage = Table(title="tasks by stage", show_lines=False)
    per_stage.add_column("stage")
    for col in ("pending", "running", "done", "failed"):
        per_stage.add_column(col)
    by_stage: dict[str, dict[str, int]] = {}
    for x in tasks:
        row = by_stage.setdefault(x.source, {"pending": 0, "running": 0, "done": 0, "failed": 0})
        if x.status in row:
            row[x.status] += 1
    for stage in sorted(by_stage):
        row = by_stage[stage]
        per_stage.add_row(stage, *(str(row[c]) for c in ("pending", "running", "done", "failed")))
    console.print(per_stage)

    costs = Table(title="cost by stage ($)", show_lines=False)
    costs.add_column("stage"); costs.add_column("usd")
    for row in db.stage_costs(run_id):
        costs.add_row(row["stage"], f"{row['usd'] or 0:.4f}")
    console.print(costs)

    unvalidated = len(db.get_unvalidated_findings(run_id))
    untraced = len(canonical) - len(reachable)
    remaining = Table(title="finalize remaining", show_lines=False)
    remaining.add_column("work"); remaining.add_column("count")
    remaining.add_row("findings to validate", str(max(0, len(findings) - len(
        [f for f in findings if f.validation_status is not None]))))
    remaining.add_row("canonicals to trace", str(max(0, len(canonical) - len(reachable))))
    console.print(remaining)
    if unvalidated:
        console.print(f"[yellow]{unvalidated} finding(s) never validated — "
                      "run the pipeline (or --resume --finalize) to grade them[/yellow]")


def _md_inline(s: str) -> str:
    r"""Escape markdown control characters for interpolation into a line.

    Taint is per-report, not per-field: finding fields come from a model
    reading attacker-influenced target code, so a field added later must be
    safe by default. Code fences still use _code_fence.

    Line terminators are folded to spaces. Escaping the character class alone
    was not enough: `=` is not in it, so a field carrying a newline followed by
    `===` turned the preceding line into a setext heading. Every caller
    interpolates into a single line by construction, so folding loses nothing
    and removes the whole class of block-level injection."""
    text = re.sub(r"[\r\n\v\f\x85\u2028\u2029]+", " ", str(s))
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>])", r"\\\1", text)


def _code_fence(content: str) -> str:
    """A backtick fence long enough to survive any run of backticks inside
    `content`: target-influenced evidence that contains ``` must not be able
    to close the code block early and inject markdown into the
    rendered report."""
    longest = 0
    run = 0
    for ch in content:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def _render_markdown_report(report: dict) -> str:
    # The bytes here are untrusted input, not "the report we just wrote":
    # _write_report deliberately writes payloads that fail validation (flagged
    # degraded) and a report.json on disk can predate the current schema. Warn
    # rather than refuse, because a flagged invalid report beats no report.
    errors = validate_schema(report, SCHEMAS / "report.schema.json")
    if errors:
        click.echo(
            "warning: report does not validate against report.schema.json "
            f"({len(errors)} error(s); first: {errors[0]})",
            err=True,
        )

    lines: list[str] = []
    if report.get("degraded"):
        # Without this the markdown of a degraded run is byte-identical in
        # shape to a clean one, so a consumer reading stdout cannot tell that
        # the finding set is incomplete.
        lines.append("**DEGRADED REPORT** — the finding set below is incomplete.")
        if report.get("degraded_reason"):
            lines.append(f"Reason: {_md_inline(report['degraded_reason'])}")
        lines.append("")
    lines.append(f"# Vulnerability report — `{_md_inline(report['run_id'])}`")
    lines.append(f"Target: `{_md_inline(report['target']['repo_path'])}`  ")
    s = report["summary"]
    by = s.get("by_severity", {})
    total = _md_inline(s["total"])
    counts = ", ".join(f"{_md_inline(k)}: {_md_inline(v)}" for k, v in by.items())
    lines.append(f"**Total findings: {total}** — {counts}" if by
                 else f"**Total findings: {total}**")
    lines.append("")
    for f in report["findings"]:
        lines.append(f"## {_md_inline(f['title'])}")
        lines.append(f"- **Severity**: {_md_inline(f['severity'])}  ")
        lines.append(f"- **Class**: {_md_inline(f['vuln_class'])}"
                     + (f" ({_md_inline(f['cwe'])})" if f.get("cwe") else ""))
        lines.append(
            f"- **Location**: `{_md_inline(f['file'])}:"
            f"{_md_inline(f['line_start'])}-{_md_inline(f['line_end'])}`  "
        )
        lines.append("")
        # description is prose from the target-influenced model output:
        # escaped like every other inline field, so headings, images and
        # links cannot inject structure into the rendered document
        lines.append(_md_inline(f["description"]))
        lines.append("")
        fence = _code_fence(f["evidence"])
        lines.append(fence)
        lines.append(f["evidence"])
        lines.append(fence)
        lines.append("")
        ep = f["trace"].get("entry_points", [])
        if ep:
            lines.append("**Entry points**:")
            for e in ep:
                lines.append(f"- `{_md_inline(e['kind'])}` at `{_md_inline(e['location'])}`")
            lines.append("")
        cc = f["trace"].get("call_chain", [])
        if cc:
            lines.append("**Call chain**:")
            for frame in cc:
                lines.append(
                    f"1. `{_md_inline(frame['file'])}:{_md_inline(frame['line'])}`"
                    f" — `{_md_inline(frame['function'])}()`"
                )
            lines.append("")
        lines.append(f"**Recommendation**: {_md_inline(f['recommendation'])}")
        lines.append("")
        if f.get("variants"):
            lines.append(f"_Variants_: {', '.join(_md_inline(v) for v in f['variants'])}")
            lines.append("")
        lines.append("---")
        lines.append("")
    untraced = report.get("untraced_findings") or []
    if untraced:
        # Confirmed but never traced. Naming them here matters because an
        # omission reads as "nothing found", when the truth is "never assessed".
        lines.append("## Confirmed findings never traced")
        lines.append("")
        lines.append(
            "The tracer failed or quota ended the stage before these were "
            "assessed, so they are neither reported as reachable nor cleared. "
            "Re-run with --resume --finalize to grade them."
        )
        lines.append("")
        for fid in untraced:
            lines.append(f"- `{_md_inline(fid)}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
