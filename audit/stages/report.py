"""Stage 8: Report — schema-validated final document."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from audit.json_utils import validate_schema
from audit.runner import AgentRunError, QuotaExhaustedError, TransientAgentError, run_agent
from audit.state import StateDB
from audit.stages._common import SCHEMAS, StageContext

log = logging.getLogger(__name__)


async def run_report(ctx: StageContext, db: StateDB) -> Path:
    reachable = db.get_reachable_canonical_findings(ctx.run_id)
    # Trace failures are retryable and persist no row, so confirmed
    # canonicals can exist without a trace. They must be named in the
    # report — a silent omission reads as "no finding here" when the truth
    # is "never assessed".
    untraced = db.untraced_canonical_ids(ctx.run_id)
    ready = []
    for f, trace in reachable:
        ready.append({
            "finding": f.raw_json,
            "validation": f.validation_json,
            "trace": trace,
            "variants": _group_members_excluding(db, ctx.run_id, f.group_id, f.finding_id)
                       if f.group_id else [],
        })

    sc = ctx.stage("report")
    target = {"repo_path": str(ctx.repo_path)}
    user_input = {"run_id": ctx.run_id, "target": target, "ready_findings": ready,
                  **ctx.extras()}

    out_path = ctx.results_dir("report") / "report.json"

    if not ready:
        # No reachable findings — emit a minimal empty report without burning an agent call.
        empty = {
            "run_id": ctx.run_id,
            "target": target,
            "summary": {"total": 0, "by_severity": {}},
            "findings": [],
            "untraced_findings": untraced,
            "degraded": bool(untraced),
            "degraded_reason": (
                f"{len(untraced)} confirmed canonical(s) have no trace row "
                "(tracer failed or quota killed the stage); they are not assessed"
            ) if untraced else None,
        }
        out_path.write_text(json.dumps(empty, indent=2))
        log.info("[%s] report: no reachable findings — wrote empty report to %s",
                 ctx.run_id, out_path)
        return out_path

    try:
        result = await run_agent(
            stage="report",
            prompt_file=ctx.prompt("08-report"),
            user_input=user_input,
            schema_file=ctx.schema("report"),
            allowed_tools=sc.tools,
            model=sc.model,
            cwd=ctx.repo_path,
            add_dirs=[ctx.repo_path],
            max_turns=sc.max_turns,
            permission_mode=sc.permission_mode,
            artifact_dir=ctx.results_dir("report"),
            artifact_name="report_agent",
            repair_attempts=max(sc.repair_attempts, 2),  # report MUST validate
            on_attempt=lambda msg: db.record_cost(ctx.run_id, "report", None, msg),
        )
    except (AgentRunError, TransientAgentError, QuotaExhaustedError) as e:
        # The fallback report is deterministic (rendered from state.db), so a
        # quota-killed report agent still yields the reachable finding set -
        # only the prose is lost. Marked degraded: a CI consumer must be able
        # to tell this from a clean run, and the orchestrator must not mark
        # the run plainly "completed".
        log.error("[%s] report agent failed: %s — emitting fallback report",
                  ctx.run_id, e)
        fallback = _build_fallback_report(ctx, db, reachable, target)
        fallback["untraced_findings"] = untraced
        fallback["degraded"] = True
        fallback["degraded_reason"] = f"report agent failed: {str(e)[:300]}"
        # The fallback bypasses the report agent and its repair budget, so
        # it must be validated the way the agent output is: an invalid
        # fallback document fails every downstream consumer silently.
        errors = validate_schema(fallback, SCHEMAS / "report.schema.json")
        if errors:
            fallback["degraded_reason"] += " | schema errors: " + "; ".join(errors[:5])
            log.error("[%s] fallback report violates report schema: %s",
                      ctx.run_id, errors[:5])
        out_path.write_text(json.dumps(fallback, indent=2))
        return out_path

    db.add_artifact(ctx.run_id, "report", None, "jsonl", str(result.artifact_path))
    result.payload["untraced_findings"] = untraced
    result.payload.setdefault("degraded", False)
    out_path.write_text(json.dumps(result.payload, indent=2))
    log.info("[%s] report: %d findings written to %s",
             ctx.run_id, len(result.payload.get("findings", [])), out_path)
    return out_path


def _group_members_excluding(db: StateDB, run_id: str, group_id: str,
                             exclude: str) -> list[str]:
    rows = db._conn.execute(  # type: ignore[attr-defined]
        "SELECT finding_id FROM findings WHERE run_id = ? AND group_id = ? AND finding_id != ?",
        (run_id, group_id, exclude),
    ).fetchall()
    return [r["finding_id"] for r in rows]


def _project_trace_for_report(trace: dict) -> dict:
    """Project a trace.schema.json trace onto the keys report.schema.json
    allows (additionalProperties: false on both): entry points lose
    auth_required, call-chain frames lose note."""
    return {
        "entry_points": [
            {k: ep[k] for k in ("kind", "location", "controllable_by") if k in ep}
            for ep in trace.get("entry_points", [])
        ],
        "call_chain": [
            {k: fr[k] for k in ("file", "function", "line") if k in fr}
            for fr in trace.get("call_chain", [])
        ],
    }


def _build_fallback_report(ctx: StageContext, db: StateDB,
                           reachable, target: dict) -> dict:
    by_sev: dict[str, int] = {}
    findings_out = []
    for f, trace in reachable:
        sev = f.severity
        by_sev[sev] = by_sev.get(sev, 0) + 1
        description = f.description
        while len(description) < 30:
            # report.schema.json requires minLength 30; hunt findings are
            # free-form. Extend with a truthful pointer, never invent.
            description += " (detail in evidence)"
        findings_out.append({
            "finding_id": f.finding_id,
            "title": f"{f.vuln_class} in {f.file}",
            "severity": sev,
            "vuln_class": f.vuln_class,
            "file": f.file,
            "line_start": f.line_start,
            "line_end": f.line_end,
            "description": description,
            "evidence": f.evidence,
            "trace": _project_trace_for_report(trace),
            "recommendation": "Review the sink and add input validation / use a safe API.",
            "variants": _group_members_excluding(db, ctx.run_id, f.group_id, f.finding_id)
                        if f.group_id else [],
        })
    return {
        "run_id": ctx.run_id,
        "target": target,
        "summary": {"total": len(findings_out), "by_severity": by_sev},
        "findings": findings_out,
    }
