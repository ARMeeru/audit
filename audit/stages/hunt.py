"""Stage 2: Hunt — concurrent single-attack-class hunters."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from audit.runner import (
    AgentRunError,
    QuotaExhaustedError,
    TransientAgentError,
    run_agent,
)
from audit.state import StateDB, Task

# Per-task estimate when this run has no hunt history yet. Upper-bound by
# intent; a historical max replaces it as soon as one task completes.
DEFAULT_TASK_ESTIMATE_USD = 1.0
from audit.stages._common import StageContext, truncated_recon_summary

log = logging.getLogger(__name__)


async def run_hunt(
    ctx: StageContext,
    db: StateDB,
    budget_check: Callable[[str], None] | None = None,
) -> int:
    """Run all pending Hunt tasks concurrently. Returns the number of
    findings emitted. If `budget_check` is provided, it is invoked
    before each task and may raise to abort the stage early."""
    pending = db.get_pending_tasks(ctx.run_id)
    if not pending:
        log.info("[%s] hunt: no pending tasks", ctx.run_id)
        return 0

    sc = ctx.stage("hunt")
    recon_summary = db.get_recon_output(ctx.run_id) or {}
    sem = asyncio.Semaphore(sc.concurrency)
    aborted = asyncio.Event()
    # In-flight reservation: the cap reads committed spend, but hunt runs
    # up to `concurrency` agents at once and committed only grows when a
    # task COMPLETES. Reserving the per-task estimate at dispatch keeps the
    # cap conservative: overrun is bounded by one task's estimate, not by
    # concurrency x actual.
    estimate = db.max_stage_cost(ctx.run_id, "hunt") or DEFAULT_TASK_ESTIMATE_USD
    in_flight = [0.0]

    log.info(
        "[%s] hunt: dispatching %d tasks (concurrency=%d, model=%s)",
        ctx.run_id, len(pending), sc.concurrency, sc.model,
    )

    counters = {"findings": 0, "tasks_done": 0, "tasks_failed": 0, "skipped": 0}

    async def _one(task: Task) -> None:
        async with sem:
            if aborted.is_set():
                counters["skipped"] += 1
                return
            if budget_check is not None:
                # check-and-increment with no await between: single-threaded
                # event loop makes this atomic; a yield in the middle would
                # let every concurrent task read in_flight = 0 and pass.
                in_flight[0] += estimate
                try:
                    budget_check(f"hunt/{task.task_id}", in_flight[0])
                except Exception as e:
                    in_flight[0] -= estimate
                    log.warning("[%s] hunt aborting: %s", ctx.run_id, e)
                    aborted.set()
                    counters["skipped"] += 1
                    return
            db.begin_task(ctx.run_id, task.task_id)
            scratch = ctx.work_dir("hunt", task.task_id)
            subsystem_hint = task.target_files[0] if task.target_files else None
            user_input = {
                "task_id": task.task_id,
                "attack_class": task.attack_class,
                "scope_hint": task.scope_hint,
                "target_files": task.target_files,
                "rationale": task.rationale,
                "repo_path": str(ctx.repo_path),
                "scratch_dir": str(scratch),
                "recon_summary": truncated_recon_summary(recon_summary, subsystem_hint),
                **ctx.extras(),
            }
            try:
                result = await run_agent(
                    stage="hunt",
                    prompt_file=ctx.prompt("02-hunt"),
                    user_input=user_input,
                    schema_file=ctx.schema("finding"),
                    allowed_tools=sc.tools,
                    model=sc.model,
                    cwd=scratch,
                    add_dirs=[ctx.repo_path],
                    max_turns=sc.max_turns,
                    permission_mode=sc.permission_mode,
                    artifact_dir=ctx.results_dir("hunt"),
                    artifact_name=task.task_id,
                    repair_attempts=sc.repair_attempts,
                    on_attempt=lambda msg, _ref=task.task_id: db.record_cost(
                        ctx.run_id, "hunt", _ref, msg),
                )
            except QuotaExhaustedError as quota_error:
                # Subscription quota/session limit hit mid-flight. Don't burn
                # this task to 'failed' (which resume skips) — leave it
                # 'pending' and propagate so the pipeline aborts cleanly into
                # a resumable state. See orchestrator's QuotaExhaustedError
                # handler.
                log.error(
                    "[%s] hunt task %s hit subscription quota — aborting stage",
                    ctx.run_id, task.task_id,
                )
                db.update_task_status(ctx.run_id, task.task_id, "pending")
                aborted.set()
                raise
            except (AgentRunError, TransientAgentError) as e:
                log.warning("[%s] hunt task %s failed: %s", ctx.run_id, task.task_id, e)
                db.update_task_status(ctx.run_id, task.task_id, "failed")
                counters["tasks_failed"] += 1
                # decrement only after the callback has recorded the final
                # attempt (on_attempt fires inside run_agent before return)
                in_flight[0] -= estimate
                return
            except Exception as e:
                log.error("[%s] hunt task %s unexpected error: %s", ctx.run_id, task.task_id, e)
                db.update_task_status(ctx.run_id, task.task_id, "failed")
                counters["tasks_failed"] += 1
                in_flight[0] -= estimate
                return

            payload = result.payload
            findings = payload.get("findings", []) or []
            # findings + done flip are one transaction: a crash between the
            # writes used to leave the task 'running', and the resume
            # re-dispatch re-inserted every finding as duplicates.
            prepared = []
            for f in findings:
                fid = db.add_finding(ctx.run_id, task.task_id, f)
                prepared.append((fid, f))
                counters["findings"] += 1
            db.complete_task(ctx.run_id, task.task_id, prepared)
            db.add_artifact(ctx.run_id, "hunt", task.task_id, "jsonl",
                            str(result.artifact_path))
            db.add_artifact(ctx.run_id, "hunt", task.task_id, "scratch_dir",
                            str(scratch))
            counters["tasks_done"] += 1
            in_flight[0] -= estimate
            log.info(
                "[%s] hunt %s: %d findings (cost=$%.4f)",
                ctx.run_id, task.task_id, len(findings), result.cost_usd or 0.0,
            )

    await asyncio.gather(*(_one(t) for t in pending))
    log.info(
        "[%s] hunt: done=%d failed=%d skipped=%d findings=%d",
        ctx.run_id, counters["tasks_done"], counters["tasks_failed"],
        counters["skipped"], counters["findings"],
    )
    return counters["findings"]
