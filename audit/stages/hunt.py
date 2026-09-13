"""Stage 2: Hunt — concurrent single-attack-class hunters."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from audit.paths import UnsafeIdentifier, safe_component
from audit.runner import (
    AgentRunError,
    QuotaExhaustedError,
    TransientAgentError,
    run_agent,
)
from audit.state import StateDB, Task

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
    # In-flight reservation: the cap reads committed spend, but hunt runs up
    # to `concurrency` agents at once and committed only grows when a task
    # COMPLETES. Reserving a configured per-task estimate at dispatch makes
    # the trip conservative.
    #
    # The estimate is a configured CONSTANT (config: est_cost_usd), not a
    # derivation from observed spend. That is a deliberate scope choice: the
    # requirement here is a bound, not an estimate, and the estimator this
    # replaced (prior chain plus self-correction plus a cold-start ramp) cost
    # more in defects than the precision bought. It was only ever correct
    # because the constant already exceeded every observed task cost (hunt
    # max $0.676 against a $1.50 default); the derivation could not have
    # improved on that. Raise est_cost_usd if a stage's cost profile changes.
    estimate = sc.est_cost_usd
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
            reserved = 0.0
            if budget_check is not None:
                # Reserve a SNAPSHOT rather than reading `estimate` at
                # release time. The estimate is a constant today, so the two
                # are equal; the snapshot keeps release correct if it ever
                # becomes dynamic again (a release that reads a raised
                # estimate returns more than the task reserved, driving
                # in_flight negative and making the cap permissive).
                # Check-and-increment stay sync: a yield in the middle would
                # let every task read in_flight = 0.
                reserved = estimate
                in_flight[0] += reserved
                try:
                    budget_check(f"hunt/{task.task_id}", in_flight[0])
                except Exception as e:
                    in_flight[0] -= reserved
                    log.warning("[%s] hunt aborting: %s", ctx.run_id, e)
                    aborted.set()
                    counters["skipped"] += 1
                    return
            db.begin_task(ctx.run_id, task.task_id)
            # `task_id` becomes both a scratch directory name and an artifact
            # filename. The agent path is already covered: every producer of
            # tasks $refs hunt_task.schema.json, which pins task_id to
            # ^[a-z0-9_-]{1,64}$, so a traversing id never survives validation.
            # This is the backstop for an id that arrives by another route (a
            # tampered or hand-edited DB, a future importer), and it turns that
            # into one failed task instead of a stage-level crash or an escape.
            #
            # After begin_task on purpose: that is what spends the attempt, and
            # a rejection that spends none is re-queued by every resume forever
            # while staying invisible to the abandonment count, which filters on
            # attempts >= the ceiling.
            try:
                safe_component(task.task_id, kind="task_id")
            except UnsafeIdentifier as bad_id:
                log.error("[%s] hunt task %r rejected: %s", ctx.run_id,
                          task.task_id, bad_id)
                db.update_task_status(ctx.run_id, task.task_id, "failed")
                counters["tasks_failed"] += 1
                in_flight[0] -= reserved
                return
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
                    sandbox=sc.sandbox,
                    network_allow=ctx.network_allow(),
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
                db.release_task(ctx.run_id, task.task_id)
                aborted.set()
                in_flight[0] -= reserved
                raise
            except (AgentRunError, TransientAgentError) as e:
                log.warning("[%s] hunt task %s failed: %s", ctx.run_id, task.task_id, e)
                db.update_task_status(ctx.run_id, task.task_id, "failed")
                counters["tasks_failed"] += 1
                # release the snapshot, not the live box (see reserve above)
                in_flight[0] -= reserved
                return
            except Exception as e:
                log.error("[%s] hunt task %s unexpected error: %s", ctx.run_id, task.task_id, e)
                db.update_task_status(ctx.run_id, task.task_id, "failed")
                counters["tasks_failed"] += 1
                in_flight[0] -= reserved
                return

            payload = result.payload
            findings = payload.get("findings", []) or []
            # findings + done flip are one transaction: a crash between the
            # writes used to leave the task 'running', and the resume
            # re-dispatch re-inserted every finding as duplicates.
            inserted = db.complete_task(ctx.run_id, task.task_id, findings)
            counters["findings"] += inserted
            db.add_artifact(ctx.run_id, "hunt", task.task_id, "jsonl",
                            str(result.artifact_path))
            db.add_artifact(ctx.run_id, "hunt", task.task_id, "scratch_dir",
                            str(scratch))
            counters["tasks_done"] += 1
            in_flight[0] -= reserved
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
