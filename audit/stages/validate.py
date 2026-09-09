"""Stage 3: Validate — adversarial review, different model from Hunt."""

from __future__ import annotations

import asyncio
import logging

from audit.runner import AgentRunError, QuotaExhaustedError, TransientAgentError, run_agent
from audit.state import Finding, StateDB
from audit.stages._common import StageContext

log = logging.getLogger(__name__)


async def run_validate(ctx: StageContext, db: StateDB) -> int:
    """Validate every finding that hasn't been validated yet. Returns
    count of confirmed findings."""
    unvalidated = db.get_unvalidated_findings(ctx.run_id)
    if not unvalidated:
        log.info("[%s] validate: nothing to validate", ctx.run_id)
        return 0

    sc = ctx.stage("validate")
    sem = asyncio.Semaphore(sc.concurrency)

    log.info(
        "[%s] validate: %d findings (concurrency=%d, model=%s)",
        ctx.run_id, len(unvalidated), sc.concurrency, sc.model,
    )

    tasks_by_id = {t.task_id: t for t in db.get_all_tasks(ctx.run_id)}
    counters = {"confirmed": 0, "rejected": 0, "needs_more_info": 0, "failed": 0, "skipped": 0}
    aborted = asyncio.Event()
    running: list[asyncio.Task] = []

    async def _one(f: Finding) -> None:
        async with sem:
            if aborted.is_set():
                counters["skipped"] += 1
                return
            task = tasks_by_id.get(f.task_id)
            ctx_block = {
                "attack_class": task.attack_class if task else f.vuln_class,
                "scope_hint": task.scope_hint if task else "",
                "rationale": task.rationale if task else "",
            }
            user_input = {
                "finding": f.raw_json,
                "task_context": ctx_block,
                "repo_path": str(ctx.repo_path),
                **ctx.extras(),
            }
            try:
                result = await run_agent(
                    stage="validate",
                    prompt_file=ctx.prompt("03-validate"),
                    user_input=user_input,
                    schema_file=ctx.schema("validation"),
                    allowed_tools=sc.tools,
                    model=sc.model,
                    cwd=ctx.repo_path,
                    add_dirs=[ctx.repo_path],
                    max_turns=sc.max_turns,
                    permission_mode=sc.permission_mode,
                    artifact_dir=ctx.results_dir("validate"),
                    artifact_name=f.finding_id,
                    repair_attempts=sc.repair_attempts,
                    on_attempt=lambda msg, _fid=f.finding_id: db.record_cost(
                        ctx.run_id, "validate", _fid, msg),
                )
            except QuotaExhaustedError:
                # Quota is the pipeline's stop signal, not this finding's
                # failure: stop dispatching siblings AND cancel the ones
                # already in flight (gating only the queued ones left them
                # running to completion past the abort). Re-raise so the
                # run aborts into a resumable state. The finding stays
                # unvalidated and is re-attempted on resume.
                log.error(
                    "[%s] validate %s hit subscription quota — aborting stage",
                    ctx.run_id, f.finding_id,
                )
                aborted.set()
                for t in running:
                    t.cancel()
                raise

            except (AgentRunError, TransientAgentError) as e:
                log.warning("[%s] validate %s failed: %s", ctx.run_id, f.finding_id, e)
                counters["failed"] += 1
                # Persist NO verdict: get_unvalidated_findings filters on
                # validation_status IS NULL, so a persisted needs_more_info
                # would bury this finding for every future resume (the
                # validate-side twin of the trace fix). NULL is not
                # confirmed either — "avoid silently confirming" still
                # holds. A deterministically failing validation stops
                # re-burning spend via the dispatch attempts ceiling.
                return

            verdict = result.payload.get("verdict", "needs_more_info")
            db.set_finding_validation(ctx.run_id, f.finding_id, verdict, result.payload)
            db.add_artifact(ctx.run_id, "validate", f.finding_id, "jsonl",
                            str(result.artifact_path))
            counters[verdict] = counters.get(verdict, 0) + 1

    running.extend(asyncio.ensure_future(_one(f)) for f in unvalidated)
    results = await asyncio.gather(*running, return_exceptions=True)
    quota_hit = None
    for r in results:
        if isinstance(r, asyncio.CancelledError):
            counters["skipped"] += 1
        elif isinstance(r, QuotaExhaustedError):
            quota_hit = r
        elif isinstance(r, BaseException):
            raise r
    log.info(
        "[%s] validate: confirmed=%d rejected=%d needs_more_info=%d failed=%d skipped=%d",
        ctx.run_id,
        counters.get("confirmed", 0),
        counters.get("rejected", 0),
        counters.get("needs_more_info", 0),
        counters["failed"],
        counters["skipped"],
    )
    if quota_hit is not None:
        raise quota_hit
    return counters.get("confirmed", 0)
