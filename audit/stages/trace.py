"""Stage 6: Trace — reachability from entry point to sink, per canonical finding."""

from __future__ import annotations

import asyncio
import logging

from audit.runner import AgentRunError, QuotaExhaustedError, TransientAgentError, run_agent
from audit.state import Finding, StateDB
from audit.stages._common import StageContext, truncated_recon_summary

log = logging.getLogger(__name__)


async def run_trace(ctx: StageContext, db: StateDB) -> int:
    canonicals = db.get_findings(ctx.run_id, validation_status="confirmed",
                                 canonical_only=True)
    if not canonicals:
        log.info("[%s] trace: no canonical findings to trace", ctx.run_id)
        return 0

    sc = ctx.stage("trace")
    sem = asyncio.Semaphore(sc.concurrency)
    recon_summary = db.get_recon_output(ctx.run_id) or {}

    log.info(
        "[%s] trace: %d canonicals (concurrency=%d, model=%s)",
        ctx.run_id, len(canonicals), sc.concurrency, sc.model,
    )
    counters = {"reachable": 0, "unreachable": 0, "failed": 0, "skipped": 0}
    aborted = asyncio.Event()
    running: list[asyncio.Task] = []

    async def _one(f: Finding) -> None:
        async with sem:
            if aborted.is_set():
                counters["skipped"] += 1
                return
            if db.get_trace(ctx.run_id, f.finding_id) is not None:
                return  # already traced (resume)
            user_input = {
                "finding": f.raw_json,
                "recon_summary": truncated_recon_summary(recon_summary),
                "repo_path": str(ctx.repo_path),
                **ctx.extras(),
            }
            try:
                result = await run_agent(
                    stage="trace",
                    prompt_file=ctx.prompt("06-trace"),
                    user_input=user_input,
                    schema_file=ctx.schema("trace"),
                    allowed_tools=sc.tools,
                    model=sc.model,
                    cwd=ctx.repo_path,
                    add_dirs=[ctx.repo_path],
                    max_turns=sc.max_turns,
                    permission_mode=sc.permission_mode,
                    sandbox=sc.sandbox,
                    artifact_dir=ctx.results_dir("trace"),
                    artifact_name=f.finding_id,
                    repair_attempts=sc.repair_attempts,
                    on_attempt=lambda msg, _fid=f.finding_id: db.record_cost(
                        ctx.run_id, "trace", _fid, msg),
                )
            except QuotaExhaustedError:
                # Quota is the pipeline's stop signal: stop dispatching
                # siblings AND cancel the ones already in flight. Nothing
                # is persisted for the killed tracer — resume re-attempts
                # the finding.
                log.error(
                    "[%s] trace %s hit subscription quota — aborting stage",
                    ctx.run_id, f.finding_id,
                )
                aborted.set()
                for t in running:
                    t.cancel()
                raise

            except (AgentRunError, TransientAgentError) as e:
                log.warning("[%s] trace %s failed: %s", ctx.run_id, f.finding_id, e)
                counters["failed"] += 1
                # Do NOT persist an unreachable verdict for a failed tracer:
                # that would permanently hide the finding from every future
                # report (resume skips findings that already have a trace
                # row). Leaving no trace lets --resume re-attempt it. The
                # real API spend still gets recorded.
                return

            except ValueError as e:
                # An identifier that cannot become a filename. Fail this
                # finding, not the run: an exception escaping the gather marks
                # every other trace as lost too. No row, so --resume retries it.
                log.warning("[%s] trace %s unusable identifier: %s",
                            ctx.run_id, f.finding_id, e)
                counters["failed"] += 1
                return

            db.add_trace(ctx.run_id, f.finding_id, result.payload)
            db.add_artifact(ctx.run_id, "trace", f.finding_id, "jsonl",
                            str(result.artifact_path))
            if result.payload.get("reachable"):
                counters["reachable"] += 1
            else:
                counters["unreachable"] += 1

    running.extend(asyncio.ensure_future(_one(f)) for f in canonicals)
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
        "[%s] trace: reachable=%d unreachable=%d failed=%d skipped=%d",
        ctx.run_id, counters["reachable"], counters["unreachable"],
        counters["failed"], counters["skipped"],
    )
    if quota_hit is not None:
        raise quota_hit
    return counters["reachable"]
