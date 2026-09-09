"""Pipeline driver: Recon → (Hunt → Validate → Gapfill)* → Dedupe → Trace
                  → Feedback → (Hunt → Validate → Dedupe → Trace)* → Report

Resume semantics: loop-iteration budgets are derived from the artifacts already
on disk, so a resumed run converges (total Gapfill/Feedback passes respect the
configured bounds) instead of re-granting exploration budgets. `--finalize`
skips exploration entirely and closes the run from current state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from audit import stages
from audit.config import HarnessConfig
from audit.runner import QuotaExhaustedError
from audit.state import StateDB
from audit.stages._common import StageContext

log = logging.getLogger(__name__)


class CostExceeded(RuntimeError):
    pass


async def run_pipeline(
    *,
    repo_path: Path,
    run_id: str,
    db: StateDB,
    config: HarnessConfig,
    max_cost_usd: float | None = None,
    finalize: bool = False,
    finalize_cost_usd: float | None = None,
    resume: bool = False,
    max_recon_tasks: int | None = None,
    live_target: dict | None = None,
    scope_notes: str | None = None,
) -> Path:
    ctx = StageContext(
        run_id=run_id,
        repo_path=repo_path.resolve(),
        config=config,
        live_target=live_target,
        scope_notes=scope_notes,
    )

    if db.get_run(run_id) is None:
        if finalize:
            raise RuntimeError(
                f"run_id {run_id!r} not found; there is nothing to finalize."
            )
        db.create_run(str(repo_path.resolve()), run_id)
        log.info("[%s] starting fresh pipeline run against %s", run_id, repo_path)
    elif resume or finalize:
        # Flip status back to 'running' so subsequent /status calls don't
        # report a stale 'aborted'/'failed' while resume work is ongoing.
        db._conn.execute(  # type: ignore[attr-defined]
            "UPDATE runs SET status = 'running', finished_at = NULL WHERE run_id = ?",
            (run_id,),
        )
        db._conn.commit()  # type: ignore[attr-defined]
        # Re-queue incomplete work so resume actually re-attempts it instead
        # of skipping it - Hunt only dispatches 'pending' tasks. Re-queueing
        # respects the attempts ceiling: a task that has failed deterministically
        # too many times stays failed instead of re-burning spend every resume.
        # Finalize skips this: it never dispatches hunt, so re-queueing would
        # only burn the attempts ceiling on tasks with no consumer.
        if not finalize:
            requeued = db.reset_incomplete_tasks(run_id)
            if requeued:
                log.info("[%s] resume: re-queued %d interrupted/failed tasks", run_id, requeued)
        if finalize:
            log.info("[%s] resuming in finalize mode (no exploration)", run_id)
        else:
            log.info("[%s] resuming existing run", run_id)
    else:
        raise RuntimeError(
            f"run_id {run_id!r} already exists; pass --resume to continue it."
        )

    def _expansion_budget_check(stage_name: str) -> None:
        if max_cost_usd is None:
            return
        spent = db.total_cost(run_id)
        if spent >= max_cost_usd:
            raise CostExceeded(
                f"[{run_id}] budget exhausted before {stage_name}: "
                f"${spent:.4f} >= ${max_cost_usd:.4f}"
            )

    # Finalize's cap bounds a single invocation, never the cumulative run:
    # a tripped finalize cap leaves the run resumable (--resume --finalize
    # resets the per-invocation accounting), so the cap can never lock the
    # user out of their report.
    finalize_start_cost = db.total_cost(run_id)

    def _finalize_budget_check(stage_name: str) -> None:
        if finalize_cost_usd is None:
            return
        spent = db.total_cost(run_id) - finalize_start_cost
        if spent >= finalize_cost_usd:
            raise CostExceeded(
                f"[{run_id}] finalize budget exhausted before {stage_name}: "
                f"${spent:.4f} of ${finalize_cost_usd:.4f} in this invocation "
                f"(--resume --finalize continues from here)"
            )

    _check = _finalize_budget_check if finalize else _expansion_budget_check

    try:
        if not finalize:
            # ---- Stage 1: Recon ----
            # Recon is exploration: --finalize on a run that died during
            # recon must close from current state, not launch a full opus
            # recon pass and queue hunts it will never dispatch.
            _check("recon")
            recon_kwargs = {} if max_recon_tasks is None else {"max_tasks": max_recon_tasks}
            await stages.run_recon(ctx, db, **recon_kwargs)

            # ---- Stages 2-3-4 loop: Hunt → Validate → Gapfill ----
            # Iterations already consumed are derived from the artifacts on
            # disk (one Gapfill agent call per iteration), so across resumes
            # the total number of Gapfill passes respects the configured
            # bound instead of being re-granted every resume.
            consumed_gapfill = db.count_artifacts(run_id, "gapfill")
            if consumed_gapfill:
                log.info(
                    "[%s] expansion: %d gapfill iteration(s) already consumed",
                    run_id, consumed_gapfill,
                )
            while True:
                _check("hunt")
                findings_added = await stages.run_hunt(ctx, db, budget_check=_check)
                _check("validate")
                await stages.run_validate(ctx, db)

                if findings_added == 0 and consumed_gapfill > 0:
                    log.info("[%s] no new findings — exiting Hunt/Gapfill loop", run_id)
                    break
                if consumed_gapfill >= config.gapfill_iterations:
                    log.info("[%s] gapfill budget consumed — exiting Hunt/Gapfill loop", run_id)
                    break
                _check("gapfill")
                new_tasks = await stages.run_gapfill(ctx, db)
                consumed_gapfill += 1
                if new_tasks == 0:
                    log.info("[%s] gapfill produced 0 tasks — exiting loop", run_id)
                    break
        else:
            # ---- Finalize: grade the remaining pile, never expand ----
            _finalize_budget_check("finalize-validate")
            await stages.run_validate(ctx, db)

        # ---- Stage 5: Dedupe ----
        _check("dedupe")
        await stages.run_dedupe(ctx, db)

        # ---- Stage 6: Trace ----
        _check("trace")
        await stages.run_trace(ctx, db)

        # ---- Stage 7: Feedback (re-runs Hunt/Validate/Dedupe/Trace) ----
        # Finalize never expands: feedback only spawns new hunts.
        if not finalize:
            consumed_feedback = db.count_artifacts(run_id, "feedback")
            for i in range(max(0, config.feedback_iterations - consumed_feedback)):
                _check(f"feedback(iter={i})")
                new_tasks = await stages.run_feedback(ctx, db)
                if new_tasks == 0:
                    break
                _check(f"feedback-hunt(iter={i})")
                await stages.run_hunt(ctx, db, budget_check=_check)
                _check(f"feedback-validate(iter={i})")
                await stages.run_validate(ctx, db)
                _check(f"feedback-dedupe(iter={i})")
                await stages.run_dedupe(ctx, db)
                _check(f"feedback-trace(iter={i})")
                await stages.run_trace(ctx, db)

        # ---- Stage 8: Report ----
        _check("report")
        report_path = await stages.run_report(ctx, db)

        # A report that names untraced canonicals or was built by the
        # fallback is not a clean completion: mark it partial so operators
        # and CI can tell "everything assessed" from "closed with gaps"
        # (--resume re-attempts the missing pieces).
        payload = json.loads(report_path.read_text())
        if payload.get("degraded") or payload.get("untraced_findings"):
            db.finish_run(run_id, "partial")
            log.warning(
                "[%s] pipeline closed PARTIAL (report degraded or %s untraced "
                "canonical(s)): total cost $%.4f — report at %s",
                run_id, len(payload.get("untraced_findings", [])),
                db.total_cost(run_id), report_path,
            )
        else:
            db.finish_run(run_id, "completed")
            log.info(
                "[%s] pipeline complete: total cost $%.4f — report at %s",
                run_id, db.total_cost(run_id), report_path,
            )
        return report_path

    except CostExceeded as e:
        log.error(str(e))
        db.finish_run(run_id, "aborted")
        raise
    except QuotaExhaustedError as e:
        # Subscription quota exhausted — surface clearly; user must wait
        # for the reset window. Run is resumable via --resume once quota
        # returns.
        log.error(
            "[%s] subscription quota exhausted — aborting (resumable with --resume): %s",
            run_id, str(e)[:300],
        )
        db.finish_run(run_id, "aborted")
        raise
    except Exception:
        db.finish_run(run_id, "failed")
        raise
