"""Stage 5: Dedupe — cluster confirmed findings by root cause."""

from __future__ import annotations

import hashlib
import logging

from audit.runner import AgentRunError, QuotaExhaustedError, TransientAgentError, run_agent
from audit.state import StateDB
from audit.stages._common import StageContext

log = logging.getLogger(__name__)


def _prepare_groups(run_id: str, groups: list[dict],
                    confirmed_ids: set[str]) -> list[tuple[dict, list[str], str]]:
    """Validate agent-emitted groups against this run's confirmed set.

    Drops unknown members; a canonical that is not among the (surviving)
    members falls back to the first member — a hallucinated canonical would
    otherwise make `fid == canonical` false for every member and empty the
    report's canonical set."""
    prepared = []
    claimed: set[str] = set()
    for g in groups:
        # dict.fromkeys dedupes inside the group; `claimed` enforces that a
        # finding belongs to at most one group -- a finding named twice was
        # assigned last-write-wins and lost canonical status entirely, the
        # same emptied-report outcome F4 causes through the canonical door.
        members = [fid for fid in dict.fromkeys(g.get("member_finding_ids", []))
                   if fid in confirmed_ids and fid not in claimed]
        if not members:
            log.warning("[%s] dedupe: group %s has no unclaimed known members — dropped",
                        run_id, g.get("group_id"))
            continue
        claimed.update(members)
        canonical = g.get("canonical_finding_id")
        if canonical not in members:
            log.warning("[%s] dedupe: canonical %s not among members; using %s",
                        run_id, canonical, members[0])
            canonical = members[0]
        prepared.append((g, members, canonical))
    return prepared


async def run_dedupe(ctx: StageContext, db: StateDB) -> int:
    confirmed = db.get_findings(ctx.run_id, validation_status="confirmed")
    if not confirmed:
        log.info("[%s] dedupe: no confirmed findings to cluster", ctx.run_id)
        return 0

    confirmed_ids = {f.finding_id for f in confirmed}
    set_hash = hashlib.sha256(
        "\n".join(sorted(confirmed_ids)).encode()
    ).hexdigest()

    # Repeat invocations over an unchanged confirmed set are idempotent:
    # --finalize is designed for repeated use, and re-paying the dedupe
    # agent call for an identical input is pure waste.
    if db.latest_artifact_path(ctx.run_id, "dedupe", "confirmed_set_hash") == set_hash:
        n = db.count_dedupe_groups(ctx.run_id)
        log.info("[%s] dedupe: confirmed set unchanged since last pass "
                 "(%d groups) — skipping", ctx.run_id, n)
        return n

    sc = ctx.stage("dedupe")
    payload = []
    for f in confirmed:
        payload.append({
            **f.raw_json,
            "validation": f.validation_json,
        })

    log.info("[%s] dedupe: clustering %d confirmed findings", ctx.run_id, len(confirmed))
    try:
        result = await run_agent(
            stage="dedupe",
            prompt_file=ctx.prompt("05-dedupe"),
            user_input={"confirmed_findings": payload, **ctx.extras()},
            schema_file=ctx.schema("dedupe_output"),
            allowed_tools=sc.tools,
            model=sc.model,
            cwd=ctx.repo_path,
            add_dirs=[ctx.repo_path],
            max_turns=sc.max_turns,
            permission_mode=sc.permission_mode,
            artifact_dir=ctx.results_dir("dedupe"),
            artifact_name="dedupe",
            repair_attempts=sc.repair_attempts,
            on_attempt=lambda msg: db.record_cost(ctx.run_id, "dedupe", None, msg),
        )
    except QuotaExhaustedError:
        raise
    except (AgentRunError, TransientAgentError) as e:
        log.warning("[%s] dedupe failed: %s — treating each finding as its own group",
                    ctx.run_id, e)
        # Fallback: one group per finding, all canonical.
        fallback = []
        for f in confirmed:
            gid = f"g_{f.finding_id[2:]}" if f.finding_id.startswith("f_") else f"g_{f.finding_id}"
            fallback.append({
                "group_id": gid,
                "root_cause": f.description[:200],
                "canonical_finding_id": f.finding_id,
                "member_finding_ids": [f.finding_id],
            })
        prepared = _prepare_groups(ctx.run_id, fallback, confirmed_ids)
        applied = db.apply_dedupe_groups(ctx.run_id, prepared)
        db.add_artifact(ctx.run_id, "dedupe", None, "confirmed_set_hash", set_hash)
        return applied

    groups = result.payload.get("groups", [])
    db.add_artifact(ctx.run_id, "dedupe", None, "jsonl", str(result.artifact_path))
    prepared = _prepare_groups(ctx.run_id, groups, confirmed_ids)
    applied = db.apply_dedupe_groups(ctx.run_id, prepared)
    db.add_artifact(ctx.run_id, "dedupe", None, "confirmed_set_hash", set_hash)

    log.info("[%s] dedupe: %d findings → %d groups", ctx.run_id, len(confirmed), applied)
    return applied
