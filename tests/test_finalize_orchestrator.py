"""Orchestrator lifecycle sensors: finalize semantics, derived loop bounds,
and cap behavior. Stages are stubbed - no API calls.

Verdict convention: each test asserts the CORRECT behavior; a failure today
would mean a defect, and the test doubles as a green-when-fixed sensor."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from audit import orchestrator, stages
from audit.config import HarnessConfig, load_config
from audit.orchestrator import CostExceeded
from audit.state import StateDB


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import audit.stages._common as common_mod
    # ctx.results_dir() must resolve into tmp_path: _common resolves
    # RESULTS/WORK from REPO_ROOT, and an unpatched fixture once let a
    # stub overwrite a real run's report.json.
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    cfg = load_config()
    calls: list[str] = []
    spend = {"usd": 0.0}

    def stub_stage(name: str, findings=0, new_tasks=0, spend_usd=0.0):
        async def _fn(ctx, db, **kwargs):
            calls.append(name)
            if spend_usd:
                spend["usd"] += spend_usd
                db.record_cost(ctx.run_id, name, None,
                               {"total_cost_usd": spend_usd, "usage": {}})
            if name == "run_report":
                out = ctx.results_dir("report") / "report.json"
                out.write_text("{}")
                return out
            if name == "run_recon":
                return {}
            return findings if name == "run_hunt" else new_tasks
        return _fn

    def install(mapping):
        for name, fn in mapping.items():
            monkeypatch.setattr(stages, name, fn)

    return db, cfg, tmp_path, calls, stub_stage, install, spend


def _run(db, cfg, tmp_path, run_id="t", gapfill_iterations=None, **kwargs):
    if gapfill_iterations is not None:
        cfg.gapfill_iterations = gapfill_iterations
    return asyncio.run(orchestrator.run_pipeline(
        repo_path=tmp_path / "repo", run_id=run_id, db=db, config=cfg, **kwargs))


def test_finalize_skips_expansion_and_bypasses_expansion_cap(env):
    """--finalize must never dispatch hunt/gapfill/feedback, and must complete
    even when the expansion cap is already exhausted."""
    db, cfg, tmp, calls, stub, install, _ = env
    install({
        "run_recon": stub("run_recon", {}),
        "run_hunt": stub("run_hunt", findings=0),
        "run_validate": stub("run_validate"),
        "run_gapfill": stub("run_gapfill", new_tasks=0),
        "run_dedupe": stub("run_dedupe"),
        "run_trace": stub("run_trace"),
        "run_feedback": stub("run_feedback", new_tasks=0),
        "run_report": stub("run_report"),
    })
    db.create_run(str(tmp), "t")
    db.record_cost("t", "hunt", "x", {"total_cost_usd": 1.0, "usage": {}})

    _run(db, cfg, tmp, max_cost_usd=0.001, finalize=True)

    assert "run_hunt" not in calls and "run_gapfill" not in calls \
        and "run_feedback" not in calls


def test_finalize_grades_the_remaining_pile(env):
    """--finalize validates the remaining pile before dedupe/trace/report."""
    db, cfg, tmp, calls, stub, install, _ = env
    install({
        "run_recon": stub("run_recon", {}),
        "run_hunt": stub("run_hunt", findings=0),
        "run_validate": stub("run_validate"),
        "run_gapfill": stub("run_gapfill", new_tasks=0),
        "run_dedupe": stub("run_dedupe"),
        "run_trace": stub("run_trace"),
        "run_feedback": stub("run_feedback", new_tasks=0),
        "run_report": stub("run_report"),
    })
    db.create_run(str(tmp), "t")
    _run(db, cfg, tmp, finalize=True)
    # finalize never explores: no recon. It grades the remaining pile
    # (validate), then the closing stages. (An earlier revision of this
    # test asserted recon runs first in finalize mode — that pinned the
    # defect where --finalize fires a full opus recon on a run that died
    # during recon.)
    assert "run_recon" not in calls, "finalize must not launch recon"
    assert "run_validate" in calls
    assert calls[-1] == "run_report"


def test_finalize_requires_existing_run(env):
    db, cfg, tmp, calls, stub, install, _ = env
    with pytest.raises(RuntimeError, match="nothing to finalize"):
        _run(db, cfg, tmp, run_id="ghost", finalize=True)


def test_finalize_cap_is_per_invocation(env):
    """A tripped finalize cap must leave the run resumable: the second
    --resume --finalize invocation resets the accounting and completes."""
    db, cfg, tmp, calls, stub, install, _ = env
    dedupe_calls = []
    def dedupe_stub(spend_usd=0.0):
        async def _fn(ctx, db, **kwargs):
            calls.append("run_dedupe")
            dedupe_calls.append(1)
            if len(dedupe_calls) == 1 and spend_usd:
                db.record_cost(ctx.run_id, "run_dedupe", None,
                               {"total_cost_usd": spend_usd, "usage": {}})
            return 0
        return _fn

    install({
        "run_recon": stub("run_recon", {}),
        "run_hunt": stub("run_hunt", findings=0),
        "run_validate": stub("run_validate"),
        "run_gapfill": stub("run_gapfill", new_tasks=0),
        "run_dedupe": dedupe_stub(spend_usd=0.10),
        "run_trace": stub("run_trace"),
        "run_feedback": stub("run_feedback", new_tasks=0),
        "run_report": stub("run_report"),
    })
    db.create_run(str(tmp), "t")

    with pytest.raises(CostExceeded):
        _run(db, cfg, tmp, run_id="t", finalize=True, resume=True,
             finalize_cost_usd=0.05)

    # second invocation: per-invocation accounting resets, pipeline completes
    _run(db, cfg, tmp, run_id="t", finalize=True, resume=True,
         finalize_cost_usd=0.05)
    assert db._conn.execute(
        "SELECT status FROM runs WHERE run_id='t'").fetchone()["status"] == "completed"


def test_expansion_cap_still_aborts_non_finalize(env):
    """The run-level cap keeps its old meaning outside finalize: abort before
    spending when the ledger is already over."""
    db, cfg, tmp, calls, stub, install, _ = env
    install({
        "run_recon": stub("run_recon", {}),
        "run_hunt": stub("run_hunt", findings=0),
    })
    db.record_cost("t", "prior", None, {"total_cost_usd": 1.0, "usage": {}})

    with pytest.raises(CostExceeded):
        _run(db, cfg, tmp, run_id="t", max_cost_usd=0.50)
    assert "run_hunt" not in calls


def test_resume_does_not_re_grant_expansion(env):
    """Regression: resume used to re-grant gapfill/feedback budgets, so a
    'quick report' resume turned into another exploration round. With bounds
    derived from artifacts, fully-consumed loops are not re-run."""
    db, cfg, tmp, calls, stub, install, _ = env
    install({
        "run_recon": stub("run_recon", {}),
        "run_hunt": stub("run_hunt", findings=0),
        "run_validate": stub("run_validate"),
        "run_gapfill": stub("run_gapfill", new_tasks=0),
        "run_dedupe": stub("run_dedupe"),
        "run_trace": stub("run_trace"),
        "run_feedback": stub("run_feedback", new_tasks=0),
        "run_report": stub("run_report"),
    })
    db.create_run(str(tmp), "t")
    # two gapfill agent calls already consumed the configured bound (2)
    db.add_artifact("t", "gapfill", None, "jsonl", "/tmp/g0.jsonl")
    db.add_artifact("t", "gapfill", None, "jsonl", "/tmp/g1.jsonl")

    _run(db, cfg, tmp, run_id="t", resume=True, gapfill_iterations=2)

    assert "run_gapfill" not in calls, "resume re-granted an exhausted expansion budget"
    assert "run_report" in calls


def test_resume_runs_one_remaining_expansion_iteration(env):
    """Bounds partially consumed leave exactly the remaining iterations."""
    db, cfg, tmp, calls, stub, install, _ = env
    hunt_findings = [1]
    def hunt_stub():
        async def _fn(ctx, db, **kwargs):
            calls.append("run_hunt")
            return hunt_findings[0]
        return _fn
    install({
        "run_recon": stub("run_recon", {}),
        "run_hunt": hunt_stub(),
        "run_validate": stub("run_validate"),
        "run_gapfill": stub("run_gapfill", new_tasks=0),
        "run_dedupe": stub("run_dedupe"),
        "run_trace": stub("run_trace"),
        "run_feedback": stub("run_feedback", new_tasks=0),
        "run_report": stub("run_report"),
    })
    db.create_run(str(tmp), "t")
    db.add_artifact("t", "gapfill", None, "jsonl", "/tmp/g0.jsonl")

    _run(db, cfg, tmp, run_id="t", resume=True, gapfill_iterations=2)

    gapfill_calls = [c for c in calls if c == "run_gapfill"]
    assert len(gapfill_calls) == 1, "exactly the one remaining gapfill iteration ran"
    assert "run_report" in calls
