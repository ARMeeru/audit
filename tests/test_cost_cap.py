"""Cost-cap sensors: the composite cap semantics (F11) and the in-flight
reservation property (F12). Every test asserts the property an operator
cares about, not the mechanism — the mechanism can be redesigned, the
property cannot."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import audit.stages._common as common_mod
import audit.stages.hunt as hunt_mod
from audit import orchestrator, stages
from audit.config import load_config
from audit.orchestrator import CostExceeded
from audit.state import StateDB
from audit.stages._common import StageContext
from tests.test_stage_failures import stage_env  # noqa: F401


def test_finalize_does_not_void_the_run_budget(tmp_path: Path, monkeypatch):
    """F11: --finalize used to replace the cumulative cap entirely, so the
    expensive finalize stages (validate, trace: opus at concurrency 10) ran
    uncapped. A run budget that is NOT already blown must bound the
    finalize invocation too."""
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    cfg = load_config()
    spend = {"usd": 0.0}

    def expensive_stub(name, cost):
        async def _fn(ctx, db, **kwargs):
            spend["usd"] += cost
            db.record_cost(ctx.run_id, name, None,
                           {"total_cost_usd": cost, "usage": {}})
            if name == "run_report":
                out = ctx.results_dir("report") / "report.json"
                out.write_text("{}")
                return out
            return 0
        return _fn

    for name, cost in [("run_recon", 0.0), ("run_validate", 10.0),
                       ("run_dedupe", 0.0), ("run_trace", 10.0),
                       ("run_report", 0.0)]:
        monkeypatch.setattr(stages, name, expensive_stub(name, cost))

    db.create_run(str(tmp_path), "t")
    with pytest.raises(CostExceeded):
        asyncio.run(orchestrator.run_pipeline(
            repo_path=tmp_path, run_id="t", db=db, config=cfg,
            max_cost_usd=1.0, finalize=True))
    assert spend["usd"] <= 1.0 + 10.0, (
        "one in-flight stage may finish after the trip; more is a voided cap"
    )


def test_finalize_still_closes_a_blown_budget(tmp_path: Path, monkeypatch):
    """The escape hatch is preserved verbatim: a run whose budget is
    already gone must remain closable by --finalize (the cap can never
    lock the user out of their report)."""
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    db = StateDB(tmp_path / "state.db")
    cfg = load_config()

    async def cheap(ctx, db, **kwargs):
        if kwargs.get("_name") == "run_report":
            out = ctx.results_dir("report") / "report.json"
            out.write_text("{}")
            return out
        return 0

    for name in ("run_recon", "run_validate", "run_dedupe", "run_trace",
                 "run_report"):
        async def _fn(ctx, db, **kwargs):
            return 0
        monkeypatch.setattr(stages, name, _fn)

    async def report_fn(ctx, db, **kwargs):
        out = ctx.results_dir("report") / "report.json"
        out.write_text("{}")
        return out
    monkeypatch.setattr(stages, "run_report", report_fn)

    db.create_run(str(tmp_path), "t")
    db.record_cost("t", "hunt", "x", {"total_cost_usd": 1.0, "usage": {}})
    asyncio.run(orchestrator.run_pipeline(
        repo_path=tmp_path, run_id="t", db=db, config=cfg,
        max_cost_usd=0.001, finalize=True))
    assert db._conn.execute(
        "SELECT status FROM runs WHERE run_id='t'").fetchone()["status"] == "completed"


def test_hunt_overrun_bounded_by_one_task_estimate(
    stage_env, monkeypatch
) -> None:
    """F12 property: with N tasks whose actual spend exceeds the cap, total
    spend stays under cap + one task estimate. A test asserting an exact
    agent count would not survive a reservation redesign; this one does.
    (stage_env fixture from test_stage_failures provides tmp isolation.)"""
    db, ctx = stage_env
    for i in range(6):
        db.add_task("poc", {"task_id": f"t_{i}", "attack_class": "sqli",
                            "scope_hint": "x", "target_files": ["a.py"],
                            "rationale": "r", "priority": 1, "source": "recon"})
    # history makes the per-task estimate an honest upper bound: $2/task
    db.record_cost("poc", "hunt", "hist", {"total_cost_usd": 2.0, "usage": {}})

    def spending_agent(**kwargs):
        on_attempt = kwargs.get("on_attempt")
        on_attempt({"total_cost_usd": 2.0, "usage": {"input_tokens": 10}})
        class R:
            payload = {"findings": []}
            raw_result_message = {"total_cost_usd": 2.0, "usage": {}}
            from pathlib import Path as _P
            artifact_path = _P("/tmp/x.jsonl")
            cost_usd = 2.0
        return R()

    monkeypatch.setattr(hunt_mod, "run_agent", spending_agent)
    cap = 5.0
    def budget_check(name, in_flight_usd=0.0):
        if db.total_cost("poc") + in_flight_usd >= cap:
            raise CostExceeded(name)
    asyncio.run(hunt_mod.run_hunt(ctx, db, budget_check=budget_check))

    assert db.total_cost("poc") <= cap + 2.0, (
        "overrun must be bounded by one task's estimate, not by concurrency"
    )

def test_first_hunt_stage_with_no_history_stays_bounded(
    stage_env, monkeypatch
) -> None:
    """F6/R2: the property must hold in the configuration the code runs in
    most often — a run's FIRST hunt stage has no history rows, so the
    estimate starts at the hard default. With per-task actuals far above
    the default, self-correction plus reservation must still bound the
    overrun (this configuration violated the property before the prior
    chain and self-correction landed)."""
    db, ctx = stage_env
    for i in range(60):
        db.add_task("poc", {"task_id": f"t_{i}", "attack_class": "sqli",
                            "scope_hint": "x", "target_files": ["a.py"],
                            "rationale": "r", "priority": 1, "source": "recon"})
    # deliberately NO history: estimate starts at DEFAULT_TASK_ESTIMATE_USD.
    # Unproven configuration, stated per the sensor-arrangement rule: with a
    # fully concurrent first wave and actuals far above the default, the
    # one-estimate bound cannot hold even post-fix (all wave reservations
    # use the default before any completion corrects it). This sensor
    # exercises the synchronous-completion configuration, where the
    # self-correction demonstrably tightens the trip point.
    from audit.stages.hunt import DEFAULT_TASK_ESTIMATE_USD

    def spending_agent(**kwargs):
        on_attempt = kwargs.get("on_attempt")
        on_attempt({"total_cost_usd": 3.0, "usage": {"input_tokens": 10}})
        class R:
            payload = {"findings": []}
            raw_result_message = {"total_cost_usd": 3.0, "usage": {}}
            from pathlib import Path as _P
            artifact_path = _P("/tmp/x.jsonl")
            cost_usd = 3.0
        return R()

    monkeypatch.setattr(hunt_mod, "run_agent", spending_agent)
    cap = 10.0
    def budget_check(name, in_flight_usd=0.0):
        if db.total_cost("poc") + in_flight_usd >= cap:
            raise CostExceeded(name)
    asyncio.run(hunt_mod.run_hunt(ctx, db, budget_check=budget_check))

    estimate_ceiling = max(DEFAULT_TASK_ESTIMATE_USD, 3.0)
    assert db.total_cost("poc") <= cap + estimate_ceiling, (
        f"overrun must stay bounded by one (self-corrected) estimate: "
        f"spent {db.total_cost('poc')}"
    )
