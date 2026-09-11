"""Cost-cap sensors: the composite cap semantics (F11) and the in-flight
reservation property (F12). Every test asserts the property an operator
cares about, not the mechanism — the mechanism can be redesigned, the
property cannot."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.conftest import install_run_agent

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


# Highest single-task cost ever recorded on the live runs (state.db, 410
# cost rows): hunt $0.676 over 145 tasks, validate $1.26 over 222, trace
# $0.95 over 43. These are the numbers the reservation has to clear.
OBSERVED_STAGE_MAX_USD = {"hunt": 0.676, "validate": 1.26, "trace": 0.95}


def test_configured_estimate_exceeds_observed_task_costs() -> None:
    """The in-flight reservation is only conservative while est_cost_usd
    sits ABOVE real per-task cost: below it, `committed + in_flight`
    under-states spend and the cap becomes more permissive than committed
    spend alone. This is the property that made the deleted estimator
    unnecessary -- it was never more accurate than the constant, because
    the constant already cleared every observed task. Red if the config
    drops below reality (a stage moves to opus, max_turns rises)."""
    cfg = load_config()
    for stage, observed_max in OBSERVED_STAGE_MAX_USD.items():
        assert cfg.get(stage).est_cost_usd > observed_max, (
            f"{stage}'s est_cost_usd must exceed the highest observed task "
            f"cost (${observed_max}); below it the reservation under-states "
            f"spend and --max-cost-usd becomes permissive"
        )


def test_hunt_trip_is_conservative_when_estimate_clears_actual(
    stage_env, monkeypatch
) -> None:
    """With the estimate above per-task actual, the trip fires early rather
    than late: total spend stays within cap + one wave of actual, and the
    reservation is never the reason spend exceeded the cap."""
    db, ctx = stage_env
    for i in range(12):
        db.add_task("poc", {"task_id": f"t_{i:02d}", "attack_class": "sqli",
                            "scope_hint": "x", "target_files": ["a.py"],
                            "rationale": "r", "priority": 1, "source": "recon"})

    actual = 0.5   # well under the configured $1.50 estimate

    async def spending_agent(**kwargs):
        kwargs["on_attempt"]({"total_cost_usd": actual, "usage": {}})
        class R:
            payload = {"findings": []}
            raw_result_message = {"total_cost_usd": actual, "usage": {}}
            from pathlib import Path as _P
            artifact_path = _P("/tmp/x.jsonl")
            cost_usd = actual
        return R()

    install_run_agent(monkeypatch, hunt_mod, spending_agent)
    cap = 4.0

    def budget_check(name, in_flight_usd=0.0):
        if db.total_cost("poc") + in_flight_usd >= cap:
            raise CostExceeded(name)

    asyncio.run(hunt_mod.run_hunt(ctx, db, budget_check=budget_check))
    spent = db.total_cost("poc")
    assert spent <= cap + actual, (
        f"conservative trip must not overspend beyond cap + one task: {spent}"
    )


@pytest.mark.parametrize("pre_spent,stage_cost,expect_completed", [
    (0.0, 0.0, True),    # fresh run: finalize runs, cap bounds the invocation
    (4.0, 0.0, True),    # below half: runs
    (6.0, 0.0, True),    # the old remainder bug tripped here (half-band)
    (9.0, 0.0, True),    # just under: runs when the invocation spends nothing
    (10.0, 0.0, True),   # at cap: escape hatch, closes uncapped
    (12.0, 0.0, True),   # over cap: escape hatch
    (9.0, 2.0, False),   # invocation spend pushes cumulative past the cap
])
def test_finalize_cap_across_the_spend_band(
    stage_env, monkeypatch, pre_spent, stage_cost, expect_completed
):
    """F3/R5: the remainder-with-zero-baseline pairing aborted whenever
    cumulative spend sat in [cap/2, cap) -- the band where closing out a
    run is most plausible. Finalize must run for every pre-spend below
    the cap, close uncapped at or above it, and abort only when the
    invocation itself pushes cumulative spend past the cap."""
    from audit import orchestrator
    from audit.state import StateDB
    from audit.config import load_config
    from pathlib import Path

    tmp = stage_env[2] if len(stage_env) > 2 else None
    # stage_env yields (db, ctx); build our own isolated pieces
    db, ctx = stage_env
    import audit.stages._common as common_mod
    tmp_path = Path(str(ctx.repo_path)).parent

    cfg = load_config()
    ran = {"stages": False}

    def cheap_stub(name, cost=0.0):
        async def _fn(c, d, **kwargs):
            ran["stages"] = True
            if cost:
                d.record_cost(c.run_id, name, None,
                              {"total_cost_usd": cost, "usage": {}})
            return 0
        return _fn

    async def report_fn(c, d, **kwargs):
        ran["stages"] = True
        out = c.results_dir("report") / "report.json"
        out.write_text('{"run_id": "%s", "target": {}, "summary": '
                       '{"total": 0, "by_severity": {}}, "findings": []}' % c.run_id)
        return out

    for name in ("run_recon", "run_validate", "run_dedupe", "run_trace",
                 "run_feedback", "run_gapfill", "run_hunt"):
        monkeypatch.setattr(orchestrator.stages, name,
                            cheap_stub(name, stage_cost if name == "run_validate" else 0.0))
    monkeypatch.setattr(orchestrator.stages, "run_report", report_fn)

    db.create_run("/r", "band")
    if pre_spent:
        db.record_cost("band", "hunt", "x",
                       {"total_cost_usd": pre_spent, "usage": {}})

    from audit.orchestrator import run_pipeline, CostExceeded
    from audit.orchestrator import run_pipeline, CostExceeded
    if expect_completed:
        asyncio.run(run_pipeline(
            repo_path=tmp_path, run_id="band", db=db, config=cfg,
            max_cost_usd=10.0, finalize=True, resume=True))
        assert ran["stages"]
        status = db._conn.execute(
            "SELECT status FROM runs WHERE run_id='band'").fetchone()["status"]
        assert status == "completed"
    else:
        with pytest.raises(CostExceeded):
            asyncio.run(run_pipeline(
                repo_path=tmp_path, run_id="band", db=db, config=cfg,
                max_cost_usd=10.0, finalize=True, resume=True))
        assert ran["stages"], "validate runs before the cap trips at dedupe"


def test_in_flight_never_goes_negative(stage_env, monkeypatch):
    """Invariant: budget_check must never see a negative in_flight --
    releasing more than the task reserved understates spend to the cap.

    Currently VACUOUS, stated here rather than left to look like coverage:
    the estimate is a configured constant, so releasing it equals releasing
    the reserved snapshot and no drift is possible. Mutation-confirmed --
    swapping the release back to a live read leaves this test green, because
    there is nothing to read differently. It becomes load-bearing the moment
    the estimate turns dynamic again (which the reservation comment warns
    against); the live sensor for this design is
    test_configured_estimate_exceeds_observed_task_costs."""
    db, ctx = stage_env
    for i in range(60):
        db.add_task("poc", {"task_id": f"t_{i:02d}", "attack_class": "sqli",
                            "scope_hint": "x", "target_files": ["a.py"],
                            "rationale": "r", "priority": 1, "source": "recon"})

    seen_in_flight: list[float] = []

    async def spending_agent(**kwargs):
        import asyncio as _aio
        on_attempt = kwargs.get("on_attempt")
        on_attempt({"total_cost_usd": 5.0, "usage": {"input_tokens": 10}})
        # yield so the first wave stacks its reservations before any
        # completion; the late reserves then observe post-release drift
        await _aio.sleep(0)
        class R:
            payload = {"findings": []}
            raw_result_message = {"total_cost_usd": 5.0, "usage": {}}
            from pathlib import Path as _P
            artifact_path = _P("/tmp/x.jsonl")
            cost_usd = 5.0
        return R()

    install_run_agent(monkeypatch, hunt_mod, spending_agent)

    def budget_check(name, in_flight_usd=0.0):
        seen_in_flight.append(in_flight_usd)

    asyncio.run(hunt_mod.run_hunt(ctx, db, budget_check=budget_check))
    assert min(seen_in_flight) >= 0.0, (
        f"negative in_flight understates spend to the cap: "
        f"min={min(seen_in_flight)}"
    )
