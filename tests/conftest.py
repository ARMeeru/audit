"""Suite-wide guards.

audit.stages._common resolves RESULTS/WORK from REPO_ROOT, so an
unpatched stage test lets ctx.results_dir() resolve into the real repo
tree — a stub's write_text once wiped a real run's report.json that way.
The autouse fixture below points both globals at tmp_path for EVERY test:
stage tests that need nothing special get isolation for free, and any
test that thinks it wants the real tree has to opt out explicitly (there
are none, and there should stay none)."""

from __future__ import annotations

import asyncio
import inspect

import pytest

import audit.stages._common as common_mod


@pytest.fixture(autouse=True)
def _isolate_stage_output_paths(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    yield


def install_run_agent(monkeypatch: pytest.MonkeyPatch, mod, stub):
    """Install a run_agent stub on a stage module.

    `await run_agent(...)` on a sync stub that RETURNS raises TypeError,
    which every stage's `except Exception` swallows -- the test then
    silently exercises only the error path while staying green (both cost
    sensors shipped that way). Sync stubs are therefore wrapped in an
    async shim so they work as intended; a warning marks them so the
    author can convert to `async def`."""
    import warnings

    if inspect.iscoroutinefunction(stub):
        monkeypatch.setattr(mod, "run_agent", stub)
        return stub

    async def _shim(**kwargs):
        return stub(**kwargs)

    monkeypatch.setattr(mod, "run_agent", _shim)
    warnings.warn(
        f"run_agent stub {stub.__name__ if hasattr(stub, '__name__') else stub} "
        "is sync; wrap in async def so await run_agent exercises the "
        "success path (sync-returning stubs used to route tests through "
        "the error handler invisibly)",
        stacklevel=2,
    )
    return _shim
