"""Suite-wide guards.

audit.stages._common resolves RESULTS/WORK from REPO_ROOT, so an
unpatched stage test lets ctx.results_dir() resolve into the real repo
tree — a stub's write_text once wiped a real run's report.json that way.
The autouse fixture below points both globals at tmp_path for EVERY test:
stage tests that need nothing special get isolation for free, and any
test that thinks it wants the real tree has to opt out explicitly (there
are none, and there should stay none)."""

from __future__ import annotations

import pytest

import audit.stages._common as common_mod


@pytest.fixture(autouse=True)
def _isolate_stage_output_paths(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(common_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(common_mod, "WORK", tmp_path / "work")
    yield
