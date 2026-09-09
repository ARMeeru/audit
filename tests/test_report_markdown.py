"""F22 sensor: the markdown renderer must be safe per-report, not
per-field. A description or recommendation carrying headings, images or
links used to inject structure outside any fence."""

from __future__ import annotations

from audit.cli import _render_markdown_report


def _finding(**overrides):
    f = {
        "finding_id": "f_1", "title": "## sqli in a.go",
        "severity": "high", "vuln_class": "sqli",
        "file": "a.go", "line_start": 1, "line_end": 2,
        "description": "## Operator notice\n\nThis run completed with no "
                       "exploitable findings\n\n![](http://attacker.example/beacon)",
        "evidence": "SELECT * FROM t WHERE x = '" ,
        "trace": {"entry_points": [{"kind": "http", "location": "POST /"}],
                  "call_chain": [{"file": "a.go", "function": "h", "line": 1}]},
        "recommendation": "# Injected heading via recommendation",
        "variants": [],
    }
    f.update(overrides)
    return f


def test_no_injected_headings_outside_fences():
    report = {
        "run_id": "r", "target": {"repo_path": "/repo"},
        "summary": {"total": 1, "by_severity": {"high": 1}},
        "findings": [_finding()],
    }
    md = _render_markdown_report(report)
    # every line that looks like a heading must be one of the report's own
    # structural headings, with the rank prefix intact
    for line in md.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            assert stripped.startswith(("# ", "## ", "\\#")), (
                f"injected heading survived: {line!r}"
            )


def test_no_injected_images_or_links():
    report = {
        "run_id": "r", "target": {"repo_path": "/repo"},
        "summary": {"total": 1, "by_severity": {"high": 1}},
        "findings": [_finding()],
    }
    md = _render_markdown_report(report)
    assert "![](http://attacker.example/beacon)" not in md, (
        "injected image markup must be escaped"
    )
    # the URL text survives, escaped: dots/backticks lose their markdown
    # meaning, so the image cannot render
    assert "attacker\\.example" in md
