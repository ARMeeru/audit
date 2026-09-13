"""Path policy: one validator for every string that becomes a filesystem
component, and the roots the tool guard protects.

Two trust levels feed path components:

  * operator input (`--run-id`), and
  * model output derived from the target repo (`task_id`, artifact names).

Both are untrusted. Before this module existed, `RESULTS / run_id` and
`WORK / run_id / "hunt" / task_id` were joined raw and followed by
`mkdir(parents=True, exist_ok=True)`, so one `..` in either produced a directory
creation and a file write outside the harness. `safe_component` is the single
place that decides what may become a path component; both bugs die there and any
future call site has somewhere obvious to go.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"
SCHEMAS = REPO_ROOT / "schemas"
CONFIG = REPO_ROOT / "config"
RESULTS = REPO_ROOT / "results"
WORK = REPO_ROOT / "work"
STATE_DB = REPO_ROOT / "state.db"
ENV_FILE = REPO_ROOT / ".env"

# The subscription credential file. auth.py imports this rather than defining
# its own, so the guard and the login path can never drift apart.
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"


# 128 rather than 64: finding.schema.json pins finding_id to
# `^f_[a-z0-9_-]{1,64}$`, so a schema-legal id runs to 66 characters, and
# _resolve_finding_id appends a `_N` suffix on a collision. A ceiling below the
# schemas' own limits turns a legal id into a crash, which is what this used to
# do. The character class and the traversal rejection are what carry the
# security property; the length is a sanity bound.
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def safe_component(value: str, *, kind: str = "identifier") -> str:
    """Return `value` if it is safe as a single path component, else raise.

    Accepts the shapes the harness generates (`run_ab12cd34`, `f_1`, `task-01`)
    and nothing else: no separators, no traversal, no control characters, no
    non-ASCII, at most 128 characters. `kind` names the offending field in the
    message so an operator knows what to change and a repair loop knows what to
    re-emit.
    """
    if not isinstance(value, str) or not _COMPONENT_RE.match(value):
        raise ValueError(
            f"unsafe {kind} {value!r}: a path component must match "
            f"{_COMPONENT_RE.pattern} (letters, digits, dot, underscore, hyphen; "
            "1-128 characters)"
        )
    if value in (".", ".."):
        raise ValueError(f"unsafe {kind} {value!r}: path traversal")
    return value


def guarded_paths() -> dict[str, Path]:
    """The harness's own mutable state and secrets: off limits to every tool.

    Read from this module at call time rather than captured at import, so a test
    (or a future --state-dir) can move them. `WORK` is deliberately absent: the
    Hunt scratch dir lives under it and the agent is told to compile and run
    PoCs there, so a write ban would break the product. Nothing under `work/`
    carries integrity: it is transient scratch.
    """
    return {
        "the harness state database": STATE_DB,
        "the harness results tree": RESULTS,
        "the harness .env": ENV_FILE,
    }


def write_only_guards() -> dict[str, Path]:
    """Files a self-audit legitimately READS but must never rewrite.

    The prompts become the next stages' system prompts and the schemas decide
    what counts as valid, so a rewrite mid-run redirects the pipeline's own
    judgement. They are absent from guarded_paths() on purpose: auditing this
    repository means reading them, and a text filter cannot tell a read from a
    write. Tool calls that carry an explicit path can, so these are refused for
    Write/Edit only.
    """
    return {
        "the harness prompts": PROMPTS,
        "the harness schemas": SCHEMAS,
        "the harness stage config": CONFIG,
    }


def guard_hit(text: str) -> str | None:
    """Describe what `text` is reaching for, or None if it looks harmless.

    This is a FILTER over a string, not a boundary. `p=$(printf %s ...|base64
    -d); sqlite3 "$p" ...` walks straight past it, and so does a script that
    builds the path at runtime. The sandbox in runner._build_options is what
    actually stops that; this exists for the platforms where no sandbox can
    start, and for the self-audit case where the target repo IS the harness, so
    no path-based boundary can separate them.

    It fails closed on purpose: the harness DB is matched by basename, so a
    target repo with its own file called `state.db` is refused too. The denial
    names a safe alternative so the run continues instead of stalling.
    """
    return _hit(text, guarded_paths())


def write_guard_hit(text: str) -> str | None:
    """guard_hit, plus the files that are read-allowed but write-forbidden."""
    return _hit(text, {**guarded_paths(), **write_only_guards()})


def _hit(text: str, roots: dict[str, Path]) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    if "state.db" in lowered:
        return "the harness's state database (matched 'state.db')"
    if CREDENTIALS_FILE.name in lowered or str(CREDENTIALS_FILE.parent).lower() in lowered:
        return f"Claude credentials (matched {CREDENTIALS_FILE.name})"
    for label, root in roots.items():
        if str(root).lower() in lowered:
            return f"{label} (matched {root})"
    return None
