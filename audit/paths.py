"""Path policy: one validator for every string that becomes a filesystem
component, and the guard that keeps agents out of the harness's own files.

The design is deliberately small. Two review rounds grew a Bash filter that
matched globs, braces, quoting, `cd` and `--flag=` forms; it was still bypassable
by eight ordinary inputs, it regressed behaviour it had previously got right, and
a 103-character glob in a PreToolUse hook blocked every concurrent agent for 41
seconds. The OS sandbox in runner._build_options is the boundary. What is left
here is the part that is exact by construction, plus a documented speed bump.

`safe_component` is the single place that decides what may become a path
component. Two trust levels feed it: operator input (`--run-id`) and model output
derived from the target repo (`task_id`, artifact names).

`structured_path_hit` answers one question exactly: does the path this tool is
about to use reach a guarded file? It resolves the path and compares by
filesystem identity, so case-folded spellings, hardlinks, symlinks, `..`,
relative spellings and `~` are all handled by the filesystem. There is no string
table and no pattern language.

`bash_guard_hit` is a FILTER and nothing more. It expands the shell's home
spellings and refuses the guarded names. It does not model globbing, quoting, `cd`
or runtime construction, and three review rounds are enough evidence that no
amount of that belongs in a permission hook. Its denials redirect the agent; the
sandbox is what actually holds, and the README says which geometries have neither.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"
SCHEMAS = REPO_ROOT / "schemas"
CONFIG = REPO_ROOT / "config"
RESULTS = REPO_ROOT / "results"
WORK = REPO_ROOT / "work"
STATE_DB = REPO_ROOT / "state.db"
ENV_FILE = REPO_ROOT / ".env"

# The subscription credential file, unresolved on purpose: a symlinked
# credentials file must not move the guarded directory to its target.
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"


class UnsafeIdentifier(ValueError):
    """A string that may not become a path component.

    Narrow on purpose. The stage handlers used to catch bare `ValueError`, which
    also swallowed `json.JSONDecodeError` from reading a schema file and reported
    it as an unusable identifier, sending an operator after an id that was fine.
    """


# 128 rather than 64: finding.schema.json pins finding_id to
# `^f_[a-z0-9_-]{1,64}$`, so a schema-legal id runs to 66 characters, and
# _resolve_finding_id appends a `_N` suffix on a collision. A ceiling below the
# schemas' own limits turns a legal id into a crash.
_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


def safe_component(value: str, *, kind: str = "identifier") -> str:
    """Return `value` if it is safe as a single path component, else raise.

    `fullmatch`, not `match`: in Python `$` also matches before a trailing
    newline, so an anchored `match` let `"..\\n"` and `"run_x\\n"` through, and
    those became real directory names on disk.
    """
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value):
        raise UnsafeIdentifier(
            f"unsafe {kind} {value!r}: a path component must match "
            f"^[A-Za-z0-9._-]{{1,128}}$ (letters, digits, dot, underscore, "
            "hyphen; 1-128 characters)"
        )
    if value in (".", ".."):
        raise UnsafeIdentifier(f"unsafe {kind} {value!r}: path traversal")
    return value


def new_run_id() -> str:
    """The default run identifier, minted in one place."""
    return f"run_{uuid.uuid4().hex[:8]}"


# ---------- what is guarded ----------


def guarded_paths() -> dict[str, Path]:
    """The harness's state and secrets.

    Read at call time rather than captured at import so a test (or a future
    --state-dir) can move them. `WORK` is deliberately absent: the Hunt scratch
    dir lives under it, the prompt tells the agent to compile and run PoCs there,
    and nothing under `work/` carries integrity.
    """
    return {
        "the harness state database": STATE_DB,
        "the harness results tree": RESULTS,
        "the harness .env": ENV_FILE,
    }


def write_only_guards() -> dict[str, Path]:
    """Harness files an agent may read but never rewrite.

    The whole checkout rather than three subdirectories: `audit/*.py` is what the
    next invocation loads and what renders a report the operator will read, so
    rewriting it forges a result as directly as rewriting a prompt does. `WORK` is
    carved out by `_carved_out`, because that is where PoCs are compiled.

    Readable on purpose: auditing this repository means reading it.
    """
    return {"the harness source and repository": REPO_ROOT}


def credentials_entries() -> dict[str, Path]:
    """Every Claude-side path holding a credential or the operator's context.

    Explicit rather than derived from `CREDENTIALS_FILE.parent`: one directory
    plus a regex could not express "this directory, and this config file beside
    it, but not `.claude-backup`". Most specific first, so a denial names the file
    the agent actually referenced rather than the directory containing it.
    """
    home = Path.home()
    return {
        "Claude credentials": home / ".claude" / ".credentials.json",
        "the Claude config backup": home / ".claude.json.backup",
        "the Claude config file": home / ".claude.json",
        "the Claude state directory": home / ".claude",
    }


# Entries in credentials_entries() that are directories: those are matched with a
# path boundary, everything else by filename.
_DIR_ENTRY_NAMES = frozenset({".claude"})


def _writable_exceptions() -> tuple[Path, ...]:
    """Where a write is allowed even though the tree around it is guarded."""
    return (WORK,)


# ---------- containment ----------


def canonical_path(path: Path) -> Path:
    """Resolve a path and normalise the macOS firmlink prefix.

    `/System/Volumes/Data/Users/x` and `/Users/x` are the same directory, and
    `resolve()` collapses symlinks but leaves a firmlinked spelling alone, so the
    two compared as different paths.
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(path)
    text = str(resolved)
    if text.startswith("/System/Volumes/Data/"):
        return Path(text[len("/System/Volumes/Data"):])
    return resolved


def _same_file(left: Path, right: Path) -> bool:
    """True when both paths are the same inode.

    `resolve()` is not enough on a case-insensitive filesystem: APFS folds case,
    so `<repo>/STATE.DB` and `<repo>/state.db` are different `Path` objects for
    the same file. A hardlink has no path relationship at all and is caught here
    too.
    """
    try:
        a, b = left.stat(), right.stat()
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _inside(child: Path, root: Path) -> bool:
    """Containment that survives case folding.

    Identity for the same file, then a case-folded prefix comparison for
    everything else, including paths that do not exist yet (a Write to a new file
    inside a guarded directory). Casefolding over-blocks at worst, which is the
    direction to fail in.
    """
    if _same_file(child, root):
        return True
    child_text = str(child).casefold().rstrip("/")
    root_text = str(root).casefold().rstrip("/")
    return child_text == root_text or child_text.startswith(root_text + "/")


def _carved_out(path: Path) -> bool:
    return any(_inside(path, hole) for hole in _writable_exceptions())


def _is_harness_db(path: Path) -> bool:
    """The state database or one of its WAL sidecars, in the checkout.

    Containment cannot see a sidecar: `state.db-wal` is not inside `state.db` as
    a path, and the files only exist while a run has the database open, so a
    prefix comparison on a resolved path misses exactly when it matters. The
    parent check keeps a target's own `state.db-wal` readable.
    """
    name = path.name.casefold()
    db = STATE_DB.name.casefold()
    if name != db and not name.startswith(db + "-"):
        return False
    parent = path.parent
    if parent != path:  # not the filesystem root
        return _inside(parent, REPO_ROOT)
    return True


def _resolve(raw: str, base: Path) -> Path:
    """Absolute, canonical path for a value a tool is about to use."""
    text = str(raw)
    try:
        path = Path(text).expanduser()
    except (RuntimeError, ValueError):
        return canonical_path(base / text)
    if not path.is_absolute():
        path = base / path
    return canonical_path(path)


# ---------- which roots apply to which tool ----------

WRITE_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})

# Path-bearing fields, per tool. Glob's `pattern` is a path expression and Grep's
# `glob` is a path filter (`rg --glob`), while Grep's own `pattern` is a regex
# over file CONTENTS and is deliberately not inspected.
_TOOL_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Read": {"targets": ("file_path",)},
    "Write": {"targets": ("file_path",)},
    "Edit": {"targets": ("file_path",)},
    "NotebookEdit": {"targets": ("notebook_path",)},
    "Glob": {"targets": ("path",), "patterns": ("pattern",)},
    "Grep": {"targets": ("path",), "patterns": ("glob",)},
}

_BASH_TOOL = "Bash"
GUARDED_TOOLS = frozenset(_TOOL_INPUTS) | {_BASH_TOOL}


def _roots_for(tool_name: str) -> dict[str, Path]:
    """Whichever roots apply to a tool.

    Read tools take the integrity-and-secrets set. Write tools and Bash take that
    plus the whole checkout, because a shell command can write to anything it can
    read and no shipped stage has Write or Edit at all.
    """
    roots = {**guarded_paths(), **credentials_entries()}
    if tool_name in WRITE_TOOLS or tool_name == _BASH_TOOL:
        roots = {**roots, **write_only_guards()}
    return roots


def _resolved(map_: dict[str, Path]) -> list[tuple[str, Path]]:
    """Resolve a guarded map once, as (label, path) pairs."""
    return [(label, _resolve(str(entry), REPO_ROOT)) for label, entry in map_.items()]


# ---------- the structured-tool check ----------


def _literal_prefix(value: str) -> str:
    """The part of a pattern before its first wildcard.

    `results/**/*.json` is judged on `results`: the prefix is a real path and the
    wildcards after it cannot move it. No brace expansion, no glob matching, no
    pattern language to get wrong.
    """
    literal = value
    for meta in ("*", "?", "[", "{"):
        literal = literal.split(meta, 1)[0]
    return literal.rstrip("/")


def structured_path_hit(
    tool_name: str, tool_input: dict, *, cwd: Path, workspace: list[Path]
) -> str | None:
    """Does this call reach a guarded file?

    Targets are resolved and compared by containment. A Grep gets one additional
    rule, because reading is not sandboxed: a search rooted above a guarded file
    reads it, and ripgrep passes `--hidden` with only VCS directories excluded.
    That is the rule which refuses `Grep {path: <home>}`, and its cost is that a
    search from a root containing a guarded file has to be narrowed by the caller.
    """
    credentials = _resolved(credentials_entries())
    roots = _resolved(_roots_for(tool_name))
    fields = _TOOL_INPUTS.get(tool_name, {})
    pattern_keys = fields.get("patterns", ())
    call_roots = [
        _resolve(str(tool_input[k]), workspace[0])
        for k in ("path",)
        if tool_input.get(k)
    ] or [canonical_path(p) for p in workspace]

    def _check(path: Path) -> str | None:
        if _carved_out(path):
            return None
        if _is_harness_db(path):
            return "the harness state database (WAL sidecars included)"
        # Most specific first, so the denial names what was referenced.
        for label, entry in credentials:
            if _inside(path, entry):
                return f"{label} ({entry})"
        for label, entry in roots:
            if _inside(path, entry):
                return f"{label} ({entry})"
        return None

    for key in fields.get("targets", ()):
        raw = tool_input.get(key)
        if raw:
            hit = _check(_resolve(str(raw), cwd))
            if hit:
                return hit

    for key in pattern_keys:
        raw = tool_input.get(key)
        if not raw:
            continue
        literal = _literal_prefix(str(raw))
        if literal:
            for root in call_roots:
                hit = _check(_resolve(literal, root))
                if hit:
                    return hit
    if tool_name == "Grep":
        # This search reads every file under its root, and a root enclosing a
        # guarded file therefore leaks it. Runs whether or not a glob was given:
        # the case this exists for is the Grep with no glob at all.
        for root in call_roots:
            for label, entry in (*credentials, *roots):
                if _inside(entry, root):
                    return (
                        f"{label} ({entry}, reachable from {root}: this search "
                        "has no glob excluding it)"
                    )
    return None


# ---------- the shell filter ----------


def _expand_shell_home(text: str) -> str:
    """Expand `~`, `~user` and `$HOME`, quoted or not.

    Without this the name rules only matched a fully expanded absolute path,
    which structured tools produce and shell strings do not.
    """
    home = str(Path.home())
    text = re.sub(r"""["']?\$\{?HOME\}?["']?""", home, text)
    return re.sub(r"""["']?~[A-Za-z_][A-Za-z0-9_-]*["']?|["']?~["']?""", home, text)


def bash_guard_hit(command: str, *, cwd: Path) -> str | None:
    """The guarded name a shell command mentions, or None.

    A speed bump with a written scope: it expands the home spellings a shell
    normally uses and refuses the guarded names. It does not model globbing
    (`cat .cla*/.creden*`), quoting (`cat .cla""ude/x`), `--flag=<path>`, `cd`,
    `cp -R .` or anything built at runtime. Three review rounds of adding those
    turned this function into a hundred lines with eight known bypasses, a hook
    that could block every concurrent agent for 41 seconds, and two behaviours it
    had previously got right. The sandbox is what protects a run; the README says
    which geometries have neither layer.
    """
    if not command:
        return None
    folded = _expand_shell_home(command).casefold()

    if STATE_DB.name in folded:
        # Deliberately fail-closed on the basename, so a spelling nothing here can
        # see (a sqlite URI, the WAL sidecars, a name assembled from a variable)
        # is refused too. The cost is that a target shipping its own `state.db`
        # cannot be reached by a shell command; Read compares by identity and does
        # allow it.
        return "the harness state database (matched 'state.db')"
    for label, entry in _resolved(credentials_entries()):
        needle = str(entry).casefold()
        if entry.name in _DIR_ENTRY_NAMES:
            # The state directory needs a path boundary, so `.claude-backup` and
            # `.claudette` are not mistaken for it while `~/.claude/x` still is.
            if f"{needle}/" in folded or folded.endswith(needle):
                return f"{label} ({entry})"
        elif entry.name.casefold() in folded:
            return f"{label} ({entry})"
    if re.search(r"(?<![\w.-])\.env(?![\w.-])", folded) or str(ENV_FILE).casefold() in folded:
        return "the harness .env (matched '.env')"
    for label, entry in _resolved(write_only_guards()):
        if str(entry).casefold() in folded:
            return f"{label} ({entry})"
    return None
