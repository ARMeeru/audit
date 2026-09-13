"""Path policy: one validator for every string that becomes a filesystem
component, and the guard that keeps agents out of the harness's own files.

Three things live here.

`safe_component` is the single place that decides what may become a path
component. Two trust levels feed it: operator input (`--run-id`) and model output
derived from the target repo (`task_id`, artifact names). Before it existed,
`RESULTS / run_id` and `WORK / run_id / "hunt" / task_id` were joined raw and
followed by `mkdir(parents=True, exist_ok=True)`, so one `..` in either produced a
directory creation and a write outside the harness.

`structured_path_hit` decides, for a tool that carries an explicit path, whether
that path lands in the harness's own files. It resolves the path and compares by
filesystem identity, so case-folded spellings, hardlinks, `..`, relative
spellings, `~` and symlinks are all handled by the filesystem rather than by a
string table.

`bash_guard_hit` does the same for a shell command. It is a FILTER and nothing
more: it expands the shell's home spellings, applies the name rules, and
containment-checks the path-shaped tokens it can see, resolved against the
directory the command runs in. A path built at runtime
(`p=$(printf %s <b64>|base64 -d); sqlite3 "$p"`) walks past it. The OS sandbox in
runner._build_options is the boundary; this exists for the two geometries where no
sandbox can separate harness from target (a self-audit, and a platform where the
sandbox cannot start), and its denials are written to redirect the agent.
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

    Deliberately narrow. The stage handlers used to catch bare `ValueError`,
    which also swallowed `json.JSONDecodeError` and `UnicodeDecodeError` from
    reading a prompt or schema file and reported them as an unusable identifier,
    sending an operator after an id that was fine.
    """


# 128 rather than 64: finding.schema.json pins finding_id to
# `^f_[a-z0-9_-]{1,64}$`, so a schema-legal id runs to 66 characters, and
# _resolve_finding_id appends a `_N` suffix on a collision. A ceiling below the
# schemas' own limits turns a legal id into a crash, which is what this used to
# do. The character class and the traversal rejection carry the security
# property; the length is a sanity bound.
_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


def safe_component(value: str, *, kind: str = "identifier") -> str:
    """Return `value` if it is safe as a single path component, else raise.

    Accepts the shapes the harness generates (`run_ab12cd34`, `f_1`, `task-01`)
    and nothing else: no separators, no traversal, no control characters, no
    non-ASCII, at most 128 characters. `kind` names the offending field in the
    message so an operator knows what to change and a repair loop knows what to
    re-emit.

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
    """The harness's state and secrets: off limits to every tool.

    Read at call time rather than captured at import so a test (or a future
    --state-dir) can move them. `WORK` is deliberately absent here: the Hunt
    scratch dir lives under it and the prompt tells the agent to compile and run
    PoCs there, so a write ban would break the product, and nothing under
    `work/` carries integrity.
    """
    return {
        "the harness state database": STATE_DB,
        "the harness results tree": RESULTS,
        "the harness .env": ENV_FILE,
    }


def write_only_guards() -> dict[str, Path]:
    """Harness files an agent may read but never rewrite.

    The whole checkout, not three subdirectories. Prompts become the next
    stages' system prompts, schemas decide what counts as valid, and the stage
    config carries the confinement switch; but `audit/*.py` is what the next
    invocation loads and what renders a report the operator will read, so
    rewriting it forges a result just as directly. `WORK` is carved out by
    `_carved_out`: it is the one writable hole, because that is where PoCs are
    compiled.

    Readable on purpose: auditing this repository means reading it.
    """
    return {"the harness source and repository": REPO_ROOT}


def credentials_entries() -> dict[str, Path]:
    """Every Claude-side path holding a credential or the operator's context.

    Explicit rather than derived from `CREDENTIALS_FILE.parent`: one directory
    plus a regex could not express "this directory, and this config file beside
    it, but not `.claude-backup`", so widening the boundary to admit the backups
    let `~/.claude.json` through. The directory itself is here because it holds
    settings and per-project prompt history.
    """
    home = Path.home()
    return {
        "the Claude state directory": home / ".claude",
        "Claude credentials": home / ".claude" / ".credentials.json",
        "the Claude config file": home / ".claude.json",
        "the Claude config backup": home / ".claude.json.backup",
    }


def _sensitive_files() -> dict[str, Path]:
    """Guarded entries that are FILES, so a filename pattern can name them."""
    return {
        "the harness state database": STATE_DB,
        "the harness .env": ENV_FILE,
        **credentials_entries(),
    }


def _writable_exceptions() -> tuple[Path, ...]:
    """Where a write is allowed even though the tree around it is guarded."""
    return (WORK,)


# ---------- name rules ----------


def sensitive_name_hit(name: str) -> str | None:
    """Match a bare filename against the guarded names, by stem where it matters.

    `state.db-wal` and `state.db-shm` carry rows the main file has not
    checkpointed (WAL mode is on) and containment cannot see them:
    `Path("state.db-wal").is_relative_to(Path("state.db"))` is False.
    """
    lowered = name.casefold()
    if lowered == STATE_DB.name or lowered.startswith(STATE_DB.name + "-"):
        return "the harness state database (WAL sidecars included)"
    for label, entry in credentials_entries().items():
        if lowered == entry.name.casefold():
            return label
    # `.env` is deliberately NOT a name rule. The harness's copy is guarded by
    # containment (it sits inside the checkout), while a target's own `.env` is
    # target data an agent may legitimately read.
    return None


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
    the same file, which let a shift key walk past the guard. A hardlink has no
    path relationship at all and is caught here too.
    """
    try:
        a, b = left.stat(), right.stat()
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _inside(child: Path, root: Path) -> bool:
    """Containment that survives case folding.

    Identity for the same file, then a casefolded prefix comparison for
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


def _resolved(map_: dict[str, Path]) -> dict[str, Path]:
    """Resolve a guarded-map once, rather than per comparison inside a loop."""
    return {label: _resolve(str(entry), REPO_ROOT) for label, entry in map_.items()}


def _hits_guarded(
    path: Path,
    roots: dict[str, Path],
    *,
    credentials: dict[str, Path] | None = None,
    resolved_roots: dict[str, Path] | None = None,
) -> str | None:
    """What `path` reaches, or None.

    `_inside` is enough for both kinds of credentials entry: for a file it
    reduces to "the same file", for a directory it covers everything under it,
    and it works for a directory that does not exist yet.
    """
    for label, entry in (credentials or _resolved(credentials_entries())).items():
        if _inside(path, entry):
            return f"{label} ({entry})"
    name_hit = sensitive_name_hit(path.name)
    if name_hit is not None:
        return f"{name_hit} (matched {path.name!r})"
    for label, entry in (resolved_roots or _resolved(roots)).items():
        if _inside(path, entry) and not _carved_out(path):
            return f"{label} ({entry})"
    return None


# ---------- which roots apply to which tool ----------

WRITE_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})

_TOOL_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Read": {"targets": ("file_path",)},
    "Write": {"targets": ("file_path",)},
    "Edit": {"targets": ("file_path",)},
    "NotebookEdit": {"targets": ("notebook_path",)},
    # Glob's `pattern` is a path expression; Grep's `pattern` is a regex over
    # file CONTENTS and its `glob` is a path filter (`rg --glob`). Inspecting the
    # right field per tool is the difference between a guard and decoration.
    "Glob": {"targets": ("path",), "patterns": ("pattern",)},
    "Grep": {"targets": ("path",), "patterns": ("glob",)},
}

_BASH_TOOL = "Bash"
GUARDED_TOOLS = frozenset(_TOOL_INPUTS) | {_BASH_TOOL}


def _roots_for(tool_name: str) -> dict[str, Path]:
    """Whichever roots apply to a tool.

    Read tools take the integrity-and-secrets set. Write tools and Bash take that
    plus the whole checkout, because a shell command can write to anything it can
    read, and no shipped stage has Write or Edit at all, which is what made the
    old write-only branch unreachable.
    """
    roots = {**guarded_paths(), **credentials_entries()}
    if tool_name in WRITE_TOOLS or tool_name == _BASH_TOOL:
        roots = {**roots, **write_only_guards()}
    return roots


# ---------- pattern handling ----------

# A pattern that names everything. Meaningful for Grep's content rule,
# meaningless for deciding whether a file was named.
_MATCH_EVERYTHING = ("*", "**", "*.*", "**/*", "**/.*", ".*")


def expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b}` alternation the way ripgrep and the CLI's glob do.

    Python's fnmatch has no brace syntax: `fnmatch(".credentials.json",
    "{.credentials.json,zzz}")` is False, so a brace-wrapped pattern naming a
    guarded file was allowed. The CLI preserves the brace form rather than
    splitting it and ripgrep expands it, so it has to be expanded before
    matching.
    """
    match = re.search(r"\{([^{}]*)\}", pattern)
    if match is None:
        return [pattern]
    expanded: list[str] = []
    for branch in match.group(1).split(","):
        expanded.extend(
            expand_braces(pattern[: match.start()] + branch + pattern[match.end():])
        )
    return expanded


def _pattern_names_guarded(pattern: str, *, contents: bool) -> str | None:
    """Does this file-selection pattern name a guarded file?

    `contents` is True for Grep, where reading a guarded file leaks it, and
    False for Glob, where a name comes back and a bare wildcard is ordinary
    enumeration: only a pattern that names something more specific counts.
    """
    if not pattern:
        return None
    for branch in expand_braces(pattern):
        name = branch.rstrip("/").split("/")[-1]
        literal = name
        for meta in ("*", "?", "[", "{"):
            literal = literal.split(meta, 1)[0]
        for label, entry in _sensitive_files().items():
            target = entry.name.casefold()
            if name.casefold() == target:
                return f"{label} (matched {pattern!r})"
            # `state.db-*` names the WAL sidecars through their stem.
            if literal and target.startswith(literal.casefold()) and len(literal) > 3:
                return f"{label} (matched {pattern!r})"
            if contents and name.casefold() in _MATCH_EVERYTHING:
                return f"{label} (a pattern that matches every file)"
    if "{" in pattern and pattern.count("{") != pattern.count("}"):
        # A form this module cannot expand: fail closed rather than guess.
        return "a search pattern in a form this guard cannot parse"
    return None


def _search_roots(tool_input: dict, workspace: list[Path]) -> list[Path]:
    """Where a slash-bearing pattern is resolved from.

    The call's own `path` when it has one: ripgrep and Glob match a relative
    pattern against the call's root, not the session's working directory, so
    resolving against cwd let `glob: "results/*/report/report.json"` reach the
    guarded results tree from hunt's scratch dir. The session workspace is the
    fallback, and every root is tried, so a self-audit's relative pattern still
    lands on the checkout.
    """
    raw = [tool_input[k] for k in ("path",) if tool_input.get(k)]
    if raw:
        return [_resolve(str(r), workspace[0]) for r in raw]
    return [canonical_path(p) for p in workspace]


def _is_absolute(text: str) -> bool:
    return str(text).startswith("/") or str(text).startswith("~")


# ---------- the structured-tool check ----------


def structured_path_hit(
    tool_name: str, tool_input: dict, *, cwd: Path, workspace: list[Path]
) -> str | None:
    """Containment check for the paths a structured tool carries."""
    roots = _roots_for(tool_name)
    fields = _TOOL_INPUTS.get(tool_name, {})
    pattern_keys = fields.get("patterns", ())
    contents = tool_name == "Grep"

    credentials = _resolved(credentials_entries())
    resolved_roots = _resolved(roots)

    def _check(candidates: list[Path]) -> str | None:
        for candidate in candidates:
            hit = _hits_guarded(
                canonical_path(candidate),
                roots,
                credentials=credentials,
                resolved_roots=resolved_roots,
            )
            if hit:
                return hit
        return None

    # A direct path resolves against the session's working directory, which is
    # what the tool itself does with a relative spelling.
    for key in fields.get("targets", ()):
        raw = tool_input.get(key)
        if not raw:
            continue
        hit = _check(_literal_candidates(str(raw), base=cwd))
        if hit:
            return hit

    # A pattern is judged on its literal prefix (`results/**/*.json` on
    # `results`), resolved from the root the tool would use: ripgrep and Glob
    # match a relative pattern against the call's own root, not the session cwd,
    # so resolving from cwd let `glob: "results/*/report/report.json"` reach the
    # guarded results tree from hunt's scratch dir.
    search_roots = _search_roots(tool_input, workspace)
    for key in pattern_keys:
        raw = tool_input.get(key)
        if not raw:
            continue
        named = _pattern_names_guarded(str(raw), contents=contents)
        if named is not None:
            return named
        hit = _check(_literal_candidates(str(raw), base=search_roots))
        if hit:
            return hit

    # A Grep with no glob reads every file under its root, guarded ones included:
    # ripgrep passes --hidden with only VCS directories excluded. A root that
    # encloses a guarded file therefore leaks it, and the caller's fix is a
    # narrowing glob.
    if contents and (not pattern_keys or not tool_input.get(pattern_keys[0])):
        for base in search_roots:
            for label, entry in _sensitive_files().items():
                resolved = _resolve(str(entry), REPO_ROOT)
                if _inside(resolved, base) and not _carved_out(resolved):
                    return (
                        f"{label} ({resolved}, reachable from {base}: this search "
                        "has no glob excluding it)"
                    )
    return None


def _literal_candidates(value: str, *, base) -> list[Path]:
    """Where a path, or a pattern's literal prefix, could land.

    An absolute value stands alone. A relative one comes off `base`, which is
    either a single directory or the roots a search tool would use.
    """
    literal = value
    for meta in ("**", "*", "?", "[", "{"):
        literal = literal.split(meta, 1)[0]
    literal = literal.rstrip("/")
    roots = list(base) if isinstance(base, list) else [base]
    if not literal:
        return roots
    if _is_absolute(literal):
        return [Path(literal)]
    return [root / literal for root in roots]


# ---------- the shell filter ----------


def _expand_shell_home(text: str) -> str:
    """Expand the shell's home spellings so the name rules can see them.

    `~` and `$HOME` are how a shell normally spells home, and they were invisible
    to a matcher that only knew absolute paths. Quoted forms and `~user` are
    included: `cat "$HOME"/.claude/x` is the shellcheck-recommended spelling, and
    substituting inside the quotes left a stray `"` in the path.
    """
    home = str(Path.home())
    text = re.sub(r"""["']?\$\{?HOME\}?["']?""", home, text)
    text = re.sub(r"""["']?~(?:[A-Za-z_][A-Za-z0-9_-]*)?["']?""", home, text)
    return text


_TOKEN_SPLIT_RE = re.compile(r"""[;&|()<>\s"']+""")
_PATH_SHAPED_RE = re.compile(
    r"^(?:.*/)?[\w.@+-]+(?:/[\w.@*?\[\]{}+-]+)+/?$|^[\w@+-]*\.[A-Za-z0-9]+$"
)


def _cd_base(command: str, cwd: Path) -> Path:
    """The directory a later relative token resolves against.

    A `cd X && ...` moves the shell before the rest of the command runs, so
    resolving every token against the session cwd both missed
    `cd <checkout> && cat .env` and refused `cd <target> && cat config/app.yml`.
    Only `cd` occurrences are honoured; anything cleverer is obfuscation and out
    of scope by design.
    """
    base = cwd
    for match in re.finditer(r"(?:^|[;&|]\s*)cd\s+([^\s;&|]+)", command):
        base = _resolve(match.group(1), base)
    return base


def bash_guard_hit(command: str, *, cwd: Path) -> str | None:
    """What a shell command reaches for, or None if it looks harmless.

    A FILTER, not a boundary: see the module docstring. It fails closed on the
    guarded names and on the paths it can resolve. Anything it cannot see (a
    substitution, a variable, a path assembled at runtime) is why the sandbox
    exists.
    """
    if not command:
        return None
    text = _expand_shell_home(command)
    lowered = text.casefold()
    roots = _roots_for(_BASH_TOOL)

    # The harness database is matched by basename, deliberately fail-closed. It
    # is the one name worth the over-block: it catches spellings the token walk
    # cannot see (a sqlite URI, a name built from a variable, the WAL sidecars),
    # and the cost is that a target shipping its own `state.db` cannot be read by
    # a shell command. The structured tools reach that file by containment, so
    # the agent is pointed at Read rather than left stuck.
    if STATE_DB.name in lowered:
        return "the harness state database (matched 'state.db')"

    base = _cd_base(command, cwd)
    credentials = _resolved(credentials_entries())
    resolved_roots = _resolved(roots)
    for token in _TOKEN_SPLIT_RE.split(text):
        if not token or token.startswith("-"):
            continue
        if "/" not in token and not _PATH_SHAPED_RE.match(token):
            continue
        if not _is_absolute(token):
            token = str(base / token)
        hit = _hits_guarded(
            canonical_path(token),
            roots,
            credentials=credentials,
            resolved_roots=resolved_roots,
        )
        if hit:
            return hit
    return None
