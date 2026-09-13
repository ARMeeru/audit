"""Path policy: one validator for every string that becomes a filesystem
component, and the guard that keeps agents out of the harness's own files.

Three things live here.

`safe_component` is the single place that decides what may become a path
component. Two trust levels feed it: operator input (`--run-id`) and model output
derived from the target repo (`task_id`, artifact names). Before it existed,
`RESULTS / run_id` and `WORK / run_id / "hunt" / task_id` were joined raw and
followed by `mkdir(parents=True, exist_ok=True)`, so one `..` in either produced
a directory creation and a write outside the harness.

`structured_path_hit` decides, for a tool that carries an explicit path
(`file_path`, `path`, `glob`, ...), whether that path lands in the harness's own
files. It resolves the path and compares containment, so `..`, relative
spellings and `~` are all handled by the filesystem rather than by a string
table.

`bash_guard_hit` is the same question for a shell command, where there is no
structure to resolve. It is a FILTER over a string and nothing more: a path built
at runtime (`p=$(printf %s <b64>|base64 -d); sqlite3 "$p"`), a variable, or any
other indirection walks past it. The OS sandbox in runner._build_options is the
boundary. This exists for the two cases where no sandbox can separate harness
from target (a self-audit, and a platform where the sandbox cannot start), and
its denials are written to redirect the agent rather than stall it.
"""

from __future__ import annotations

import re
import uuid
from fnmatch import fnmatch
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
# do. The character class and the traversal rejection are what carry the
# security property; the length is a sanity bound.
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


def guarded_paths() -> dict[str, Path]:
    """The harness's own state and secrets: off limits to every tool.

    Read at call time rather than captured at import so a test (or a future
    --state-dir) can move them. `WORK` is deliberately absent: the Hunt scratch
    dir lives under it and the prompt tells the agent to compile and run PoCs
    there, so a write ban would break the product, and nothing under `work/`
    carries integrity.
    """
    return {
        "the harness state database": STATE_DB,
        "the harness results tree": RESULTS,
        "the harness .env": ENV_FILE,
    }


def write_only_guards() -> dict[str, Path]:
    """Harness files an agent may read but never rewrite.

    The prompts become the next stages' system prompts and the schemas decide
    what counts as valid, and `_run_agent_once` re-reads both per dispatch, so a
    rewrite mid-run redirects the pipeline's own judgement. They are readable on
    purpose: auditing this repository means reading them.
    """
    return {
        "the harness prompts": PROMPTS,
        "the harness schemas": SCHEMAS,
        "the harness stage config": CONFIG,
    }


def _bash_guarded() -> dict[str, Path]:
    """Everything a shell command may not touch.

    Bash gets the union, including the read-allowed files, because a shell
    command can write to anything it can read and its text carries no read/write
    signal. A `cat prompts/08-report.md` in self-audit geometry is refused and
    the agent is pointed at Read instead, which is the tool that can only read.
    """
    return {**guarded_paths(), **write_only_guards()}


# Tools that can write, and the tool_input fields each guarded tool carries.
# runner.py derives its PreToolUse matcher from the keys here, so a tool whose
# calls the guard inspects cannot be left out of the matcher: a matcher that
# omits a tool means the CLI never invokes the callback for it, and the check
# then exists in the code and can never fire.
WRITE_TOOLS = frozenset({"Write", "Edit", "NotebookEdit"})

_TOOL_INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Read": {"targets": ("file_path",)},
    "Write": {"targets": ("file_path",)},
    "Edit": {"targets": ("file_path",)},
    "NotebookEdit": {"targets": ("notebook_path",)},
    # Glob's `pattern` is a path expression; Grep's `pattern` is a regex over
    # file contents and its `glob` is a path filter (`rg --glob`). Inspecting
    # the right field per tool is the difference between a guard and decoration:
    # a Grep pattern of "state.db" is a hunter grepping the target for a string,
    # while its `glob` of ".credentials.json" is a search for a file.
    "Glob": {"targets": ("path",), "patterns": ("pattern",)},
    "Grep": {"targets": ("path",), "patterns": ("glob",)},
}

_BASH_TOOL = "Bash"
GUARDED_TOOLS = frozenset(_TOOL_INPUTS) | {_BASH_TOOL}


def _guarded_roots(writes: bool) -> dict[str, Path]:
    """Roots off limits for a tool that writes, or for one that only reads.

    Read tools get the integrity-and-secrets set. Write tools get that plus the
    files a self-audit must be able to read (prompts, schemas, stage config),
    which are re-read per dispatch and would otherwise let a rewrite redirect
    the pipeline's own judgement. Bash appears to be a read tool and is not: a
    shell command can write to anything it can read, so it takes the union.
    """
    return {**guarded_paths(), **write_only_guards()} if writes else guarded_paths()


# macOS firmlinks: /System/Volumes/Data/Users/x and /Users/x are the same
# directory, and `resolve()` collapses symlinks but leaves a firmlinked spelling
# alone, so the two compared as different paths and the self-audit check missed
# the firmlinked form of its own checkout.
_FIRMLINK_PREFIX = "/System/Volumes/Data"


def canonical_path(path: Path) -> Path:
    """Resolve a path and normalise the macOS firmlink prefix."""
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(path)
    text = str(resolved)
    if text.startswith(_FIRMLINK_PREFIX + "/"):
        return Path(text[len(_FIRMLINK_PREFIX):])
    return resolved


def _resolve(raw: str, cwd: Path) -> Path | None:
    """Best-effort absolute path for a value a tool is about to use."""
    try:
        path = Path(str(raw)).expanduser()
    except (RuntimeError, ValueError):
        # expanduser raises RuntimeError for a malformed ~user on some systems
        return None
    if not path.is_absolute():
        path = Path(cwd) / path
    try:
        return canonical_path(path)
    except (OSError, ValueError):
        return None


def _encloses(root: Path, target: Path) -> bool:
    return root == target or target.is_relative_to(root)


def _single_files() -> dict[str, Path]:
    """Guarded entries that are files rather than trees.

    Only these can be named by a filename glob, so only these matter to the
    pattern rule below.
    """
    return {
        "the harness state database": STATE_DB,
        "the harness .env": ENV_FILE,
        "Claude credentials": CREDENTIALS_FILE,
    }


def structured_path_hit(
    tool_name: str, tool_input: dict, *, cwd: Path, workspace: list[Path]
) -> str | None:
    """Containment check for the paths a structured tool carries.

    Resolving first is what makes `..`, relative spellings, `~` and symlinks
    work without a string table, and it is why `ls ~/.claude-backup` is no
    longer mistaken for the credentials directory.

    Two rules, because tools differ:

    * a target value that lands in a guarded tree is refused outright;
    * a search tool whose filename pattern can match a guarded FILE, from a
      root that encloses it, is refused. That is the `Grep {path: <home>,
      glob: ".credentials.json"}` case, and it is deliberately narrow: it
      matches the pattern against the guarded filenames, so an ordinary
      `Grep {path: <repo>, pattern: "sqlite3"}` in a self-audit still works.
    """
    roots = _guarded_roots(writes=tool_name in WRITE_TOOLS)
    fields = _TOOL_INPUTS.get(tool_name, {})
    credentials = _resolve(str(CREDENTIALS_FILE), cwd) or CREDENTIALS_FILE

    def _check(value_path: Path) -> str | None:
        if value_path == credentials or value_path.is_relative_to(credentials.parent):
            return f"Claude credentials (under {credentials.parent})"
        for label, root in roots.items():
            root_res = _resolve(str(root), cwd) or root
            if value_path == root_res or value_path.is_relative_to(root_res):
                return f"{label} ({root_res})"
        return None

    for key in (*fields.get("targets", ()), *fields.get("patterns", ())):
        raw = tool_input.get(key)
        if not raw:
            continue
        # A pattern contributes its literal prefix: `<results>/**/*.json` is
        # judged on `<results>`, not on the pattern text, and a bare `state.db`
        # is judged relative to the session's cwd.
        text = str(raw)
        for meta in ("**", "*", "?", "[", "{"):
            text = text.split(meta, 1)[0]
        resolved = _resolve(text.rstrip("/") or ".", cwd)
        if resolved is not None:
            hit = _check(resolved)
            if hit:
                return hit

    pattern_keys = fields.get("patterns", ())
    raw_roots = [tool_input[k] for k in fields.get("targets", ()) if tool_input.get(k)]
    search_roots = (
        [Path(str(r)) for r in raw_roots] if raw_roots else [Path(p) for p in workspace]
    )
    for key in pattern_keys:
        pattern = str(tool_input.get(key) or "")
        if not pattern:
            continue
        name = pattern.rstrip("/").split("/")[-1]
        for label, guarded in _single_files().items():
            guarded_res = _resolve(str(guarded), cwd) or guarded
            if not fnmatch(guarded_res.name, name):
                continue
            for root in search_roots:
                root_res = _resolve(str(root), cwd)
                if root_res is not None and _encloses(root_res, guarded_res):
                    return f"{label} (reachable from {root_res}, matched {name!r})"
    return None


_CREDENTIALS_DIR_RE = re.compile(
    re.escape(str(CREDENTIALS_FILE.parent)) + r"(?![\w.-])"
)


def _expand_shell_home(text: str) -> str:
    """Expand `~` and `$HOME` so the natural shell spellings can be matched.

    Without this the credentials branch only ever matched a fully expanded
    absolute path, which structured tools produce and shell strings do not:
    `cat ~/.claude/*` was allowed while `cat /Users/x/.claude/.credentials.json`
    was denied.
    """
    home = str(Path.home())
    text = re.sub(r"\$\{HOME\}|\$HOME\b", home, text)
    text = re.sub(r"(?<![\w~])~(?=[/\s]|$)", home, text)
    return text


def bash_guard_hit(command: str, *, cwd: Path) -> str | None:
    """What a shell command is reaching for, or None if it looks harmless.

    A FILTER, not a boundary: see the module docstring. Fails closed on the
    harness DB by basename, so a target repo shipping its own `state.db` is
    refused too; the denial names Read/Grep as the way on.
    """
    if not command:
        return None
    text = _expand_shell_home(command)
    lowered = text.lower()

    if "state.db" in lowered:
        return "the harness's state database (matched 'state.db')"
    if _CREDENTIALS_DIR_RE.search(text) or CREDENTIALS_FILE.name in lowered:
        return f"Claude credentials (matched {CREDENTIALS_FILE.parent})"

    roots = _bash_guarded()
    for label, root in roots.items():
        if str(root).lower() in lowered:
            return f"{label} (matched {root})"

    # Relative spellings. A shell resolves `results/x` against its own cwd, so
    # the spelling only reaches the harness tree when that cwd is at or inside
    # it, which is the self-audit geometry. Matching requires a real path shape
    # (`results/`, not the bare word) to keep the over-block off commands that
    # merely mention the word. A `cd` earlier in the same command can move the
    # shell afterwards, so this is a heuristic on top of a heuristic and the
    # sandbox remains the thing that actually holds.
    cwd_res = _resolve(str(cwd), REPO_ROOT)
    if cwd_res is not None and (cwd_res == REPO_ROOT or cwd_res.is_relative_to(REPO_ROOT)):
        for label, root in roots.items():
            try:
                rel = root.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            pattern = (
                rf"(?<![\w./-]){re.escape(rel)}/"
                if root.is_dir()
                else rf"(?<![\w./-]){re.escape(rel)}(?![\w.-])"
            )
            if re.search(pattern, text):
                return f"{label} (matched relative '{rel}' from {cwd_res})"
    return None
