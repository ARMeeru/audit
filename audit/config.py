"""Load per-stage configuration from config/stages.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The tools a stage may be granted. An unknown name used to be forwarded
# verbatim into ClaudeAgentOptions.allowed_tools, so a typo'd or invented entry
# silently became an approved tool.
KNOWN_TOOLS = frozenset(
    {"Read", "Write", "Edit", "NotebookEdit", "Bash", "Grep", "Glob",
     "WebFetch", "WebSearch", "Task", "TodoWrite"}
)

# The SDK's permission modes. `bypassPermissions` is deliberately excluded:
# stages.yaml has always carried "# never bypassPermissions" as a comment, and a
# comment is not an invariant.
KNOWN_PERMISSION_MODES = frozenset({"default", "acceptEdits", "plan", "dontAsk", "auto"})
FORBIDDEN_PERMISSION_MODES = {"bypassPermissions"}


@dataclass
class StageConfig:
    name: str
    model: str
    concurrency: int
    tools: list[str]
    max_turns: int
    permission_mode: str
    repair_attempts: int
    # Per-task cost estimate used to reserve in-flight spend against the cap.
    # Must be an UPPER BOUND on observed per-task cost: reserving less than
    # actual makes the cap permissive rather than conservative.
    est_cost_usd: float = 1.5
    # Whether to run this stage's agents under the SDK's OS sandbox. On by
    # default because the sandbox is the only real boundary between an agent's
    # Bash and the harness's own state.db; turn it off only on a platform where
    # it cannot start, and read the residual-risk note in FORK-NOTES.md first.
    sandbox: bool = True


@dataclass
class HarnessConfig:
    stages: dict[str, StageConfig] = field(default_factory=dict)
    gapfill_iterations: int = 2
    feedback_iterations: int = 1

    def get(self, stage: str) -> StageConfig:
        try:
            return self.stages[stage]
        except KeyError:
            raise KeyError(
                f"Unknown stage {stage!r}. Known: {sorted(self.stages)}"
            ) from None

    def cap_concurrency(self, cap: int) -> None:
        """Mutate every stage's concurrency to min(current, cap). Useful
        for cost-contained test runs."""
        if cap < 1:
            raise ValueError("concurrency cap must be >= 1")
        for sc in self.stages.values():
            sc.concurrency = min(sc.concurrency, cap)


def _validate_stage(name: str, spec: dict, defaults: dict) -> None:
    """Fail loudly on a stage that would run with a tool set or permission mode
    nobody intended. Unknown keys are still ignored (forward compatibility), but
    the keys we do consume are checked."""
    tools = spec.get("tools", [])
    for tool in tools:
        if tool not in KNOWN_TOOLS:
            raise ValueError(
                f"stage {name!r}: unknown tool {tool!r}. Known tools: "
                f"{sorted(KNOWN_TOOLS)}"
            )
    mode = spec.get("permission_mode", defaults.get("permission_mode", "acceptEdits"))
    if mode in FORBIDDEN_PERMISSION_MODES:
        raise ValueError(
            f"stage {name!r}: permission_mode {mode!r} is never allowed — every "
            "permission check is what keeps an agent inside its scratch dir."
        )
    if mode not in KNOWN_PERMISSION_MODES:
        raise ValueError(
            f"stage {name!r}: unknown permission_mode {mode!r}. Known: "
            f"{sorted(KNOWN_PERMISSION_MODES)}"
        )
    # The confinement switch is the one key whose silent misparse removes the
    # boundary rather than tightening it. YAML is helpful enough that
    # `sandbox:` (null), `0`, `[]` and `{}` all coerce to False through bool(),
    # and a quoted "false" coerces to True: both directions are wrong and
    # neither said anything. Require a real boolean.
    for where, block in (("defaults", defaults), (f"stage {name!r}", spec)):
        if "sandbox" in block and not isinstance(block["sandbox"], bool):
            raise ValueError(
                f"{where}: sandbox must be an unquoted true or false, got "
                f"{block['sandbox']!r}. A quoted \"false\" does not disable it."
            )


def load_config(path: Path | None = None) -> HarnessConfig:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "stages.yaml"
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {}) or {}
    stages: dict[str, StageConfig] = {}
    for name, spec in (raw.get("stages") or {}).items():
        _validate_stage(name, spec, defaults)
        stages[name] = StageConfig(
            name=name,
            model=spec["model"],
            concurrency=int(spec["concurrency"]),
            tools=list(spec["tools"]),
            max_turns=int(spec.get("max_turns", defaults.get("max_turns", 25))),
            permission_mode=spec.get(
                "permission_mode", defaults.get("permission_mode", "acceptEdits")
            ),
            repair_attempts=int(
                spec.get("repair_attempts", defaults.get("repair_attempts", 1))
            ),
            est_cost_usd=float(
                spec.get("est_cost_usd", defaults.get("est_cost_usd", 1.5))
            ),
            sandbox=bool(spec.get("sandbox", defaults.get("sandbox", True))),
        )
    loops = raw.get("loops", {}) or {}
    return HarnessConfig(
        stages=stages,
        gapfill_iterations=int(loops.get("gapfill_iterations", 2)),
        feedback_iterations=int(loops.get("feedback_iterations", 1)),
    )
