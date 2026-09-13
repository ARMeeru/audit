"""Shared helpers for stage modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from audit.config import HarnessConfig, StageConfig
from audit.paths import PROMPTS, RESULTS, SCHEMAS, WORK, safe_component

# Re-exported so a future module imports the paths policy instead of rebuilding
# it. Rebound from audit.paths, not duplicated: tests patch these names on this
# module to keep stage output out of the real tree, and a second definition
# would silently ignore that.

@dataclass
class StageContext:
    run_id: str
    repo_path: Path
    config: HarnessConfig
    # Optional operator context — when set, downstream prompts use them.
    live_target: dict | None = None    # {"url": "...", "credentials": {...}}
    scope_notes: str | None = None     # verbatim text appended to user_input

    def stage(self, name: str) -> StageConfig:
        return self.config.get(name)

    def extras(self) -> dict:
        """Optional fields merged into every agent's user_input."""
        out: dict = {}
        if self.live_target:
            out["live_target"] = self.live_target
        if self.scope_notes:
            out["scope_notes"] = self.scope_notes
        return out

    def network_allow(self) -> list[str]:
        """Egress the sandbox should permit: the operator's live target, if any.

        Measured: with the sandbox on and no network config, every outbound
        connection is refused, loopback included. A static run therefore reaches
        nothing, which is the property worth keeping; a live-target run has to
        name its target or the reproduce step cannot execute. Only the host is
        allowed, not the whole URL, and a target that redirects to a CDN needs
        the stage's sandbox switched off.
        """
        url = str((self.live_target or {}).get("url", ""))
        try:
            host = urlparse(url).hostname
        except ValueError:
            # Malformed authority (e.g. `http://[::1:8888`). cli.run validates
            # the flag, so this is belt-and-braces: this method is evaluated
            # inside every stage's argument list, and raising here used to abort
            # a whole stage after its upstream spend was already sunk.
            return []
        return [host] if host else []

    def prompt(self, name: str) -> Path:
        path = PROMPTS / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt: {path}")
        return path

    def schema(self, name: str) -> Path:
        path = SCHEMAS / f"{name}.schema.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing schema: {path}")
        return path

    def results_dir(self, stage: str) -> Path:
        d = RESULTS / safe_component(self.run_id, kind="run_id") / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def work_dir(self, stage: str, ref: str | None = None) -> Path:
        # `ref` is a task id: `WORK / run_id / "hunt" / ref` with a `..` in it
        # created a directory anywhere the operator can write. The agent path is
        # already blocked by hunt_task.schema.json's task_id pattern; this is
        # the storage-side backstop, for an id that arrives by another route.
        # None means "no reference, use the shared default"; an empty string is
        # a value the caller got wrong, and silently aliasing it onto "default"
        # would let every such task share one scratch directory.
        ref_component = "default" if ref is None else safe_component(
            ref, kind="work-dir reference"
        )
        d = WORK / safe_component(self.run_id, kind="run_id") / stage / ref_component
        d.mkdir(parents=True, exist_ok=True)
        return d



def truncated_recon_summary(full: dict, subsystem_filter: str | None = None) -> dict:
    """Pass only the architecture facts downstream agents need."""
    out: dict = {
        "architecture": full.get("architecture", {}),
        "subsystems": full.get("subsystems", []),
    }
    if subsystem_filter is not None:
        match = next(
            (s for s in out["subsystems"] if s.get("name") == subsystem_filter
             or subsystem_filter.startswith(s.get("path", "##nope##"))),
            None,
        )
        out["subsystem_for_task"] = match
    return out
