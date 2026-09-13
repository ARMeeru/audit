> **This fork** (`ARMeeru/audit`, branch `fix/self-audit-hardening`) diverges from
> upstream `evilsocket/audit` in the following ways:
>
> - **Run-scoped state keys**: findings, traces and dedupe groups are keyed
>   `(run_id, id)` instead of globally, so concurrent runs sharing one `state.db`
>   can no longer silently drop each other's findings. Legacy databases migrate
>   on open (`PRAGMA user_version` gates future migrations).
> - **Failure-cost ledger**: API spend on failed agent attempts (retries, repair
>   turns, quota aborts) is recorded, so `--max-cost-usd` no longer under-counts
>   failing runs. Caps are enforced conservatively: hunt reserves a per-task
>   estimate at dispatch (this run's max completed hunt cost), so overrun is
>   bounded by one task's estimate rather than by concurrency.
>   runs that are going wrong.
> - **`--finalize`**: skip exploration entirely and close a run from current
>   state (validate remaining -> dedupe -> trace -> report). The optional
>   `--finalize-cost-usd` caps a single finalize invocation (per-invocation, not
>   cumulative - a tripped cap leaves the run resumable).
> - **Resume converges**: expansion-loop budgets are derived from artifacts, so
>   resuming no longer re-grants exploration rounds. A deterministically failing
>   task stops being re-queued after 3 attempts.
> - **Trace failures are retryable**: a quota-killed tracer no longer persists a
>   permanent unreachable verdict.
> - **Quota deaths degrade safely**: quota-killed validate/dedupe/report stages
>   fall back to deterministic behavior (retryable work, or a report built
>   straight from state.db) instead of persisting corrupt state.
> - **Exact-host gateway check** and fail-closed behavior for a gateway base URL
>   without an auth token.
> - **Report evidence is fence-safe**: target-influenced evidence renders inside
>   a fence it cannot break; `AUDIT_ALLOW_API_KEY=no/off` parse as opt-out.
> - **Egress correction**: network egress is NOT restricted to the target host
>   by any enforced mechanism. It is a prompt-level instruction only; recon
>   credentials are stored plaintext in result artifacts. Run in a disposable
>   VM or container when the target is sensitive.
>
> The original MIT LICENSE applies; changes are documented here rather than in
> the license file. The bundled Claude Code CLI inside `claude_agent_sdk` remains
> Anthropic proprietary and is never committed or redistributed by this fork.

# audit

An 8-stage vulnerability-discovery agent, driven by your **Claude Pro / Max
subscription** through the official Claude Code Agent SDK. Many narrow agents,
deliberate disagreement, and an explicit reachability gate.

MIT-licensed. No API key needed if you already use `claude login`.

## Origin

This project is a from-scratch reimplementation of the pipeline described in
Cloudflare's [Project Glasswing](https://blog.cloudflare.com/cyber-frontier-models/)
post, which tested Anthropic's Mythos preview LLM against Cloudflare's own
codebase. The blog argues that real-world vulnerability discovery does **not**
come from asking one big model "find bugs here" — it comes from:

1. **Many narrow agents** working in parallel on tightly-scoped questions
   ("Look for command injection in this specific function, with this trust
   boundary above it") rather than one exhaustive agent.
2. **Deliberate disagreement** — a second agent, on a different model, that
   tries to *disprove* the first agent's findings.
3. **A reachability trace** as the gating step — most "is this code buggy?"
   findings are noise unless an attacker-controlled input can actually reach
   the sink from outside the system.
4. **A feedback loop** so reachable bugs in one place automatically seed
   hunts for the same pattern elsewhere.

This repo packages that pipeline into a runnable agent. The Cloudflare post
showed the architecture; this codebase ships the prompts, schemas, state
store, and orchestrator.

## The 8 stages

![Vulnerability discovery harness — 8 stages](https://raw.githubusercontent.com/evilsocket/audit/main/docs/pipeline.png)

<sub>Diagram from Cloudflare's [Project Glasswing](https://blog.cloudflare.com/cyber-frontier-models/) post, reproduced here for reference.</sub>

| # | Stage    | Default model | Purpose |
|---|----------|---------------|---------|
| 1 | Recon    | Opus 4.7  | Map the repo, emit narrowly-scoped Hunt tasks |
| 2 | Hunt     | Sonnet 4.6 | One attack class per agent; compile/run PoCs |
| 3 | Validate | Opus 4.7  | Adversarial re-read; tries to **disprove** (different model from Hunt) |
| 4 | Gapfill  | Sonnet 4.6 | Re-queue under-covered areas |
| 5 | Dedupe   | Sonnet 4.6 | Cluster findings by root cause |
| 6 | Trace    | Opus 4.7  | Prove attacker-controlled input reaches the sink |
| 7 | Feedback | Sonnet 4.6 | Turn reachable traces into new Hunt tasks |
| 8 | Report   | Sonnet 4.6 | Schema-validated structured report |

Each stage is one markdown prompt in `prompts/` + one JSON Schema in
`schemas/`. The orchestrator passes the schema into the system prompt so
every output is shape-stable on the first try.

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Auth (pick one)
#    (a) Already logged in via claude login? You're done.
#    (b) Or generate a 1-year OAuth token for CI / non-interactive use:
claude setup-token
echo "CLAUDE_CODE_OAUTH_TOKEN=<paste>" > .env

# 3. Verify
audit auth-check

# 4. Run
audit run --repo /path/to/target --run-id my-run
audit status --run-id my-run
audit report --run-id my-run --format md > report.md
```

By default the agent uses **subscription billing** via your Claude.ai
login — it does **not** call the metered Anthropic API. The on-disk auth
module scrubs `ANTHROPIC_API_KEY` from the environment so it can't
silently route around the OAuth flow.

## Using a different model / provider

The auth module picks one of three modes, in this order:

1. **LLM gateway** (OpenRouter, custom proxy, etc.) — when
   `ANTHROPIC_BASE_URL` points away from `anthropic.com` AND
   `ANTHROPIC_AUTH_TOKEN` is set. The gateway env is left intact;
   only `ANTHROPIC_API_KEY` is scrubbed (it would otherwise outrank the
   gateway token).
2. **Subscription OAuth (headless)** — `CLAUDE_CODE_OAUTH_TOKEN` from
   `claude setup-token`. Best for CI.
3. **Subscription OAuth (interactive)** — `~/.claude/.credentials.json`
   from `claude login`. Best for local dev.

### OpenRouter

OpenRouter exposes Claude-compatible Anthropic-API endpoints behind its
own credit system; that lets you spend OpenRouter credits instead of an
Anthropic subscription, and gives you access to Sonnet/Opus *and* other
models through the same SDK path. See [OpenRouter's Agent SDK guide](https://openrouter.ai/docs/guides/community/anthropic-agent-sdk).

```bash
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""           # must be explicitly empty / unset
# optional: pick a non-Anthropic model
export ANTHROPIC_MODEL="anthropic/claude-sonnet-4-6"
# or e.g.: ANTHROPIC_MODEL="openai/gpt-5"
#         ANTHROPIC_MODEL="google/gemini-2.5-pro"
#         ANTHROPIC_MODEL="qwen/qwen3-coder-480b"

audit auth-check                       # confirms "using LLM gateway at https://openrouter.ai/api"
audit run --repo /path/to/target --run-id orun --max-cost-usd 30
```

Caveats:
- Per-stage model overrides in `config/stages.yaml` are model **names**
  (e.g. `claude-opus-4-7`); OpenRouter accepts slash-prefixed forms like
  `anthropic/claude-opus-4-7`. Edit the YAML if you want different
  providers per stage. Otherwise `ANTHROPIC_MODEL` forces every stage
  onto one model.
- Non-Claude models may not produce schema-compliant JSON as reliably.
  The runner's schema-validation + repair turn still applies; quality
  varies by model.
- Tool-use semantics (Read/Grep/Glob/Bash) are part of the Claude Code
  CLI, not the model — they work as long as the gateway speaks the
  Anthropic Messages API.

### Other gateways / cloud providers

Same recipe — anything that exposes the Anthropic Messages API at a URL
+ a bearer token works:

```bash
export ANTHROPIC_BASE_URL="https://your-proxy.example.com"
export ANTHROPIC_AUTH_TOKEN="$YOUR_TOKEN"
unset ANTHROPIC_API_KEY
```

For Amazon Bedrock / Google Vertex / Microsoft Foundry, Claude Code has
first-class env-var flags (`CLAUDE_CODE_USE_BEDROCK=1` etc.) that
outrank everything else. See the [Claude Code auth docs](https://code.claude.com/docs/en/authentication).

## Cost containment

A real production codebase can produce 15-50 Hunt tasks and 25+ findings to
validate. At default concurrency this gets expensive. Flags to keep it sane:

```bash
audit run --repo /path/to/target \
  --max-concurrency 1 \           # one claude subprocess at a time
  --max-recon-tasks 15 \          # cap initial Hunt fanout
  --max-cost-usd 30               # abort cleanly if exceeded
```

The budget guard fires between *and* within stages — a per-task check in
Hunt cooperatively aborts rather than running 30 more tasks past the cap.

## Live-target reproduction (optional)

If the target has a running deployment, point the agents at it. Hunt
**reproduces** each finding against the live service instead of compiling
a local PoC, and Trace **confirms** reachability with real HTTP
round-trips. Validate has no Bash in any mode: it judges reproduction from
the code and from what Hunt recorded. The static path remains available,
and these flags are opt-in.

Egress is not a prompt instruction any more. The sandbox refuses every
outbound connection by default, and `--target-url` adds that host to the
allowlist for the run, so a static run reaches nothing at all and a
live-target run reaches only its target. A target that redirects to a
third-party host needs that stage's `sandbox` switched off.

```bash
audit run --repo /path/to/target --run-id live \
  --max-concurrency 1 --max-cost-usd 30 \
  --target-url http://server.local:8888 \
  --target-creds email=admin@system.com \
  --target-creds password=changechangeme
```

Rules the agents follow when `--target-url` is set:
- Network egress is NOT enforced to that host: it is a prompt-level
  instruction only, and the CLI itself contacts isbndb/OpenLibrary-style
  services and your gateway. Run sensitive targets inside a disposable VM
  or container.
- A finding that doesn't reproduce against the live target is dropped or
  rejected (depending on stage) — "no fabrication".
- Credentials flow into every relevant stage's user_input as a dict.

## Scope notes (optional)

Targets often have intentionally-loose-by-design surfaces that aren't bugs
(e.g. plaintext API keys when that's a feature, test-only Mailpit endpoints,
anonymous-analytics ingest). Drop them in a text file and pass it in — the
notes are appended verbatim to every stage's user_input, and Recon / Hunt /
Validate honor exclusions you list.

```bash
audit run --repo /path/to/target --scope-notes target_scope.md
```

Example `target_scope.md`:

```markdown
- Mailpit (port 1025) is test-only; ignore.
- Plaintext API keys in the database are a required feature.
- Don't flag rate-limit absence on anonymous /ping endpoints.
- Only consider critical/high severity.
```

## Recon mines git history

Recon greps the git history for past security patches
(`CVE`, `sec:`, `fix.*auth`, `sanitize`, …) — patched files are hardened,
but **sibling files with the same idiom often aren't**. Findings get seeded
against the unpatched copies. Adds zero cost on repos without that pattern;
catches real cross-component bugs on repos that have it.

## Logic chains

The pipeline's default is one-attack-class-per-task (the Cloudflare paper's
narrow-scope rule). Recon can also emit `logic_chain` tasks for high-impact
multi-component paths (auth-bypass + IDOR + path-traversal that compose into
RCE, etc.) — one chain per task, with the `scope_hint` naming the specific
chain. This is the one allowed exception to single-attack-class scoping.

## Layout

```
prompts/        8 stage prompts (markdown, loaded as system prompts)
schemas/        9 JSON schemas — every agent output is validated
config/         stages.yaml — model + concurrency + tool allowlist per stage
audit/          Python package
  auth.py       OAuth check + ANTHROPIC_API_KEY scrubbing
  state.py      SQLite DAO (runs, tasks, findings, traces, dedupe, costs)
  runner.py     claude-agent-sdk wrapper with schema validation + repair turn
  orchestrator.py pipeline driver
  stages/       one module per stage
work/           per-Hunt-task scratch dirs (sandbox for PoC compile/run)
results/        JSONL artifacts per stage + final report.json
state.db        SQLite (gitignored)
```

## Safety

Hunt agents have Bash and run inside per-task scratch dirs. They also run under
the SDK's OS sandbox (macOS and Linux), enabled by default and configurable per
stage via `sandbox` in `config/stages.yaml`:

- writes outside the agent's working directory are refused by the sandbox, so an
  agent cannot rewrite `state.db` or the `results/` tree even when the code it
  is reading talks it into trying;
- a PreToolUse filter additionally refuses tool calls naming `state.db`, the
  results tree, the harness `.env` or the Claude credentials file, and says why.
  It is a filter over a command string, **not** a boundary: a path built at
  runtime, base64-encoded, or held in a variable walks straight past it.

Two cases where the sandbox is not the boundary, both worth knowing before you
rely on it:

- **A self-audit** (`--repo` pointing at this checkout) hands the harness tree to
  the agent as its own working directory, so no path-based restriction can
  separate the two. Only the filter applies. Such a run logs a warning.
- **A platform where the sandbox cannot start** (Linux without a working
  sandbox) falls back to the filter alone.

For an untrusted target, run the audit inside a disposable VM or container
anyway: a target with malicious build scripts could otherwise execute on your
host during PoC compilation, which is outside anything the harness can enforce.

The agent reads everything you `--add-dir`, including any `.env` or
`secrets/` directories in the target. Outputs land in `results/<run-id>/`
which is `.gitignore`d but **not** scrubbed of those reads.

### What the confinement does not cover

Written down because the alternative is an operator assuming a guarantee they do
not have. `scripts/mutation-check.py` is the executable form of the claims above:
it breaks one protection at a time and fails if the sensor that should notice
stays green.

- **Bash commands are filtered by string, not by structure.** A path built at
  runtime (`p=$(printf %s <base64> | base64 -d); sqlite3 "$p" ...`), a variable
  or any other indirection walks past the filter. Structured tools are checked by
  resolving the path and comparing containment, so those are exact.
- **A self-audit is unconfined by construction.** `--repo` pointing at this
  checkout, or at a directory containing it, hands the agent the tree the
  harness keeps `state.db` and `results/` in. The run logs a warning. Keeping
  those paths outside the audited tree is the structural fix and is not done yet.
- **A platform where the sandbox cannot start** leaves the filter alone. That is
  the one case where `sandbox: false` is the honest setting.
- **Prompt files can still be rewritten by a shell command in self-audit
  geometry.** Write and Edit are refused for the prompts, schemas and stage
  config, and Bash is refused by name, but a Bash command that builds the path at
  runtime is not caught. Prompts and schemas are re-read per dispatch, so a
  rewrite lands in a later stage's system prompt within the same run.
- **MCP servers are suppressed, not filtered.** Sessions run with no MCP
  servers and `strict_mcp_config`, because the CLI otherwise loads the
  operator's own user, project and plugin servers. Those clients run inside the
  harness process, which is outside the sandbox, so an MCP server with write
  access is an exfiltration route the sandbox cannot see. If you want a stage to
  have one, pass it explicitly through `mcp_servers` rather than relying on your
  global configuration.
- **Reads are not restricted at all in a normal run.** An agent can read any file
  the user can, including anything under `--add-dir`. The filter covers the
  harness's own secrets and state, not the rest of the filesystem.
- **`--run-id` must match `[A-Za-z0-9._-]{1,128}`** and is rejected at the CLI
  otherwise. It becomes a directory name, so a run id with a colon or a space
  will not resolve.
- **Case-only differences collide.** macOS and Windows fold case, so `Foo` and
  `foo` share one directory while SQLite keeps two rows; `create_run` refuses the
  second one.

## License

[MIT](LICENSE). Reuse freely. No warranty.

## Acknowledgements

- The pipeline design is from Cloudflare's [Project Glasswing](https://blog.cloudflare.com/cyber-frontier-models/)
  blog post. The credit for the architecture goes there.
- Built on the official [Claude Code Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).
