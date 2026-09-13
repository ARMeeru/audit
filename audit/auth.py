"""Auth setup for the Claude Code Agent SDK.

Claude Code's authentication-precedence list
(https://code.claude.com/docs/en/authentication#authentication-precedence)
is:

  1. Cloud provider credentials (Bedrock / Vertex / Foundry, when their
     respective `CLAUDE_CODE_USE_*` flag is set)
  2. ANTHROPIC_AUTH_TOKEN  (Bearer-token mode — used by LLM gateways
     like OpenRouter, custom proxies, etc.)
  3. ANTHROPIC_API_KEY      (the canonical metered Anthropic API)
  4. apiKeyHelper
  5. CLAUDE_CODE_OAUTH_TOKEN (long-lived subscription token)
  6. Subscription OAuth credentials from `claude login`

This module supports four modes, picked in this order:

  - **gateway**: `ANTHROPIC_BASE_URL` points away from anthropic.com AND
    `ANTHROPIC_AUTH_TOKEN` is set. Used for OpenRouter and similar.
    We leave those two env vars intact but still scrub `ANTHROPIC_API_KEY`
    (it'd outrank the gateway token).

  - **api_key**: `ANTHROPIC_API_KEY` is set with no gateway configured,
    AND the caller passed `allow_api_key=True`. Metered Anthropic API
    billing. We leave the key in place; the SDK uses it natively.
    ANTHROPIC_AUTH_TOKEN is scrubbed so a stale value can't outrank
    the key.

    This mode is opt-in to protect users who set ANTHROPIC_API_KEY in
    their shell for other tools (e.g. anthropic-sdk-python) but expect
    subscription billing here. By default, the API key is scrubbed and
    one of the subscription modes wins instead — matching the behavior
    before this mode was added.

  - **oauth_token**: `CLAUDE_CODE_OAUTH_TOKEN` is set (Pro/Max/Team/Enterprise
    subscription, ideal for CI). We scrub `ANTHROPIC_API_KEY` (unless
    api_key mode was selected above) and `ANTHROPIC_AUTH_TOKEN` so they
    can't outrank the OAuth token.

  - **keychain_login**: `~/.claude/.credentials.json` exists from
    `claude login`. Same scrubbing as oauth_token.

Anything else raises AuthError.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from audit.paths import CREDENTIALS_FILE


@dataclass
class AuthStatus:
    auth_mode: str            # "gateway" | "api_key" | "oauth_token" | "keychain_login" | "macos_keychain_login"
    api_key_scrubbed: bool
    auth_token_scrubbed: bool
    claude_cli_path: str | None
    claude_cli_version: str | None
    credentials_file: Path | None
    gateway_base_url: str | None
    gateway_model: str | None  # value of ANTHROPIC_MODEL if set, for display


class AuthError(RuntimeError):
    pass


CREDENTIALS_PATH = CREDENTIALS_FILE


def _base_url_rejection_reason(url: str) -> str:
    """Why `url` is unusable as ANTHROPIC_BASE_URL, or "" when it is fine.

    Judged on the raw string, deliberately independent of what any one parser
    decides, because the two parsers in play do not agree. For
    `https://evil.com\\@api.anthropic.com`:

      * Python's urlparse splits userinfo on the LAST "@" and reports the
        hostname as api.anthropic.com, so this module classified the value as
        the canonical Anthropic API and the fail-closed guard never fired, while
      * the WHATWG parser the Claude CLI consumes turns the backslash into a
        path separator and connects to evil.com.

    The subscription token would have gone to the attacker's host. A
    classification that both parsers must agree on cannot be assembled out of
    one parser's opinion, so anything that is not a boring URL is refused here
    instead. Only string-shape problems are handled: a value that fails to parse
    at all is left to _is_gateway_base, which already fails closed on it.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if "\\" in raw:
        return (
            "it contains a backslash. Python and WHATWG URL parsers disagree "
            "about one in the authority (path separator vs. userinfo), which is "
            "how a host you did not choose receives your credentials"
        )
    if not raw.isascii():
        return "it contains non-ASCII characters, which IDNA may reinterpret as another host"
    control = sorted({f"U+{ord(c):04X}" for c in raw if c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F})
    if control:
        return f"it contains whitespace or control characters ({', '.join(control)})"

    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parts = urlparse(candidate)
        host = parts.hostname or ""
    except ValueError:
        # An unparseable authority (e.g. `http://[`). Left to _is_gateway_base,
        # which catches this and fails closed with its own message.
        return ""
    if parts.username is not None or parts.password is not None:
        return (
            "it carries userinfo (user[:password]@host). Credentials in a URL "
            "get logged, and the parsers disagree about where the host begins"
        )
    if not host:
        return "it has no host"
    try:
        # A non-numeric port, an empty one, or one out of range raises here
        # rather than at parse time. A broken port is a misconfiguration, so it
        # is refused like any other: this branch must not assume the value has
        # already been caught upstream, because it has not.
        port = parts.port
    except ValueError:
        return f"its port is not a valid port number ({parts.netloc!r})"
    # The authority must be exactly host[:port]. Anything else (the '@' Python
    # already split off, percent-encoding, a stray character) is a place where
    # one parser can see a different host than the other.
    if ":" in host:  # IPv6 literal, which urlparse keeps bracketed in netloc
        expected = f"[{host}]:{port}" if port is not None else f"[{host}]"
    else:
        expected = f"{host}:{port}" if port is not None else host
    if parts.netloc.lower() != expected.lower():
        return (
            f"its authority {parts.netloc!r} is not exactly the host "
            f"{expected!r}"
        )
    return ""


def _is_gateway_base(url: str) -> bool:
    """A non-empty BASE_URL whose hostname is not canonical Anthropic
    counts as 'gateway mode'.

    The comparison is on the parsed hostname, not a substring: a lookalike
    like `https://api.anthropic.com.evil.net` must classify as a gateway,
    otherwise configure_auth scrubs ANTHROPIC_AUTH_TOKEN but keeps the
    hostile base URL active for the SDK's CLI."""
    u = (url or "").strip().lower()
    if not u:
        return False
    if "://" not in u:
        u = f"https://{u}"
    try:
        host = urlparse(u).hostname or ""
    except ValueError:
        # Unparseable URL (e.g. malformed IPv6 literal): treat as a
        # foreign host and fail closed rather than raising past the
        # AuthError-handling call sites.
        return True
    # Deliberate single-host allowlist: only the canonical API host counts
    # as Anthropic. Console/docs hosts are not API endpoints, so pointing
    # BASE_URL at them is a misconfiguration and fails closed.
    return host != "api.anthropic.com"


def configure_auth(
    env_file: Path | None = None,
    *,
    allow_api_key: bool = False,
) -> AuthStatus:
    """Load .env, decide auth mode, scrub conflicting env vars accordingly.

    Args:
        env_file: Optional .env file to load before reading env vars.
        allow_api_key: When True, ANTHROPIC_API_KEY is honored as a valid
            auth path (api_key mode, metered billing). When False (default),
            the key is scrubbed in favor of subscription auth — matching the
            original "subscription only" behavior. Wire this from a CLI flag
            or AUDIT_ALLOW_API_KEY=1 in the env.

    Returns an AuthStatus describing what was picked. Raises AuthError if
    no usable auth path is available.
    """
    if env_file is not None and env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Before anything else, and before any mode is chosen: this value decides
    # which host receives a credential. Every mode below is downstream of it,
    # so a value the parsers can disagree about is refused for all of them
    # rather than only where the gateway guard happens to sit.
    base_url_raw = os.environ.get("ANTHROPIC_BASE_URL", "")
    url_problem = _base_url_rejection_reason(base_url_raw)
    if url_problem:
        raise AuthError(
            f"ANTHROPIC_BASE_URL is unusable: {url_problem}.\n"
            f"  value: {base_url_raw!r}\n"
            "Refusing to start: unset ANTHROPIC_BASE_URL to use subscription "
            "billing, or set it to a plain https://host[:port] form."
        )

    cli_path = shutil.which("claude")
    if cli_path is None:
        raise AuthError(
            "`claude` CLI not found on PATH. Install Claude Code first: "
            "https://code.claude.com/docs/en/setup"
        )

    api_key_was_set = "ANTHROPIC_API_KEY" in os.environ
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    gateway = _is_gateway_base(base_url) and bool(auth_token)

    # Credential-independent rule, checked above the mode fork: a
    # non-Anthropic BASE_URL without a gateway token must never proceed.
    # The subscription branch used to check this, but the --allow-api-key
    # branch skipped it, so a real API key would be sent to the hostile
    # host with a green preflight. Validate once, for every mode.
    if _is_gateway_base(base_url) and not auth_token:
        raise AuthError(
            f"ANTHROPIC_BASE_URL points at a non-Anthropic host ({base_url})\n"
            "but ANTHROPIC_AUTH_TOKEN is not set. Credentials are never sent\n"
            "to a custom host without an explicit gateway token: either set\n"
            "ANTHROPIC_AUTH_TOKEN for the gateway, or unset ANTHROPIC_BASE_URL."
        )

    api_key_scrubbed = False
    auth_token_was_scrubbed = False
    creds_file: Path | None = None

    if gateway:
        # Gateway path (OpenRouter / custom proxy / etc.): keep
        # ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, but still drop
        # ANTHROPIC_API_KEY (rung 3 would outrank the gateway token).
        if api_key_was_set:
            del os.environ["ANTHROPIC_API_KEY"]
            api_key_scrubbed = True
        mode = "gateway"
    elif allow_api_key and api_key_was_set:
        # Explicit API key path (metered Anthropic billing). Leave the
        # key in place; the SDK uses it natively at precedence rung 3.
        # Scrub ANTHROPIC_AUTH_TOKEN so a stale value can't outrank the
        # key (rung 2 > rung 3). No claude login credentials are needed.
        if "ANTHROPIC_AUTH_TOKEN" in os.environ:
            del os.environ["ANTHROPIC_AUTH_TOKEN"]
            auth_token_was_scrubbed = True
        mode = "api_key"
    else:
        # Subscription paths: scrub both API-key vars so subscription
        # OAuth wins precedence. (When allow_api_key=False, this is the
        # only place ANTHROPIC_API_KEY can land — and we always scrub it,
        # matching the pre-opt-in behavior.)
        if api_key_was_set:
            del os.environ["ANTHROPIC_API_KEY"]
            api_key_scrubbed = True
        if "ANTHROPIC_AUTH_TOKEN" in os.environ:
            del os.environ["ANTHROPIC_AUTH_TOKEN"]
            auth_token_was_scrubbed = True

        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
        creds_file = CREDENTIALS_PATH if CREDENTIALS_PATH.exists() else None
        if token:
            mode = "oauth_token"
        elif creds_file is not None:
            mode = "keychain_login"
        elif platform.system() == "Darwin":
            # Claude Code stores interactive /login credentials in the macOS
            # Keychain, not in ~/.claude/.credentials.json. Do not reject this
            # path during preflight; let Claude Code / the Agent SDK use the
            # active Keychain-backed first-party login.
            mode = "macos_keychain_login"
        else:
            hint = ""
            if api_key_was_set:
                hint = (
                    "\n\nNote: ANTHROPIC_API_KEY was set but ignored. To use\n"
                    "metered API billing, re-run with --allow-api-key (or set\n"
                    "AUDIT_ALLOW_API_KEY=1 in the env)."
                )
            raise AuthError(
                "No auth available. Pick one of:\n"
                "  (a) Subscription OAuth (interactive): run `claude login`.\n"
                "  (b) Subscription OAuth (headless): run `claude setup-token` "
                "and paste into .env as CLAUDE_CODE_OAUTH_TOKEN.\n"
                "  (c) LLM gateway (OpenRouter / proxy): set "
                "ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN.\n"
                "  (d) Direct Anthropic API key (metered): set "
                "ANTHROPIC_API_KEY and pass --allow-api-key."
                + hint
            )

    cli_version: str | None = None
    try:
        out = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            cli_version = out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass

    return AuthStatus(
        auth_mode=mode,
        api_key_scrubbed=api_key_scrubbed,
        auth_token_scrubbed=auth_token_was_scrubbed,
        claude_cli_path=cli_path,
        claude_cli_version=cli_version,
        credentials_file=creds_file,
        gateway_base_url=base_url or None,
        gateway_model=os.environ.get("ANTHROPIC_MODEL") if mode == "gateway" else None,
    )
