"""Auth setup tests — env scrubbing + the auth modes.

Modes: gateway, api_key (opt-in), oauth_token, keychain_login,
macos_keychain_login.

The api_key mode requires the caller to pass `allow_api_key=True` to
configure_auth(). Without it, ANTHROPIC_API_KEY is scrubbed in favor of
subscription auth, matching the original "subscription only" behavior.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from audit import auth as auth_mod
from audit.auth import AuthError, configure_auth


def _empty_env(tmp_path: Path) -> Path:
    p = tmp_path / ".env"
    p.write_text("")
    return p


def _require_claude_cli() -> None:
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")


def _clear_all_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe every env var that influences auth-mode selection."""
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

def _force_non_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tests that expect no implicit local login fallback to run as Linux."""
    monkeypatch.setattr(auth_mod.platform, "system", lambda: "Linux")

# ---------- absence ----------


def test_missing_everything_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_all_auth_env(monkeypatch)
    _force_non_macos(monkeypatch)
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    with pytest.raises(AuthError, match="No auth available"):
        configure_auth(env_file=_empty_env(tmp_path))


def test_missing_claude_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-test-token")
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(AuthError, match="claude.*CLI"):
        configure_auth(env_file=_empty_env(tmp_path))


# ---------- default behavior (allow_api_key=False, preserves upstream) ----------


def test_default_scrubs_api_key_with_oauth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default behavior: ANTHROPIC_API_KEY is scrubbed even when OAuth is
    present. Subscription auth wins. Matches upstream evilsocket/audit."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-test-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-deleted")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path))
    assert status.auth_mode == "oauth_token"
    assert status.api_key_scrubbed is True
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_default_scrubs_api_key_even_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default behavior: ANTHROPIC_API_KEY alone (no other auth, no opt-in)
    is scrubbed and yields AuthError with a hint about --allow-api-key."""
    _clear_all_auth_env(monkeypatch)
    _force_non_macos(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-deleted")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    with pytest.raises(AuthError, match="--allow-api-key"):
        configure_auth(env_file=_empty_env(tmp_path))
    # And the key was scrubbed before the raise.
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_oauth_token_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OAuth token alone selects oauth_token mode."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-test-token")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path))
    assert status.auth_mode == "oauth_token"
    assert status.api_key_scrubbed is False


def test_keychain_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_all_auth_env(monkeypatch)
    creds = tmp_path / "creds.json"
    creds.write_text("{}")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", creds)
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path))
    assert status.auth_mode == "keychain_login"
    assert status.credentials_file == creds

def test_macos_keychain_login_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On macOS, Claude Code /login credentials live in the macOS Keychain,
    not in ~/.claude/.credentials.json. Allow this path through preflight so
    the Claude Agent SDK can use the active first-party login."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    monkeypatch.setattr(auth_mod.platform, "system", lambda: "Darwin")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path))
    assert status.auth_mode == "macos_keychain_login"
    assert status.credentials_file is None


# ---------- opt-in api_key mode ----------


def test_api_key_mode_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ANTHROPIC_API_KEY with allow_api_key=True selects api_key mode
    and leaves the key in the env so the SDK can use it."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path), allow_api_key=True)
    assert status.auth_mode == "api_key"
    assert status.api_key_scrubbed is False
    # CRITICAL: the key MUST still be in the env so the SDK can use it
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-api03-fake"


def test_api_key_outranks_oauth_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With allow_api_key=True, ANTHROPIC_API_KEY wins over
    CLAUDE_CODE_OAUTH_TOKEN. Matches SDK precedence (rung 3 > rung 5)."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-oauth-token")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path), allow_api_key=True)
    assert status.auth_mode == "api_key"
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-api03-fake"


def test_api_key_scrubs_stale_auth_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In api_key mode, a stale ANTHROPIC_AUTH_TOKEN must be scrubbed
    so it can't outrank the API key (rung 2 > rung 3)."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "stale-token-must-go")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path), allow_api_key=True)
    assert status.auth_mode == "api_key"
    assert status.auth_token_scrubbed is True
    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-api03-fake"


def test_allow_api_key_with_no_key_falls_back_to_oauth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Passing allow_api_key=True without setting ANTHROPIC_API_KEY is
    a no-op — falls through to subscription auth normally."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-test-token")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path), allow_api_key=True)
    assert status.auth_mode == "oauth_token"


# ---------- gateway mode ----------


def test_gateway_mode_openrouter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When ANTHROPIC_BASE_URL points at a non-anthropic host AND
    ANTHROPIC_AUTH_TOKEN is set, leave the gateway env intact and
    don't scrub the token."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "or-sk-xxx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-deleted")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path))
    assert status.auth_mode == "gateway"
    assert status.gateway_base_url == "https://openrouter.ai/api"
    assert status.api_key_scrubbed is True
    assert status.auth_token_scrubbed is False
    # CRITICAL: the gateway token MUST still be in the env so the SDK can use it
    assert os.environ.get("ANTHROPIC_AUTH_TOKEN") == "or-sk-xxx"
    assert os.environ.get("ANTHROPIC_BASE_URL") == "https://openrouter.ai/api"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_gateway_beats_api_key_even_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gateway mode outranks api_key mode even with allow_api_key=True.
    Mirrors SDK precedence (ANTHROPIC_AUTH_TOKEN at rung 2 > API key at 3)."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "or-sk-xxx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path), allow_api_key=True)
    assert status.auth_mode == "gateway"
    assert status.api_key_scrubbed is True
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_gateway_mode_requires_both_url_and_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A base URL without a token doesn't trigger gateway mode."""
    _clear_all_auth_env(monkeypatch)
    _force_non_macos(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setattr(auth_mod, "CREDENTIALS_PATH", tmp_path / "no_creds.json")
    _require_claude_cli()
    with pytest.raises(AuthError):
        configure_auth(env_file=_empty_env(tmp_path))


def test_anthropic_base_url_does_not_trigger_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A base URL pointing AT anthropic.com is normal — not gateway mode.
    Subscription scrubbing should still happen for the auth token."""
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "should-be-scrubbed")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-token")
    _require_claude_cli()
    status = configure_auth(env_file=_empty_env(tmp_path))
    assert status.auth_mode == "oauth_token"
    assert status.auth_token_scrubbed is True
    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ


# ---------- gateway base-url classification (lookalike hosts) ----------


def test_is_gateway_base_lookalike_host_is_gateway() -> None:
    """A hostname that merely CONTAINS 'anthropic.com' is not Anthropic:
    the substring check used to fail open and leave a hostile base URL
    active while scrubbing ANTHROPIC_AUTH_TOKEN."""
    assert auth_mod._is_gateway_base("https://api.anthropic.com.evil.net") is True
    assert auth_mod._is_gateway_base("https://api.anthropic.com") is False
    assert auth_mod._is_gateway_base("") is False
    # scheme-less values are treated as https
    assert auth_mod._is_gateway_base("api.z.ai/api/anthropic") is True


def test_configure_auth_lookalike_base_url_uses_gateway_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a lookalike base URL + auth token, gateway mode must win (the
    user-configured token goes to that host; subscription credentials must
    never be attached to it)."""
    _require_claude_cli()
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com.evil.net")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-user-token")
    status = configure_auth(env_file=_empty_env(tmp_path), allow_api_key=False)
    assert status.auth_mode == "gateway"
    assert status.gateway_base_url == "https://api.anthropic.com.evil.net"


def test_configure_auth_base_url_without_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-Anthropic BASE_URL with no AUTH_TOKEN is a misconfiguration:
    subscription credentials must never be pointed at a foreign host, so
    configure_auth raises instead of scrubbing the token and proceeding."""
    _require_claude_cli()
    _clear_all_auth_env(monkeypatch)
    _force_non_macos(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.example")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(AuthError, match="non-Anthropic host"):
        configure_auth(env_file=_empty_env(tmp_path), allow_api_key=False)

# ---------- F1/F27: lookalike base URL fails closed in every mode ----------

def test_lookalike_base_url_raises_in_all_three_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-Anthropic BASE_URL without a gateway token must raise AuthError
    in subscription, --allow-api-key, and gateway-without-token modes alike.
    Each case builds a fresh environment: a prior configure_auth call deletes
    ANTHROPIC_API_KEY from os.environ and would mask the api_key case."""
    _require_claude_cli()
    hostile = "https://api.anthropic.com.evil.net"

    # (a) subscription mode (the original fix)
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", hostile)
    _force_non_macos(monkeypatch)
    with pytest.raises(auth_mod.AuthError, match="non-Anthropic host"):
        auth_mod.configure_auth()

    # (b) --allow-api-key mode: the real key must never reach the hostile host
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-REAL-USER-KEY")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", hostile)
    with pytest.raises(auth_mod.AuthError, match="non-Anthropic host"):
        auth_mod.configure_auth(allow_api_key=True)
    # fail closed means the key never left the process, but the run refused
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-REAL-USER-KEY"

    # (c) gateway base without token lands in the same guard
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com")
    _force_non_macos(monkeypatch)
    with pytest.raises(auth_mod.AuthError, match="non-Anthropic host"):
        auth_mod.configure_auth()


def test_api_key_mode_surfaces_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A green preflight in api_key mode must say WHERE the key is going:
    AuthStatus carries the base URL in every mode, not just gateway."""
    _require_claude_cli()
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    status = auth_mod.configure_auth(allow_api_key=True)
    assert status.auth_mode == "api_key"
    assert status.gateway_base_url == "https://api.anthropic.com"


def test_unparseable_base_url_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed BASE_URL (raises ValueError inside urlparse) must classify
    as a gateway host and fail closed with AuthError — not escape as a raw
    ValueError past the AuthError-handling call sites (F27)."""
    _require_claude_cli()
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://[")
    _force_non_macos(monkeypatch)
    with pytest.raises(auth_mod.AuthError, match="non-Anthropic host"):
        auth_mod.configure_auth()


# ---------- the BASE_URL is judged as a string, not by one parser ----------

# `https://evil.com\@api.anthropic.com` is the demonstrated leak: Python's
# urlparse splits userinfo on the last "@" and reports hostname
# "api.anthropic.com", while the WHATWG parser the CLI consumes turns the
# backslash into a path separator and connects to "evil.com". The harness read
# it as Anthropic, skipped the fail-closed guard, and handed the subscription
# token to evil.com.
#
# The rest of the table is the surrounding class: hosts the two parsers can
# disagree about. The property under test is not "the backslash is blocked" but
# "no URL naming a foreign transport host is ever classified as Anthropic".
FOREIGN_HOST_BASE_URLS = [
    "https://evil.com\\@api.anthropic.com",
    "https://api.anthropic.com\\@evil.com",
    "https://user@api.anthropic.com",
    "https://user:pass@api.anthropic.com",
    "https://api.anthropic.com\t.evil.com",
    "https://api.anthropic.com\n.evil.com",
    "https://api.anthropic.com\r.evil.com",
    "https://api.anthropic.com .evil.com",
    "https://\u0430pi.anthropic.com",  # Cyrillic a: IDNA reinterprets the host
    "https://api.anthropic.com%2e.evil.com",
    "https://api.anthropic.com.evil.net",
    "http://[",
]

# A NUL cannot be stored in os.environ at all (setenv raises ValueError), so it
# can only be exercised at the string level. The validator still refuses it, in
# case a future config path hands it one.
STRING_LEVEL_REJECTIONS = FOREIGN_HOST_BASE_URLS + [
    "https://api.anthropic.com\x00.evil.com",
]

LEGIT_BASE_URLS = [
    "https://api.anthropic.com",
    "https://api.anthropic.com/v1",
    "https://api.anthropic.com:443",
    "https://openrouter.ai/api",
    "api.z.ai/api/anthropic",
    "http://localhost:8080",
    "https://gw.example.com:8443/api",
    "https://[::1]:8080",
]


@pytest.mark.parametrize("url", STRING_LEVEL_REJECTIONS)
def test_foreign_host_base_url_is_never_classified_as_anthropic(url: str) -> None:
    """A URL naming a foreign host must come out of classification as a gateway
    (so the no-token guard fires) or be rejected outright. Silence here is what
    leaked a credential."""
    reason = auth_mod._base_url_rejection_reason(url)
    assert reason or auth_mod._is_gateway_base(url), (
        f"hostile BASE_URL passed as Anthropic: {url!r}"
    )


@pytest.mark.parametrize("url", FOREIGN_HOST_BASE_URLS)
def test_foreign_host_base_url_fails_closed_end_to_end(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Same values through the real entry point, with every auth path that could
    carry a credential available. None of them may proceed."""
    _require_claude_cli()
    _clear_all_auth_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-subscription-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-REAL-USER-KEY")
    with pytest.raises(auth_mod.AuthError):
        auth_mod.configure_auth(allow_api_key=True)


@pytest.mark.parametrize("url", LEGIT_BASE_URLS)
def test_legitimate_base_urls_are_still_accepted(url: str) -> None:
    """The anti-overblocking direction: gateways, ports and IPv6 literals are
    normal configurations, and a validator that rejects them is a breakage."""
    assert auth_mod._base_url_rejection_reason(url) == ""


@pytest.mark.parametrize("url", ["", "   "])
def test_empty_base_url_is_not_a_rejection(url: str) -> None:
    """No BASE_URL set is the common case (subscription billing)."""
    assert auth_mod._base_url_rejection_reason(url) == ""


@pytest.mark.parametrize("url", [
    "https://api.anthropic.com:abc",
    "https://api.anthropic.com:99999",
    "https://api.anthropic.com:",
])
def test_base_url_with_a_broken_port_is_rejected(url: str) -> None:
    """`parts.port` raises for a non-numeric, empty or out-of-range port, and
    that raise is not the same thing as an unparseable authority: it must be
    refused here rather than falling through as a valid Anthropic URL."""
    reason = auth_mod._base_url_rejection_reason(url)
    assert reason, f"a broken port was accepted: {url!r}"


@pytest.mark.parametrize("url", ["https://api.anthropic.com:443", "http://localhost:8080"])
def test_valid_ports_are_still_accepted(url: str) -> None:
    """Control: the port branch must not swallow correct ports."""
    assert auth_mod._base_url_rejection_reason(url) == ""


def test_backslash_base_url_reason_names_the_character() -> None:
    """The message has to name the problem: this is a config error an operator
    has to fix, and "invalid URL" sends them hunting."""
    reason = auth_mod._base_url_rejection_reason("https://evil.com\\@api.anthropic.com")
    assert "backslash" in reason.lower(), reason


def test_userinfo_base_url_reason_names_userinfo() -> None:
    reason = auth_mod._base_url_rejection_reason("https://user@api.anthropic.com")
    assert "userinfo" in reason.lower() or "credential" in reason.lower(), reason

