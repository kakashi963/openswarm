"""Per-install auth token for the localhost API.

OpenSwarm's backend runs a FastAPI server on `127.0.0.1:<random-port>`
and streams sensitive agent data (tool inputs, approval requests,
messages) over WebSockets. Without auth, any webpage loaded in any
browser on the same machine can connect to those endpoints — WebSockets
aren't subject to Same-Origin Policy — and impersonate the user.

This module issues a cryptographically random token at backend startup,
writes it 0600 to `<DATA_ROOT>/auth.token`, and provides validation
helpers. The token changes every backend restart. Only code running as
the same OS user can read the file.

Delivery to legitimate consumers:

- Electron main process reads the file and exposes it to the renderer
  via a contextBridge method in preload.js (NOT plain window global).
- Our Python MCP subprocesses receive it via env var
  `OPENSWARM_AUTH_TOKEN` that agent_manager passes when spawning.
- The Claude Code CLI we spawn receives it as `ANTHROPIC_API_KEY` in
  env; the anthropic-proxy route trusts that value.

None of those paths are accessible from a third-party webpage.
"""

from __future__ import annotations

import logging
import os
import secrets

from backend.config.paths import AUTH_TOKEN_FILE, DATA_ROOT

logger = logging.getLogger(__name__)

_TOKEN: str = ""


def _write_atomic(path: str, data: str, mode: int = 0o600) -> None:
    """Write `data` to `path` atomically with the given file mode.

    Uses `os.open(..., O_CREAT|O_WRONLY|O_TRUNC, mode)` + rename so the
    final file is never world-readable and never left half-written if
    the backend crashes mid-write. Windows-safe (rename of a file over
    an existing one works on NTFS when the source was just closed).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
    try:
        os.write(fd, data.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(tmp, mode)
    except Exception:
        pass
    os.replace(tmp, path)


def init_auth_token() -> str:
    """Generate a fresh token, persist to disk, return it.

    Called once at backend startup before the HTTP port is bound.
    """
    global _TOKEN
    _TOKEN = secrets.token_urlsafe(32)
    try:
        _write_atomic(AUTH_TOKEN_FILE, _TOKEN, mode=0o600)
        logger.info(f"auth: wrote token to {AUTH_TOKEN_FILE} (mode 0600)")
    except Exception as e:
        # Fail open is NOT an option here — if we can't write the file,
        # Electron can't read it, and the user sees a broken app. But
        # don't hard-crash the backend either; log loudly.
        logger.error(f"auth: failed to write token file: {e}")
    return _TOKEN


def get_auth_token() -> str:
    """Return the current token. Empty string if init_auth_token() hasn't run."""
    return _TOKEN


# Paths that never require auth. These are the public surface.
_AUTH_EXEMPT_EXACT = {
    # External OAuth providers redirect the user's browser here. The
    # browser has no way to inject our bearer token (it's a 302 from
    # Google/Anthropic/etc). The `state` query param is already a
    # one-time nonce validated against `_pending_oauth`.
    "/api/subscriptions/callback",
    # Same pattern for the per-tool OAuth flow (Notion / Google Workspace /
    # Airtable / HubSpot / Discord). The browser hits this with ?code=...&state=...
    # after the user approves on the provider's site; the `state` param is
    # the tool_id which we cross-check against _pending_oauth in tools_lib.py.
    # Without this exemption the redirect lands a 401 page in the user's
    # browser — see tools_lib.py:1156 where redirect_uri is constructed.
    "/api/tools/oauth/callback",
    "/api/version",
}

# Path prefixes that never require auth. Trailing slash optional.
_AUTH_EXEMPT_PREFIX = (
    # Electron's boot handshake polls /api/health/check before it has a
    # token (the HTTP port is up before main.js calls loadAuthToken()).
    # Use a prefix so /api/health/check — and any future sub-route — is
    # covered without re-introducing the bootstrap deadlock that an
    # exact "/api/health" match caused.
    "/api/health",
    # FastAPI's default health/docs/schema surface (packaged app never
    # ships /docs, but be defensive).
    "/docs",
    "/openapi",
    "/redoc",
    "/favicon",
)


def is_path_exempt(path: str) -> bool:
    """True if this request path bypasses token auth."""
    if path in _AUTH_EXEMPT_EXACT:
        return True
    for p in _AUTH_EXEMPT_PREFIX:
        if path.startswith(p):
            return True
    return False


def extract_bearer(header_value: str | None) -> str:
    """Pull the token out of `Authorization: Bearer <token>`."""
    if not header_value:
        return ""
    if header_value.startswith("Bearer "):
        return header_value[len("Bearer "):].strip()
    if header_value.startswith("bearer "):
        return header_value[len("bearer "):].strip()
    return ""


def request_matches_token(request_headers: dict, query_params: dict | None = None) -> bool:
    """Validate that an incoming HTTP / WS request carries our token.

    Accepts any of:
      - `Authorization: Bearer <token>`
      - `x-openswarm-token: <token>` (custom header for callers that
        can't easily set Authorization — e.g. future CLI clients)
      - `?token=<token>` query param (WS only; browsers can't easily
        set custom WS headers, so the token rides in the URL)

    The token comparison is constant-time via `secrets.compare_digest`.
    """
    if not _TOKEN:
        # Backend started without auth init — fail closed. This should
        # only happen in test fixtures that intentionally bypass main.
        return False

    candidates: list[str] = []

    auth = request_headers.get("authorization") or request_headers.get("Authorization")
    bearer = extract_bearer(auth)
    if bearer:
        candidates.append(bearer)

    openswarm_header = (
        request_headers.get("x-openswarm-token")
        or request_headers.get("X-OpenSwarm-Token")
    )
    if openswarm_header:
        candidates.append(openswarm_header.strip())

    if query_params:
        qp_token = query_params.get("token")
        if qp_token:
            candidates.append(qp_token)

    for candidate in candidates:
        if secrets.compare_digest(candidate, _TOKEN):
            return True
    return False


# Origin allowlist for WS handshakes. Electron's renderer loads from
# `file://` when packaged; `http://localhost:3000` (Vite dev server) and
# `http://127.0.0.1:3000` in dev. A bare `null` Origin is sent by some
# Electron contexts.
_ORIGIN_ALLOWLIST_DEV = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Electron may load prod build from file:// or an app:// scheme.
    "file://",
    "null",
}


def is_origin_allowed(origin: str | None) -> bool:
    """True if the WS connection's Origin header is from our app."""
    if origin is None:
        # No Origin header = curl / native WS client / MCP subprocess.
        # Token check is still required, so allow.
        return True
    if origin in _ORIGIN_ALLOWLIST_DEV:
        return True
    # file:// origins in Electron prod sometimes include paths like
    # file:///Applications/OpenSwarm.app/... — match by prefix.
    if origin.startswith("file://"):
        return True
    # localhost + any port (dev servers, tools the developer is running).
    if origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
        return True
    return False
