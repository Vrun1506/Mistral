"""
OAuth 2.0 + PKCE handler for Notion MCP Server.

Handles the full OAuth lifecycle:
  - RFC 9470 / RFC 8414 endpoint discovery
  - PKCE code_verifier / code_challenge generation (S256)
  - Dynamic client registration (RFC 7591)
  - Local HTTP callback server to capture the authorization code
  - Token exchange and refresh
  - Persistent token/credential storage (tokens.json)

Usage:
  python oauth_handler.py          # Run interactive OAuth flow
  python oauth_handler.py --refresh # Force token refresh
"""

import base64
import hashlib
import json
import secrets
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests as http_requests

from config import (
    CLIENT_FILE,
    MCP_SERVER_URL,
    OAUTH_CALLBACK_PORT as CALLBACK_PORT,
    REDIRECT_URI,
    TOKEN_FILE,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OAuthMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: Optional[str] = None
    code_challenge_methods_supported: list[str] = field(default_factory=list)


@dataclass
class ClientCredentials:
    client_id: str
    client_secret: Optional[str] = None


@dataclass
class TokenSet:
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: str = "Bearer"
    obtained_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        if not self.expires_in:
            return False
        return time.time() > (self.obtained_at + self.expires_in - 60)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) using S256."""
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def generate_state() -> str:
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# OAuth discovery (RFC 9470 then RFC 8414)
# ---------------------------------------------------------------------------


def discover_oauth_metadata(mcp_server_url: str = MCP_SERVER_URL) -> OAuthMetadata:
    # Step 1 — Protected Resource Metadata (RFC 9470)
    pr_url = f"{mcp_server_url}/.well-known/oauth-protected-resource"
    print(f"  [discovery] GET {pr_url}")
    pr_resp = http_requests.get(pr_url, timeout=15)
    pr_resp.raise_for_status()
    pr_data = pr_resp.json()

    auth_servers = pr_data.get("authorization_servers", [])
    if not auth_servers:
        raise RuntimeError("No authorization_servers in protected resource metadata")
    auth_server_url = auth_servers[0]

    # Step 2 — Authorization Server Metadata (RFC 8414)
    as_url = f"{auth_server_url}/.well-known/oauth-authorization-server"
    print(f"  [discovery] GET {as_url}")
    as_resp = http_requests.get(as_url, timeout=15)
    as_resp.raise_for_status()
    as_data = as_resp.json()

    if not as_data.get("authorization_endpoint") or not as_data.get("token_endpoint"):
        raise RuntimeError("Missing required OAuth endpoints")

    return OAuthMetadata(
        issuer=as_data.get("issuer", auth_server_url),
        authorization_endpoint=as_data["authorization_endpoint"],
        token_endpoint=as_data["token_endpoint"],
        registration_endpoint=as_data.get("registration_endpoint"),
        code_challenge_methods_supported=as_data.get("code_challenge_methods_supported", []),
    )


# ---------------------------------------------------------------------------
# Dynamic client registration (RFC 7591)
# ---------------------------------------------------------------------------


def register_client(
    metadata: OAuthMetadata,
    redirect_uri: str = REDIRECT_URI,
    client_name: str = "Flashcard-Pipeline-Notion-MCP",
) -> ClientCredentials:
    if not metadata.registration_endpoint:
        raise RuntimeError("Server does not support dynamic client registration")

    payload = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }

    print(f"  [registration] POST {metadata.registration_endpoint}")
    resp = http_requests.post(
        metadata.registration_endpoint,
        json=payload,
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    creds = ClientCredentials(client_id=data["client_id"], client_secret=data.get("client_secret"))
    CLIENT_FILE.write_text(json.dumps(asdict(creds), indent=2))
    print(f"  [registration] Registered: {creds.client_id}")
    return creds


# ---------------------------------------------------------------------------
# Local callback server
# ---------------------------------------------------------------------------


class _CallbackHandler(BaseHTTPRequestHandler):
    authorization_code: Optional[str] = None
    received_state: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if "error" in params:
            _CallbackHandler.error = params["error"][0]
            body = f"<h2>OAuth Error</h2><p>{params.get('error_description', ['Unknown'])[0]}</p>"
        elif "code" in params:
            _CallbackHandler.authorization_code = params["code"][0]
            _CallbackHandler.received_state = params.get("state", [None])[0]
            body = "<h2>Authorization successful!</h2><p>You can close this tab.</p>"
        else:
            body = "<h2>Unexpected callback</h2>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


def _wait_for_callback(port: int = CALLBACK_PORT, timeout: int = 120) -> str:
    server = HTTPServer(("localhost", port), _CallbackHandler)
    server.timeout = timeout
    _CallbackHandler.authorization_code = None
    _CallbackHandler.received_state = None
    _CallbackHandler.error = None

    print(f"  [callback] Listening on http://localhost:{port}/callback ...")
    while _CallbackHandler.authorization_code is None and _CallbackHandler.error is None:
        server.handle_request()
    server.server_close()

    if _CallbackHandler.error:
        raise RuntimeError(f"OAuth error: {_CallbackHandler.error}")
    return _CallbackHandler.authorization_code


# ---------------------------------------------------------------------------
# Token exchange & refresh
# ---------------------------------------------------------------------------


def exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    metadata: OAuthMetadata,
    client: ClientCredentials,
    redirect_uri: str = REDIRECT_URI,
) -> TokenSet:
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client.client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if client.client_secret:
        payload["client_secret"] = client.client_secret

    resp = http_requests.post(
        metadata.token_endpoint,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} — {resp.text}")

    data = resp.json()
    tokens = TokenSet(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        expires_in=data.get("expires_in"),
        token_type=data.get("token_type", "Bearer"),
        obtained_at=time.time(),
    )
    _save_tokens(tokens)
    return tokens


def refresh_access_token(
    tokens: TokenSet,
    metadata: OAuthMetadata,
    client: ClientCredentials,
) -> TokenSet:
    if not tokens.refresh_token:
        raise RuntimeError("No refresh token — re-authentication required")

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "client_id": client.client_id,
    }
    if client.client_secret:
        payload["client_secret"] = client.client_secret

    resp = http_requests.post(
        metadata.token_endpoint,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=15,
    )
    if not resp.ok:
        try:
            err = json.loads(resp.text)
            if err.get("error") == "invalid_grant":
                raise RuntimeError("REAUTH_REQUIRED: Refresh token invalid/expired")
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Token refresh failed: {resp.status_code} — {resp.text}")

    data = resp.json()
    new_tokens = TokenSet(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", tokens.refresh_token),
        expires_in=data.get("expires_in"),
        token_type=data.get("token_type", "Bearer"),
        obtained_at=time.time(),
    )
    _save_tokens(new_tokens)
    return new_tokens


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_tokens(tokens: TokenSet):
    TOKEN_FILE.write_text(json.dumps(asdict(tokens), indent=2))


def load_tokens() -> Optional[TokenSet]:
    if not TOKEN_FILE.exists():
        return None
    return TokenSet(**json.loads(TOKEN_FILE.read_text()))


def load_client() -> Optional[ClientCredentials]:
    if not CLIENT_FILE.exists():
        return None
    return ClientCredentials(**json.loads(CLIENT_FILE.read_text()))


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def run_oauth_flow() -> TokenSet:
    """Complete interactive OAuth flow (opens browser)."""
    print("\n=== Notion MCP OAuth Flow ===\n")

    print("[1/5] Discovering endpoints ...")
    metadata = discover_oauth_metadata()

    print("[2/5] Registering client ...")
    client = load_client() or register_client(metadata)

    print("[3/5] Generating PKCE ...")
    code_verifier, code_challenge = generate_pkce_pair()
    state = generate_state()

    print("[4/5] Opening browser ...")
    auth_params = urlencode({
        "response_type": "code",
        "client_id": client.client_id,
        "redirect_uri": REDIRECT_URI,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "prompt": "consent",
    })
    auth_url = f"{metadata.authorization_endpoint}?{auth_params}"
    webbrowser.open(auth_url)

    auth_code = _wait_for_callback()
    if _CallbackHandler.received_state != state:
        raise RuntimeError("State mismatch — possible CSRF attack")

    print("[5/5] Exchanging code for tokens ...")
    tokens = exchange_code_for_tokens(auth_code, code_verifier, metadata, client)
    print(f"\n=== Done. Token saved to {TOKEN_FILE} ===\n")
    return tokens


def ensure_valid_tokens() -> TokenSet:
    """Load existing tokens, refresh if expired, or run full flow if needed."""
    tokens = load_tokens()
    if tokens is None:
        return run_oauth_flow()

    if tokens.is_expired:
        print("Access token expired, refreshing ...")
        metadata = discover_oauth_metadata()
        client = load_client()
        if not client:
            return run_oauth_flow()
        try:
            return refresh_access_token(tokens, metadata, client)
        except RuntimeError as e:
            if "REAUTH_REQUIRED" in str(e):
                return run_oauth_flow()
            raise

    remaining = int(tokens.expires_in - (time.time() - tokens.obtained_at)) if tokens.expires_in else "?"
    print(f"Token valid (~{remaining}s remaining)")
    return tokens


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        t = load_tokens()
        if t and t.refresh_token:
            m = discover_oauth_metadata()
            c = load_client()
            refresh_access_token(t, m, c)
        else:
            run_oauth_flow()
    else:
        run_oauth_flow()