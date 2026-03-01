"""Shared Supabase auth dependency — extracts user_id from the @supabase/ssr auth cookie."""

from __future__ import annotations

import base64
import json

from fastapi import HTTPException, Request
from supabase import create_client

from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

COOKIE_NAME = "sb-koiaoajdcnxsarfpsfau-auth-token"


async def get_current_user_id(request: Request) -> str:
    """FastAPI dependency: returns the authenticated Supabase user_id."""
    raw = request.cookies.get(COOKIE_NAME, "")
    if not raw:
        chunks: list[str] = []
        i = 0
        while True:
            chunk = request.cookies.get(f"{COOKIE_NAME}.{i}", "")
            if not chunk:
                break
            chunks.append(chunk)
            i += 1
        raw = "".join(chunks)

    if not raw:
        raise HTTPException(status_code=401, detail="Missing auth cookie")

    try:
        if raw.startswith("base64-"):
            raw = base64.b64decode(raw[len("base64-"):] + "==").decode()
        session = json.loads(raw)
        access_token: str = session["access_token"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Malformed auth cookie") from exc

    user_response = supabase.auth.get_user(access_token)
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user_response.user.id
