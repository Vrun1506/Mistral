"""Fetch Claude.ai conversations with full messages → conversations.json"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from curl_cffi import requests as sync_requests
from curl_cffi.requests import AsyncSession

MAX_CONVERSATIONS = 50

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://claude.ai",
    "referer": "https://claude.ai/recents",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
}

# --- Load cookies (sync module-level, used only by CLI main()) ---
_cookies: dict[str, Any] | None = None
_ORG: str | None = None
_BASE: str | None = None


def _ensure_cookies() -> tuple[dict[str, Any], str, str]:
    global _cookies, _ORG, _BASE
    if _cookies is not None and _ORG is not None and _BASE is not None:
        return _cookies, _ORG, _BASE
    try:
        with open("cookies.json") as f:
            _cookies = json.load(f)
    except FileNotFoundError:
        print("ERROR: cookies.json not found.")
        sys.exit(1)
    _ORG = _cookies.get("lastActiveOrg")
    if not _ORG:
        print("ERROR: lastActiveOrg cookie missing")
        sys.exit(1)
    _BASE = f"https://claude.ai/api/organizations/{_ORG}/chat_conversations"
    return _cookies, _ORG, _BASE


def api_get(url, params):
    """GET with retries on transient errors."""
    cookies, _, _ = _ensure_cookies()
    for attempt in range(3):
        try:
            resp = sync_requests.get(url, headers=HEADERS, cookies=cookies, params=params, impersonate="chrome")
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} (attempt {attempt + 1}/3)")
                time.sleep(2)
                continue
            return resp.json()
        except Exception as e:
            print(f"  Network error (attempt {attempt + 1}/3): {e}")
            time.sleep(2)
    return None


def fetch_conversation_list(limit):
    """Fetch the most recent `limit` conversations."""
    _, _, base = _ensure_cookies()
    all_convos: list[object] = []
    cursor = None
    while len(all_convos) < limit:
        batch_size = min(50, limit - len(all_convos))
        params = {"limit": batch_size, "starred": "false", "consistency": "eventual"}
        if cursor:
            params["cursor"] = cursor

        data = api_get(base, params)
        if not isinstance(data, list) or len(data) == 0:
            break

        all_convos.extend(data)
        print(f"  Listed {len(all_convos)} conversations...")

        if len(data) < batch_size:
            break
        cursor = data[-1].get("uuid")

    return all_convos[:limit]


def extract_messages(chat_messages):
    """Extract sender + text from raw chat_messages list."""
    msgs = []
    for msg in chat_messages:
        sender = msg.get("sender", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("text"))
        else:
            text = str(content)
        if text.strip():
            msgs.append({"sender": sender, "text": text})
    return msgs


def fetch_full_conversation(uuid):
    """Fetch full messages for a single conversation."""
    _, _, base = _ensure_cookies()
    url = f"{base}/{uuid}"
    params = {
        "tree": "True",
        "rendering_mode": "messages",
        "render_all_tools": "true",
        "consistency": "eventual",
    }
    data = api_get(url, params)
    if not data or (isinstance(data, dict) and data.get("type") == "error"):
        return []
    return extract_messages(data.get("chat_messages", []))


# ---------------------------------------------------------------------------
# Async fetcher — used by the API pipeline endpoint
# ---------------------------------------------------------------------------


async def _async_api_get(
    session: AsyncSession,
    url: str,
    params: dict[str, Any],
    cookie_dict: dict[str, str],
) -> Any:
    """Async GET with retries."""
    for attempt in range(3):
        try:
            resp = await session.get(url, headers=HEADERS, cookies=cookie_dict, params=params)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} (attempt {attempt + 1}/3)")
                await asyncio.sleep(2)
                continue
            return resp.json()
        except Exception as e:
            print(f"  Network error (attempt {attempt + 1}/3): {e}")
            await asyncio.sleep(2)
    return None


async def async_fetch_conversations(
    session_key: str,
    last_active_org: str,
    on_progress: Callable[..., Any] | None = None,
    max_conversations: int = 2000,
) -> list[dict[str, Any]]:
    """Fetch all conversations asynchronously using curl_cffi AsyncSession.

    Parameters are passed directly — no cookies.json needed.
    Uses Semaphore(8) for concurrent full-conversation fetches.
    """
    from models.schemas import PipelinePhase

    cookie_dict = {"sessionKey": session_key, "lastActiveOrg": last_active_org}
    base = f"https://claude.ai/api/organizations/{last_active_org}/chat_conversations"

    async with AsyncSession(impersonate="chrome") as session:
        # 1. Fetch conversation list (sequential, paginated)
        all_convos: list[dict[str, Any]] = []
        cursor: str | None = None
        while len(all_convos) < max_conversations:
            batch_size = min(50, max_conversations - len(all_convos))
            params: dict[str, Any] = {"limit": batch_size, "starred": "false", "consistency": "eventual"}
            if cursor:
                params["cursor"] = cursor

            data = await _async_api_get(session, base, params, cookie_dict)
            if not isinstance(data, list) or len(data) == 0:
                break

            all_convos.extend(data)
            if on_progress:
                on_progress(
                    PipelinePhase.fetching,
                    f"Listed {len(all_convos)} conversations (max {max_conversations})",
                    0.15,
                )

            if len(data) < batch_size:
                break
            cursor = data[-1].get("uuid")

        convos = all_convos[:max_conversations]
        total = len(convos)

        # 2. Fetch full conversations concurrently with semaphore
        sem = asyncio.Semaphore(15)
        results: list[dict[str, Any] | None] = [None] * total

        async def fetch_one(idx: int, conv: dict[str, Any]) -> None:
            async with sem:
                conv_uuid = conv["uuid"]
                url = f"{base}/{conv_uuid}"
                params = {
                    "tree": "True",
                    "rendering_mode": "messages",
                    "render_all_tools": "true",
                    "consistency": "eventual",
                }
                data = await _async_api_get(session, url, params, cookie_dict)
                messages: list[dict[str, str]] = []
                if data and not (isinstance(data, dict) and data.get("type") == "error"):
                    messages = extract_messages(data.get("chat_messages", []))

                if messages:
                    results[idx] = {
                        "uuid": conv_uuid,
                        "name": conv.get("name", "(untitled)"),
                        "messages": messages,
                    }

                if on_progress:
                    done = sum(1 for r in results if r is not None)
                    on_progress(
                        PipelinePhase.fetching,
                        f"Fetched {done}/{total} conversations",
                        0.3 + (done / total) * 0.7,  # 30-100% for fetching
                    )

        await asyncio.gather(*[fetch_one(i, c) for i, c in enumerate(convos)])

    return [r for r in results if r is not None]


def main():
    print(f"Fetching latest {MAX_CONVERSATIONS} conversations...")
    convos = fetch_conversation_list(MAX_CONVERSATIONS)
    print(f"Got {len(convos)} conversation headers.\n")

    results = []
    for i, c in enumerate(convos):
        uuid = c["uuid"]
        name = c.get("name", "(untitled)")
        print(f"[{i + 1}/{len(convos)}] {name}")
        messages = fetch_full_conversation(uuid)
        if messages:
            results.append({"uuid": uuid, "name": name, "messages": messages})
            print(f"  → {len(messages)} messages")
        else:
            print("  → skipped (no messages)")

    with open("conversations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} conversations to conversations.json")


if __name__ == "__main__":
    main()
