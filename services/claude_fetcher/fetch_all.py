"""Fetch Claude.ai conversations with full messages → conversations.json"""

import json
import sys
import time

from curl_cffi import requests

MAX_CONVERSATIONS = 50

# --- Load cookies ---
try:
    with open("cookies.json") as f:
        cookies = json.load(f)
except FileNotFoundError:
    print("ERROR: cookies.json not found.")
    sys.exit(1)

ORG = cookies.get("lastActiveOrg")
if not ORG:
    print("ERROR: lastActiveOrg cookie missing")
    sys.exit(1)

BASE = f"https://claude.ai/api/organizations/{ORG}/chat_conversations"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://claude.ai",
    "referer": "https://claude.ai/recents",
    "user-agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
}


def api_get(url, params):
    """GET with retries on transient errors."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, cookies=cookies, params=params, impersonate="chrome")
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
    all_convos: list[object] = []
    cursor = None
    while len(all_convos) < limit:
        batch_size = min(50, limit - len(all_convos))
        params = {"limit": batch_size, "starred": "false", "consistency": "eventual"}
        if cursor:
            params["cursor"] = cursor

        data = api_get(BASE, params)
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
    url = f"{BASE}/{uuid}"
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
