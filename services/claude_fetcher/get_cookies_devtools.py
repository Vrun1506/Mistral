# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "curl_cffi",
# ]
# ///

"""
Loads cookies from cookies.json (pasted from DevTools) and
tests them against the claude.ai API.

Steps:
  1. Open claude.ai in your browser, log in
  2. Open DevTools (F12) → Application → Cookies → claude.ai
  3. Copy sessionKey + other HttpOnly cookies
  4. Save to cookies.json
  5. Run: python get_cookies_devtools.py
"""

import json
import sys

from curl_cffi import requests

COOKIES_FILE = "cookies.json"

try:
    with open(COOKIES_FILE) as f:
        cookies = json.load(f)
except FileNotFoundError:
    print(f"ERROR: {COOKIES_FILE} not found.")
    print("Paste the DevTools output into cookies.json first.")
    sys.exit(1)

session_key = cookies.get("sessionKey")
org = cookies.get("lastActiveOrg")

if not session_key:
    print("ERROR: sessionKey not found in cookies.")
    sys.exit(1)

if not org:
    print("ERROR: lastActiveOrg not found in cookies.")
    sys.exit(1)

print(f"sessionKey: {session_key[:15]}...")
print(f"lastActiveOrg: {org}")

# Quick test — fetch conversations
url = f"https://claude.ai/api/organizations/{org}/chat_conversations"
headers = {
    "accept": "application/json",
    "origin": "https://claude.ai",
    "referer": "https://claude.ai/recents",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    ),
}

resp = requests.get(url, headers=headers, cookies=cookies, params={"limit": 5}, impersonate="chrome")
print(f"\nAPI test: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    if isinstance(data, list):
        print(f"SUCCESS — found {len(data)} conversations")
        for c in data:
            print(f"  - {c.get('name', 'untitled')}")
    else:
        print("Unexpected response:", json.dumps(data, indent=2)[:500])
else:
    print("Failed. Response:", resp.text[:500])
    print("\nThe sessionKey may be expired. Re-run the DevTools snippet.")
