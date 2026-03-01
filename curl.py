# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Quick test script: sends session_key + org_id to /get-cookies with Supabase auth."""
import json
import requests

with open("headersupabase.json") as f:
    auth_cookies = json.load(f)

with open("cookies.json") as f:
    claude_cookies = json.load(f)

cookie_dict = {c["name"]: c["value"] for c in claude_cookies}

resp = requests.post(
    "http://localhost:8000/get-cookies",
    json={
        "session_key": cookie_dict["sessionKey"],
        "last_active_org": cookie_dict["lastActiveOrg"],
    },
    cookies=auth_cookies,
    stream=True,
)

for line in resp.iter_lines(decode_unicode=True):
    if line:
        print(line)
