import asyncio
from typing import AsyncGenerator

from curl_cffi.requests import AsyncSession



HEADERS = {
        "accept": "*/*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "anthropic-client-platform": "web_claude_ai",
        "anthropic-client-sha": "d3bdd8e5ec6aaae32685b9babd3e6e1ae18153c8",
        "anthropic-client-version": "1.0.0",
        "content-type": "application/json",
        "referer": "https://claude.ai/recents",
        "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
}


class ClaudeFetcher:

    def __init__(self, cookies: list[dict]):
        cookie_jar = {c["name"]: c["value"] for c in cookies}

        org_id = cookie_jar.get("lastActiveOrg")
        if not org_id:
            raise ValueError("lastActiveOrg not found in cookies")

        headers = {**HEADERS, "anthropic-device-id": cookie_jar["anthropic-device-id"]}

        self.org_id = org_id
        self.session = AsyncSession(headers=headers, cookies=cookie_jar, impersonate="chrome")

    async def close(self):
        await self.session.close()



    # Gets all the number of conversations
    async def get_all_conversations(self) -> int:
        url = f"https://claude.ai/api/organizations/{self.org_id}/chat_conversations/count_all"
        resp = await self.session.get(url)

        if resp.status_code != 200:
            raise RuntimeError(f"count_all returned HTTP {resp.status_code}: {resp.text[:200]}")

        return resp.json()["count"]

    async def fetch_conversation_list(self, total: int) -> list[str]:
        BASE = f"https://claude.ai/api/organizations/{self.org_id}/chat_conversations"
        all_convos: list[dict] = []
        cursor = None

        while len(all_convos) < total:
            batch_size = min(50, total - len(all_convos))
            params = {"limit": batch_size, "starred": "false", "consistency": "eventual"}
            if cursor:
                params["cursor"] = cursor

            resp = await self.session.get(BASE, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"chat_conversations returned HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            all_convos.extend(data)

            if len(data) < batch_size:
                break
            cursor = data[-1].get("uuid")

        return [c["uuid"] for c in all_convos]

    def _extract_messages(self, chat_messages: list[dict]) -> list[dict]:
        msgs = []
        for msg in chat_messages:
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("text")
                )
            else:
                text = str(content)
            if text.strip():
                msgs.append({"sender": sender, "text": text})
        return msgs

    async def _fetch_single_conversation(self, uuid: str, sem: asyncio.Semaphore) -> dict | None:
        async with sem:
            url = f"https://claude.ai/api/organizations/{self.org_id}/chat_conversations/{uuid}"
            params = {"tree": "True", "rendering_mode": "messages", "render_all_tools": "true", "consistency": "eventual"}
            resp = await self.session.get(url, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if isinstance(data, dict) and data.get("type") == "error":
                return None
            messages = self._extract_messages(data.get("chat_messages", []))
            if not messages:
                return None
            return {"uuid": uuid, "name": data.get("name", "(untitled)"), "messages": messages}

    async def fetch_all_conversation_details(self, uuids: list[str]) -> AsyncGenerator[dict, None]:
        sem = asyncio.Semaphore(10)
        completed = 0
        total = len(uuids)

        tasks = [asyncio.create_task(self._fetch_single_conversation(uuid, sem)) for uuid in uuids]

        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            if result:
                yield {"type": "info", "message": f"Downloading conversations {completed}/{total}", "conversation": result}
            else:
                yield {"type": "info", "message": f"Downloading conversations {completed}/{total} (skipped)"}
