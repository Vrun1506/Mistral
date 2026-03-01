"""Rate-limited Discord webhook alerts for observability."""

from __future__ import annotations

import asyncio

import httpx

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1477640922983305281/OAXepDGfTC46zbzOKMVxJ-jTgSjmjuL5oY5SRKsHhkxJsk4LMaqecTe7OxhG2lBXFlHj"

_queue: asyncio.Queue[str] | None = None
_worker_task: asyncio.Task | None = None


async def _worker() -> None:
    """Drain the queue, sending one message per second."""
    async with httpx.AsyncClient() as client:
        while True:
            msg = await _queue.get()
            try:
                resp = await client.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                if resp.status_code == 429:
                    retry = resp.json().get("retry_after", 1)
                    await asyncio.sleep(retry)
                    await client.post(DISCORD_WEBHOOK_URL, json={"content": msg})
                elif resp.status_code >= 400:
                    print(f"[discord] webhook returned {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                print(f"[discord] webhook failed: {e}")
            finally:
                _queue.task_done()
            await asyncio.sleep(0.6)


def fire_discord(msg: str) -> None:
    """Queue a Discord message. Starts the background worker on first call."""
    global _queue, _worker_task
    if _queue is None:
        _queue = asyncio.Queue()
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker())
    _queue.put_nowait(msg)
