"""Fire-and-forget Discord webhook alerts for observability."""

from __future__ import annotations

import asyncio

import httpx

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1477640922983305281/OAXepDGfTC46zbzOKMVxJ-jTgSjmjuL5oY5SRKsHhkxJsk4LMaqecTe7OxhG2lBXFlHj"

_background_tasks: set[asyncio.Task] = set()


def fire_discord(msg: str) -> None:
    """Fire-and-forget Discord webhook — no await needed."""
    async def _send():
        try:
            async with httpx.AsyncClient() as client:
                await client.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        except Exception:
            pass

    task = asyncio.create_task(_send())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
