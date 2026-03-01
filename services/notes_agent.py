"""Mistral + Exa agentic loop for enriched note generation.

3-iteration max: Mistral generates notes, optionally calls exa_search
to enrich with relevant sources, then returns final markdown.
"""

from __future__ import annotations

import json
import os

from exa_py import AsyncExa
from mistralai import Mistral

from services.discord import fire_discord

MODEL = "mistral-large-latest"
MAX_ITERATIONS = 3

_mistral_client: Mistral | None = None
_exa_client: AsyncExa | None = None


def _get_mistral() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return _mistral_client


def _get_exa() -> AsyncExa:
    global _exa_client
    if _exa_client is None:
        _exa_client = AsyncExa(api_key=os.environ["EXA_API_KEY"])
    return _exa_client


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": (
                "Search the web for current information, articles, documentation, "
                "or references related to a topic. Returns titles, URLs, and text snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant web content.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


async def _execute_exa_search(query: str) -> str:
    """Run an Exa search and return formatted JSON results."""
    try:
        exa = _get_exa()
        results = await exa.search(
            query=query,
            type="auto",
            num_results=5,
            text={"max_characters": 2000},
        )
        if not results.results:
            return json.dumps({"results": [], "message": "No results found."})

        formatted = []
        for r in results.results:
            formatted.append({
                "title": r.title,
                "url": r.url,
                "snippet": (r.text[:500] if r.text else ""),
            })
        return json.dumps({"results": formatted})
    except Exception as e:
        return json.dumps({"error": str(e)})


_TOOL_DISPATCH = {
    "exa_search": _execute_exa_search,
}


async def run_notes_agent_async(
    system_prompt: str, user_prompt: str, label: str = "",
) -> str:
    """Run the Mistral + Exa agent loop (up to MAX_ITERATIONS).

    Returns the final markdown string from the assistant.
    """
    tag = label or "unknown"
    fire_discord(f"🟢 Agent start: **{tag}**")

    client = _get_mistral()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    assistant_message = None

    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            fire_discord(f"🔁 Iteration {iteration}/{MAX_ITERATIONS}: **{tag}**")

            response = await client.chat.complete_async(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                fire_discord(
                    f"💬 No tool calls, final answer: **{tag}** "
                    f"(iteration {iteration})"
                )
                break

            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                fn_params = json.loads(tool_call.function.arguments)

                fire_discord(f"🔍 Tool call [{tag}]: `{fn_name}({fn_params})`")

                fn = _TOOL_DISPATCH.get(fn_name)
                if fn is None:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                    fire_discord(f"⚠️ Unknown tool [{tag}]: {fn_name}")
                else:
                    result = await fn(**fn_params)
                    result_parsed = json.loads(result)
                    n_results = len(result_parsed.get("results", []))
                    fire_discord(f"📊 Exa returned {n_results} results [{tag}]")

                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": result,
                    "tool_call_id": tool_call.id,
                })

        content = assistant_message.content if assistant_message else ""
        fire_discord(
            f"✅ Agent done: **{tag}** — {len(content)} chars, "
            f"{iteration} iteration(s)"
        )
        return content

    except Exception as exc:
        fire_discord(f"❌ Agent error [{tag}]: `{exc}`")
        raise
