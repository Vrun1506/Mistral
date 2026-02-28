"""
Notion MCP Client wrapper.

Connects to the hosted Notion MCP server (https://mcp.notion.com) using
the OAuth tokens from oauth_handler.py. Provides a simple interface to:
  - List available MCP tools
  - Call any MCP tool by name with arguments
  - Auto-refresh tokens on 401

Uses the official MCP Python SDK with Streamable HTTP transport (fallback to SSE).
"""

import asyncio
import json
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

from config import MCP_SERVER_URL
from oauth_handler import (
    TokenSet,
    ensure_valid_tokens,
    discover_oauth_metadata,
    load_client,
    refresh_access_token,
)


class NotionMCPClient:
    """
    Wrapper around the MCP Python SDK that handles:
      - Transport selection (Streamable HTTP → SSE fallback)
      - Authentication header injection
      - Token refresh on expiry
      - Tool listing and invocation
    """

    def __init__(self, tokens: Optional[TokenSet] = None):
        self.tokens = tokens or ensure_valid_tokens()
        self.server_url = MCP_SERVER_URL

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.tokens.access_token}",
            "User-Agent": "Flashcard-Pipeline-Notion-MCP/1.0",
        }

    async def list_tools(self) -> list[dict]:
        """Return all tools exposed by the Notion MCP server."""
        try:
            return await self._list_tools_streamable_http()
        except Exception as e:
            print(f"[mcp] Streamable HTTP failed ({e}), falling back to SSE ...")
            return await self._list_tools_sse()

    async def _list_tools_streamable_http(self) -> list[dict]:
        url = f"{self.server_url}/mcp"
        async with streamablehttp_client(url, headers=self._auth_headers()) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in result.tools
                ]

    async def _list_tools_sse(self) -> list[dict]:
        url = f"{self.server_url}/sse"
        async with sse_client(url, headers=self._auth_headers()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in result.tools
                ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a named MCP tool with the given arguments.
        Returns the raw result from the MCP server.
        """
        try:
            return await self._call_tool_streamable_http(tool_name, arguments)
        except Exception as e:
            print(f"[mcp] Streamable HTTP failed ({e}), falling back to SSE ...")
            return await self._call_tool_sse(tool_name, arguments)

    async def _call_tool_streamable_http(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        url = f"{self.server_url}/mcp"
        async with streamablehttp_client(url, headers=self._auth_headers()) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)

    async def _call_tool_sse(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        url = f"{self.server_url}/sse"
        async with sse_client(url, headers=self._auth_headers()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)

    async def refresh_and_reconnect(self):
        """Refresh OAuth tokens."""
        metadata = discover_oauth_metadata()
        client = load_client()
        self.tokens = refresh_access_token(self.tokens, metadata, client)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def extract_text_from_result(result: Any) -> str:
    """Pull plain text out of an MCP tool call result."""
    if hasattr(result, "content"):
        for block in result.content:
            if hasattr(block, "text"):
                return block.text
    return str(result)


def extract_json_from_result(result: Any) -> Optional[dict]:
    """Try to parse JSON from an MCP tool call result. Returns None on failure."""
    text = extract_text_from_result(result)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Convenience: synchronous wrappers for use in non-async code
# ---------------------------------------------------------------------------


def list_notion_tools(tokens: Optional[TokenSet] = None) -> list[dict]:
    """Synchronous wrapper to list all available Notion MCP tools."""

    async def _run():
        client = NotionMCPClient(tokens)
        return await client.list_tools()

    return asyncio.run(_run())


# Alias used by page_resolver.py
list_tools = list_notion_tools


def call_notion_tool(
    tool_name: str,
    arguments: dict[str, Any],
    tokens: Optional[TokenSet] = None,
) -> Any:
    """Synchronous wrapper to call a single Notion MCP tool."""

    async def _run():
        client = NotionMCPClient(tokens)
        return await client.call_tool(tool_name, arguments)

    return asyncio.run(_run())


# Alias used by page_resolver.py
call_tool = call_notion_tool


# ---------------------------------------------------------------------------
# CLI: list tools to verify connection works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Notion MCP — Tool Discovery ===\n")
    tools = list_notion_tools()
    print(f"Found {len(tools)} tools:\n")
    for t in tools:
        print(f"  • {t['name']}")
        if t.get("description"):
            print(f"    {t['description'][:100]}")
        print()