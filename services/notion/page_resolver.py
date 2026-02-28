"""
Page resolver — finds or creates the parent page in the user's Notion workspace.

After OAuth, the user's workspace is accessible via MCP tools. This module:
  1. Searches for existing pages the user granted access to
  2. Presents them as a numbered list in the terminal
  3. Lets the user pick one, or creates a new page

This replaces the need for a NOTION_PARENT_PAGE_ID environment variable.
"""

import json
from typing import Optional

from oauth_handler import TokenSet
from notion_mcp_client import (
    call_tool,
    list_tools,
    extract_text_from_result,
    extract_json_from_result,
)


# ---------------------------------------------------------------------------
# Tool name discovery
# ---------------------------------------------------------------------------

SEARCH_TOOL = "notion-search"
CREATE_PAGE_TOOL = "notion-create-pages"


def discover_tool_names(tokens: TokenSet) -> dict[str, str]:
    """
    Query the MCP server for available tools and map them to our needs.
    Returns {"search": "actual_name", "create_page": "actual_name"}.
    """
    tools = list_tools(tokens)
    tool_names = [t["name"] for t in tools]

    print(f"\n  Available MCP tools: {tool_names}\n")

    mapping = {}

    # Find search tool
    for candidate in ["notion-search", "notion_search", "search"]:
        if candidate in tool_names:
            mapping["search"] = candidate
            break
    if "search" not in mapping:
        for name in tool_names:
            if "search" in name.lower():
                mapping["search"] = name
                break

    # Find create page tool
    for candidate in [
        "notion-create-pages",
        "notion-create-page",
        "notion_create_page",
        "notion_create_a_page",
        "notion_pages_create",
        "notion-pages-create",
        "create_page",
        "createPage",
    ]:
        if candidate in tool_names:
            mapping["create_page"] = candidate
            break
    if "create_page" not in mapping:
        for name in tool_names:
            if "create" in name.lower() and "page" in name.lower():
                mapping["create_page"] = name
                break

    if "search" not in mapping:
        raise RuntimeError(
            "Could not find a search tool on the Notion MCP server.\n"
            f"  Available tools: {tool_names}"
        )
    if "create_page" not in mapping:
        raise RuntimeError(
            "Could not find a create-page tool on the Notion MCP server.\n"
            f"  Available tools: {tool_names}"
        )

    print(f"  Mapped -> search='{mapping['search']}', create_page='{mapping['create_page']}'\n")
    return mapping


# ---------------------------------------------------------------------------
# Search & select
# ---------------------------------------------------------------------------


def search_workspace(query: str, tokens: TokenSet, tool_name: str = SEARCH_TOOL) -> list[dict]:
    """
    Search the user's Notion workspace via MCP.
    Returns a list of pages: [{"id": "...", "title": "..."}, ...]
    """
    result = call_tool(tool_name, {"query": query}, tokens=tokens)
    text = extract_text_from_result(result)

    pages = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                pages.append({
                    "id": item.get("id", item.get("page_id", "")),
                    "title": item.get("title", item.get("name", "Untitled")),
                })
        elif isinstance(data, dict):
            results = data.get("results", data.get("pages", [data]))
            for item in results:
                title = "Untitled"
                props = item.get("properties", {})
                if "title" in props:
                    title_arr = props["title"].get("title", [])
                    if title_arr:
                        title = title_arr[0].get("plain_text", "Untitled")
                elif "Name" in props:
                    title_arr = props["Name"].get("title", [])
                    if title_arr:
                        title = title_arr[0].get("plain_text", "Untitled")
                else:
                    title = item.get("title", item.get("name", "Untitled"))
                pages.append({"id": item.get("id", ""), "title": title})
    except (json.JSONDecodeError, TypeError):
        print(f"\n  Raw search result:\n{text[:500]}")

    return pages


def create_new_page(title: str, tokens: TokenSet, tool_name: str = CREATE_PAGE_TOOL) -> str:
    """
    Create a new top-level page using notion-create-pages.
    The tool expects a 'pages' list; we create one page and return its ID.
    """
    result = call_tool(
        tool_name,
        {
            "pages": [
                {
                    "properties": {
                        "title": title
                    },
                    "content": "Auto-created by Flashcard Pipeline",
                }
            ]
        },
        tokens=tokens,
    )

    text = extract_text_from_result(result)

    # Try JSON first
    data = extract_json_from_result(result)
    if data:
        # Response might be a list of created pages or a single dict
        if isinstance(data, list) and data:
            return data[0].get("id", data[0].get("page_id", str(data[0])))
        if isinstance(data, dict):
            return data.get("id", data.get("page_id", str(data)))

    # Fallback: return the raw text (may be a page URL or ID)
    return text.strip()


# ---------------------------------------------------------------------------
# Interactive resolver
# ---------------------------------------------------------------------------


def resolve_parent_page(tokens: TokenSet) -> str:
    """
    Interactive terminal flow to find or create the parent page.
    Returns the Notion page ID.
    """
    print("\n--- Where should we create your study notes? ---\n")

    tool_map = discover_tool_names(tokens)
    search_name = tool_map["search"]
    create_name = tool_map["create_page"]

    print("  Searching your workspace ...")
    pages = search_workspace(" ", tokens, tool_name=search_name)

    if pages:
        print(f"\n  Found {len(pages)} pages:\n")
        for i, page in enumerate(pages, 1):
            print(f"    [{i}] {page['title'] or 'Untitled'}")
        print(f"    [0] Create a new page")
        print(f"    [m] Enter a page ID manually")
        print()

        choice = input("  Select (number): ").strip().lower()

        if choice == "m":
            return input("  Paste Notion page ID: ").strip()
        elif choice == "0" or not choice:
            title = input("  New page title [Study Notes]: ").strip() or "Study Notes"
            page_id = create_new_page(f"📚 {title}", tokens, tool_name=create_name)
            print(f"  Created: {page_id}")
            return page_id
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(pages):
                selected = pages[idx]
                print(f"  Selected: {selected['title']} ({selected['id']})")
                return selected["id"]
            else:
                print("  Invalid selection. Creating new page.")
                page_id = create_new_page("📚 Study Notes", tokens, tool_name=create_name)
                return page_id
    else:
        print("  No pages found (or search returned unparseable results).")
        print()
        choice = input("  [c] Create new page  |  [m] Enter page ID manually: ").strip().lower()

        if choice == "m":
            return input("  Paste Notion page ID: ").strip()
        else:
            title = input("  New page title [Study Notes]: ").strip() or "Study Notes"
            page_id = create_new_page(f"📚 {title}", tokens, tool_name=create_name)
            print(f"  Created: {page_id}")
            return page_id


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from oauth_handler import ensure_valid_tokens

    tokens = ensure_valid_tokens()
    page_id = resolve_parent_page(tokens)
    print(f"\nResolved parent page ID: {page_id}")