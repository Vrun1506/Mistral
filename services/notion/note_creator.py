"""
Note Creator — Pipeline Step 7 → Notion Pages via MCP.

Takes the topic hierarchy from the clustering pipeline and creates
structured notes in Notion using the notion-create-pages tool.
"""

import json
from typing import Any, Optional

from mistralai import Mistral

from config import MISTRAL_API_KEY, MISTRAL_MODEL
from notion_mcp_client import call_notion_tool
from oauth_handler import TokenSet, ensure_valid_tokens


# ---------------------------------------------------------------------------
# Mistral summarizer
# ---------------------------------------------------------------------------


def summarize_subtopic(subtopic_label: str, segments: list[dict]) -> str:
    """Use Mistral to generate a 2-3 sentence summary of the subtopic content."""
    if not MISTRAL_API_KEY:
        return f"Notes on: {subtopic_label}"

    content_parts = []
    for seg in segments:
        for turn in seg["turns"]:
            prefix = "Q" if turn["role"] == "user" else "A"
            content_parts.append(f"{prefix}: {turn['content']}")

    content_block = "\n\n".join(content_parts)

    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study notes summarizer. Given Q&A conversation excerpts "
                    "on a subtopic, write a concise 2-3 sentence summary that captures "
                    "the key concepts. Be direct and informative."
                ),
            },
            {
                "role": "user",
                "content": f"Subtopic: {subtopic_label}\n\nContent:\n{content_block}",
            },
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Notion block builders
# ---------------------------------------------------------------------------


def _build_subtopic_markdown(subtopic_label: str, summary: str, segments: list[dict]) -> str:
    """Build markdown content for a subtopic page."""
    lines = []
    lines.append(f"## {subtopic_label}\n")
    lines.append(f"**Summary:** {summary}\n")
    lines.append("---\n")

    for i, seg in enumerate(segments):
        if i > 0:
            lines.append("---\n")
        lines.append(f"*Source: conversation {seg['convo_id']}, segment {seg['segment_id']}*\n")
        for turn in seg["turns"]:
            if turn["role"] == "user":
                lines.append(f"**Q:** {turn['content']}\n")
            else:
                lines.append(f"**A:** {turn['content']}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notion page creation
# ---------------------------------------------------------------------------


def _extract_page_id(mcp_result: Any) -> str:
    """
    Extract the page ID from a notion-create-pages MCP result.
    The tool returns: {"pages": [{"id": "...", "url": "...", "properties": {...}}]}
    """
    if hasattr(mcp_result, "content"):
        for block in mcp_result.content:
            if hasattr(block, "text"):
                try:
                    data = json.loads(block.text)
                    # notion-create-pages wraps results in {"pages": [...]}
                    if isinstance(data, dict) and "pages" in data:
                        pages = data["pages"]
                        if pages and isinstance(pages, list):
                            return pages[0]["id"]
                    # Bare list fallback
                    if isinstance(data, list) and data:
                        return data[0].get("id", str(data[0]))
                    if isinstance(data, dict) and "id" in data:
                        return data["id"]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
                # Plain UUID string
                text = block.text.strip()
                if len(text) == 36 and text.count("-") == 4:
                    return text

    return str(mcp_result)


def _create_page(parent_page_id: str, title: str, content: str, tokens: TokenSet) -> str:
    """
    Call notion-create-pages to create a single page under parent_page_id.
    Returns the new page's ID.
    """
    result = call_notion_tool(
        tool_name="notion-create-pages",
        arguments={
            "pages": [
                {
                    "properties": {
                        "title": title,
                    },
                    "content": content,
                }
            ],
            "parent_id": parent_page_id,
        },
        tokens=tokens,
    )
    return _extract_page_id(result)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def create_topic_notes(
    topic_hierarchy: list[dict],
    parent_page_id: str,
    tokens: Optional[TokenSet] = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    Create Notion pages for the entire topic hierarchy.

    Args:
        topic_hierarchy: Output of pipeline Step 7 (list of topics with subtopics)
        parent_page_id: Notion page ID where top-level topic pages will be created
        tokens: OAuth tokens (auto-loaded if not provided)
        dry_run: If True, print what would be created without calling Notion

    Returns:
        List of results: [{ topic, subtopic, status, page_id_or_error }]
    """
    if not tokens and not dry_run:
        tokens = ensure_valid_tokens()

    results = []

    for topic in topic_hierarchy:
        topic_label = topic["topic_label"]
        print(f"\n📁 Creating topic page: {topic_label}")

        if dry_run:
            topic_page_id = f"dry-run-{topic['cluster_id']}"
            print(f"  [DRY RUN] Would create page under {parent_page_id}")
        else:
            topic_page_id = _create_page(
                parent_page_id=parent_page_id,
                title=topic_label,
                content=f"Study notes cluster: {topic_label}",
                tokens=tokens,
            )
            print(f"  ✅ Created topic page: {topic_page_id}")

        for subtopic in topic["subtopics"]:
            subtopic_label = subtopic["subtopic_label"]
            segments = subtopic["segments"]

            print(f"  📄 Creating subtopic: {subtopic_label} ({len(segments)} segments)")

            summary = summarize_subtopic(subtopic_label, segments)
            print(f"    Summary: {summary[:80]}...")

            content = _build_subtopic_markdown(subtopic_label, summary, segments)

            if dry_run:
                print(f"    [DRY RUN] Would create page under {topic_page_id}")
                print(f"    Content preview:\n{content[:200]}...\n")
                results.append({
                    "topic": topic_label,
                    "subtopic": subtopic_label,
                    "status": "dry_run",
                    "content_length": len(content),
                })
            else:
                try:
                    sub_page_id = _create_page(
                        parent_page_id=topic_page_id,
                        title=subtopic_label,
                        content=content,
                        tokens=tokens,
                    )
                    print(f"    ✅ Created: {sub_page_id}")
                    results.append({
                        "topic": topic_label,
                        "subtopic": subtopic_label,
                        "status": "created",
                        "page_id": sub_page_id,
                    })
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    results.append({
                        "topic": topic_label,
                        "subtopic": subtopic_label,
                        "status": "error",
                        "error": str(e),
                    })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from sample_data import SAMPLE_TOPIC_HIERARCHY

    parent_id = os.environ.get("NOTION_PARENT_PAGE_ID")

    if not parent_id:
        print("Running in DRY RUN mode (no NOTION_PARENT_PAGE_ID set)\n")
        results = create_topic_notes(SAMPLE_TOPIC_HIERARCHY, "fake-parent-id", dry_run=True)
    else:
        print(f"Creating notes under Notion page: {parent_id}\n")
        results = create_topic_notes(SAMPLE_TOPIC_HIERARCHY, parent_id)

    print("\n=== Results ===")
    for r in results:
        print(f"  {r['status']:>8} | {r['topic']} -> {r['subtopic']}")