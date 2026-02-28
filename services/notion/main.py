"""
Main entry point for the Notion MCP PoC.

Runs the full flow:
  1. Ensure OAuth tokens are valid (interactive flow if needed)
  2. Connect to Notion MCP and list available tools
  3. Take sample pipeline output (or real data) and create notes in Notion

Usage:
  python main.py                    # Full flow with sample data
  python main.py --dry-run          # Preview without creating Notion pages
  python main.py --list-tools       # Just list available MCP tools
  python main.py --input data.json  # Use a custom JSON topic hierarchy
"""

import argparse
import json

from .note_creator import create_topic_notes
from .notion_mcp_client import list_notion_tools
from .oauth_handler import ensure_valid_tokens
from .page_resolver import resolve_parent_page
from .sample_data import SAMPLE_TOPIC_HIERARCHY


def main() -> None:
    parser = argparse.ArgumentParser(description="Notion MCP PoC — Create study notes from pipeline output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without calling Notion",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available Notion MCP tools and exit",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file with topic hierarchy (default: sample data)",
    )
    args = parser.parse_args()

    # --- Step 1: Authentication ---
    print("=" * 60)
    print("STEP 1: Authentication")
    print("=" * 60)

    if args.dry_run:
        print("Dry run mode — skipping OAuth\n")
        tokens = None
    else:
        tokens = ensure_valid_tokens()
        print()

    # --- Step 2: Tool discovery (optional) ---
    if args.list_tools:
        print("=" * 60)
        print("STEP 2: MCP Tool Discovery")
        print("=" * 60)
        tools = list_notion_tools(tokens)
        print(f"\nFound {len(tools)} tools:\n")
        for t in tools:
            print(f"  {t['name']}")
            if t.get("description"):
                desc = t["description"][:120]
                print(f"    {desc}")
            print()
        return

    # --- Step 3: Load topic hierarchy ---
    print("=" * 60)
    print("STEP 2: Load Topic Hierarchy")
    print("=" * 60)

    if args.input:
        with open(args.input) as f:
            hierarchy = json.load(f)
        print(f"Loaded {len(hierarchy)} topics from {args.input}\n")
    else:
        hierarchy = SAMPLE_TOPIC_HIERARCHY
        print(f"Using sample data: {len(hierarchy)} topics\n")

    # Show preview
    for topic in hierarchy:
        subtopic_count = len(topic["subtopics"])
        segment_count = sum(len(st["segments"]) for st in topic["subtopics"])
        print(f"  📁 {topic['topic_label']} ({subtopic_count} subtopics, {segment_count} segments)")
        for st in topic["subtopics"]:
            print(f"    📄 {st['subtopic_label']} ({len(st['segments'])} segments)")
    print()

    # --- Step 4: Resolve parent page ---
    print("=" * 60)
    print("STEP 3: Select Destination Page")
    print("=" * 60)

    if args.dry_run:
        parent_page_id = "dry-run-parent"
        print("Dry run — skipping page selection\n")
    else:
        # Interactive search-and-select via MCP (no env var needed)
        assert tokens is not None
        parent_page_id = resolve_parent_page(tokens)
        print(f"\nUsing parent page: {parent_page_id}\n")

    # --- Step 5: Create notes ---
    print("=" * 60)
    print("STEP 4: Create Notion Notes")
    print("=" * 60)

    results = create_topic_notes(
        topic_hierarchy=hierarchy,
        parent_page_id=parent_page_id,
        tokens=tokens,
        dry_run=args.dry_run,
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    created = sum(1 for r in results if r["status"] == "created")
    errors = sum(1 for r in results if r["status"] == "error")
    dry = sum(1 for r in results if r["status"] == "dry_run")

    for r in results:
        icon = {"created": "✅", "error": "❌", "dry_run": "🔸"}.get(r["status"], "?")
        print(f"  {icon} {r['topic']} → {r['subtopic']} [{r['status']}]")

    print(f"\nTotal: {len(results)} | Created: {created} | Errors: {errors} | Dry run: {dry}")


if __name__ == "__main__":
    main()
