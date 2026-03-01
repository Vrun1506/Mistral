"""Notes API: generate markdown notes from pipeline results via Mistral."""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user_id
from services.pipeline.pipeline import CHAT_MODEL, _get_llm_sem, get_async_client
from store import UserData, get_user

router = APIRouter(prefix="/api", tags=["notes"])

NOTES_SYSTEM_PROMPT = (
    "You are an expert note-taker. You transform raw conversation excerpts into "
    "clean, well-structured Markdown notes suitable for a knowledge base like Obsidian. "
    "Use headings, bullet points, and code blocks where appropriate. "
    "Be concise but preserve all key information and insights. "
    "Add wiki-style [[links]] to other topic labels when they are referenced."
)


def _find_related_labels(label: str, hierarchy: dict[str, Any]) -> list[str]:
    """Find sibling topic labels within the same subcategory."""
    for _root, subcats in hierarchy.items():
        for _sub, labels in subcats.items():
            if label in labels:
                return [lb for lb in labels if lb != label]
    return []


async def _generate_note_for_root(
    root_name: str,
    subcats: dict[str, list[str]],
    topic_groups: dict[str, Any],
    other_roots: list[str],
) -> str:
    """Call Mistral to synthesize a root category into one consolidated markdown note."""
    topic_summaries: list[str] = []

    for sub_name, labels in subcats.items():
        for label in labels:
            info = topic_groups.get(label, {})
            keywords = info.get("keywords", [])[:5]
            segments = info.get("segments", [])
            sample_texts: list[str] = []
            for seg in segments[:3]:
                messages = seg.get("messages", [])
                sample_texts.append(" ".join(m["text"][:200] for m in messages[:3]))
            sample = " | ".join(sample_texts)[:600]
            topic_summaries.append(
                f"### {label} (under {sub_name})\nKeywords: {', '.join(keywords)}\nSample: {sample}\n"
            )

    topics_block = "\n".join(topic_summaries[:30])
    related_str = ", ".join(f"[[{r}]]" for r in other_roots) if other_roots else "None"

    prompt = (
        f"# Category: {root_name}\n"
        f"**Subcategories:** {', '.join(subcats.keys())}\n"
        f"**Related categories:** {related_str}\n\n"
        f"Below are topics grouped under this category with keywords and sample excerpts. "
        f"Synthesize them into a single, comprehensive Markdown note.\n\n"
        f"Requirements:\n"
        f"- Start with a `# {root_name}` heading\n"
        f"- Add a brief summary paragraph of what this category covers\n"
        f"- Use `## Subcategory Name` headings for each subcategory\n"
        f"- Under each subcategory, summarize the key topics with bullet points\n"
        f"- Use `code blocks` for any code/commands mentioned\n"
        f"- Add a `## Related` section at the end linking to: {related_str}\n"
        f"- Aim for 300-600 words\n\n"
        f"---\n\n{topics_block}"
    )

    client = get_async_client()
    sem = _get_llm_sem()

    async with sem:
        for attempt in range(1, 4):
            try:
                resp = await client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": NOTES_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                )
                return str(resp.choices[0].message.content or "").strip()
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    await asyncio.sleep(2 * attempt)
                    continue
                raise

    return f"# {root_name}\n\nFailed to generate note."


async def generate_all_notes(user: UserData) -> None:
    """Fire-and-forget: generate one note per root category and store on user object."""
    if not user.topic_groups or not user.hierarchy:
        return

    user.notes_generating = True
    try:
        root_names = list(user.hierarchy.keys())
        tasks: list[tuple[str, asyncio.Task[str]]] = []

        for root_name, subcats in user.hierarchy.items():
            other_roots = [r for r in root_names if r != root_name]
            task = asyncio.create_task(_generate_note_for_root(root_name, subcats, user.topic_groups, other_roots))
            tasks.append((root_name, task))

        for root_name, task in tasks:
            try:
                md = await task
            except Exception as e:
                md = f"# {root_name}\n\nError generating note: {e}"
            user.notes[root_name] = md

        print(f"  Notes generation complete: {len(user.notes)} root category notes.")
    except Exception:
        traceback.print_exc()
    finally:
        user.notes_generating = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/notes/{label}")
async def get_note(label: str, user_id: str = Depends(get_current_user_id)) -> dict[str, Any]:
    """Return a generated note for a root category. Generates on demand if not cached."""
    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        raise HTTPException(status_code=404, detail="No pipeline results yet")

    if label not in user.hierarchy:
        raise HTTPException(status_code=404, detail="Root category not found")

    if label in user.notes:
        return {"label": label, "markdown": user.notes[label], "cached": True}

    root_names = list(user.hierarchy.keys())
    other_roots = [r for r in root_names if r != label]
    markdown = await _generate_note_for_root(label, user.hierarchy[label], user.topic_groups, other_roots)
    user.notes[label] = markdown

    return {"label": label, "markdown": markdown, "cached": False}


@router.get("/notes")
async def get_all_notes(user_id: str = Depends(get_current_user_id)) -> dict[str, Any]:
    """Return all generated notes. If still generating, returns what's available so far."""
    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        raise HTTPException(status_code=404, detail="No pipeline results yet")

    return {
        "notes": user.notes,
        "count": len(user.notes),
        "generating": user.notes_generating,
    }
