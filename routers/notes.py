"""Notes API: generate markdown notes from pipeline results via Mistral."""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user_id
from services.pipeline.pipeline import CHAT_MODEL, get_async_client, get_llm_semaphore
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
                return [l for l in labels if l != label]
    return []


async def _generate_note_for_topic(
    label: str,
    info: dict[str, Any],
    related_labels: list[str],
) -> str:
    """Call Mistral to synthesize a topic's segments into a markdown note."""
    keywords = info.get("keywords", [])[:10]
    segments = info.get("segments", [])

    segment_texts: list[str] = []
    for seg in segments[:8]:
        convo_name = seg.get("conversation_name", "Unknown")
        messages = seg.get("messages", [])
        msg_text = "\n".join(f"  {m['sender']}: {m['text'][:500]}" for m in messages[:6])
        segment_texts.append(f"Conversation: {convo_name}\n{msg_text}")

    segments_block = "\n---\n".join(segment_texts)
    related_str = ", ".join(f"[[{r}]]" for r in related_labels[:10]) if related_labels else "None"

    prompt = (
        f"# Topic: {label}\n"
        f"**Keywords:** {', '.join(keywords)}\n"
        f"**Related topics:** {related_str}\n\n"
        f"Below are conversation excerpts grouped under this topic. "
        f"Synthesize them into a single, well-organized Markdown note.\n\n"
        f"Requirements:\n"
        f"- Start with a `# {label}` heading\n"
        f"- Add a brief summary paragraph\n"
        f"- Organize key points under `##` subheadings\n"
        f"- Use bullet points for details\n"
        f"- Use `code blocks` for any code/commands mentioned\n"
        f"- Add a `## Related` section at the end linking to: {related_str}\n"
        f"- Keep it concise — aim for 200-400 words\n\n"
        f"---\n\n{segments_block}"
    )

    client = get_async_client()
    sem = get_llm_semaphore()

    async with sem:
        for attempt in range(1, 4):
            try:
                resp = await client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": NOTES_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    await asyncio.sleep(2 * attempt)
                    continue
                raise

    return f"# {label}\n\nFailed to generate note."


async def generate_all_notes(user: UserData) -> None:
    """Fire-and-forget: generate notes for all topics and store on user object."""
    from main import _is_sensitive

    if not user.topic_groups or not user.hierarchy:
        return

    user.notes_generating = True
    try:
        tasks: list[tuple[str, asyncio.Task[str]]] = []

        for _root, subcats in user.hierarchy.items():
            for _sub, labels in subcats.items():
                for label in labels:
                    info = user.topic_groups.get(label)
                    if not info:
                        continue
                    keywords = info.get("keywords", [])[:5]
                    if _is_sensitive(label, keywords):
                        continue
                    related = _find_related_labels(label, user.hierarchy)
                    task = asyncio.create_task(_generate_note_for_topic(label, info, related))
                    tasks.append((label, task))

        for label, task in tasks:
            try:
                md = await task
            except Exception as e:
                md = f"# {label}\n\nError generating note: {e}"
            user.notes[label] = md

        print(f"  Notes generation complete: {len(user.notes)} notes.")
    except Exception:
        traceback.print_exc()
    finally:
        user.notes_generating = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/notes/{label}")
async def get_note(label: str, user_id: str = Depends(get_current_user_id)) -> dict[str, Any]:
    """Return a generated note for a single topic. Generates on demand if not cached."""
    user = get_user(user_id)
    if not user.topic_groups or not user.hierarchy:
        raise HTTPException(status_code=404, detail="No pipeline results yet")

    if label not in user.topic_groups:
        raise HTTPException(status_code=404, detail="Topic not found")

    if label in user.notes:
        return {"label": label, "markdown": user.notes[label], "cached": True}

    info = user.topic_groups[label]
    related = _find_related_labels(label, user.hierarchy)
    markdown = await _generate_note_for_topic(label, info, related)
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
