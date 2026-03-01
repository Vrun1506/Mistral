"""Notes API: generate markdown notes from pipeline results via Mistral."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user_id
from services.discord import fire_discord
from services.notes_agent import run_notes_agent_async
from store import UserData, get_user

router = APIRouter(prefix="/api", tags=["notes"])

NOTES_SYSTEM_PROMPT = (
    "You are an expert note-taker. You transform raw conversation excerpts into "
    "clean, minimal Markdown notes for Obsidian.\n\n"
    "STRICT FORMATTING RULES:\n"
    "- Use # for the title only\n"
    "- Use ## for section headings only\n"
    "- Use **bold** for emphasis on key terms\n"
    "- Use plain text for everything else\n"
    "- DO NOT use bullet points, lists, tables, code blocks, or --- dividers\n"
    "- DO NOT use ###, ####, or deeper headings\n"
    "- Write in short paragraphs, not lists\n"
    "- Keep it conversational and readable, like clean study notes\n"
    "- Use [[wiki-links]] when referencing other topic labels\n\n"
    "TOOL USAGE:\n"
    "You have access to exa_search. Use it to find real information that ENRICHES "
    "your notes — facts, explanations, context, definitions, or examples.\n"
    "When you get search results, READ the snippets and weave the useful information "
    "directly into your note paragraphs as if you already knew it. Integrate the "
    "knowledge naturally into the text. Do NOT just list links at the end.\n"
    "If a source is particularly useful, you can inline-link it like "
    "[topic](url) within the paragraph where you used its information.\n"
    "The final output must be a complete, self-contained note — not a link dump."
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
        f"- Start with # {root_name} as the title\n"
        f"- Write a brief summary paragraph\n"
        f"- Use ## for each subcategory heading\n"
        f"- Write in short paragraphs, NOT bullet points or lists\n"
        f"- Use **bold** for key terms only\n"
        f"- NO code blocks, NO tables, NO ---, NO ### or deeper headings\n"
        f"- End with ## Related followed by a paragraph linking to: {related_str}\n"
        f"- Aim for 300-600 words\n\n"
        f"---\n\n{topics_block}"
    )

    return await run_notes_agent_async(NOTES_SYSTEM_PROMPT, prompt, label=root_name)


async def generate_all_notes(user: UserData) -> None:
    """Fire-and-forget: generate one note per root category and store on user object."""
    if not user.topic_groups or not user.hierarchy:
        return

    user.notes_generating = True
    root_names = list(user.hierarchy.keys())
    fire_discord(f"📝 Starting notes generation: **{len(root_names)}** root categories")
    try:
        tasks: list[tuple[str, asyncio.Task[str]]] = []

        for root_name, subcats in user.hierarchy.items():
            other_roots = [r for r in root_names if r != root_name]
            task = asyncio.create_task(_generate_note_for_root(root_name, subcats, user.topic_groups, other_roots))
            tasks.append((root_name, task))

        for root_name, task in tasks:
            try:
                md = await task
            except Exception:
                logger.exception("Note generation failed for %s", root_name)
                fire_discord(f"❌ Note failed [{root_name}]")
                md = f"# {root_name}\n\nError generating note. Please try again."
            user.notes[root_name] = md

        fire_discord(f"🎉 All notes complete: **{len(user.notes)}** notes generated")
    except Exception:
        fire_discord(f"💀 generate_all_notes crashed: `{traceback.format_exc()[-500:]}`")
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

    fire_discord(f"📝 On-demand note requested: **{label}**")
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
