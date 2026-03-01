"""Flashcards API router.

Endpoints:
    GET    /api/decks                       → list user's decks
    GET    /api/decks/{deck_id}             → get deck + cards
    POST   /api/decks                       → create deck from raw text/notes
    POST   /api/decks/from-topic/{label}    → generate deck from pipeline topic
    DELETE /api/decks/{deck_id}             → delete a deck
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from auth import get_current_user_id
from services.flashcard_generator import (
    generate_flashcards_for_topic,
    generate_flashcards_from_text,
)
from services.supabase_generator import (
    delete_deck,
    get_deck_with_cards,
    list_decks,
    save_deck,
)
from store import get_user

router = APIRouter(prefix="/api", tags=["flashcards"])


class CreateDeckRequest(BaseModel):
    label: str
    content: str = ""


@router.get("/decks")
async def get_decks(user_id: str = Depends(get_current_user_id)) -> dict[str, Any]:
    """List all decks for the current user."""
    decks = list_decks(user_id)
    return {"decks": decks}


@router.get("/decks/{deck_id}")
async def get_deck_detail(
    deck_id: int,
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Get a single deck with its cards."""
    data = get_deck_with_cards(deck_id)
    if not data or not data["deck"]:
        raise HTTPException(status_code=404, detail="Deck not found")
    if data["deck"]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not your deck")
    return data


@router.post("/decks")
async def create_deck_from_text(
    req: CreateDeckRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Create a new deck by generating cards from raw text or a topic label."""
    if not req.label.strip():
        raise HTTPException(status_code=400, detail="Label is required")
    try:
        cards = await generate_flashcards_from_text(req.label.strip(), req.content.strip())
    except Exception:
        logger.exception("Flashcard generation failed for label=%s", req.label)
        raise HTTPException(status_code=500, detail="Card generation failed") from None
    deck_id = save_deck(user_id, req.label.strip(), cards)
    return {"deck_id": deck_id, "card_count": len(cards)}


@router.post("/decks/from-topic/{label}")
async def create_deck_from_topic(
    label: str,
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Generate a flashcard deck from an existing pipeline topic."""
    user = get_user(user_id)
    if not user.topic_groups:
        raise HTTPException(status_code=404, detail="No pipeline results yet. Run the pipeline first.")
    topic_info = user.topic_groups.get(label)
    if not topic_info:
        raise HTTPException(status_code=404, detail="Topic not found")
    segments = topic_info.get("segments", [])
    if not segments:
        raise HTTPException(status_code=400, detail="Topic has no segments")
    try:
        cards = await generate_flashcards_for_topic(label, segments)
    except Exception:
        logger.exception("Flashcard generation failed for topic=%s", label)
        raise HTTPException(status_code=500, detail="Card generation failed") from None
    deck_id = save_deck(user_id, label, cards)
    return {"deck_id": deck_id, "topic_label": label, "card_count": len(cards)}


@router.delete("/decks/{deck_id}")
async def remove_deck(
    deck_id: int,
    user_id: str = Depends(get_current_user_id),
) -> dict[str, str]:
    """Delete a deck and all its cards."""
    data = get_deck_with_cards(deck_id)
    if not data or not data["deck"]:
        raise HTTPException(status_code=404, detail="Deck not found")
    if data["deck"]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not your deck")
    delete_deck(deck_id)
    return {"status": "deleted"}
