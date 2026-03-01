"""Flashcard generation via Mistral on NVIDIA NIM.

Reuses the async client + semaphore from services/pipeline/pipeline.py
so no extra API key or config is needed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from services.pipeline.pipeline import CHAT_MODEL, _get_llm_sem, get_async_client

FLASHCARD_SYSTEM_PROMPT = """\
You are an expert educational assistant. Extract concepts and create flashcards.

CLOZE DELETION RULES (CRITICAL):
- Use the format: {{c1::hidden text}}.
- Every hidden part MUST increment the index: {{c1::first item}}, {{c2::second item}}, etc.
- DO NOT reuse the same index for two different blanks in the same card.
- The 'answer' field for a Cloze card must list the hidden values.

CARD TYPES:
- Basic: Standard Q&A.  type = "Basic"
- Cloze: Fill-in-the-blank with {{cN::...}} syntax.  type = "Cloze"
- Multiple Choice: Question text followed by options A, B, C, D on separate lines.
  Answer = correct letter + text, e.g. "A: Mitochondria".  type = "Multiple Choice"
- True/False: Statement ending with 'True or False?'.
  Answer is exactly "True" or "False".  type = "True/False"

Return ONLY a JSON object:
{
  "cards": [
    {"type": "Basic", "question": "...", "answer": "..."},
    ...
  ]
}

Aim for 8-15 high-quality cards. Mix card types. Prioritise understanding.
"""


async def generate_flashcards_for_topic(
    label: str,
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Generate flashcard dicts from pipeline topic segments."""
    context_parts = [f"TOPIC: {label}\n"]
    for seg in segments[:5]:
        conv_name = seg.get("conversation_name", "unknown")
        context_parts.append(f"\n--- {conv_name} ---")
        for msg in seg.get("messages", [])[:10]:
            sender = msg.get("sender", "unknown").upper()
            text = msg.get("text", "")[:500]
            context_parts.append(f"{sender}: {text}")
    context_text = "\n".join(context_parts)

    client = get_async_client()
    sem = _get_llm_sem()

    async with sem:
        for attempt in range(1, 4):
            try:
                resp = await client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": FLASHCARD_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Create flashcards for:\n\n{context_text}"},
                    ],
                    max_tokens=3000,
                    temperature=0.4,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if not raw:
                    raise ValueError("Empty response from Mistral")
                data = json.loads(raw)
                cards = data.get("cards", [])
                validated = [
                    {"type": c["type"], "question": c["question"], "answer": c["answer"]}
                    for c in cards
                    if all(k in c for k in ("type", "question", "answer"))
                ]
                if not validated:
                    raise ValueError("No valid cards in response")
                return validated
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    await asyncio.sleep(2 * attempt)
                    continue
                if attempt == 3:
                    raise
    raise RuntimeError("Flashcard generation failed after 3 attempts")


async def generate_flashcards_from_text(
    label: str,
    content: str,
) -> list[dict[str, Any]]:
    """Generate flashcards from raw user-pasted text/notes."""
    if not content.strip():
        segments = [
            {
                "conversation_name": "notes",
                "messages": [{"sender": "user", "text": f"Create flashcards about: {label}"}],
            }
        ]
    else:
        segments = [
            {
                "conversation_name": "notes",
                "messages": [{"sender": "user", "text": content}],
            }
        ]
    return await generate_flashcards_for_topic(label, segments)
