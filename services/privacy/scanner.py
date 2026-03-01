"""Privacy pre-scan: GLiNER-PII local entity detection.

Runs *before* any cloud embedding/LLM calls so conversations containing
personal information can be excluded from processing entirely.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from models.schemas import PipelinePhase, PrivacyCategory, ScanResult

# GLiNER-PII entity labels and their human-friendly names
PII_LABELS: list[str] = [
    "PERSON",
    "EMAIL",
    "PHONE_NUMBER",
    "SSN",
    "CREDIT_CARD",
    "ADDRESS",
    "MEDICAL_CONDITION",
    "DATE_OF_BIRTH",
    "PASSPORT_NUMBER",
    "BANK_ACCOUNT",
    "IP_ADDRESS",
    "PASSWORD",
]

PII_FRIENDLY_NAMES: dict[str, str] = {
    "PERSON": "Person Names",
    "EMAIL": "Email Addresses",
    "PHONE_NUMBER": "Phone Numbers",
    "SSN": "Social Security Numbers",
    "CREDIT_CARD": "Credit Card Numbers",
    "ADDRESS": "Physical Addresses",
    "MEDICAL_CONDITION": "Medical Conditions",
    "DATE_OF_BIRTH": "Dates of Birth",
    "PASSPORT_NUMBER": "Passport Numbers",
    "BANK_ACCOUNT": "Bank Account Numbers",
    "IP_ADDRESS": "IP Addresses",
    "PASSWORD": "Passwords",
}

# ---------------------------------------------------------------------------
# GLiNER model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_gliner_model = None
_gliner_lock = asyncio.Lock()

_gliner_available: bool | None = None


def _check_gliner_available() -> bool:
    """Check if gliner + torch are installed."""
    global _gliner_available
    if _gliner_available is None:
        try:
            import gliner  # noqa: F401

            _gliner_available = True
        except ImportError:
            _gliner_available = False
            print("[privacy] gliner not installed — PII scanning disabled")
    return _gliner_available


async def _get_gliner_model() -> Any:
    """Lazily load the GLiNER-PII model (thread-safe, uses GPU if available)."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    if not _check_gliner_available():
        return None

    async with _gliner_lock:
        if _gliner_model is not None:
            return _gliner_model

        def _load() -> Any:
            import torch
            from gliner import GLiNER

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[privacy] Loading GLiNER-PII model (nvidia/gliner-PII) on {device}...")
            model = GLiNER.from_pretrained("nvidia/gliner-PII")
            if device == "cuda":
                model = model.to(device)
            print("[privacy] GLiNER-PII model loaded.")
            return model

        _gliner_model = await asyncio.to_thread(_load)
        return _gliner_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_messages(messages: list[dict[str, str]], n: int = 3) -> str:
    """Concatenate first n + last n messages (capped at 500 chars each).

    This gives a representative sample of the conversation without
    sending the entire text to the scanner.
    """
    selected = messages if len(messages) <= n * 2 else messages[:n] + messages[-n:]

    parts: list[str] = []
    for msg in selected:
        text = msg.get("text", "")[:500]
        sender = msg.get("sender", "unknown")
        parts.append(f"[{sender}] {text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# GLiNER-PII scanning (local)
# ---------------------------------------------------------------------------


async def scan_conversation_pii(
    text: str,
    model: Any,
    sem: asyncio.Semaphore,
) -> set[str]:
    """Run GLiNER-PII entity detection on text. Returns set of detected entity labels."""
    async with sem:

        def _predict() -> set[str]:
            entities = model.predict_entities(text, PII_LABELS, threshold=0.3)
            return {e["label"] for e in entities}

        return await asyncio.to_thread(_predict)


async def _batch_scan_pii(
    texts: dict[str, str],  # uuid -> sampled text
    model: Any,
    sem: asyncio.Semaphore,
    on_progress: Callable[..., None] | None = None,
) -> dict[str, set[str]]:
    """Scan all conversations for PII. Returns uuid -> set of PII labels."""
    results: dict[str, set[str]] = {}
    total = len(texts)
    done = 0

    async def _scan_one(uuid: str, text: str) -> None:
        nonlocal done
        labels = await scan_conversation_pii(text, model, sem)
        results[uuid] = labels
        done += 1
        if on_progress and total > 0:
            on_progress(
                PipelinePhase.scanning,
                f"PII scan: {done}/{total}",
                done / total,
            )

    await asyncio.gather(*[_scan_one(uid, txt) for uid, txt in texts.items()])
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def scan_conversations(
    conversations: list[dict[str, Any]],
    on_progress: Callable[..., None] | None = None,
) -> ScanResult:
    """Run GLiNER-PII privacy scan.

    Returns a ScanResult with per-category conversation counts.
    """
    if on_progress:
        on_progress(PipelinePhase.scanning, "Loading privacy scanner...", 0.0)

    # Sample messages for each conversation
    texts: dict[str, str] = {}
    for conv in conversations:
        uuid = conv["uuid"]
        texts[uuid] = _sample_messages(conv.get("messages", []))

    # Run GLiNER-PII (local, GPU if available) — skipped if gliner not installed
    gliner_model = await _get_gliner_model()
    if gliner_model is not None:
        if on_progress:
            on_progress(PipelinePhase.scanning, "Running PII detection...", 0.05)
        pii_sem = asyncio.Semaphore(10)
        pii_results = await _batch_scan_pii(texts, gliner_model, pii_sem, on_progress)
    else:
        if on_progress:
            on_progress(PipelinePhase.scanning, "PII scanning unavailable (gliner not installed)", 0.5)
        pii_results = {}

    # Build conversation_flags and categories from PII results
    conversation_flags: dict[str, list[str]] = defaultdict(list)
    category_uuids: dict[str, set[str]] = defaultdict(set)

    for uuid, pii_labels in pii_results.items():
        for label in pii_labels:
            cat_id = f"pii:{label}"
            conversation_flags[uuid].append(cat_id)
            category_uuids[cat_id].add(uuid)

    # Build PrivacyCategory list
    categories: list[PrivacyCategory] = []

    for label in PII_LABELS:
        cat_id = f"pii:{label}"
        if cat_id in category_uuids:
            categories.append(
                PrivacyCategory(
                    id=cat_id,
                    name=PII_FRIENDLY_NAMES.get(label, label),
                    source="gliner",
                    conversation_count=len(category_uuids[cat_id]),
                    conversation_uuids=sorted(category_uuids[cat_id]),
                )
            )

    flagged_uuids = {uid for uid, flags in conversation_flags.items() if flags}

    result = ScanResult(
        total_conversations=len(conversations),
        flagged_conversations=len(flagged_uuids),
        categories=categories,
        conversation_flags=dict(conversation_flags),
    )

    if on_progress:
        on_progress(
            PipelinePhase.scanning,
            f"Scan complete: {len(flagged_uuids)} flagged in {len(categories)} categories",
            1.0,
        )

    return result
