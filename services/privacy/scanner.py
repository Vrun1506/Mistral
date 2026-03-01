"""Privacy pre-scan: GLiNER-PII (local) + NemoGuard Content Safety (NIM API).

Runs *before* any cloud embedding/LLM calls so sensitive conversations
can be excluded from processing entirely.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from models.schemas import PipelinePhase, PrivacyCategory, ScanResult

load_dotenv()

NIM_BASE = "https://integrate.api.nvidia.com/v1"

# NemoGuard safety category codes
NEMOGUARD_CATEGORIES: dict[str, str] = {
    "S1": "Violence",
    "S2": "Sexual Content",
    "S3": "Criminal Planning",
    "S4": "Weapons",
    "S5": "Controlled Substances",
    "S6": "Self-Harm",
    "S7": "Hate Speech",
    "S8": "Harassment",
    "S9": "Threats",
    "S10": "Child Safety",
    "S11": "Deception",
    "S12": "Malware/Hacking",
    "S13": "Privacy Violation",
    "S14": "Defamation",
    "S15": "Political Lobbying",
    "S16": "Copyright Violation",
    "S17": "Financial Advice",
    "S18": "Medical Advice",
    "S19": "Legal Advice",
    "S20": "Government Decisions",
    "S21": "Professional Misconduct",
    "S22": "Discrimination",
    "S23": "Misinformation",
}

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


async def _get_gliner_model() -> Any:
    """Lazily load the GLiNER-PII model (thread-safe, runs on CPU)."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

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
                done / total * 0.5,  # PII is first half of scan progress
            )

    await asyncio.gather(*[_scan_one(uid, txt) for uid, txt in texts.items()])
    return results


# ---------------------------------------------------------------------------
# NemoGuard Content Safety scanning (NIM API)
# ---------------------------------------------------------------------------


def _get_nemoguard_client() -> AsyncOpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY in .env file")
        sys.exit(1)
    return AsyncOpenAI(base_url=NIM_BASE, api_key=api_key)


async def scan_conversation_nemoguard(
    client: AsyncOpenAI,
    text: str,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> set[str]:
    """Call NemoGuard Content Safety for topic classification.

    Returns set of violated category codes (e.g. {"S1", "S7"}).
    """
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="nvidia/llama-3.1-nemoguard-8b-content-safety",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Check the following text for safety concerns. "
                                "Return a JSON object with keys 'safe' (boolean) "
                                "and 'categories' (list of violated category codes like S1, S2, etc).\n\n"
                                f"Text: {text[:2000]}"
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                content = response.choices[0].message.content or ""
                return _parse_nemoguard_response(content)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"[nemoguard] 429 rate limit, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if attempt == max_retries - 1:
                    print(f"[nemoguard] Failed after {max_retries} attempts: {e}")
                    return set()
                raise
    return set()


def _parse_nemoguard_response(content: str) -> set[str]:
    """Extract violated category codes from NemoGuard response text."""
    # Try JSON parse first
    try:
        # Find JSON object in response
        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if not data.get("safe", True):
                cats = data.get("categories", [])
                return {c for c in cats if c in NEMOGUARD_CATEGORIES}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: look for S-codes in the text
    codes = set(re.findall(r"\bS\d{1,2}\b", content))
    valid = {c for c in codes if c in NEMOGUARD_CATEGORIES}

    # If we found codes and the response says "unsafe", return them
    if valid and ("unsafe" in content.lower() or "not safe" in content.lower()):
        return valid

    return set()


async def _batch_scan_nemoguard(
    texts: dict[str, str],  # uuid -> sampled text
    sem: asyncio.Semaphore,
    on_progress: Callable[..., None] | None = None,
) -> dict[str, set[str]]:
    """Scan all conversations with NemoGuard. Returns uuid -> set of category codes."""
    client = _get_nemoguard_client()
    results: dict[str, set[str]] = {}
    total = len(texts)
    done = 0

    async def _scan_one(uuid: str, text: str) -> None:
        nonlocal done
        cats = await scan_conversation_nemoguard(client, text, sem)
        results[uuid] = cats
        done += 1
        if on_progress and total > 0:
            on_progress(
                PipelinePhase.scanning,
                f"Content safety scan: {done}/{total}",
                0.5 + done / total * 0.5,  # Second half of scan progress
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
    """Run full privacy scan: GLiNER-PII + NemoGuard.

    Returns a ScanResult with per-category conversation counts.
    """
    if on_progress:
        on_progress(PipelinePhase.scanning, "Loading privacy scanner...", 0.0)

    # Sample messages for each conversation
    texts: dict[str, str] = {}
    for conv in conversations:
        uuid = conv["uuid"]
        texts[uuid] = _sample_messages(conv.get("messages", []))

    # Run GLiNER-PII (local, GPU if available)
    if on_progress:
        on_progress(PipelinePhase.scanning, "Running PII detection...", 0.05)

    gliner_model = await _get_gliner_model()
    pii_sem = asyncio.Semaphore(10)
    pii_results = await _batch_scan_pii(texts, gliner_model, pii_sem, on_progress)

    # Run NemoGuard (NIM API)
    if on_progress:
        on_progress(PipelinePhase.scanning, "Running content safety scan...", 0.5)

    nemoguard_sem = asyncio.Semaphore(5)
    nemoguard_results = await _batch_scan_nemoguard(texts, nemoguard_sem, on_progress)

    # Merge results: build conversation_flags and categories
    conversation_flags: dict[str, list[str]] = defaultdict(list)
    category_uuids: dict[str, set[str]] = defaultdict(set)

    for uuid, pii_labels in pii_results.items():
        for label in pii_labels:
            cat_id = f"pii:{label}"
            conversation_flags[uuid].append(cat_id)
            category_uuids[cat_id].add(uuid)

    for uuid, safety_codes in nemoguard_results.items():
        for code in safety_codes:
            cat_id = f"safety:{code}"
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

    for code, name in NEMOGUARD_CATEGORIES.items():
        cat_id = f"safety:{code}"
        if cat_id in category_uuids:
            categories.append(
                PrivacyCategory(
                    id=cat_id,
                    name=name,
                    source="nemoguard",
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
