"""Privacy pre-scan: remote GLiNER-PII entity detection.

Calls an external GLiNER-PII inference server over HTTP.  If the server
is unreachable (or GLINER_SERVER_URL is not set), scanning is skipped
gracefully so the pipeline can continue without PII filtering.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import httpx

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

GLINER_SERVER_URL: str = os.getenv("GLINER_SERVER_URL", "")

# ---------------------------------------------------------------------------
# Remote inference helpers
# ---------------------------------------------------------------------------


async def _check_server_health(client: httpx.AsyncClient) -> bool:
    """GET /health with a short timeout. Returns True if the server is up."""
    try:
        resp = await client.get(f"{GLINER_SERVER_URL}/health", timeout=3.0)
        return bool(resp.status_code == 200)
    except (httpx.RequestError, httpx.TimeoutException):
        return False


async def _batch_predict_remote(
    client: httpx.AsyncClient,
    texts: dict[str, str],
) -> dict[str, list[str]]:
    """POST /predict with all sampled texts. Returns uuid -> list of labels."""
    resp = await client.post(
        f"{GLINER_SERVER_URL}/predict",
        json={"texts": texts, "labels": PII_LABELS, "threshold": 0.3},
        timeout=120.0,
    )
    resp.raise_for_status()
    results: dict[str, list[str]] = resp.json()["results"]
    return results


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
# Orchestrator
# ---------------------------------------------------------------------------


async def scan_conversations(
    conversations: list[dict[str, Any]],
    on_progress: Callable[..., None] | None = None,
) -> ScanResult:
    """Run remote GLiNER-PII privacy scan.

    Returns a ScanResult with per-category conversation counts.
    Skips gracefully if the inference server is unreachable or URL is not set.
    """
    if on_progress:
        on_progress(PipelinePhase.scanning, "Starting privacy scan...", 0.0)

    # Early exit if no server URL configured
    if not GLINER_SERVER_URL:
        if on_progress:
            on_progress(PipelinePhase.scanning, "PII scanning skipped (GLINER_SERVER_URL not set)", 1.0)
        return ScanResult(total_conversations=len(conversations))

    # Sample messages for each conversation
    texts: dict[str, str] = {}
    for conv in conversations:
        uuid = conv["uuid"]
        texts[uuid] = _sample_messages(conv.get("messages", []))

    # Check server health
    async with httpx.AsyncClient() as client:
        if on_progress:
            on_progress(PipelinePhase.scanning, "Checking PII inference server...", 0.05)

        if not await _check_server_health(client):
            if on_progress:
                on_progress(PipelinePhase.scanning, "PII scanning skipped (inference server unreachable)", 1.0)
            return ScanResult(total_conversations=len(conversations))

        # Run remote prediction
        if on_progress:
            on_progress(PipelinePhase.scanning, "Running PII detection...", 0.1)

        try:
            pii_results = await _batch_predict_remote(client, texts)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            print(f"[privacy] Remote PII scan failed: {exc}")
            if on_progress:
                on_progress(PipelinePhase.scanning, "PII scanning failed (server error)", 1.0)
            return ScanResult(total_conversations=len(conversations))

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
