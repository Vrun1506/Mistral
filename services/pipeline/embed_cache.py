"""File-based embedding cache: .embed_cache/<conv_uuid>/<content_hash>.npy"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(".embed_cache")


def _conv_cache_key(conv: dict) -> str:
    """SHA-256 of concatenated '[sender] text[:2000]' strings."""
    parts = []
    for msg in conv["messages"]:
        parts.append(f"[{msg['sender']}] {msg['text'][:2000]}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def get_cached_embeddings(conv: dict) -> np.ndarray | None:
    """Return cached embeddings for a conversation, or None if not cached."""
    conv_dir = CACHE_DIR / conv["uuid"]
    if not conv_dir.exists():
        return None
    content_hash = _conv_cache_key(conv)
    cache_file = conv_dir / f"{content_hash}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    return None


def save_embeddings(conv: dict, embeddings: np.ndarray) -> None:
    """Save embeddings for a conversation. Removes old hash files for same UUID."""
    conv_dir = CACHE_DIR / conv["uuid"]
    conv_dir.mkdir(parents=True, exist_ok=True)
    # Clean old cache files for this conversation
    for old_file in conv_dir.glob("*.npy"):
        old_file.unlink()
    content_hash = _conv_cache_key(conv)
    np.save(conv_dir / f"{content_hash}.npy", embeddings)
