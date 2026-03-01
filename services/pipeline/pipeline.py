"""
Segmentation + Clustering pipeline.

1. Embed messages via NVIDIA NIM API (baai/bge-m3)
2. Segment conversations via DeepTiling (cosine similarity depth scores)
3. Mean-pool segment embeddings
4. BERTopic: UMAP → HDBSCAN → c-TF-IDF
5. Label clusters via Mistral on NVIDIA NIM
6. Build 3-level topic hierarchy via Mistral
"""

from __future__ import annotations

import asyncio
import json
import os
import re as _re
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

NIM_BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "baai/bge-m3"
CHAT_MODEL = "mistralai/mistral-small-3.1-24b-instruct-2503"

EMBED_BATCH_SIZE = 100
DEPTH_THRESHOLD_FACTOR = 1.0  # boundary if depth > mean + factor * stdev


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY in .env file")
        sys.exit(1)
    return OpenAI(base_url=NIM_BASE, api_key=api_key)


# ---------------------------------------------------------------------------
# Step 1: Embed
# ---------------------------------------------------------------------------


def embed_texts(client, texts):
    """Embed a list of texts via NIM API, batching to stay within limits."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            input=batch,
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": "passage"},
        )
        all_embeddings.extend([d.embedding for d in resp.data])
    return np.array(all_embeddings)


def prepare_message_texts(conversations):
    """Concatenate human+assistant per turn into embeddable strings.
    Returns flat list of texts and a parallel structure tracking provenance."""
    texts = []
    provenance = []  # (conv_index, msg_index)
    for ci, conv in enumerate(conversations):
        for mi, msg in enumerate(conv["messages"]):
            t = f"[{msg['sender']}] {msg['text']}"
            # Truncate to ~2000 chars to avoid token limit
            texts.append(t[:2000])
            provenance.append((ci, mi))
    return texts, provenance


# ---------------------------------------------------------------------------
# Step 2: DeepTiling segmentation
# ---------------------------------------------------------------------------


def deep_tiling_segment(embeddings, window=3):
    """Segment a sequence of embeddings using depth-score boundaries.
    Returns list of (start, end) index pairs for each segment."""
    n = len(embeddings)
    if n <= 1:
        return [(0, n)]

    # Cosine similarity between consecutive embeddings (smoothed with window)
    sims = []
    for i in range(n - 1):
        lo = max(0, i - window + 1)
        hi = min(n, i + window + 1)
        left_block = embeddings[lo : i + 1].mean(axis=0, keepdims=True)
        right_block = embeddings[i + 1 : hi].mean(axis=0, keepdims=True)
        sim = cosine_similarity(left_block, right_block)[0, 0]
        sims.append(sim)

    sims_arr = np.array(sims)

    # Depth scores
    depths = np.zeros(len(sims_arr))
    for i in range(len(sims_arr)):
        # left peak
        left_peak = sims_arr[i]
        for j in range(i - 1, -1, -1):
            if sims_arr[j] > left_peak:
                left_peak = sims_arr[j]
            else:
                break
        # right peak
        right_peak = sims_arr[i]
        for j in range(i + 1, len(sims_arr)):
            if sims_arr[j] > right_peak:
                right_peak = sims_arr[j]
            else:
                break
        depths[i] = (left_peak - sims_arr[i]) + (right_peak - sims_arr[i])

    # Place boundaries where depth > mean + factor * stdev
    threshold = depths.mean() + DEPTH_THRESHOLD_FACTOR * depths.std()
    boundaries = [i for i, d in enumerate(depths) if d > threshold]

    # Build segments
    segments = []
    prev = 0
    for b in boundaries:
        segments.append((prev, b + 1))
        prev = b + 1
    segments.append((prev, n))

    return segments


# ---------------------------------------------------------------------------
# Step 3: Build segments with mean-pooled embeddings
# ---------------------------------------------------------------------------


def build_segments(conversations, all_embeddings, provenance, on_progress=None):
    """Run DeepTiling per conversation, return list of segment dicts."""
    segments = []

    # Group embeddings by conversation
    conv_msgs: dict[int, list[tuple[int, Any]]] = {}  # conv_index → [(msg_index, embedding)]
    for idx, (ci, mi) in enumerate(provenance):
        conv_msgs.setdefault(ci, []).append((mi, all_embeddings[idx]))

    total_convs = len([ci for ci in range(len(conversations)) if ci in conv_msgs])
    processed = 0

    for ci, conv in enumerate(conversations):
        if ci not in conv_msgs:
            continue
        items = sorted(conv_msgs[ci], key=lambda x: x[0])
        msg_indices = [x[0] for x in items]
        embs = np.array([x[1] for x in items])

        seg_ranges = deep_tiling_segment(embs)

        for start, end in seg_ranges:
            seg_msg_indices = msg_indices[start:end]
            seg_messages = [conv["messages"][mi] for mi in seg_msg_indices]
            seg_embedding = embs[start:end].mean(axis=0)
            segments.append(
                {
                    "conversation_name": conv["name"],
                    "conversation_uuid": conv["uuid"],
                    "messages": seg_messages,
                    "embedding": seg_embedding,
                }
            )

        processed += 1
        if on_progress and total_convs > 0:
            on_progress(processed, total_convs, len(segments))

    return segments


# ---------------------------------------------------------------------------
# Step 4: BERTopic clustering
# ---------------------------------------------------------------------------


def cluster_segments(segments, on_progress=None):
    """Cluster segment embeddings with BERTopic.
    Returns (topics, topic_model, docs) where docs are the text representations."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    embeddings = np.array([s["embedding"] for s in segments])

    # Build text representation for c-TF-IDF
    docs = []
    for s in segments:
        text = " ".join(m["text"][:200] for m in s["messages"])
        docs.append(text)

    n = len(embeddings)
    n_neighbors = min(15, n - 1)
    n_components = min(5, n - 1)
    min_cluster = max(2, min(3, n // 2))

    # Wrap UMAP/HDBSCAN to emit progress when BERTopic calls them internally
    class _ProgressUMAP(UMAP):  # type: ignore[misc]
        def fit_transform(self, X, y=None):
            if on_progress:
                on_progress("reducing", f"Reducing {n} segments to {n_components}D with UMAP...", 0.0)
            result = super().fit_transform(X, y)
            if on_progress:
                on_progress("reducing", "Dimension reduction complete", 0.33)
            return result

    class _ProgressHDBSCAN(HDBSCAN):  # type: ignore[misc]
        def fit(self, X, y=None):
            if on_progress:
                on_progress("clustering", f"Finding clusters with HDBSCAN ({n} segments)...", 0.33)
            result = super().fit(X, y)
            n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            if on_progress:
                on_progress("clustering", f"Found {n_clusters} clusters, extracting topics...", 0.66)
            return result

    init = "random" if n < 10 else "spectral"
    umap_model = _ProgressUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        init=init,
        random_state=42,
    )
    hdbscan_model = _ProgressHDBSCAN(
        min_cluster_size=min_cluster, min_samples=1, metric="euclidean", prediction_data=True
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    if on_progress:
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        on_progress("done", f"{n_topics} topics discovered", 1.0)

    return topics, topic_model, docs


# ---------------------------------------------------------------------------
# Step 5: Label clusters via Mistral
# ---------------------------------------------------------------------------


def label_clusters(client, topics, topic_model, segments, docs):
    """Use Mistral to generate human-readable labels for each cluster.
    Returns dict: {label: {keywords, segments}}."""
    topic_info = topic_model.get_topic_info()
    unique_topics = [t for t in topic_info["Topic"] if t != -1]

    result = {}

    for topic_id in unique_topics:
        # Get keywords from c-TF-IDF
        topic_words = topic_model.get_topic(topic_id)
        keywords = [w for w, _ in topic_words[:10]]

        # Get representative segments
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        rep_indices = indices[:3]
        rep_texts = []
        for idx in rep_indices:
            text = " ".join(m["text"][:300] for m in segments[idx]["messages"])
            rep_texts.append(text[:500])

        # Ask Mistral for a label
        prompt = (
            "Given these keywords and example text segments from a conversation cluster, "
            "generate a concise topic label (3-7 words). Reply with ONLY the label.\n\n"
            f"Keywords: {', '.join(keywords)}\n\n"
            "Example segments:\n" + "\n---\n".join(rep_texts)
        )

        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.3,
            )
            label = resp.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            print(f"  Warning: Mistral labeling failed for topic {topic_id}: {e}")
            label = f"Topic {topic_id}: {', '.join(keywords[:3])}"

        # Deduplicate label if needed
        if label in result:
            label = f"{label} ({topic_id})"

        seg_list = []
        for idx in indices:
            seg_list.append(
                {
                    "conversation_name": segments[idx]["conversation_name"],
                    "messages": segments[idx]["messages"],
                }
            )

        result[label] = {"keywords": keywords, "segments": seg_list}
        print(f"  Topic {topic_id} → {label} ({len(seg_list)} segments)")

    # Handle outliers (topic -1)
    outlier_indices = [i for i, t in enumerate(topics) if t == -1]
    if outlier_indices:
        seg_list = []
        for idx in outlier_indices:
            seg_list.append(
                {
                    "conversation_name": segments[idx]["conversation_name"],
                    "messages": segments[idx]["messages"],
                }
            )
        result["Uncategorized"] = {"keywords": [], "segments": seg_list}
        print(f"  Outliers → Uncategorized ({len(seg_list)} segments)")

    return result


# ---------------------------------------------------------------------------
# Step 6: Build topic hierarchy via Mistral
# ---------------------------------------------------------------------------


def build_hierarchy(client: OpenAI, topic_groups: dict[str, Any]) -> dict[str, Any]:
    """Send all topic labels + keywords to Mistral and get back a 3-level category tree.
    topic_groups: dict {label: {keywords: [...], ...}}
    Returns dict: {root_category: {subcategory: [topic_label, ...]}}."""
    import re

    # Build label list with keywords for disambiguation
    entries = []
    for label, info in topic_groups.items():
        kws = info.get("keywords", [])[:6]
        if kws:
            entries.append(f'- "{label}" [keywords: {", ".join(kws)}]')
        else:
            entries.append(f'- "{label}"')
    entries_text = "\n".join(entries)
    n = len(entries)

    prompt = (
        "You are an expert librarian organizing technical topics into a knowledge taxonomy.\n\n"
        f"Given these {n} topic labels with their associated keywords, organize them into a "
        "hierarchical JSON tree: root category → subcategory → topic labels.\n\n"
        "For each topic, first identify its true technical domain from the keywords, "
        "then assign it to the most specific matching category. "
        "Prioritize keyword evidence over surface-level label text when they conflict — "
        "for example, a topic with keywords like [allocator, pointer, free] belongs in "
        "systems programming, not multimedia, regardless of other words in the label.\n\n"
        "Return ONLY valid JSON (no markdown fences) with this structure:\n"
        "{\n"
        '  "Root Category": {\n'
        '    "Subcategory": ["Topic Label 1", "Topic Label 2"]\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Create as many root categories and subcategories as needed to organize the topics naturally\n"
        "- Every topic label must appear EXACTLY once as a leaf\n"
        "- Use the EXACT label text in quotes — do not rename or modify any label\n\n"
        f"Topics:\n{entries_text}"
    )

    print(f"  Sending {n} labels to Mistral for categorization...")
    # Scale max_tokens with topic count — ~50 tokens per label in the JSON output
    max_tokens = max(2000, n * 50)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert librarian and taxonomist. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    raw_content = resp.choices[0].message.content
    raw = raw_content.strip() if raw_content else ""

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    hierarchy = json.loads(raw)

    # Normalize: if LLM returned 2-level (root → [labels]), wrap as 3-level
    for root_name, value in list(hierarchy.items()):
        if isinstance(value, list):
            hierarchy[root_name] = {root_name: value}

    # Validate: every label appears exactly once
    found = set()
    for _root, subcats in hierarchy.items():
        for subcat, labels in subcats.items():
            if isinstance(labels, str):
                # Single label not wrapped in a list
                subcats[subcat] = [labels]
                labels = [labels]
            for label in labels:
                found.add(label)

    expected = set(topic_groups.keys())
    missing = expected - found
    extra = found - expected

    if missing:
        print(f"  Warning: {len(missing)} labels missing from hierarchy: {missing}")
        # Put missing labels under an "Other" subcategory in the last root
        last_root = list(hierarchy.keys())[-1]
        hierarchy[last_root].setdefault("Other", []).extend(sorted(missing))

    if extra:
        print(f"  Warning: {len(extra)} extra labels in hierarchy — removing: {extra}")
        for root_name, subcats in list(hierarchy.items()):
            for sub_name, labels in list(subcats.items()):
                subcats[sub_name] = [lb for lb in labels if lb not in extra]
                if not subcats[sub_name]:
                    del subcats[sub_name]
            if not subcats:
                del hierarchy[root_name]

    return hierarchy


# ---------------------------------------------------------------------------
# Async versions — used by the API pipeline endpoint
# ---------------------------------------------------------------------------


def get_async_client() -> AsyncOpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return AsyncOpenAI(base_url=NIM_BASE, api_key=api_key)


async def async_embed_texts(
    client: AsyncOpenAI,
    texts: list[str],
    on_progress: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Embed texts asynchronously with concurrent batches."""
    from models.schemas import PipelinePhase

    batches = [texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    total = len(batches)
    results: list[list[list[float]]] = [[] for _ in range(total)]
    completed_batches = 0
    batch_lock = asyncio.Lock()

    sem = asyncio.Semaphore(10)

    async def embed_batch(idx: int, batch: list[str]) -> None:
        nonlocal completed_batches
        async with sem:
            resp = await client.embeddings.create(
                input=batch,
                model=EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": "passage"},
            )
            results[idx] = [d.embedding for d in resp.data]
            async with batch_lock:
                completed_batches += 1
                done = completed_batches
            if on_progress:
                remaining = total - done
                if remaining <= 2:
                    msg = f"Finalizing embeddings ({remaining} remaining)..."
                else:
                    msg = f"Embedded batch {done}/{total}"
                on_progress(
                    PipelinePhase.embedding,
                    msg,
                    done / total,
                )

    await asyncio.gather(*[embed_batch(i, b) for i, b in enumerate(batches)])

    all_embeddings: list[list[float]] = []
    for batch_result in results:
        all_embeddings.extend(batch_result)
    return np.array(all_embeddings)


async def async_label_clusters(
    client: AsyncOpenAI,
    topics: list[int],
    topic_model: Any,
    segments: list[dict[str, Any]],
    docs: list[str],
    on_progress: Callable[..., Any] | None = None,
    sensitive_filter: Callable[[str, list[str]], bool] | None = None,
) -> dict[str, Any]:
    """Label clusters concurrently via Mistral. Returns {label: {keywords, segments}}."""
    from models.schemas import GraphNode, PipelinePhase

    topic_info = topic_model.get_topic_info()
    unique_topics = [t for t in topic_info["Topic"] if t != -1]
    total = len(unique_topics)

    result: dict[str, Any] = {}
    label_lock = asyncio.Lock()
    completed_count = 0
    sem = asyncio.Semaphore(5)

    async def label_one(idx: int, topic_id: int) -> None:
        nonlocal completed_count
        async with sem:
            topic_words = topic_model.get_topic(topic_id)
            keywords = [w for w, _ in topic_words[:10]]

            indices = [i for i, t in enumerate(topics) if t == topic_id]
            rep_indices = indices[:3]
            rep_texts = []
            for r_idx in rep_indices:
                text = " ".join(m["text"][:300] for m in segments[r_idx]["messages"])
                rep_texts.append(text[:500])

            prompt = (
                "Given these keywords and example text segments from a conversation cluster, "
                "generate a concise topic label (3-7 words). Reply with ONLY the label.\n\n"
                f"Keywords: {', '.join(keywords)}\n\n"
                "Example segments:\n" + "\n---\n".join(rep_texts)
            )

            try:
                resp = await client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                label = content.strip().strip('"').strip("'") if content else f"Topic {topic_id}"
            except Exception as e:
                print(f"  Warning: Mistral labeling failed for topic {topic_id}: {e}")
                label = f"Topic {topic_id}: {', '.join(keywords[:3])}"

            seg_list = []
            for s_idx in indices:
                seg_list.append(
                    {
                        "conversation_name": segments[s_idx]["conversation_name"],
                        "messages": segments[s_idx]["messages"],
                    }
                )

            async with label_lock:
                if label in result:
                    label = f"{label} ({topic_id})"
                result[label] = {"keywords": keywords, "segments": seg_list}
                completed_count += 1
                done = completed_count

            print(f"  Topic {topic_id} → {label} ({len(seg_list)} segments)")

            # Emit progress node if not sensitive
            is_sensitive = sensitive_filter(label, keywords) if sensitive_filter else False
            if on_progress and not is_sensitive:
                on_progress(
                    PipelinePhase.labeling,
                    f"Labeled {done}/{total}: {label}",
                    done / total,
                    node=GraphNode(
                        id=f"topic::{label}",
                        name=label,
                        level=2,
                        segment_count=len(seg_list),
                        type="topic",
                        keywords=keywords[:5],
                    ),
                )
            elif on_progress:
                on_progress(
                    PipelinePhase.labeling,
                    f"Labeled {done}/{total}",
                    done / total,
                )

    await asyncio.gather(*[label_one(i, t) for i, t in enumerate(unique_topics)])

    # Handle outliers
    outlier_indices = [i for i, t in enumerate(topics) if t == -1]
    if outlier_indices:
        seg_list = []
        for idx in outlier_indices:
            seg_list.append(
                {
                    "conversation_name": segments[idx]["conversation_name"],
                    "messages": segments[idx]["messages"],
                }
            )
        result["Uncategorized"] = {"keywords": [], "segments": seg_list}
        print(f"  Outliers → Uncategorized ({len(seg_list)} segments)")

    return result


HIERARCHY_BATCH_SIZE = 30  # max topics per LLM call


async def _hierarchy_call(
    client: AsyncOpenAI,
    entries_text: str,
    n: int,
    on_progress: Callable[..., Any] | None = None,
    progress_label: str = "",
) -> dict[str, Any]:
    """Single hierarchy LLM call with retry logic. Returns parsed JSON."""
    from models.schemas import PipelinePhase

    prompt = (
        "You are an expert librarian organizing technical topics into a knowledge taxonomy.\n\n"
        f"Given these {n} topic labels with their associated keywords, organize them into a "
        "hierarchical JSON tree: root category → subcategory → topic labels.\n\n"
        "For each topic, first identify its true technical domain from the keywords, "
        "then assign it to the most specific matching category. "
        "Prioritize keyword evidence over surface-level label text when they conflict — "
        "for example, a topic with keywords like [allocator, pointer, free] belongs in "
        "systems programming, not multimedia, regardless of other words in the label.\n\n"
        "Return ONLY valid JSON (no markdown fences) with this structure:\n"
        "{\n"
        '  "Root Category": {\n'
        '    "Subcategory": ["Topic Label 1", "Topic Label 2"]\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Create as many root categories and subcategories as needed\n"
        "- Every topic label must appear EXACTLY once as a leaf\n"
        "- Use the EXACT label text — do not rename or modify any label\n\n"
        f"Topics:\n{entries_text}"
    )

    max_tokens = max(2000, n * 60)
    sys_msg = "You are an expert librarian and taxonomist. Respond with JSON only."

    for attempt in range(1, 4):
        if on_progress:
            if attempt == 1:
                msg = f"Organizing {progress_label or f'{n} topics'}..."
            else:
                msg = f"Retrying {progress_label or 'hierarchy'} (attempt {attempt}/3)..."
            on_progress(PipelinePhase.hierarchy, msg, 0.0)

        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                ),
                timeout=120,
            )
        except TimeoutError:
            print(f"  Hierarchy {progress_label} attempt {attempt} timed out")
            if on_progress:
                on_progress(
                    PipelinePhase.hierarchy,
                    f"Timeout on attempt {attempt}/3, retrying...",
                    0.0,
                )
            if attempt == 3:
                raise
            continue
        except Exception as exc:
            # Catch 429 rate limits and other API errors
            exc_str = str(exc)
            print(f"  Hierarchy {progress_label} attempt {attempt} error: {exc_str}")
            if "429" in exc_str:
                wait = 5 * attempt
                if on_progress:
                    on_progress(
                        PipelinePhase.hierarchy,
                        f"Rate limited, waiting {wait}s (attempt {attempt}/3)...",
                        0.0,
                    )
                await asyncio.sleep(wait)
                if attempt == 3:
                    raise
                continue
            raise

        raw_content = resp.choices[0].message.content
        raw = raw_content.strip() if raw_content else ""
        raw = _re.sub(r"^```(?:json)?\s*", "", raw)
        raw = _re.sub(r"\s*```$", "", raw)

        # Check for finish_reason indicating truncation
        finish = resp.choices[0].finish_reason
        if finish == "length":
            print(f"  Hierarchy {progress_label} attempt {attempt}: output truncated (max_tokens={max_tokens})")
            if on_progress:
                on_progress(
                    PipelinePhase.hierarchy,
                    f"Response truncated on attempt {attempt}/3, retrying...",
                    0.0,
                )
            if attempt == 3:
                # Try to salvage truncated JSON
                break
            max_tokens = int(max_tokens * 1.5)
            continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  Hierarchy {progress_label} attempt {attempt} invalid JSON: {exc}")
            print(f"  Raw (first 500): {raw[:500]}")
            if on_progress:
                on_progress(
                    PipelinePhase.hierarchy,
                    f"Invalid JSON on attempt {attempt}/3, retrying...",
                    0.0,
                )
            if attempt == 3:
                raise
            continue

    # Should not reach here normally, but handle truncated last attempt
    raise ValueError(f"Hierarchy failed after 3 attempts for {progress_label}")


async def async_build_hierarchy(
    client: AsyncOpenAI,
    topic_groups: dict[str, Any],
    on_progress: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build 3-level topic hierarchy asynchronously via Mistral.

    For large topic sets (>HIERARCHY_BATCH_SIZE), splits into batches,
    builds sub-hierarchies concurrently, then merges.
    """
    from models.schemas import PipelinePhase

    if on_progress:
        on_progress(PipelinePhase.hierarchy, "Building topic hierarchy...", 0.0)

    # Build entries list
    all_labels = list(topic_groups.keys())
    n = len(all_labels)

    def _make_entries(labels: list[str]) -> str:
        entries = []
        for label in labels:
            info = topic_groups.get(label, {})
            kws = info.get("keywords", [])[:6]
            if kws:
                entries.append(f'- "{label}" [keywords: {", ".join(kws)}]')
            else:
                entries.append(f'- "{label}"')
        return "\n".join(entries)

    hierarchy: dict[str, Any]

    if n <= HIERARCHY_BATCH_SIZE:
        # Small enough for a single call
        entries_text = _make_entries(all_labels)
        hierarchy = await _hierarchy_call(
            client,
            entries_text,
            n,
            on_progress,
            progress_label=f"{n} topics",
        )
    else:
        # Split into batches and build concurrently
        batches: list[list[str]] = []
        for i in range(0, n, HIERARCHY_BATCH_SIZE):
            batches.append(all_labels[i : i + HIERARCHY_BATCH_SIZE])

        if on_progress:
            on_progress(
                PipelinePhase.hierarchy,
                f"Splitting {n} topics into {len(batches)} batches...",
                0.0,
            )

        async def _do_batch(idx: int, batch_labels: list[str]) -> dict[str, Any]:
            entries_text = _make_entries(batch_labels)
            return await _hierarchy_call(
                client,
                entries_text,
                len(batch_labels),
                on_progress,
                progress_label=f"batch {idx + 1}/{len(batches)}",
            )

        # Run sequentially to avoid rate limits
        results = []
        for i, b in enumerate(batches):
            results.append(await _do_batch(i, b))

        # Merge sub-hierarchies
        hierarchy = {}
        for sub_h in results:
            # Normalize 2-level → 3-level before merging
            for rn, val in list(sub_h.items()):
                if isinstance(val, list):
                    sub_h[rn] = {rn: val}
            for root_name, subcats in sub_h.items():
                if root_name not in hierarchy:
                    hierarchy[root_name] = {}
                for sub_name, labels in subcats.items():
                    if isinstance(labels, str):
                        labels = [labels]
                    existing = hierarchy[root_name].get(sub_name, [])
                    hierarchy[root_name][sub_name] = existing + labels

        if on_progress:
            on_progress(
                PipelinePhase.hierarchy,
                f"Merged {len(batches)} batches into hierarchy",
                0.8,
            )

    # Normalize 2-level → 3-level
    for root_name, value in list(hierarchy.items()):
        if isinstance(value, list):
            hierarchy[root_name] = {root_name: value}

    # Validate labels
    found: set[str] = set()
    for _root, subcats in hierarchy.items():
        for subcat, labels in subcats.items():
            if isinstance(labels, str):
                subcats[subcat] = [labels]
                labels = [labels]
            for label in labels:
                found.add(label)

    expected = set(topic_groups.keys())
    missing = expected - found
    extra = found - expected

    if missing:
        print(f"  Warning: {len(missing)} labels missing from hierarchy")
        last_root = list(hierarchy.keys())[-1]
        hierarchy[last_root].setdefault("Other", []).extend(sorted(missing))
    if extra:
        print(f"  Warning: {len(extra)} extra labels — removing")
        for root_name, subcats in list(hierarchy.items()):
            for sub_name, labels in list(subcats.items()):
                subcats[sub_name] = [lb for lb in labels if lb not in extra]
                if not subcats[sub_name]:
                    del subcats[sub_name]
            if not subcats:
                del hierarchy[root_name]

    if on_progress:
        on_progress(PipelinePhase.hierarchy, "Hierarchy complete", 1.0)

    return hierarchy


async def run_pipeline_async(
    conversations: list[dict[str, Any]],
    on_progress: Callable[..., Any] | None = None,
    sensitive_filter: Callable[[str, list[str]], bool] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full pipeline asynchronously. Returns (topic_groups, hierarchy)."""
    from models.schemas import PipelinePhase

    client = get_async_client()

    # Step 1: Embed (async, concurrent batches)
    if on_progress:
        on_progress(PipelinePhase.embedding, "Preparing messages...", 0.0)
    texts, provenance = prepare_message_texts(conversations)
    print(f"  {len(texts)} messages to embed.")
    all_embeddings = await async_embed_texts(client, texts, on_progress)
    print(f"  Embeddings shape: {all_embeddings.shape}")

    # Step 2-3: Segment (CPU, offloaded to thread)
    if on_progress:
        on_progress(PipelinePhase.segmenting, "Segmenting conversations...", 0.0)

    def _seg_progress(processed: int, total: int, seg_count: int) -> None:
        if on_progress:
            on_progress(
                PipelinePhase.segmenting,
                f"Segmented {processed}/{total} conversations ({seg_count} segments)",
                processed / total,
            )

    segments = await asyncio.to_thread(build_segments, conversations, all_embeddings, provenance, _seg_progress)
    print(f"  {len(segments)} segments created.")
    if on_progress:
        on_progress(PipelinePhase.segmenting, f"{len(segments)} segments created", 1.0)

    # Step 4: Cluster (CPU, offloaded to thread)
    if on_progress:
        on_progress(PipelinePhase.clustering, "Clustering segments...", 0.0)

    def _cluster_progress(step: str, message: str, progress: float) -> None:
        if on_progress:
            on_progress(PipelinePhase.clustering, message, progress)

    topics, topic_model, docs = await asyncio.to_thread(cluster_segments, segments, _cluster_progress)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  {n_topics} topics discovered.")
    if on_progress:
        on_progress(PipelinePhase.clustering, f"{n_topics} topics discovered", 1.0)

    # Step 5: Label (async, concurrent API calls)
    if on_progress:
        on_progress(PipelinePhase.labeling, "Labeling topics...", 0.0)
    topic_groups = await async_label_clusters(
        client, topics, topic_model, segments, docs, on_progress, sensitive_filter
    )
    print(f"  {len(topic_groups)} topic groups.")

    # Step 6: Hierarchy (single async call)
    hierarchy = await async_build_hierarchy(client, topic_groups, on_progress)
    n_roots = len(hierarchy)
    n_subs = sum(len(sc) for sc in hierarchy.values())
    print(f"  {n_roots} root categories, {n_subs} subcategories.")

    return topic_groups, hierarchy


# ---------------------------------------------------------------------------
# Main (sync CLI entry point)
# ---------------------------------------------------------------------------


def run_pipeline(conversations_path="conversations.json"):
    """Run the full pipeline and return the topic groups dict."""
    print("Loading conversations...")
    with open(conversations_path) as f:
        conversations = json.load(f)
    print(f"  {len(conversations)} conversations loaded.\n")

    client = get_client()

    # Step 1: Embed
    print("Step 1: Embedding messages...")
    texts, provenance = prepare_message_texts(conversations)
    print(f"  {len(texts)} messages to embed.")
    all_embeddings = embed_texts(client, texts)
    print(f"  Embeddings shape: {all_embeddings.shape}\n")

    # Step 2 + 3: Segment
    print("Step 2-3: Segmenting conversations (DeepTiling)...")
    segments = build_segments(conversations, all_embeddings, provenance)
    print(f"  {len(segments)} segments created.\n")

    # Step 4: Cluster
    print("Step 4: Clustering with BERTopic...")
    topics, topic_model, docs = cluster_segments(segments)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  {n_topics} topics discovered.\n")

    # Step 5: Label
    print("Step 5: Labeling clusters via Mistral...")
    topic_groups = label_clusters(client, topics, topic_model, segments, docs)
    print(f"  {len(topic_groups)} topic groups.\n")

    # Step 6: Hierarchy
    print("Step 6: Building topic hierarchy via Mistral...")
    hierarchy = build_hierarchy(client, topic_groups)
    n_roots = len(hierarchy)
    n_subs = sum(len(sc) for sc in hierarchy.values())
    print(f"  {n_roots} root categories, {n_subs} subcategories.\n")

    print("Done!")
    return topic_groups, hierarchy


if __name__ == "__main__":
    groups, hierarchy = run_pipeline()
    with open("topic_groups.json", "w") as f:
        json.dump(groups, f, indent=2, default=str)
    print("Saved topic_groups.json")
    with open("topic_hierarchy.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    print("Saved topic_hierarchy.json")
    print("\nTopics:")
    for label, info in groups.items():
        print(f"  [{len(info['segments'])} segs] {label}")
