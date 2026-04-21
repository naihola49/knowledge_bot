"""Retrieval helpers for Node 1."""

from __future__ import annotations

import math
from book_companion.config import TOP_K_RETRIEVAL


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_top_k(
    user_vec: list[float],
    chunk_vectors: list[list[float]],
    chunks: list[str],
    k: int | None = None,
) -> list[tuple[str, str, float]]:
    """Return top-k (chunk_id, chunk_text, cosine similarity). k defaults to TOP_K_RETRIEVAL."""
    if k is None:
        k = TOP_K_RETRIEVAL
    if k < 1:
        return []
    if not chunks or not chunk_vectors:
        return []
    if len(chunks) != len(chunk_vectors):
        raise ValueError(
            f"chunks and chunk_vectors length mismatch: {len(chunks)} vs {len(chunk_vectors)}"
        )

    scored: list[tuple[str, str, float]] = []
    for idx, (chunk_vec, chunk_text) in enumerate(zip(chunk_vectors, chunks)):
        score = cosine_similarity(user_vec, chunk_vec)
        scored.append((f"chunk_{idx}", chunk_text, round(score, 6)))

    scored.sort(key=lambda item: item[2], reverse=True)
    effective = min(k, len(scored))
    return scored[:effective]
