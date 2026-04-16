"""Chunking helpers for Node 1."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into overlapping word chunks."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = [] # place into array
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks

"""
How this works: 
1. clean text, strip unneeded chars
2. split texts into words
3. step function to traverse through words, preserving overlap for semantic continuity
4. build array of chunks, return this for embedding model

input will be raw content!
"""