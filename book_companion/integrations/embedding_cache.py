"""Same-calendar-day in-memory cache for embedding API calls
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Iterable

from book_companion.config import EMBEDDING_MODEL_NAME
from book_companion.integrations.embeddings import EmbeddingsClient


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class SameDayCachedEmbeddingsClient:
    """Wraps an EmbeddingsClient with a per-day dict cache keyed by content hash."""

    def __init__(self, inner: EmbeddingsClient, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self._inner = inner
        self._model_name = model_name
        self._day: date | None = None
        self._vectors: dict[str, list[float]] = {}

    def _roll_day_if_needed(self) -> None: # clear cache func
        today = date.today()
        if self._day != today:
            self._vectors.clear()
            self._day = today

    def embed_text(self, text: str) -> list[float]: # new embeddings call 
        self._roll_day_if_needed()
        key = _cache_key(self._model_name, text)
        if key in self._vectors:
            return self._vectors[key]
        vec = self._inner.embed_text(text)
        self._vectors[key] = vec
        return vec

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        self._roll_day_if_needed()
        text_list = list(texts)
        if not text_list:
            return []

        keys = [_cache_key(self._model_name, t) for t in text_list]
        out: list[list[float] | None] = [None] * len(text_list)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for i, key in enumerate(keys):
            if key in self._vectors:
                out[i] = self._vectors[key]
            else:
                missing_indices.append(i)
                missing_texts.append(text_list[i])

        if missing_texts:
            fresh = self._inner.embed_texts(missing_texts)
            if len(fresh) != len(missing_texts):
                raise RuntimeError("Inner embed_texts length mismatch.")
            for j, idx in enumerate(missing_indices):
                vec = fresh[j]
                k = _cache_key(self._model_name, text_list[idx])
                self._vectors[k] = vec
                out[idx] = vec

        filled: list[list[float]] = []
        for row in out:
            if row is None:
                raise RuntimeError("Embedding cache failed to fill all rows.")
            filled.append(row)
        return filled
