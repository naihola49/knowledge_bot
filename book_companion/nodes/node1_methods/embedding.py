"""Node 1 embedding helpers delegating to the embeddings service (external API)"""

from __future__ import annotations

from typing import Iterable

from book_companion.services.embedding_cache import SameDayCachedEmbeddingsClient
from book_companion.services.embeddings import EmbeddingsClient, HFInferenceEmbeddingsClient

_CLIENT: EmbeddingsClient | None = None


def _get_client() -> EmbeddingsClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = SameDayCachedEmbeddingsClient(HFInferenceEmbeddingsClient()) # wrap in cache
    return _CLIENT


def vectorize_text(text: str) -> list[float]:
    return _get_client().embed_text(text)


def vectorize_texts(texts: Iterable[str]) -> list[list[float]]:
    return _get_client().embed_texts(texts)

