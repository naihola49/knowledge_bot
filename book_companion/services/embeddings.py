"""Embeddings client protocol and Hugging Face Inference via huggingface_hub.InferenceClient."""

from __future__ import annotations

import math
import os
from typing import Iterable, Protocol

import numpy as np
from huggingface_hub import InferenceClient

from book_companion.config import (
    EMBEDDING_MODEL_NAME,
    HF_INFERENCE_API_TIMEOUT_SECONDS,
    HF_TOKEN_ENV_VAR,
)
from book_companion.services.hf_inference_throttle import after_hf_request, before_hf_request

# Matches HF model-card examples (Inference Providers, not legacy api-inference URL).
HF_INFERENCE_PROVIDER = "hf-inference"


class EmbeddingsClient(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _ndarray_to_flat_floats(raw: object) -> list[float]:
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr.tolist()]


class HFInferenceEmbeddingsClient:
    """Embeddings via huggingface_hub.InferenceClient"""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        token_env_var: str = HF_TOKEN_ENV_VAR,
        timeout_seconds: int = HF_INFERENCE_API_TIMEOUT_SECONDS,
        provider: str = HF_INFERENCE_PROVIDER,
    ) -> None:
        self.model_name = model_name
        self.token_env_var = token_env_var
        self.timeout_seconds = float(timeout_seconds)
        self.provider = provider
        self._client: InferenceClient | None = None

    def _get_client(self) -> InferenceClient:
        token = os.getenv(self.token_env_var)
        if not token:
            raise RuntimeError(f"Missing required environment variable: {self.token_env_var}")
        if self._client is None:
            self._client = InferenceClient(
                provider=self.provider,  # type: ignore[arg-type]
                token=token,
                timeout=self.timeout_seconds,
            )
        return self._client

    def embed_text(self, text: str) -> list[float]:
        before_hf_request()
        try:
            client = self._get_client()
            raw = client.feature_extraction(text, model=self.model_name)
        finally:
            after_hf_request()
        vec = _ndarray_to_flat_floats(raw)
        if not vec:
            raise RuntimeError("Empty embedding vector from HF feature_extraction.")
        return _normalize(vec)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        # InferenceClient.feature_extraction accepts a single str; call per chunk.
        return [self.embed_text(t) for t in text_list]
