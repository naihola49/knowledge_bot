"""Embeddings client protocol and Hugging Face inference implementation"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, Protocol
from urllib import error, request

from book_companion.config import (
    EMBEDDING_MODEL_NAME,
    HF_INFERENCE_API_TIMEOUT_SECONDS,
    HF_INFERENCE_API_URL,
    HF_TOKEN_ENV_VAR,
)


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


class HFInferenceEmbeddingsClient:
    """Embeddings client backed by Hugging Face Inference API."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        base_url: str = HF_INFERENCE_API_URL,
        token_env_var: str = HF_TOKEN_ENV_VAR,
        timeout_seconds: int = HF_INFERENCE_API_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.token_env_var = token_env_var
        self.timeout_seconds = timeout_seconds

    def _post_inference(self, payload: dict) -> list:
        token = os.getenv(self.token_env_var)
        if not token:
            raise RuntimeError(f"Missing required environment variable: {self.token_env_var}")

        req = request.Request(
            url=f"{self.base_url}/{self.model_name}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HF inference API error ({exc.code}): {body}") from exc

    def embed_text(self, text: str) -> list[float]:
        data = self._post_inference({"inputs": text, "options": {"wait_for_model": True}})
        if not isinstance(data, list) or not data:
            raise RuntimeError("Unexpected HF embedding response for single input.")
        return _normalize([float(v) for v in data])

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        data = self._post_inference({"inputs": text_list, "options": {"wait_for_model": True}})
        if not isinstance(data, list) or not data:
            raise RuntimeError("Unexpected HF embedding response for batch input.")

        if isinstance(data[0], list):
            return [_normalize([float(v) for v in row]) for row in data]

        return [_normalize([float(v) for v in data])]

