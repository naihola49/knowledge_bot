"""Anthropic adapter for deterministic topic hydration."""

from __future__ import annotations

import json
import os
from typing import Any

from book_companion.config import ANTHROPIC_API_KEY_ENV_VAR, ANTHROPIC_MODEL_NAME
from book_companion.integrations.anthropic_layer.contract import (
    build_topic_skeleton,
    merge_hydrations,
)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = next((p for p in parts if "{" in p and "}" in p), text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


class AnthropicTopicCompiler:
    """Hydrates a deterministic topic skeleton with compact LLM output."""

    def __init__(self, client: Any, *, model: str = ANTHROPIC_MODEL_NAME) -> None:
        self._client = client
        self._model = model

    def compile_topics(self, payload: dict, *, candidate_ids: list[str], max_topics: int) -> list[dict]:
        skeleton = build_topic_skeleton(candidate_ids, max_topics=max_topics)
        if not skeleton:
            return []

        prompt = (
            "Return ONLY JSON object: {\"hydrations\":[{\"id\":\"1\",\"topic\":\"...\","
            "\"error_explanation\":\"...\",\"confidence\":0.0}]}. "
            "Hydrate rows by id only. Keep topic <= 8 words and error_explanation <= 140 chars."
        )
        user_payload = {"context": payload, "rows": skeleton}
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=260,
            temperature=0.1,
            system=prompt,
            messages=[{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", "") == "text":
                text += block.text
        parsed = _extract_json_object(text)
        hydrations = parsed.get("hydrations", [])
        return merge_hydrations(skeleton, hydrations if isinstance(hydrations, list) else [])


_CLIENT: AnthropicTopicCompiler | None = None


def get_anthropic_topic_compiler(
    api_key_env_var: str = ANTHROPIC_API_KEY_ENV_VAR,
    model: str = ANTHROPIC_MODEL_NAME,
) -> AnthropicTopicCompiler | None:
    """Return compiler when configured; otherwise None for fallback path."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        return None

    try:
        from anthropic import Anthropic
    except Exception:
        return None

    _CLIENT = AnthropicTopicCompiler(Anthropic(api_key=api_key), model=model)
    return _CLIENT
