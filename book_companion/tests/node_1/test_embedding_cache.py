"""Tests for same-day embedding cache."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from book_companion.services.embedding_cache import SameDayCachedEmbeddingsClient


def test_embed_text_cache_hit_calls_inner_once() -> None:
    inner = MagicMock()
    inner.embed_text.return_value = [0.1, 0.2, 0.3]
    client = SameDayCachedEmbeddingsClient(inner, model_name="test-model")

    a = client.embed_text("same string")
    b = client.embed_text("same string")
    assert a == b == [0.1, 0.2, 0.3]
    inner.embed_text.assert_called_once_with("same string")


def test_embed_texts_batch_reuses_cached_rows() -> None:
    inner = MagicMock()
    inner.embed_texts.return_value = [[1.0, 0.0], [0.0, 1.0]]
    client = SameDayCachedEmbeddingsClient(inner, model_name="m")

    first = client.embed_texts(["alpha", "beta"])
    second = client.embed_texts(["alpha", "beta"])
    assert first == second
    assert inner.embed_texts.call_count == 1


@patch("book_companion.services.embedding_cache.date")
def test_cache_clears_on_new_calendar_day(mock_date: MagicMock) -> None:
    # _roll_day_if_needed calls today() once per embed_text.
    mock_date.today.side_effect = [
        date(2026, 1, 1),
        date(2026, 1, 1),
        date(2026, 1, 2),
    ]

    inner = MagicMock()
    inner.embed_text.return_value = [1.0]
    client = SameDayCachedEmbeddingsClient(inner, model_name="m")

    client.embed_text("x")
    client.embed_text("x")
    assert inner.embed_text.call_count == 1

    client.embed_text("x")
    assert inner.embed_text.call_count == 2
