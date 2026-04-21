"""Tests for Node 1 chunking and embedding helpers"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from book_companion.nodes.node1_methods.chunking import chunk_text, clean_text
from book_companion.nodes.node1_methods import embedding as embedding_module

# test article
ARTICLE = """
In a post on Truth Social, Trump said Lebanese President Joseph Aoun and Israeli Prime Minister Benjamin Netanyahu have agreed that in order to achieve PEACE between their Countries, they will formally begin a 10 Day CEASEFIRE at 5 P.M. EST Trump said that he had directed Vice President JD Vance and Secretary of State Rubio to work with the countries toward achieving a Lasting PEACE, adding that he had invited Aoun and Netanyahu to take part in peace talks at the White House. 
Lebanese Prime Minister Nawaf Salam welcomed the ceasefire in a post on X, saying it had been a key objective for Lebanon in talks this week. In a video statement, Netanyahu confirmed he had agreed to a temporary ceasefire, but said that Israel had not agreed to withdraw from southern Lebanon, a key demand of Hezbollah, adding that the group must be dismantled. We are remaining in Lebanon in an expanded security zone, he said, adding that this was necessary due to the danger of an invasion and to prevent fire into Israel. 
It was unclear when or if those displaced from their homes in southern Lebanon by Israel's invasion would be allowed to return. Lebanon's Parliament Speaker Nabih Berri warned people to postpone their return to their towns and villages until the situation becomes clearer, in accordance with the ceasefire agreement.
"""


class TestChunking:
    def test_clean_text_collapses_whitespace(self) -> None:
        assert clean_text("  hello   world  \n\t") == "hello world"

    def test_clean_text_empty(self) -> None:
        assert clean_text("") == ""

    def test_chunk_text_empty_returns_empty_list(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_short_single_chunk(self) -> None:
        words = " ".join([f"w{i}" for i in range(10)])
        out = chunk_text(words, chunk_size=120, overlap=20)
        assert len(out) == 1
        assert out[0] == words

    def test_chunk_text_overlap_produces_multiple_chunks(self) -> None:
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        out = chunk_text(text, chunk_size=10, overlap=2)
        assert len(out) >= 2
        assert all(isinstance(c, str) and c for c in out)
        joined = " ".join(out)
        for w in words:
            assert w in joined

    def test_chunk_article_default_produces_at_least_one_chunk(self) -> None:
        chunks = chunk_text(ARTICLE, chunk_size=40, overlap=8)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_chunk_article_words_preserved_across_chunks(self) -> None:
        cleaned = clean_text(ARTICLE)
        words = cleaned.split()
        if len(words) <= 40:
            pytest.skip("ARTICLE is short; use a longer paste to exercise overlap.")

        chunks = chunk_text(ARTICLE, chunk_size=40, overlap=8)
        joined = " ".join(chunks)
        for w in words:
            assert w in joined


class TestEmbedding:
    @pytest.fixture(autouse=True)
    def reset_embedding_client(self) -> None:
        embedding_module._CLIENT = None
        yield
        embedding_module._CLIENT = None

    @patch.object(embedding_module, "_get_client")
    def test_vectorize_text_delegates_to_client(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.embed_text.return_value = [0.25, 0.5, 1.0]
        mock_get_client.return_value = mock_client

        result = embedding_module.vectorize_text("test phrase")
        assert result == [0.25, 0.5, 1.0]
        mock_client.embed_text.assert_called_once_with("test phrase")

    @patch.object(embedding_module, "_get_client")
    def test_vectorize_texts_delegates_to_client(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = [[1.0, 0.0], [0.0, 1.0]]
        mock_get_client.return_value = mock_client

        result = embedding_module.vectorize_texts(["a", "b"])
        assert result == [[1.0, 0.0], [0.0, 1.0]]
        mock_client.embed_texts.assert_called_once()
        call_args = mock_client.embed_texts.call_args[0][0]
        assert list(call_args) == ["a", "b"]

    @patch.object(embedding_module, "_get_client")
    def test_vectorize_texts_empty_iterable(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = []
        mock_get_client.return_value = mock_client

        assert embedding_module.vectorize_texts([]) == []
        mock_client.embed_texts.assert_called_once()

    @patch.object(embedding_module, "_get_client")
    def test_vectorize_texts_article_chunks_delegates(self, mock_get_client: MagicMock) -> None:
        chunks = chunk_text(ARTICLE, chunk_size=40, overlap=8)
        if not chunks:
            pytest.skip("ARTICLE is empty; paste content into ARTICLE.")

        fake_vecs = [[float(i), float(i + 1)] for i in range(len(chunks))]
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = fake_vecs
        mock_get_client.return_value = mock_client

        result = embedding_module.vectorize_texts(chunks)
        assert result == fake_vecs
        mock_client.embed_texts.assert_called_once()
        passed = list(mock_client.embed_texts.call_args[0][0])
        assert passed == chunks
