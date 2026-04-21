from __future__ import annotations

import os
from typing import Any

import pytest

from book_companion.premise_ingestion import (
    CorrectionIntent,
    InterestIntent,
    PremiseIngestionRequest,
    build_premises_with_tavily,
)


class _FakeTavilyClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.extract_calls: list[dict[str, Any]] = []

    def search(self, *, query: str, max_results: int) -> dict[str, Any]:
        self.search_calls.append({"query": query, "max_results": max_results})
        if "misconceptions" in query:
            return {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "content": "snippet-a",
                        "score": 0.91,
                    },
                    {
                        "url": "https://example.com/b",
                        "title": "B",
                        "content": "snippet-b",
                        "score": 0.64,
                    },
                ]
            }
        return {
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A-low",
                    "content": "snippet-a-low",
                    "score": 0.31,
                }
            ]
        }

    def extract(self, *, urls: list[str], extract_depth: str | None = None) -> dict[str, Any]:
        self.extract_calls.append({"urls": urls, "extract_depth": extract_depth})
        return {
            "results": [
                {"url": "https://example.com/a", "raw_content": "full-content-a"},
                {"url": "https://example.com/b", "raw_content": "full-content-b"},
            ]
        }


def test_build_premises_with_tavily_searches_extracts_and_normalizes() -> None:
    req = PremiseIngestionRequest(
        day="2026-04-21",
        daily_interests=[InterestIntent(topic="Reinforcement learning", priority=0.4)],
        corrections=[
            CorrectionIntent(
                topic="gradient descent",
                error_explanation="confused update direction",
                confidence=0.9,
            )
        ],
        max_queries=4,
    )
    fake = _FakeTavilyClient()
    docs = build_premises_with_tavily(req, client=fake, max_results_per_query=5)

    assert len(fake.search_calls) >= 2
    assert len(fake.extract_calls) == 1
    assert len(docs) == 2

    # Dedup keeps best-scoring hit metadata for repeated URL.
    top = docs[0]
    assert top.url == "https://example.com/a"
    assert top.source_score == 0.91
    assert top.raw_content == "full-content-a"
    assert top.intent_kind in ("daily_interest", "correction_topic")


requires_tavily = pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set; live Tavily test skipped.",
)


@requires_tavily
def test_build_premises_with_tavily_live_prints_output() -> None:
    """
    Live integration check against Tavily search+extract.
    Run with `-s` to view printed output in terminal.
    """
    req = PremiseIngestionRequest(
        day="2026-04-21",
        daily_interests=[InterestIntent(topic="what is overfitting in machine learning", priority=0.6)],
        corrections=[
            CorrectionIntent(
                topic="gradient descent",
                error_explanation="misunderstood update direction and learning dynamics",
                confidence=0.8,
            )
        ],
        max_queries=2,
    )

    docs = build_premises_with_tavily(req, max_results_per_query=3)
    assert len(docs) >= 1

    print("\n=== Tavily Live Premise Docs ===")
    print(f"doc_count={len(docs)}")
    for idx, d in enumerate(docs[:5], start=1):
        preview = d.raw_content[:280].replace("\n", " ").strip()
        print(f"\n[{idx}] topic={d.topic!r} intent={d.intent_kind} score={d.source_score:.4f}")
        print(f"title: {d.title}")
        print(f"url: {d.url}")
        print(f"query: {d.query}")
        print(f"snippet: {d.snippet[:180]!r}")
        print(f"raw_preview: {preview!r}")

