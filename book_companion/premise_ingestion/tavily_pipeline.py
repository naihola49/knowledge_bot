"""Tavily search + extract pipeline for premise ingestion package"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from book_companion.config import (
    TAVILY_EXTRACT_DEPTH,
    TAVILY_MAX_RESULTS_PER_QUERY,
)
from book_companion.integrations.tavily import get_tavily_client
from book_companion.premise_ingestion.models import PremiseDoc, PremiseIngestionRequest
from book_companion.premise_ingestion.planner import build_query_plan


class TavilyLikeClient(Protocol):
    def search(self, *, query: str, max_results: int) -> Mapping[str, Any]:
        ...

    def extract(self, *, urls: list[str], extract_depth: str | None = None) -> Mapping[str, Any]:
        ...


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_premises_with_tavily(
    request: PremiseIngestionRequest,
    *,
    max_results_per_query: int = TAVILY_MAX_RESULTS_PER_QUERY,
    extract_depth: str | None = TAVILY_EXTRACT_DEPTH,
    client: TavilyLikeClient | None = None,
) -> list[PremiseDoc]:
    """
    Execute planning -> Tavily search -> Tavily extract and return normalized premise docs.

    The same URL can appear in multiple queries. Deduping is done by URL while keeping
    the highest-score search hit as the representative metadata row.
    """
    tavily_client: TavilyLikeClient = client or get_tavily_client()
    query_plan = build_query_plan(request)
    if not query_plan:
        return []

    hits_by_url: dict[str, dict[str, Any]] = {}
    for spec in query_plan:
        payload = tavily_client.search(query=spec.query, max_results=max_results_per_query)
        results = payload.get("results", [])
        if not isinstance(results, list):
            continue
        for hit in results:
            if not isinstance(hit, dict):
                continue
            url = str(hit.get("url", "")).strip()
            if not url:
                continue
            score = _to_float(hit.get("score"))
            existing = hits_by_url.get(url)
            if existing is None or score > _to_float(existing.get("score")):
                hits_by_url[url] = {
                    **hit,
                    "_topic": spec.topic,
                    "_intent_kind": spec.intent_kind,
                    "_query": spec.query,
                }

    if not hits_by_url:
        return []

    urls = list(hits_by_url.keys())
    extracted = tavily_client.extract(urls=urls, extract_depth=extract_depth)
    extracted_results = extracted.get("results", [])
    if not isinstance(extracted_results, list):
        return []

    content_by_url: dict[str, str] = {}
    for row in extracted_results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        raw_content = str(row.get("raw_content") or row.get("content") or "").strip()
        if raw_content:
            content_by_url[url] = raw_content

    docs: list[PremiseDoc] = []
    for url, hit in hits_by_url.items():
        raw = content_by_url.get(url, "")
        if not raw:
            continue
        intent_kind_raw = str(hit.get("_intent_kind", "daily_interest")).strip()
        intent_kind = "correction_topic" if intent_kind_raw == "correction_topic" else "daily_interest"
        docs.append(
            PremiseDoc(
                day=request.day,
                topic=str(hit.get("_topic", "")).strip(),
                intent_kind=intent_kind,
                query=str(hit.get("_query", "")).strip(),
                url=url,
                title=str(hit.get("title", "")).strip(),
                snippet=str(hit.get("content", "")).strip(),
                source_score=_to_float(hit.get("score")),
                raw_content=raw,
            )
        )

    docs.sort(key=lambda d: d.source_score, reverse=True)
    return docs

