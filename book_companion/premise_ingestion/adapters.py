"""Adapters between node outputs and premise-ingestion request models"""

from __future__ import annotations

from book_companion.premise_ingestion.models import (
    CorrectionIntent,
    InterestIntent,
    PremiseDoc,
    PremiseIngestionRequest,
)
from book_companion.premise_ingestion.tavily_pipeline import TavilyLikeClient, build_premises_with_tavily
from book_companion.schema.validation import validate_output_2, validate_premise_ingestion_request


def build_request_from_output_2(
    *,
    day: str,
    output_2: dict,
    daily_interests: list[dict] | None = None,
    max_queries: int = 8,
) -> PremiseIngestionRequest:
    """
    Build a validated premise-ingestion request from Node 2 output.

    This keeps Node 3 input wiring strict while allowing optional day-level interests.
    """
    validated_output_2 = validate_output_2(output_2)
    corrections = [
        CorrectionIntent(
            topic=topic["topic"],
            error_explanation=topic["error_explanation"],
            confidence=topic["confidence"],
        )
        for topic in validated_output_2["topics"]
    ]
    interests = [
        InterestIntent(
            topic=str(item.get("topic", "")).strip(),
            why_today=item.get("why_today"),
            priority=float(item.get("priority", 0.6)),
        )
        for item in (daily_interests or [])
        if str(item.get("topic", "")).strip()
    ]

    payload = validate_premise_ingestion_request(
        {
            "day": day,
            "daily_interests": [i.__dict__ for i in interests],
            "corrections": [c.__dict__ for c in corrections],
            "max_queries": max_queries,
        }
    )
    return PremiseIngestionRequest(
        day=payload["day"],
        daily_interests=[InterestIntent(**row) for row in payload["daily_interests"]],
        corrections=[CorrectionIntent(**row) for row in payload["corrections"]],
        max_queries=payload["max_queries"],
    )


def build_premises_from_output_2(
    *,
    day: str,
    output_2: dict,
    daily_interests: list[dict] | None = None,
    max_queries: int = 8,
    client: TavilyLikeClient | None = None,
) -> list[PremiseDoc]:
    """
    Node-3-facing facade:
    output_2 -> validated ingestion request -> Tavily-backed premise docs.
    """
    request = build_request_from_output_2(
        day=day,
        output_2=output_2,
        daily_interests=daily_interests,
        max_queries=max_queries,
    )
    return build_premises_with_tavily(request, client=client)
