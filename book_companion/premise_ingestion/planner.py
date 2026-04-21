"""Query planning entry point for premise ingestion

Scope:
- Build ranked, deduped query plan from interests + corrections
- Keep output provider-agnostic
"""

from __future__ import annotations

from book_companion.premise_ingestion.models import (
    CorrectionIntent,
    InterestIntent,
    PremiseIngestionRequest,
    QuerySpec,
)


def _clamp_priority(value: float) -> float:
    return max(0.0, min(1.0, value))


def _interest_to_query(intent: InterestIntent) -> QuerySpec:
    rationale = intent.why_today
    query = intent.topic.strip()
    return QuerySpec(
        query=query,
        intent_kind="daily_interest",
        topic=intent.topic.strip(),
        priority=_clamp_priority(intent.priority),
        rationale=rationale.strip() if rationale else None,
    )


def _correction_to_query(intent: CorrectionIntent) -> QuerySpec:
    # Bias correction searches higher: these directly resolve detected misunderstandings.
    priority = _clamp_priority(0.55 + 0.45 * intent.confidence)
    rationale = intent.error_explanation.strip()
    query = f"{intent.topic.strip()} explanation common misconceptions"
    return QuerySpec(
        query=query,
        intent_kind="correction_topic",
        topic=intent.topic.strip(),
        priority=priority,
        rationale=rationale,
    )


def _dedupe_by_query(specs: list[QuerySpec]) -> list[QuerySpec]:
    deduped: dict[str, QuerySpec] = {}
    for spec in specs:
        key = spec.query.lower().strip()
        if key not in deduped or spec.priority > deduped[key].priority:
            deduped[key] = spec
    return list(deduped.values())


def build_query_plan(request: PremiseIngestionRequest) -> list[QuerySpec]:
    """Entry point: produce ranked search query specs for the current day."""
    raw_specs: list[QuerySpec] = []
    raw_specs.extend(_interest_to_query(i) for i in request.daily_interests)
    raw_specs.extend(_correction_to_query(c) for c in request.corrections)

    deduped = _dedupe_by_query(raw_specs)
    deduped.sort(key=lambda s: s.priority, reverse=True)

    limit = max(1, request.max_queries)
    return deduped[:limit]

