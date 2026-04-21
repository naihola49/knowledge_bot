from __future__ import annotations

from book_companion.premise_ingestion import (
    CorrectionIntent,
    InterestIntent,
    PremiseIngestionRequest,
    build_query_plan,
)


def test_build_query_plan_prioritizes_corrections_and_caps_results() -> None:
    request = PremiseIngestionRequest(
        day="2026-04-21",
        daily_interests=[
            InterestIntent(topic="US inflation outlook", priority=0.4),
            InterestIntent(topic="Neural scaling laws", priority=0.7),
        ],
        corrections=[
            CorrectionIntent(
                topic="gradient descent",
                error_explanation="Misstated that descent increases loss first.",
                confidence=0.9,
            ),
        ],
        max_queries=2,
    )

    plan = build_query_plan(request)
    assert len(plan) == 2
    assert plan[0].intent_kind == "correction_topic"
    assert "misconceptions" in plan[0].query


def test_build_query_plan_dedupes_query_text() -> None:
    request = PremiseIngestionRequest(
        day="2026-04-21",
        daily_interests=[
            InterestIntent(topic="Reinforcement learning", priority=0.2),
            InterestIntent(topic="reinforcement learning", priority=0.8),
        ],
        corrections=[],
        max_queries=5,
    )

    plan = build_query_plan(request)
    assert len(plan) == 1
    assert plan[0].query == "reinforcement learning"
    assert plan[0].priority == 0.8

