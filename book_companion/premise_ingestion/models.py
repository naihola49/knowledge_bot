"""Typed contracts for premise-ingestion planning.

- Flexible entry inputs (beginning of workflow + node 3 research subgraph)
- Define normalized query objects for Tavily
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


IntentKind = Literal["daily_interest", "correction_topic"]


@dataclass(frozen=True)
class InterestIntent:
    """User-declared topic of interest for today's workflow kickoff."""

    topic: str
    why_today: str | None = None
    priority: float = 0.6


@dataclass(frozen=True)
class CorrectionIntent:
    """Node-2-derived correction target that needs researched clarification."""

    topic: str
    error_explanation: str
    confidence: float


@dataclass(frozen=True)
class PremiseIngestionRequest:
    """Flexible entry request accepted by premise-ingestion entry point."""

    day: str
    daily_interests: list[InterestIntent] = field(default_factory=list)
    corrections: list[CorrectionIntent] = field(default_factory=list)
    max_queries: int = 8


@dataclass(frozen=True)
class QuerySpec:
    """Normalized search query + metadata for downstream search/extract steps."""

    query: str
    intent_kind: IntentKind
    topic: str
    priority: float
    rationale: str | None = None


@dataclass(frozen=True)
class PremiseDoc:
    """Normalized premise document built from search + extract output"""

    day: str
    topic: str
    intent_kind: IntentKind
    query: str
    url: str
    title: str
    snippet: str
    source_score: float
    raw_content: str

