"""Premise ingestion package

Stack:
- tavily API for URL search-extract
- Anthropic API for Reasoning
"""

from .models import (
    CorrectionIntent,
    InterestIntent,
    PremiseDoc,
    PremiseIngestionRequest,
    QuerySpec,
)
from .adapters import build_premises_from_output_2, build_request_from_output_2
from .planner import build_query_plan
from .tavily_pipeline import build_premises_with_tavily

__all__ = [
    "InterestIntent",
    "CorrectionIntent",
    "PremiseDoc",
    "PremiseIngestionRequest",
    "QuerySpec",
    "build_request_from_output_2",
    "build_premises_from_output_2",
    "build_query_plan",
    "build_premises_with_tavily",
]