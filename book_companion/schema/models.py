"""Pydantic schemas used to validate node inputs/outputs and graph state."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Score = Field(ge=0.0, le=1.0)


class RetrievedChunkModel(BaseModel):
    chunk_id: str = Field(min_length=1)
    text: str
    similarity: float = Score


class NLIResultModel(BaseModel):
    chunk_id: str = Field(min_length=1)
    entailment: float = Score
    neutral: float = Score
    contradiction: float = Score


class Output1Model(BaseModel):
    day: str = Field(min_length=1)
    comprehension_score: float = Score
    needs_clarification: bool
    weak_topics: list[str] | None = None
    user_input: str
    retrieved_chunks: list[RetrievedChunkModel]
    nli_results: list[NLIResultModel]
    coverage_score: float = Score
    contradiction_score: float = Score


class ClarificationTopicModel(BaseModel):
    topic: str = Field(min_length=1)
    error_explanation: str = Field(min_length=1)
    confidence: float = Score


class Output2Model(BaseModel):
    topics: list[ClarificationTopicModel] = Field(default_factory=list)


class Output3Model(BaseModel):
    research_md_path: str
    prompt_user_retry: bool


class Output4Model(BaseModel):
    day: str = Field(min_length=1)
    success: bool
    synthesis_txt_path: str
    retrieved_chunk_ids: list[str]
    synthesis_text: str

"""
premise ingestion models 
"""
IntentKind = Literal["daily_interest", "correction_topic"]


class InterestIntentModel(BaseModel):
    topic: str = Field(min_length=1)
    why_today: str | None = None
    priority: float = Score


class CorrectionIntentModel(BaseModel):
    topic: str = Field(min_length=1)
    error_explanation: str = Field(min_length=1)
    confidence: float = Score


class PremiseIngestionRequestModel(BaseModel):
    day: str = Field(min_length=1)
    daily_interests: list[InterestIntentModel] = Field(default_factory=list)
    corrections: list[CorrectionIntentModel] = Field(default_factory=list)
    max_queries: int = Field(default=8, ge=1)


class QuerySpecModel(BaseModel):
    query: str = Field(min_length=1)
    intent_kind: IntentKind
    topic: str = Field(min_length=1)
    priority: float = Score
    rationale: str | None = None


class PremiseDocModel(BaseModel):
    day: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    intent_kind: IntentKind
    query: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str
    snippet: str
    source_score: float = Score
    raw_content: str = Field(min_length=1)


class GraphStateModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    day: str | None = None
    loop_count: int | None = None
    max_loops: int | None = None
    store_ready: bool | None = None
    exit_reason: Literal["continue", "done", "max_loops", "manual_review"] | None = None
    weak_topics: list[str] | None = None
    raw_content_path: str | None = None
    user_input_path: str | None = None
    output_1: Output1Model | None = None
    output_2: Output2Model | None = None
    output_3: Output3Model | None = None
    output_4: Output4Model | None = None


class ComprehensionInputStateModel(BaseModel):
    """Minimum required state shape to run Node 1."""

    model_config = ConfigDict(extra="allow")

    raw_content_path: str
    user_input_path: str
    day: str | None = None
    loop_count: int | None = None
    max_loops: int | None = None
