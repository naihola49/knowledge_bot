"""State contracts for the LangGraph workflow defined in architecture.txt."""

from typing import Literal, TypedDict
# TypedDict v1, switch to pydantic v2 for type safety!

class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    similarity: float # cosine sim! < magnitude agnostic > 


class NLIResult(TypedDict):
    chunk_id: str
    entailment: float # 2
    neutral: float # 1
    contradiction: float # 0


class Output1(TypedDict):
    day: str
    comprehension_score: float  # final Node 1 score after retrieval + NLI aggregation
    needs_clarification: bool
    weak_topics: list[str] | None
    user_input: str
    retrieved_chunks: list[RetrievedChunk]
    nli_results: list[NLIResult]
    coverage_score: float
    contradiction_score: float


class ClarificationTopic(TypedDict):
    topic: str
    error_explanation: str
    confidence: float


class Output2(TypedDict, total=False):
    topics: list[ClarificationTopic]


class Output3(TypedDict):
    research_md_path: str
    prompt_user_retry: bool


class Output4(TypedDict):
    day: str
    success: bool
    synthesis_txt_path: str
    retrieved_chunk_ids: list[str]
    synthesis_text: str


class GraphState(TypedDict, total=False):
    run_id: str
    day: str
    loop_count: int
    max_loops: int
    store_ready: bool
    exit_reason: Literal["continue", "done", "max_loops", "manual_review"]
    weak_topics: list[str]
    raw_content_path: str
    user_input_path: str
    output_1: Output1
    output_2: Output2
    output_3: Output3
    output_4: Output4
