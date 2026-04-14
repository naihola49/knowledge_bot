"""State contracts for the LangGraph workflow defined in architecture.txt"""

from typing import Literal, TypedDict


class TopicIssue(TypedDict):
    topic: str
    error_explanation: str
    confidence: float


class Output1(TypedDict):
    day: str
    comprehension_score: float
    needs_clarification: bool
    weak_topics: list[str] | None
    cleaned_summary: str


class Output2(TypedDict, total=False):
    topics: list[TopicIssue]


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
    daily_notes_path: str
    output_1: Output1
    output_2: Output2
    output_3: Output3
    output_4: Output4
