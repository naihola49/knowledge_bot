"""Tests for Node 2 clarification orchestration"""

from __future__ import annotations

from book_companion.nodes.clarification import run_clarification_node
from book_companion.nodes.comprehension import MERGED_PREMISE_CHUNK_ID
from book_companion.state import GraphState


def _base_state() -> GraphState:
    return {
        "day": "day_2",
        "loop_count": 0,
        "max_loops": 3,
        "output_1": {
            "day": "day_2",
            "comprehension_score": 0.44,
            "needs_clarification": True,
            "weak_topics": [MERGED_PREMISE_CHUNK_ID, "high contradiction overall", "chunk_0"],
            "user_input": "I think gradient descent increases the loss first.",
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk_0",
                    "text": "Gradient descent follows the negative gradient to reduce loss.",
                    "similarity": 0.73,
                }
            ],
            "nli_results": [
                {
                    "chunk_id": MERGED_PREMISE_CHUNK_ID,
                    "entailment": 0.42,
                    "neutral": 0.18,
                    "contradiction": 0.86,
                },
                {
                    "chunk_id": "chunk_0",
                    "entailment": 0.35,
                    "neutral": 0.20,
                    "contradiction": 0.78,
                },
            ],
            "coverage_score": 0.24,
            "contradiction_score": 0.86,
        },
    }


def test_run_clarification_node_builds_output2_topics() -> None:
    state = _base_state()
    out = run_clarification_node(state)

    assert "output_2" in out
    topics = out["output_2"]["topics"]
    assert len(topics) >= 1

    first = topics[0]
    assert "topic" in first
    assert "error_explanation" in first
    assert "confidence" in first
    assert isinstance(first["error_explanation"], str)
    assert len(first["error_explanation"]) > 30
    assert 0.0 <= first["confidence"] <= 1.0


def test_run_clarification_node_handles_missing_output1() -> None:
    out = run_clarification_node({"day": "day_2"})
    assert out["output_2"]["topics"] == []

