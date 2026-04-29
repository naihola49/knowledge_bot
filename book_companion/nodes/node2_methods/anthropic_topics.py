"""Compact topic generation from Output1 using Anthropic"""

from __future__ import annotations

from book_companion.config import (
    NODE2_ANTHROPIC_MAX_RETRIEVED_CHUNKS,
    NODE2_ANTHROPIC_MAX_TOPICS,
    NODE2_ANTHROPIC_MAX_USER_INPUT_CHARS,
    NODE2_ANTHROPIC_MAX_WEAK_TOPICS,
)
from book_companion.integrations import get_anthropic_topic_compiler
from book_companion.schema.validation import validate_output_2
from book_companion.state import Output1
from .topic_candidates import extract_candidate_topics


def _build_compact_payload(output_1: Output1) -> dict:
    return {
        "scores": {
            "comprehension_score": output_1.get("comprehension_score", 0.0),
            "coverage_score": output_1.get("coverage_score", 0.0),
            "contradiction_score": output_1.get("contradiction_score", 0.0),
        },
        "weak_topics": (output_1.get("weak_topics") or [])[:NODE2_ANTHROPIC_MAX_WEAK_TOPICS],
        "user_input": (output_1.get("user_input") or "")[:NODE2_ANTHROPIC_MAX_USER_INPUT_CHARS],
        "retrieved_chunks": [
            {"chunk_id": c.get("chunk_id"), "text": (c.get("text") or "")[:120]}
            for c in output_1.get("retrieved_chunks", [])[:NODE2_ANTHROPIC_MAX_RETRIEVED_CHUNKS]
        ],
        "nli_results": [
            {
                "chunk_id": r.get("chunk_id"),
                "contradiction": r.get("contradiction"),
                "entailment": r.get("entailment"),
            }
            for r in output_1.get("nli_results", [])[:NODE2_ANTHROPIC_MAX_RETRIEVED_CHUNKS]
        ],
    }


def build_topics_with_anthropic(output_1: Output1) -> list[dict] | None:
    """Return validated topics list or None when compiler unavailable/fails."""
    compiler = get_anthropic_topic_compiler()
    if compiler is None:
        return None

    try:
        payload = _build_compact_payload(output_1)
        candidate_ids = extract_candidate_topics(output_1, max_topics=NODE2_ANTHROPIC_MAX_TOPICS)
        topics = compiler.compile_topics(
            payload,
            candidate_ids=candidate_ids,
            max_topics=NODE2_ANTHROPIC_MAX_TOPICS,
        )
        validated = validate_output_2({"topics": topics})
        return validated["topics"]
    except Exception:
        return None
