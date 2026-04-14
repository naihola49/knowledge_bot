"""Node 1: Comprehension check for daily notes."""

from __future__ import annotations

import re
from pathlib import Path

from book_companion.config import (
    CLARIFICATION_TRIGGER_SCORE,
    COMPREHENSION_PASS_SCORE,
    MAX_LOOPS,
    MIN_WORD_COUNT,
)
from book_companion.state import GraphState, Output1


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _comprehension_score(text: str) -> float:
    """TODO, potentially BERT"""


def _derive_weak_topics(text: str) -> list[str] | None:
    """TODO"""
    


def build_output_1(day: str, text: str) -> Output1:
    cleaned_summary = _clean_text(text)
    comprehension_score = _comprehension_score(cleaned_summary)
    weak_topics = _derive_weak_topics(cleaned_summary)
    needs_clarification = comprehension_score < CLARIFICATION_TRIGGER_SCORE or bool(weak_topics)

    return {
        "day": day,
        "comprehension_score": comprehension_score,
        "needs_clarification": needs_clarification,
        "weak_topics": weak_topics,
        "cleaned_summary": cleaned_summary,
    }


def run_comprehension_node(state: GraphState) -> GraphState:
    """Read notes, score comprehension, and write Node 1 output into state."""
    notes_path = Path(state["daily_notes_path"])
    notes_text = notes_path.read_text(encoding="utf-8")

    word_count = _count_words(notes_text)
    day = state.get("day", "unknown_day")
    loop_count = state.get("loop_count", 0)

    if word_count < MIN_WORD_COUNT:
        output_1 = build_output_1(day=day, text=notes_text)
        output_1["needs_clarification"] = True
        return {
            **state,
            "output_1": output_1,
            "weak_topics": output_1["weak_topics"] or ["insufficient note length"],
            "store_ready": False,
            "exit_reason": "continue",
            "loop_count": loop_count,
            "max_loops": state.get("max_loops", MAX_LOOPS),
        }

    output_1 = build_output_1(day=day, text=notes_text)
    score = output_1["comprehension_score"]
    hit_max_loops = loop_count >= state.get("max_loops", MAX_LOOPS)

    if hit_max_loops:
        exit_reason = "max_loops"
        store_ready = False
    elif score >= COMPREHENSION_PASS_SCORE and not output_1["needs_clarification"]:
        exit_reason = "done"
        store_ready = True
    else:
        exit_reason = "continue"
        store_ready = False

    return {
        **state,
        "output_1": output_1,
        "weak_topics": output_1["weak_topics"] or [],
        "store_ready": store_ready,
        "exit_reason": exit_reason,
        "loop_count": loop_count,
        "max_loops": state.get("max_loops", MAX_LOOPS),
    }
