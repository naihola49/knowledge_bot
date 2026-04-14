"""Routing helpers for conditional LangGraph edges."""

from book_companion.state import GraphState


def route_after_comprehension(state: GraphState) -> str:
    if state.get("exit_reason") == "max_loops":
        return "end"
    if state.get("output_1", {}).get("needs_clarification"):
        return "clarification"
    return "synthesis"

