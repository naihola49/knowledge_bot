"""Node 2 orchestration: build clarification topics from Node 1 output"""

from __future__ import annotations

from book_companion.nodes.node2_methods import build_topic_explanations, extract_candidate_topics
from book_companion.schema.validation import validate_graph_state, validate_output_2
from book_companion.state import GraphState, Output2


def run_clarification_node(state: GraphState) -> GraphState:
    """
    Build `output_2.topics` from `output_1` using Node 2 helper methods
    - This node focuses only on producing Output2 contract payload
    """
    validated_state = validate_graph_state(state, context="clarification input")
    output_1 = validated_state.get("output_1")
    if output_1 is None:
        validated_output_2 = validate_output_2({"topics": []})
        return {**validated_state, "output_2": validated_output_2}

    topic_ids = extract_candidate_topics(output_1)
    topics = build_topic_explanations(output_1, topic_ids)
    output_2: Output2 = {"topics": topics}
    validated_output_2 = validate_output_2(output_2)
    return {**validated_state, "output_2": validated_output_2}
