"""Node 2 orchestration: build clarification topics from Node 1 output"""

from __future__ import annotations

from book_companion.nodes.node2_methods import build_topic_explanations, extract_candidate_topics
from book_companion.state import GraphState, Output2


def run_clarification_node(state: GraphState) -> GraphState:
    """
    Build `output_2.topics` from `output_1` using Node 2 helper methods
    - This node focuses only on producing Output2 contract payload
    """
    output_1 = state.get("output_1")
    if output_1 is None: # TODO: add schema val + exit condition
        return {**state, "output_2": {"topics": []}}

    topic_ids = extract_candidate_topics(output_1)
    topics = build_topic_explanations(output_1, topic_ids)
    output_2: Output2 = {"topics": topics}
    return {**state, "output_2": output_2}
