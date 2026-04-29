"""Graph assembly module"""

from book_companion.edges.routing import route_after_comprehension
from book_companion.nodes.clarification import run_clarification_node
from book_companion.nodes.comprehension import run_comprehension_node
from book_companion.nodes.research import run_research_node
from book_companion.schema.validation import validate_graph_state


def run_graph_once(initial_state: dict) -> dict:
    """
    TODO: Assemble when all nodes completed
    """
    state = validate_graph_state(initial_state, context="graph initial_state")
    state = run_comprehension_node(state)
    state = validate_graph_state(state, context="graph after_comprehension")
    next_node = route_after_comprehension(state)

    if next_node == "clarification":
        state = run_clarification_node(state)
        state = validate_graph_state(state, context="graph after_clarification")
        state = run_research_node(state)
        state = validate_graph_state(state, context="graph after_research")
        state["next_node"] = "synthesis"
    else:
        state["next_node"] = next_node

    return validate_graph_state(state, context="graph final_state")

