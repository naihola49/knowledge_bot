"""Graph assembly module."""

from book_companion.edges.routing import route_after_comprehension
from book_companion.nodes.comprehension import run_comprehension_node
from book_companion.schema.validation import validate_graph_state


def run_graph_once(initial_state: dict) -> dict:
    """
    TODO: Assemble when all nodes completed
    """
    state = validate_graph_state(initial_state, context="graph initial_state")
    state = run_comprehension_node(state)
    state = validate_graph_state(state, context="graph after_comprehension")
    state["next_node"] = route_after_comprehension(state)
    return validate_graph_state(state, context="graph final_state")

