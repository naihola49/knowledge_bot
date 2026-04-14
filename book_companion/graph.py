"""Graph assembly module."""

from book_companion.edges.routing import route_after_comprehension
from book_companion.nodes.comprehension import run_comprehension_node


def run_graph_once(initial_state: dict) -> dict:
    """
    TODO: Assemble when all nodes completed
    """
    state = run_comprehension_node(initial_state)
    state["next_node"] = route_after_comprehension(state)
    return state

