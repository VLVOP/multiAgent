from __future__ import annotations

from typing import Literal

from multiagent.state import DiscoveryState, Route


LoopRoute = Literal["accept", "refine", "backtrack", "explore", "stop"]


def budget_exhausted(state: DiscoveryState) -> bool:
    """Return whether the explicit loop budget has been exhausted."""
    return state.get("iteration", 0) >= state.get("max_iterations", 5)


def route_after_critique(state: DiscoveryState) -> LoopRoute:
    """Route the post-critique loop without changing the graph topology.

    CriticAgent proposes one of the semantic LBD routes. This edge policy applies
    graph-level safety constraints such as the explicit loop budget before the
    next node is selected.
    """
    route: Route = state.get("route", "explore")

    # A valid accepted discovery should terminate immediately rather than be
    # rejected merely because the budget is also exhausted at this step.
    if route == "accept":
        return "accept"

    if budget_exhausted(state):
        return "stop"

    return route
