from __future__ import annotations

from typing import Literal

from multiagent.state import DiscoveryState, Route


LoopRoute = Literal["accept", "refine", "backtrack", "explore", "stop"]


def budget_exhausted(state: DiscoveryState) -> bool:
    """Return whether the explicit global loop budget has been exhausted."""
    return state.get("iteration", 0) >= state.get("max_iterations", 5)


def refinement_budget_exhausted(state: DiscoveryState) -> bool:
    """Return whether the current hypothesis has consumed its local refine budget."""
    return state.get("refinement_round", 0) >= state.get("max_refinement_rounds", 3)


def route_after_critique(state: DiscoveryState) -> LoopRoute:
    """Apply graph-level loop policy after CriticAgent proposes a semantic route.

    The agent decides what the scientific state suggests; this edge policy enforces
    architecture-level resource constraints. Repeated unresolved refinement is converted
    into backtracking so one weak ABC path cannot monopolize the discovery budget.
    """
    route: Route = state.get("route", "explore")

    if route == "accept":
        return "accept"

    if budget_exhausted(state):
        return "stop"

    if route == "refine" and refinement_budget_exhausted(state):
        return "backtrack"

    return route
