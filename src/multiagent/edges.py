from __future__ import annotations

from dataclasses import dataclass

from multiagent.state import DiscoveryState, LoopRoute, Route


@dataclass(frozen=True)
class LoopDecision:
    """Final graph routing decision after architecture-level resource guards."""

    semantic_route: Route
    effective_route: LoopRoute
    overridden: bool
    reason: str


def budget_exhausted(state: DiscoveryState) -> bool:
    """Return whether the explicit global loop budget has been exhausted."""
    return state.get("iteration", 0) >= state.get("max_iterations", 5)


def refinement_budget_exhausted(state: DiscoveryState) -> bool:
    """Return whether the current hypothesis has consumed its local refine budget."""
    return state.get("refinement_round", 0) >= state.get("max_refinement_rounds", 3)


def decide_after_critique(state: DiscoveryState) -> LoopDecision:
    """Combine scientific Critic recommendation with architecture-level loop constraints."""
    route: Route = state.get("route", "explore")

    if route == "accept":
        return LoopDecision(route, "accept", False, "scientific candidate accepted")

    if budget_exhausted(state):
        return LoopDecision(
            route,
            "stop",
            True,
            "global iteration budget exhausted",
        )

    if route == "refine" and refinement_budget_exhausted(state):
        return LoopDecision(
            route,
            "backtrack",
            True,
            "local refinement budget exhausted; backtrack to another ABC path",
        )

    return LoopDecision(route, route, False, "critic recommendation executed")


def route_after_critique(state: DiscoveryState) -> LoopRoute:
    """Return the already-recorded effective edge route, or compute it as a fallback."""
    recorded = state.get("edge_route")
    if recorded in {"accept", "refine", "backtrack", "explore", "stop"}:
        return recorded  # type: ignore[return-value]
    return decide_after_critique(state).effective_route
