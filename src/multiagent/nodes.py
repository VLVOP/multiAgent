from __future__ import annotations

from multiagent.agents import CriticAgent, ExplorerAgent, HypothesisAgent, PlannerAgent, VerifierAgent
from multiagent.state import DiscoveryState


planner_agent = PlannerAgent()
explorer_agent = ExplorerAgent()
hypothesis_agent = HypothesisAgent()
verifier_agent = VerifierAgent()
critic_agent = CriticAgent()


def _trace(state: DiscoveryState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def plan_node(state: DiscoveryState) -> DiscoveryState:
    plan = planner_agent.run(state)
    return {
        **state,
        "iteration": state.get("iteration", 0),
        "plan": plan,
        "frontier": [state["target_entity"]],
        "entity_paths": [],
        "exploration_observations": [],
        "hypotheses": [],
        "failed_paths": [],
        "trace": _trace(state, "PLAN"),
    }


def explore_node(state: DiscoveryState) -> DiscoveryState:
    result = explorer_agent.run(state)
    return {
        **state,
        "entity_paths": result.get("entity_paths", []),
        "exploration_observations": result.get("exploration_observations", []),
        "trace": _trace(state, "EXPLORE"),
    }


def hypothesize_node(state: DiscoveryState) -> DiscoveryState:
    hypothesis = hypothesis_agent.run(state)
    hypotheses = list(state.get("hypotheses", []))
    if hypothesis is not None:
        hypotheses.append(hypothesis)

    return {
        **state,
        "current_hypothesis": hypothesis,
        "hypotheses": hypotheses,
        "trace": _trace(state, "HYPOTHESIZE"),
    }


def verify_node(state: DiscoveryState) -> DiscoveryState:
    verification = verifier_agent.run(state)
    return {
        **state,
        "verification": verification,
        "trace": _trace(state, "VERIFY"),
    }


def critique_node(state: DiscoveryState) -> DiscoveryState:
    route, reflection = critic_agent.run(state)
    return {
        **state,
        "reflection": reflection,
        "route": route,
        "trace": _trace(state, "CRITIQUE"),
    }


def refine_node(state: DiscoveryState) -> DiscoveryState:
    """Deterministic state-control node; not an Agent.

    For now it lowers confidence slightly before re-verification so the loop remains explicit.
    A dedicated hypothesis-repair policy can be introduced later without changing graph topology.
    """
    h = state.get("current_hypothesis")
    if h is not None:
        h = {**h, "score": max(0.0, float(h.get("score", 0.5)) - 0.05)}
    return {**state, "current_hypothesis": h, "trace": _trace(state, "REFINE")}


def backtrack_node(state: DiscoveryState) -> DiscoveryState:
    """Deterministic state-control node; mark the current ABC path as failed and loop."""
    h = state.get("current_hypothesis")
    failed_paths = list(state.get("failed_paths", []))
    if h is not None:
        failed_paths.append([h["a"], h["b"], h["c"]])

    return {
        **state,
        "failed_paths": failed_paths,
        "current_hypothesis": None,
        "iteration": state.get("iteration", 0) + 1,
        "trace": _trace(state, "BACKTRACK"),
    }
