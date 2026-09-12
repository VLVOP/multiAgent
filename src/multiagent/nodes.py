from __future__ import annotations

from typing import Any

from multiagent.agents import CriticAgent, ExplorerAgent, HypothesisAgent, PlannerAgent, VerifierAgent
from multiagent.context import AgentRole, context_manager
from multiagent.state import DiscoveryState


planner_agent = PlannerAgent()
explorer_agent = ExplorerAgent()
hypothesis_agent = HypothesisAgent()
verifier_agent = VerifierAgent()
critic_agent = CriticAgent()


def _trace(state: DiscoveryState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def _agent_view(
    state: DiscoveryState,
    role: AgentRole,
    node: str,
) -> tuple[DiscoveryState, list[dict[str, Any]]]:
    view = context_manager.project_state(role, state)
    metadata = context_manager.view_metadata(role, state)
    metadata.update({"node": node, "iteration": state.get("iteration", 0)})
    access_log = [*state.get("context_access_log", []), metadata]
    return view, access_log


def plan_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "planner", "PLAN")
    plan = planner_agent.run(view)
    return {
        **state,
        "iteration": state.get("iteration", 0),
        "termination_reason": None,
        "plan": plan,
        "frontier": [state["target_entity"]],
        "entity_paths": [],
        "exploration_observations": [],
        "hypotheses": [],
        "failed_paths": [],
        "refinement_request": None,
        "refinement_round": 0,
        "evidence_cache": {},
        "cache_stats": {"hits": 0, "misses": 0, "writes": 0},
        "context_access_log": access_log,
        "trace": _trace(state, "PLAN"),
    }


def explore_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "explorer", "EXPLORE")
    result = explorer_agent.run(view)
    return {
        **state,
        "entity_paths": result.get("entity_paths", []),
        "exploration_observations": result.get("exploration_observations", []),
        "context_access_log": access_log,
        "trace": _trace(state, "EXPLORE"),
    }


def hypothesize_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "hypothesis", "HYPOTHESIZE")
    hypothesis = hypothesis_agent.run(view)
    hypotheses = list(state.get("hypotheses", []))
    if hypothesis is not None:
        hypotheses.append(hypothesis)

    return {
        **state,
        "current_hypothesis": hypothesis,
        "hypotheses": hypotheses,
        "refinement_request": None,
        "refinement_round": 0,
        "context_access_log": access_log,
        "trace": _trace(state, "HYPOTHESIZE"),
    }


def verify_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "verifier", "VERIFY")
    verification = verifier_agent.run(view)

    evidence_cache = verification.pop("_evidence_cache", state.get("evidence_cache", {}))
    cache_stats = verification.pop("_cache_stats", state.get("cache_stats", {}))

    return {
        **state,
        "verification": verification,
        "refinement_request": None,
        "evidence_cache": evidence_cache,
        "cache_stats": cache_stats,
        "context_access_log": access_log,
        "trace": _trace(state, "VERIFY"),
    }


def critique_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "critic", "CRITIQUE")
    route, reflection = critic_agent.run(view)
    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 5)

    termination_reason: str | None = None
    if route == "accept":
        termination_reason = "accepted"
    elif iteration >= max_iterations:
        termination_reason = "max_iterations"

    return {
        **state,
        "iteration": iteration,
        "termination_reason": termination_reason,
        "reflection": reflection,
        "route": route,
        "context_access_log": access_log,
        "trace": _trace(state, "CRITIQUE"),
    }


def refine_node(state: DiscoveryState) -> DiscoveryState:
    """Translate Critic diagnosis into a targeted request for the next verification pass."""
    reflection = state.get("reflection") or {}
    target = reflection.get("refinement_target", "both")
    refinement_round = state.get("refinement_round", 0) + 1

    # Increase evidence depth gradually rather than repeating the exact same verification call.
    top_k = min(32, 8 + 4 * refinement_round)
    request = {
        "target": target,
        "reason": str(reflection.get("issue", "verification needs refinement")),
        "top_k": top_k,
        "round": refinement_round,
    }

    return {
        **state,
        "refinement_request": request,
        "refinement_round": refinement_round,
        "trace": _trace(state, f"REFINE[{target}]"),
    }


def backtrack_node(state: DiscoveryState) -> DiscoveryState:
    """Mark the current ABC path as failed while retaining reusable evidence cache entries."""
    h = state.get("current_hypothesis")
    failed_paths = list(state.get("failed_paths", []))
    if h is not None:
        failed_paths.append([h["a"], h["b"], h["c"]])

    return {
        **state,
        "failed_paths": failed_paths,
        "current_hypothesis": None,
        "refinement_request": None,
        "refinement_round": 0,
        "trace": _trace(state, "BACKTRACK"),
    }
