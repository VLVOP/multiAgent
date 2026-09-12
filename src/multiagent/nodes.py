from __future__ import annotations

from multiagent.agents import CriticAgent, ExplorerAgent, HypothesisAgent, PlannerAgent, VerifierAgent
from multiagent.context import context_manager
from multiagent.state import DiscoveryState


planner_agent = PlannerAgent()
explorer_agent = ExplorerAgent()
hypothesis_agent = HypothesisAgent()
verifier_agent = VerifierAgent()
critic_agent = CriticAgent()


def _trace(state: DiscoveryState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def plan_node(state: DiscoveryState) -> DiscoveryState:
    plan = planner_agent.run(context_manager.project_state("planner", state))
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
        "trace": _trace(state, "PLAN"),
    }


def explore_node(state: DiscoveryState) -> DiscoveryState:
    result = explorer_agent.run(context_manager.project_state("explorer", state))
    return {
        **state,
        "entity_paths": result.get("entity_paths", []),
        "exploration_observations": result.get("exploration_observations", []),
        "trace": _trace(state, "EXPLORE"),
    }


def hypothesize_node(state: DiscoveryState) -> DiscoveryState:
    hypothesis = hypothesis_agent.run(context_manager.project_state("hypothesis", state))
    hypotheses = list(state.get("hypotheses", []))
    if hypothesis is not None:
        hypotheses.append(hypothesis)

    return {
        **state,
        "current_hypothesis": hypothesis,
        "hypotheses": hypotheses,
        "refinement_request": None,
        "refinement_round": 0,
        "trace": _trace(state, "HYPOTHESIZE"),
    }


def verify_node(state: DiscoveryState) -> DiscoveryState:
    verification = verifier_agent.run(context_manager.project_state("verifier", state))
    return {
        **state,
        "verification": verification,
        "refinement_request": None,
        "trace": _trace(state, "VERIFY"),
    }


def critique_node(state: DiscoveryState) -> DiscoveryState:
    route, reflection = critic_agent.run(context_manager.project_state("critic", state))
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
    """Deterministic state-control node; mark the current ABC path as failed and loop."""
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
