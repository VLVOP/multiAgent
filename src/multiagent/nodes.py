from __future__ import annotations

from typing import Any

from multiagent.agents import CriticAgent, ExplorerAgent, HypothesisAgent, PlannerAgent, VerifierAgent
from multiagent.cache import relevant_relation_keys
from multiagent.communication import (
    append_message,
    evidence_refs_from_verification,
    make_discovery_message,
)
from multiagent.context import AgentRole, context_manager
from multiagent.edges import decide_after_critique
from multiagent.instrumentation import empty_tool_usage, merge_tool_usage
from multiagent.state import DiscoveryState


planner_agent = PlannerAgent()
explorer_agent = ExplorerAgent()
hypothesis_agent = HypothesisAgent()
verifier_agent = VerifierAgent()
critic_agent = CriticAgent()


def _trace(state: DiscoveryState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def _abc_path(state: DiscoveryState) -> list[str]:
    hypothesis = state.get("current_hypothesis")
    if hypothesis is not None:
        return [hypothesis["a"], hypothesis["b"], hypothesis["c"]]
    paths = state.get("entity_paths", [])
    return list(paths[0]) if paths else []


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


def _emit(
    state: DiscoveryState,
    message: dict[str, Any],
) -> DiscoveryState:
    if not state.get("a2a_enabled", True):
        return state
    messages, stats = append_message(state, message)
    return {**state, "agent_messages": messages, "communication_stats": stats}


def plan_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "planner", "PLAN")
    plan = planner_agent.run(view)
    next_state: DiscoveryState = {
        **state,
        "iteration": state.get("iteration", 0),
        "max_refinement_rounds": state.get("max_refinement_rounds", 3),
        "architecture_preset": state.get("architecture_preset", "custom"),
        "context_mode": state.get("context_mode", "hierarchical"),
        "cache_enabled": state.get("cache_enabled", True),
        "a2a_enabled": state.get("a2a_enabled", True),
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
        "tool_usage": empty_tool_usage(),
        "agent_messages": [],
        "communication_stats": {
            "messages": 0,
            "payload_chars": 0,
            "evidence_refs": 0,
            "cache_refs": 0,
        },
        "context_access_log": access_log,
        "trace": _trace(state, "PLAN"),
    }
    message = make_discovery_message(
        src="planner",
        dst="explorer",
        kind="plan",
        requested_action="explore_bridges",
        payload={
            "objective": plan.get("objective"),
            "exploration_focus": plan.get("exploration_focus", []),
        },
    )
    return _emit(next_state, message)


def explore_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "explorer", "EXPLORE")
    result = explorer_agent.run(view)
    tool_delta = result.pop("_tool_usage_delta", {})
    tool_usage = merge_tool_usage(
        state.get("tool_usage"),
        agent="explorer",
        delta=tool_delta,
    )
    paths = result.get("entity_paths", [])
    next_state: DiscoveryState = {
        **state,
        "entity_paths": paths,
        "exploration_observations": result.get("exploration_observations", []),
        "tool_usage": tool_usage,
        "context_access_log": access_log,
        "trace": _trace(state, "EXPLORE"),
    }
    message = make_discovery_message(
        src="explorer",
        dst="hypothesis",
        kind="candidate_paths",
        requested_action="formulate_hypothesis",
        abc_path=list(paths[0]) if paths else [],
        payload={"candidate_paths": paths[:5], "candidate_count": len(paths)},
    )
    return _emit(next_state, message)


def hypothesize_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "hypothesis", "HYPOTHESIZE")
    hypothesis = hypothesis_agent.run(view)
    hypotheses = list(state.get("hypotheses", []))
    if hypothesis is not None:
        hypotheses.append(hypothesis)

    next_state: DiscoveryState = {
        **state,
        "current_hypothesis": hypothesis,
        "hypotheses": hypotheses,
        "refinement_request": None,
        "refinement_round": 0,
        "context_access_log": access_log,
        "trace": _trace(state, "HYPOTHESIZE"),
    }
    path = (
        [hypothesis["a"], hypothesis["b"], hypothesis["c"]]
        if hypothesis is not None
        else []
    )
    message = make_discovery_message(
        src="hypothesis",
        dst="verifier",
        kind="hypothesis",
        requested_action="verify_abc",
        abc_path=path,
        relation_under_test="A-B + B-C; audit A-C novelty",
        payload={"score": hypothesis.get("score") if hypothesis else None},
    )
    return _emit(next_state, message)


def verify_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "verifier", "VERIFY")
    verification = verifier_agent.run(view)

    evidence_cache = verification.pop("_evidence_cache", state.get("evidence_cache", {}))
    cache_stats = verification.pop("_cache_stats", state.get("cache_stats", {}))
    tool_delta = verification.pop("_tool_usage_delta", {})
    tool_usage = merge_tool_usage(
        state.get("tool_usage"),
        agent="verifier",
        delta=tool_delta,
    )

    next_state: DiscoveryState = {
        **state,
        "verification": verification,
        "refinement_request": None,
        "evidence_cache": evidence_cache,
        "cache_stats": cache_stats,
        "tool_usage": tool_usage,
        "context_access_log": access_log,
        "trace": _trace(state, "VERIFY"),
    }

    path = _abc_path(next_state)
    cache_refs: list[str] = []
    if len(path) == 3:
        local_keys = relevant_relation_keys(path[0], path[1], path[2], state["cutoff_year"])
        cache_refs = sorted(key for key in local_keys if key in evidence_cache)

    confidences: list[float] = []
    for detail_key in ("ab_verification", "bc_verification", "ac_novelty"):
        detail = verification.get(detail_key) or {}
        try:
            confidences.append(float(detail.get("confidence", 0.0)))
        except (TypeError, ValueError):
            pass
    uncertainty = 1.0 - min(confidences) if confidences else 1.0

    message = make_discovery_message(
        src="verifier",
        dst="critic",
        kind="verification",
        requested_action="critique_evidence",
        abc_path=path,
        relation_under_test="ABC bridge + A-C novelty",
        evidence_refs=evidence_refs_from_verification(verification),
        cache_references=cache_refs,
        uncertainty=uncertainty,
        payload={
            "ab_supported": verification.get("ab_supported"),
            "bc_supported": verification.get("bc_supported"),
            "ac_already_known": verification.get("ac_already_known"),
            "ac_novelty_resolved": verification.get("ac_novelty_resolved"),
        },
    )
    return _emit(next_state, message)


def critique_node(state: DiscoveryState) -> DiscoveryState:
    view, access_log = _agent_view(state, "critic", "CRITIQUE")
    semantic_route, reflection = critic_agent.run(view)
    tool_delta = reflection.pop("_tool_usage_delta", {})
    tool_usage = merge_tool_usage(
        state.get("tool_usage"),
        agent="critic",
        delta=tool_delta,
    )
    iteration = state.get("iteration", 0) + 1

    provisional: DiscoveryState = {
        **state,
        "iteration": iteration,
        "reflection": reflection,
        "route": semantic_route,
    }
    decision = decide_after_critique(provisional)
    effective_route = decision.effective_route

    termination_reason: str | None = None
    if effective_route == "accept":
        termination_reason = "accepted"
    elif effective_route == "stop":
        termination_reason = "max_iterations"

    loop_decision = {
        "semantic_route": decision.semantic_route,
        "effective_route": effective_route,
        "overridden": decision.overridden,
        "reason": decision.reason,
    }

    next_state: DiscoveryState = {
        **provisional,
        "termination_reason": termination_reason,
        "edge_route": effective_route,
        "loop_decision": loop_decision,
        "tool_usage": tool_usage,
        "context_access_log": access_log,
        "trace": _trace(state, "CRITIQUE"),
    }

    dst = (
        "verifier"
        if effective_route == "refine"
        else "explorer"
        if effective_route in {"backtrack", "explore"}
        else "control"
    )
    requested_action = {
        "refine": "refine_verification",
        "backtrack": "backtrack_and_explore",
        "explore": "explore_more",
        "accept": "terminate_accept",
        "stop": "terminate_budget",
    }[effective_route]
    message = make_discovery_message(
        src="critic",
        dst=dst,
        kind="reflection",
        requested_action=requested_action,
        abc_path=_abc_path(next_state),
        payload={
            "semantic_recommendation": semantic_route,
            "effective_recommendation": effective_route,
            "loop_policy_overridden": decision.overridden,
            "loop_policy_reason": decision.reason,
            "issue": reflection.get("issue"),
            "refinement_target": reflection.get("refinement_target"),
        },
    )
    return _emit(next_state, message)


def refine_node(state: DiscoveryState) -> DiscoveryState:
    """Translate Critic diagnosis into a targeted request for the next verification pass."""
    reflection = state.get("reflection") or {}
    target = reflection.get("refinement_target", "both")
    refinement_round = state.get("refinement_round", 0) + 1

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
