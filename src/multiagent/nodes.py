from __future__ import annotations

from multiagent.state import DiscoveryState, Hypothesis
from multiagent.tools.mock_graphs import check_direct_link, expand_entity, get_relation


def _trace(state: DiscoveryState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def plan_node(state: DiscoveryState) -> DiscoveryState:
    return {
        **state,
        "iteration": state.get("iteration", 0),
        "frontier": [state["target_entity"]],
        "entity_paths": [],
        "hypotheses": [],
        "failed_paths": [],
        "trace": _trace(state, "PLAN"),
    }


def explore_node(state: DiscoveryState) -> DiscoveryState:
    target = state["target_entity"]
    cutoff = state["cutoff_year"]
    failed = {tuple(path) for path in state.get("failed_paths", [])}

    candidates: list[list[str]] = []
    for ab in expand_entity(target, cutoff):
        for bc in expand_entity(ab.tail, cutoff):
            path = [target, ab.tail, bc.tail]
            if tuple(path) not in failed:
                candidates.append(path)

    return {
        **state,
        "entity_paths": candidates,
        "trace": _trace(state, "EXPLORE"),
    }


def hypothesize_node(state: DiscoveryState) -> DiscoveryState:
    paths = state.get("entity_paths", [])
    if not paths:
        return {
            **state,
            "current_hypothesis": None,
            "trace": _trace(state, "HYPOTHESIZE"),
        }

    a, b, c = paths[0]
    hypothesis: Hypothesis = {"a": a, "b": b, "c": c, "score": 0.5}
    return {
        **state,
        "current_hypothesis": hypothesis,
        "hypotheses": [*state.get("hypotheses", []), hypothesis],
        "trace": _trace(state, "HYPOTHESIZE"),
    }


def verify_node(state: DiscoveryState) -> DiscoveryState:
    h = state.get("current_hypothesis")
    if h is None:
        verification = {"ab_supported": False, "bc_supported": False, "ac_already_known": False}
    else:
        cutoff = state["cutoff_year"]
        verification = {
            "ab_supported": get_relation(h["a"], h["b"], cutoff) is not None,
            "bc_supported": get_relation(h["b"], h["c"], cutoff) is not None,
            "ac_already_known": check_direct_link(h["a"], h["c"], cutoff),
        }

    return {**state, "verification": verification, "trace": _trace(state, "VERIFY")}


def critique_node(state: DiscoveryState) -> DiscoveryState:
    v = state.get("verification", {})
    h = state.get("current_hypothesis")

    if h is None:
        reflection = {"issue": "no_candidate_path", "recommendation": "explore"}
        route = "explore"
    elif v.get("ac_already_known"):
        reflection = {"issue": "A-C relation already known before cutoff", "recommendation": "backtrack"}
        route = "backtrack"
    elif not v.get("ab_supported") or not v.get("bc_supported"):
        reflection = {"issue": "bridge evidence incomplete", "recommendation": "refine"}
        route = "refine"
    else:
        reflection = {"issue": "candidate is supported and not directly known", "recommendation": "accept"}
        route = "accept"

    return {
        **state,
        "reflection": reflection,
        "route": route,
        "trace": _trace(state, "CRITIQUE"),
    }


def refine_node(state: DiscoveryState) -> DiscoveryState:
    h = state.get("current_hypothesis")
    if h is not None:
        h = {**h, "score": min(1.0, h["score"] + 0.1)}
    return {**state, "current_hypothesis": h, "trace": _trace(state, "REFINE")}


def backtrack_node(state: DiscoveryState) -> DiscoveryState:
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
