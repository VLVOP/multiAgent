from __future__ import annotations

import json

from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState, Hypothesis, Route
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
    hypothesis: Hypothesis = {
        "a": a,
        "b": b,
        "c": c,
        "score": 0.5,
        "rationale": f"Mock ABC bridge: {a} -> {b} -> {c}",
    }

    if deepseek_enabled():
        result = chat_json(
            system_prompt=(
                "You are the hypothesis agent in a biomedical Literature-Based Discovery system. "
                "A, B, and C are biomedical entities. Assess the proposed ABC bridge without "
                "inventing evidence. Output JSON only with keys score and rationale. "
                "score must be a number from 0 to 1."
            ),
            user_prompt=(
                "Return JSON for this candidate ABC path:\n"
                f"A={a}\nB={b}\nC={c}\n"
                f"cutoff_year={state['cutoff_year']}\n"
                "Example JSON: {\"score\": 0.6, \"rationale\": \"...\"}"
            ),
        )
        hypothesis["score"] = max(0.0, min(1.0, float(result.get("score", 0.5))))
        hypothesis["rationale"] = str(result.get("rationale", hypothesis["rationale"]))

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


def _deterministic_route(state: DiscoveryState) -> tuple[Route, str]:
    v = state.get("verification", {})
    h = state.get("current_hypothesis")

    if h is None:
        return "explore", "no_candidate_path"
    if v.get("ac_already_known"):
        return "backtrack", "A-C relation already known before cutoff"
    if not v.get("ab_supported") or not v.get("bc_supported"):
        return "refine", "bridge evidence incomplete"
    return "accept", "candidate is supported and not directly known"


def critique_node(state: DiscoveryState) -> DiscoveryState:
    route, issue = _deterministic_route(state)
    reflection = {
        "issue": issue,
        "recommendation": route,
        "rationale": "Deterministic verification-based critique.",
    }

    if deepseek_enabled() and state.get("current_hypothesis") is not None:
        result = chat_json(
            system_prompt=(
                "You are the critique/reflection agent in a biomedical Literature-Based Discovery "
                "system. Critique only from the supplied hypothesis and verification state. "
                "Do not invent literature. Output JSON only with keys recommendation, issue, "
                "and rationale. recommendation must be one of accept, refine, backtrack, explore."
            ),
            user_prompt=(
                "Critique this state and return JSON:\n"
                f"hypothesis={json.dumps(state.get('current_hypothesis'), ensure_ascii=False)}\n"
                f"verification={json.dumps(state.get('verification', {}), ensure_ascii=False)}\n"
                f"cutoff_year={state['cutoff_year']}\n"
                "Example JSON: {\"recommendation\": \"refine\", \"issue\": \"...\", "
                "\"rationale\": \"...\"}"
            ),
        )
        proposed = str(result.get("recommendation", route)).lower()
        allowed: set[Route] = {"accept", "refine", "backtrack", "explore"}
        if proposed in allowed:
            route = proposed  # type: ignore[assignment]
        reflection = {
            "issue": str(result.get("issue", issue)),
            "recommendation": route,
            "rationale": str(result.get("rationale", "")),
        }

        # Hard LBD constraints override model preference.
        verification = state.get("verification", {})
        if verification.get("ac_already_known"):
            route = "backtrack"
            reflection["recommendation"] = route
        elif not verification.get("ab_supported") or not verification.get("bc_supported"):
            if route == "accept":
                route = "refine"
                reflection["recommendation"] = route

    return {
        **state,
        "reflection": reflection,
        "route": route,
        "trace": _trace(state, "CRITIQUE"),
    }


def refine_node(state: DiscoveryState) -> DiscoveryState:
    h = state.get("current_hypothesis")
    if h is not None:
        h = {**h, "score": min(1.0, float(h.get("score", 0.5)) + 0.1)}
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
