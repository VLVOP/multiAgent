from __future__ import annotations

import json
from typing import Any

from multiagent.agents.explorer import online_tools_enabled
from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState, Route
from multiagent.tools.agent_tools import search_counter_evidence


class CriticAgent:
    """Reflect on verification results, inspect counter-evidence, and recommend a transition."""

    @staticmethod
    def _deterministic_route(state: DiscoveryState) -> tuple[Route, str]:
        verification = state.get("verification", {})
        hypothesis = state.get("current_hypothesis")

        if hypothesis is None:
            return "explore", "no_candidate_path"
        if verification.get("ac_already_known"):
            return "backtrack", "A-C relation already known before cutoff"
        if not verification.get("ab_supported") or not verification.get("bc_supported"):
            return "refine", "bridge evidence incomplete"
        return "accept", "candidate is supported and not directly known"

    def _counter_evidence(self, state: DiscoveryState) -> dict[str, Any]:
        hypothesis = state.get("current_hypothesis")
        if hypothesis is None or not online_tools_enabled():
            return {}

        cutoff = state["cutoff_year"]
        try:
            ab = search_counter_evidence.invoke(
                {
                    "entity_a": hypothesis["a"],
                    "entity_b": hypothesis["b"],
                    "before_year": cutoff,
                    "top_k": 5,
                }
            )
            bc = search_counter_evidence.invoke(
                {
                    "entity_a": hypothesis["b"],
                    "entity_b": hypothesis["c"],
                    "before_year": cutoff,
                    "top_k": 5,
                }
            )
            return {"ab": ab, "bc": bc}
        except Exception as exc:
            return {"tool_error": repr(exc)}

    def run(self, state: DiscoveryState) -> tuple[Route, dict[str, Any]]:
        route, issue = self._deterministic_route(state)
        counter = self._counter_evidence(state)

        ab_counter = bool(counter.get("ab", {}).get("counter_evidence_found"))
        bc_counter = bool(counter.get("bc", {}).get("counter_evidence_found"))
        if route == "accept" and (ab_counter or bc_counter):
            route = "refine"
            issue = "counter-evidence found for an otherwise supported bridge"

        reflection: dict[str, Any] = {
            "issue": issue,
            "recommendation": route,
            "rationale": "Verification- and counter-evidence-based critique.",
            "counter_evidence": counter,
        }

        if deepseek_enabled() and state.get("current_hypothesis") is not None:
            result = chat_json(
                system_prompt=(
                    "You are the critique/reflection agent in a biomedical Literature-Based "
                    "Discovery system. Critique only from the supplied hypothesis, verification, "
                    "counter-evidence, and prior exploration state. Do not invent literature. "
                    "Output JSON with keys recommendation, issue, rationale. recommendation must "
                    "be one of accept, refine, backtrack, explore."
                ),
                user_prompt=(
                    f"hypothesis={json.dumps(state.get('current_hypothesis'), ensure_ascii=False)}\n"
                    f"verification={json.dumps(state.get('verification', {}), ensure_ascii=False)[:14000]}\n"
                    f"counter_evidence={json.dumps(counter, ensure_ascii=False)[:10000]}\n"
                    f"failed_paths={json.dumps(state.get('failed_paths', []), ensure_ascii=False)}\n"
                    f"cutoff_year={state['cutoff_year']}"
                ),
            )
            proposed = str(result.get("recommendation", route)).lower()
            allowed: set[Route] = {"accept", "refine", "backtrack", "explore"}
            if proposed in allowed:
                route = proposed  # type: ignore[assignment]
            reflection.update(
                {
                    "issue": str(result.get("issue", issue)),
                    "recommendation": route,
                    "rationale": str(result.get("rationale", "")),
                }
            )

        # Hard LBD constraints override model preference.
        verification = state.get("verification", {})
        if verification.get("ac_already_known"):
            route = "backtrack"
            reflection["recommendation"] = route
        elif not verification.get("ab_supported") or not verification.get("bc_supported"):
            if route == "accept":
                route = "refine"
                reflection["recommendation"] = route
        elif ab_counter or bc_counter:
            if route == "accept":
                route = "refine"
                reflection["recommendation"] = route

        return route, reflection
