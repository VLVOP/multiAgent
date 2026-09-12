from __future__ import annotations

import json
from typing import Any

from multiagent.agents.explorer import online_tools_enabled
from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState, RefinementTarget, Route
from multiagent.tools.agent_tools import search_counter_evidence


class CriticAgent:
    """Reflect on verification results, inspect counter-evidence, and recommend a transition."""

    @staticmethod
    def _deterministic_route(
        state: DiscoveryState,
    ) -> tuple[Route, str, RefinementTarget | None]:
        verification = state.get("verification", {})
        hypothesis = state.get("current_hypothesis")

        if hypothesis is None:
            return "explore", "no_candidate_path", None
        if verification.get("ac_already_known"):
            return "backtrack", "A-C relation already known before cutoff", None
        if verification.get("ac_novelty_resolved", True) is False:
            return "refine", "A-C novelty status is unresolved", "ac_novelty"

        ab_supported = bool(verification.get("ab_supported"))
        bc_supported = bool(verification.get("bc_supported"))
        if not ab_supported and not bc_supported:
            return "refine", "both bridge relations need stronger evidence", "both"
        if not ab_supported:
            return "refine", "A-B bridge evidence is incomplete", "ab"
        if not bc_supported:
            return "refine", "B-C bridge evidence is incomplete", "bc"
        return "accept", "candidate is supported and not directly known", None

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
        route, issue, refinement_target = self._deterministic_route(state)
        counter = self._counter_evidence(state)

        ab_counter = bool(counter.get("ab", {}).get("counter_evidence_found"))
        bc_counter = bool(counter.get("bc", {}).get("counter_evidence_found"))
        if route == "accept" and (ab_counter or bc_counter):
            route = "refine"
            issue = "counter-evidence found for an otherwise supported bridge"
            refinement_target = "counter"

        reflection: dict[str, Any] = {
            "issue": issue,
            "recommendation": route,
            "rationale": "Verification- and counter-evidence-based critique.",
            "counter_evidence": counter,
        }
        if refinement_target is not None:
            reflection["refinement_target"] = refinement_target

        if deepseek_enabled() and state.get("current_hypothesis") is not None:
            result = chat_json(
                system_prompt=(
                    "You are the critique/reflection agent in a biomedical Literature-Based "
                    "Discovery system. Critique only from the supplied hypothesis, verification, "
                    "counter-evidence, and prior exploration state. Do not invent literature. "
                    "Output JSON with keys recommendation, issue, rationale, refinement_target. "
                    "recommendation must be one of accept, refine, backtrack, explore. "
                    "refinement_target, when recommendation is refine, should be one of "
                    "ab, bc, both, ac_novelty, counter."
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

            proposed_target = str(result.get("refinement_target", "")).lower()
            allowed_targets: set[str] = {"ab", "bc", "both", "ac_novelty", "counter"}
            if route == "refine" and proposed_target in allowed_targets:
                reflection["refinement_target"] = proposed_target

            reflection.update(
                {
                    "issue": str(result.get("issue", issue)),
                    "recommendation": route,
                    "rationale": str(result.get("rationale", "")),
                }
            )

        # Hard LBD constraints override model preference and restore the deterministic target.
        hard_route, hard_issue, hard_target = self._deterministic_route(state)
        if hard_route in {"backtrack", "refine"}:
            route = hard_route
            reflection["recommendation"] = route
            reflection["issue"] = hard_issue
            if hard_target is not None:
                reflection["refinement_target"] = hard_target
        elif ab_counter or bc_counter:
            route = "refine"
            reflection["recommendation"] = route
            reflection["refinement_target"] = "counter"

        return route, reflection
