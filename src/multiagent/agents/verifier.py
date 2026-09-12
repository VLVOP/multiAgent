from __future__ import annotations

from typing import Any

from multiagent.agents.explorer import online_tools_enabled
from multiagent.state import DiscoveryState
from multiagent.tools.agent_tools import check_novelty, verify_relation
from multiagent.tools.mock_graphs import check_direct_link, get_relation


class VerifierAgent:
    """Verify A-B/B-C relations and audit whether A-C was already known pre-cutoff."""

    def _mock_verify(self, state: DiscoveryState) -> dict[str, Any]:
        h = state.get("current_hypothesis")
        if h is None:
            return {
                "ab_supported": False,
                "bc_supported": False,
                "ac_already_known": False,
                "ac_novelty_resolved": True,
            }
        cutoff = state["cutoff_year"]
        return {
            "ab_supported": get_relation(h["a"], h["b"], cutoff) is not None,
            "bc_supported": get_relation(h["b"], h["c"], cutoff) is not None,
            "ac_already_known": check_direct_link(h["a"], h["c"], cutoff),
            "ac_novelty_resolved": True,
            "mode": "mock",
        }

    def _online_verify(self, state: DiscoveryState) -> dict[str, Any]:
        h = state.get("current_hypothesis")
        if h is None:
            return {
                "ab_supported": False,
                "bc_supported": False,
                "ac_already_known": False,
                "ac_novelty_resolved": False,
            }

        cutoff = state["cutoff_year"]
        previous = state.get("verification", {})
        request = state.get("refinement_request") or {}
        target = request.get("target")
        top_k = max(1, int(request.get("top_k", 8)))

        ab = previous.get("ab_verification", {})
        bc = previous.get("bc_verification", {})
        novelty = previous.get("ac_novelty", {})

        rerun_ab = target in {None, "ab", "both", "counter"} or not ab
        rerun_bc = target in {None, "bc", "both", "counter"} or not bc
        rerun_novelty = target in {None, "ac_novelty"} or not novelty

        if rerun_ab:
            ab = verify_relation.invoke(
                {"entity_a": h["a"], "entity_b": h["b"], "before_year": cutoff, "top_k": top_k}
            )
        if rerun_bc:
            bc = verify_relation.invoke(
                {"entity_a": h["b"], "entity_b": h["c"], "before_year": cutoff, "top_k": top_k}
            )
        if rerun_novelty:
            novelty = check_novelty.invoke(
                {"entity_a": h["a"], "entity_c": h["c"], "before_year": cutoff, "top_k": top_k}
            )

        # If semantic judgment is unavailable, do not promote co-occurrence into relation support.
        ab_supported = ab.get("relation_supported") is True
        bc_supported = bc.get("relation_supported") is True
        direct_relation_known = novelty.get("direct_relation_known")
        ac_known = direct_relation_known is True
        ac_novelty_resolved = direct_relation_known is not None

        return {
            "ab_supported": ab_supported,
            "bc_supported": bc_supported,
            "ac_already_known": ac_known,
            "ac_novelty_resolved": ac_novelty_resolved,
            "ab_verification": ab,
            "bc_verification": bc,
            "ac_novelty": novelty,
            "refinement_target": target,
            "mode": "online_targeted_verification" if target else "online_semantic_verification",
        }

    def run(self, state: DiscoveryState) -> dict[str, Any]:
        if online_tools_enabled():
            try:
                return self._online_verify(state)
            except Exception as exc:
                fallback = self._mock_verify(state)
                fallback["tool_error"] = repr(exc)
                return fallback
        return self._mock_verify(state)
