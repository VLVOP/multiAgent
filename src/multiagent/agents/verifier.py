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
            return {"ab_supported": False, "bc_supported": False, "ac_already_known": False}
        cutoff = state["cutoff_year"]
        return {
            "ab_supported": get_relation(h["a"], h["b"], cutoff) is not None,
            "bc_supported": get_relation(h["b"], h["c"], cutoff) is not None,
            "ac_already_known": check_direct_link(h["a"], h["c"], cutoff),
            "mode": "mock",
        }

    def _online_verify(self, state: DiscoveryState) -> dict[str, Any]:
        h = state.get("current_hypothesis")
        if h is None:
            return {"ab_supported": False, "bc_supported": False, "ac_already_known": False}

        cutoff = state["cutoff_year"]
        ab = verify_relation.invoke(
            {"entity_a": h["a"], "entity_b": h["b"], "before_year": cutoff, "top_k": 8}
        )
        bc = verify_relation.invoke(
            {"entity_a": h["b"], "entity_b": h["c"], "before_year": cutoff, "top_k": 8}
        )
        novelty = check_novelty.invoke(
            {"entity_a": h["a"], "entity_c": h["c"], "before_year": cutoff, "top_k": 8}
        )

        # If semantic judgment is unavailable, do not promote co-occurrence into relation support.
        ab_supported = ab.get("relation_supported") is True
        bc_supported = bc.get("relation_supported") is True
        ac_known = novelty.get("direct_relation_known") is True

        return {
            "ab_supported": ab_supported,
            "bc_supported": bc_supported,
            "ac_already_known": ac_known,
            "ab_verification": ab,
            "bc_verification": bc,
            "ac_novelty": novelty,
            "mode": "online_semantic_verification",
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
