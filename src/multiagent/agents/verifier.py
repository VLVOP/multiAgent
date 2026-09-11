from __future__ import annotations

import json
from typing import Any

from multiagent.agents.explorer import online_tools_enabled
from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState
from multiagent.tools.agent_tools import count_entity_pair_mentions, find_entity_pair_evidence
from multiagent.tools.mock_graphs import check_direct_link, get_relation


class VerifierAgent:
    """Verify A-B and B-C support and test whether A-C was already known pre-cutoff."""

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
        ab = find_entity_pair_evidence.invoke(
            {"entity_a": h["a"], "entity_b": h["b"], "before_year": cutoff, "top_k": 5}
        )
        bc = find_entity_pair_evidence.invoke(
            {"entity_a": h["b"], "entity_b": h["c"], "before_year": cutoff, "top_k": 5}
        )
        ac = find_entity_pair_evidence.invoke(
            {"entity_a": h["a"], "entity_b": h["c"], "before_year": cutoff, "top_k": 5}
        )
        ac_count = count_entity_pair_mentions.invoke(
            {"entity_a": h["a"], "entity_b": h["c"], "before_year": cutoff}
        )

        verification: dict[str, Any] = {
            "ab_supported": bool(ab),
            "bc_supported": bool(bc),
            "ac_already_known": bool(ac_count),
            "ab_evidence": ab,
            "bc_evidence": bc,
            "ac_evidence": ac,
            "ac_mention_count": int(ac_count),
            "mode": "online_cooccurrence",
        }

        if deepseek_enabled():
            result = chat_json(
                system_prompt=(
                    "You are the verification agent in a biomedical LBD system. Judge only from "
                    "the supplied pre-cutoff PubMed evidence. Co-occurrence alone is not sufficient "
                    "for a biomedical relation. Output JSON with booleans ab_supported, "
                    "bc_supported, ac_already_known and a short rationale. ac_already_known is true "
                    "only when supplied A-C evidence explicitly establishes the target association."
                ),
                user_prompt=(
                    f"hypothesis={json.dumps(h, ensure_ascii=False)}\n"
                    f"cutoff_year={cutoff}\n"
                    f"AB_evidence={json.dumps(ab, ensure_ascii=False)[:12000]}\n"
                    f"BC_evidence={json.dumps(bc, ensure_ascii=False)[:12000]}\n"
                    f"AC_evidence={json.dumps(ac, ensure_ascii=False)[:12000]}"
                ),
            )
            for key in ("ab_supported", "bc_supported", "ac_already_known"):
                if key in result:
                    verification[key] = bool(result[key])
            verification["rationale"] = str(result.get("rationale", ""))
            verification["mode"] = "online_llm_evidence_judgment"

        return verification

    def run(self, state: DiscoveryState) -> dict[str, Any]:
        if online_tools_enabled():
            try:
                return self._online_verify(state)
            except Exception as exc:
                fallback = self._mock_verify(state)
                fallback["tool_error"] = repr(exc)
                return fallback
        return self._mock_verify(state)
