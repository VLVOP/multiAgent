from __future__ import annotations

from typing import Any

from multiagent.agents.explorer import online_tools_enabled
from multiagent.cache import (
    get_novelty_entry,
    get_relation_entry,
    novelty_entry_covers,
    put_novelty_decision,
    put_relation_verification,
    relation_entry_covers,
)
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

    @staticmethod
    def _stats(state: DiscoveryState) -> dict[str, int]:
        current = state.get("cache_stats", {})
        return {
            "hits": int(current.get("hits", 0)),
            "misses": int(current.get("misses", 0)),
            "writes": int(current.get("writes", 0)),
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
        iteration = state.get("iteration", 0)
        cache_enabled = state.get("cache_enabled", True)

        cache = dict(state.get("evidence_cache", {})) if cache_enabled else {}
        stats = self._stats(state)

        def resolve_relation(
            entity_a: str,
            entity_b: str,
            previous_result: dict[str, Any],
            required_top_k: int,
        ) -> dict[str, Any]:
            nonlocal cache, stats
            entry = (
                get_relation_entry(cache, entity_a, entity_b, cutoff)
                if cache_enabled
                else None
            )
            if cache_enabled and relation_entry_covers(entry, required_top_k):
                stats["hits"] += 1
                return dict(entry.get("verification", {}))

            if (
                required_top_k == 0
                and previous_result
                and previous_result.get("relation_supported") is not None
            ):
                return previous_result

            stats["misses"] += 1
            effective_top_k = max(1, required_top_k or 8)
            result = verify_relation.invoke(
                {
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "before_year": cutoff,
                    "top_k": effective_top_k,
                }
            )
            if cache_enabled:
                cache = put_relation_verification(
                    cache,
                    result,
                    top_k=effective_top_k,
                    iteration=iteration,
                )
                stats["writes"] += 1
            return result

        def resolve_novelty(
            entity_a: str,
            entity_c: str,
            previous_result: dict[str, Any],
            required_top_k: int,
        ) -> dict[str, Any]:
            nonlocal cache, stats
            entry = (
                get_novelty_entry(cache, entity_a, entity_c, cutoff)
                if cache_enabled
                else None
            )
            if cache_enabled and novelty_entry_covers(entry, required_top_k):
                stats["hits"] += 1
                return dict(entry.get("novelty", {}))

            if (
                required_top_k == 0
                and previous_result
                and previous_result.get("direct_relation_known") is not None
            ):
                return previous_result

            stats["misses"] += 1
            effective_top_k = max(1, required_top_k or 8)
            result = check_novelty.invoke(
                {
                    "entity_a": entity_a,
                    "entity_c": entity_c,
                    "before_year": cutoff,
                    "top_k": effective_top_k,
                }
            )
            if cache_enabled:
                cache = put_novelty_decision(
                    cache,
                    result,
                    top_k=effective_top_k,
                    iteration=iteration,
                )
                stats["writes"] += 1
            return result

        ab_depth = top_k if target in {None, "ab", "both", "counter"} else 0
        bc_depth = top_k if target in {None, "bc", "both", "counter"} else 0
        novelty_depth = top_k if target in {None, "ac_novelty"} else 0

        ab = resolve_relation(
            h["a"],
            h["b"],
            dict(previous.get("ab_verification", {})),
            ab_depth,
        )
        bc = resolve_relation(
            h["b"],
            h["c"],
            dict(previous.get("bc_verification", {})),
            bc_depth,
        )
        novelty = resolve_novelty(
            h["a"],
            h["c"],
            dict(previous.get("ac_novelty", {})),
            novelty_depth,
        )

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
            "_evidence_cache": cache,
            "_cache_stats": stats,
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
