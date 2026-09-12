from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from multiagent.state import DiscoveryState


def make_discovery_message(
    *,
    src: str,
    dst: str,
    kind: str,
    requested_action: str,
    abc_path: list[str] | None = None,
    relation_under_test: str | None = None,
    evidence_refs: list[str] | None = None,
    cache_references: list[str] | None = None,
    uncertainty: float | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a compact typed message for inter-agent communication."""
    message: dict[str, Any] = {
        "src": src,
        "dst": dst,
        "kind": kind,
        "requested_action": requested_action,
        "abc_path": list(abc_path or []),
        "relation_under_test": relation_under_test,
        "evidence_refs": list(dict.fromkeys(evidence_refs or [])),
        "cache_references": list(dict.fromkeys(cache_references or [])),
        "payload": deepcopy(payload or {}),
    }
    if uncertainty is not None:
        message["uncertainty"] = max(0.0, min(1.0, float(uncertainty)))
    return message


def append_message(
    state: DiscoveryState,
    message: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Append one message and update communication-cost diagnostics."""
    messages = [*state.get("agent_messages", []), deepcopy(message)]
    stats = dict(state.get("communication_stats", {}))
    stats["messages"] = int(stats.get("messages", 0)) + 1
    stats["payload_chars"] = int(stats.get("payload_chars", 0)) + len(
        json.dumps(message, ensure_ascii=False, sort_keys=True)
    )
    stats["evidence_refs"] = int(stats.get("evidence_refs", 0)) + len(
        message.get("evidence_refs", [])
    )
    stats["cache_refs"] = int(stats.get("cache_refs", 0)) + len(
        message.get("cache_references", [])
    )
    return messages, stats


def messages_for_agent(
    state: DiscoveryState,
    role: str,
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Select recent messages addressed to an agent without exposing the full message history."""
    relevant = [
        message
        for message in state.get("agent_messages", [])
        if message.get("dst") in {role, "broadcast"}
    ]
    return deepcopy(relevant[-max(1, limit):])


def evidence_refs_from_verification(verification: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("ab_verification", "bc_verification"):
        item = verification.get(key) or {}
        refs.extend(str(pmid) for pmid in item.get("supporting_pmids", []))
    novelty = verification.get("ac_novelty") or {}
    refs.extend(str(pmid) for pmid in novelty.get("known_relation_pmids", []))
    return list(dict.fromkeys(refs))
