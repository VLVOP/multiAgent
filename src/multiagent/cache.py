from __future__ import annotations

from copy import deepcopy
from typing import Any


RelationEvidenceCache = dict[str, dict[str, Any]]


def relation_cache_key(entity_a: str, entity_b: str, before_year: int) -> str:
    """Build a stable key for a temporally grounded directed entity relation."""
    a = " ".join(entity_a.strip().lower().split())
    b = " ".join(entity_b.strip().lower().split())
    return f"{before_year}::{a}::{b}"


def get_relation_entry(
    cache: RelationEvidenceCache,
    entity_a: str,
    entity_b: str,
    before_year: int,
) -> dict[str, Any] | None:
    entry = cache.get(relation_cache_key(entity_a, entity_b, before_year))
    return deepcopy(entry) if entry is not None else None


def relation_entry_resolved(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    verification = entry.get("verification") or {}
    return verification.get("relation_supported") is not None


def relation_entry_covers(entry: dict[str, Any] | None, top_k: int) -> bool:
    """Return whether cached semantic verification is resolved at the requested depth."""
    if not relation_entry_resolved(entry):
        return False
    return int(entry.get("top_k", 0)) >= max(0, int(top_k))


def put_relation_verification(
    cache: RelationEvidenceCache,
    verification: dict[str, Any],
    *,
    top_k: int,
    iteration: int,
) -> RelationEvidenceCache:
    """Return a new cache containing one normalized relation-verification entry."""
    entity_a = str(verification.get("entity_a", ""))
    entity_b = str(verification.get("entity_b", ""))
    before_year = int(verification.get("before_year", 0))
    key = relation_cache_key(entity_a, entity_b, before_year)

    updated = deepcopy(cache)
    updated[key] = {
        "key": key,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "before_year": before_year,
        "top_k": max(0, int(top_k)),
        "relation_supported": verification.get("relation_supported"),
        "relation_type": verification.get("relation_type"),
        "confidence": verification.get("confidence", 0.0),
        "supporting_pmids": list(verification.get("supporting_pmids", [])),
        "evidence_count": len(verification.get("evidence", [])),
        "updated_iteration": int(iteration),
        "verification": deepcopy(verification),
    }
    return updated


def relevant_relation_keys(
    entity_a: str,
    entity_b: str,
    entity_c: str,
    before_year: int,
) -> set[str]:
    """Return cache keys local to the current ABC hypothesis."""
    return {
        relation_cache_key(entity_a, entity_b, before_year),
        relation_cache_key(entity_b, entity_c, before_year),
        relation_cache_key(entity_a, entity_c, before_year),
    }


def project_cache_to_abc(
    cache: RelationEvidenceCache,
    entity_a: str,
    entity_b: str,
    entity_c: str,
    before_year: int,
) -> RelationEvidenceCache:
    """Project a global relation cache onto the current ABC neighborhood."""
    keys = relevant_relation_keys(entity_a, entity_b, entity_c, before_year)
    return {key: deepcopy(value) for key, value in cache.items() if key in keys}
