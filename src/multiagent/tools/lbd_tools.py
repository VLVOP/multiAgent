from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from multiagent.llm import chat_json, deepseek_enabled
from multiagent.tools.interfaces import Record
from multiagent.tools.runtime import get_backend


def _evidence_payload(records: list[Record], limit_chars: int = 18000) -> str:
    payload = [
        {
            "pmid": record.get("pmid"),
            "year": record.get("year"),
            "title": record.get("title"),
            "abstract": record.get("abstract"),
            "mesh_terms": record.get("mesh_terms", []),
        }
        for record in records
    ]
    return json.dumps(payload, ensure_ascii=False)[:limit_chars]


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _get_entity_info_impl(entity: str, before_year: int, top_k: int = 5) -> dict[str, Any]:
    backend = get_backend()
    query = f'"{entity}"'
    return {
        "entity": entity,
        "before_year": before_year,
        "backend": backend.backend_name,
        "mesh_candidates": backend.resolve_entity(entity, top_k),
        "mention_count": backend.count_literature(query, before_year),
        "sample_papers": backend.search_literature(query, before_year, min(top_k, 10)),
    }


@tool
def get_entity_info(entity: str, before_year: int, top_k: int = 5) -> dict[str, Any]:
    """Get temporally constrained biomedical context for an entity."""
    return _get_entity_info_impl(entity, before_year, top_k)


def _verify_relation_impl(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    backend = get_backend()
    evidence = backend.get_pair_evidence(entity_a, entity_b, before_year, top_k)
    mention_count = backend.count_pair_mentions(entity_a, entity_b, before_year)

    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "before_year": before_year,
        "backend": backend.backend_name,
        "mention_count": mention_count,
        "evidence": evidence,
        "relation_supported": None,
        "relation_type": None,
        "confidence": 0.0,
        "rationale": "Evidence retrieved; semantic relation judgment requires the configured LLM.",
        "supporting_pmids": [],
    }

    if not evidence:
        result.update(
            {
                "relation_supported": False,
                "confidence": 0.0,
                "rationale": "No pre-cutoff pair evidence was retrieved.",
            }
        )
        return result

    if not deepseek_enabled():
        return result

    judged = chat_json(
        system_prompt=(
            "You are a biomedical relation-verification component inside a Literature-Based "
            "Discovery system. Use only the supplied pre-cutoff evidence. Co-occurrence alone is "
            "not sufficient. Decide whether the evidence explicitly supports a meaningful "
            "biomedical relation between entity A and entity B. Output JSON only with keys: "
            "relation_supported (boolean), relation_type (short string or null), confidence "
            "(0 to 1), rationale (short string), supporting_pmids (list of strings)."
        ),
        user_prompt=(
            f"entity_a={entity_a}\n"
            f"entity_b={entity_b}\n"
            f"cutoff_year={before_year}\n"
            f"evidence={_evidence_payload(evidence)}"
        ),
    )

    valid_pmids = {str(record.get("pmid")) for record in evidence if record.get("pmid")}
    proposed_pmids = judged.get("supporting_pmids", [])
    supporting_pmids = (
        [str(pmid) for pmid in proposed_pmids if str(pmid) in valid_pmids]
        if isinstance(proposed_pmids, list)
        else []
    )

    result.update(
        {
            "relation_supported": bool(judged.get("relation_supported", False)),
            "relation_type": judged.get("relation_type"),
            "confidence": _confidence(judged.get("confidence")),
            "rationale": str(judged.get("rationale", "")),
            "supporting_pmids": supporting_pmids,
        }
    )
    return result


@tool
def verify_relation(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Verify whether pre-cutoff evidence explicitly supports a biomedical relation."""
    return _verify_relation_impl(entity_a, entity_b, before_year, top_k)


def _search_counter_evidence_impl(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 6,
) -> dict[str, Any]:
    backend = get_backend()
    counter_terms = (
        '("no association"[Title/Abstract] OR "not associated"[Title/Abstract] OR '
        '"negative association"[Title/Abstract] OR "no effect"[Title/Abstract] OR '
        '"failed to"[Title/Abstract] OR inconsistent[Title/Abstract] OR '
        'inverse[Title/Abstract] OR contrary[Title/Abstract])'
    )
    query = f'"{entity_a}" AND "{entity_b}" AND {counter_terms}'
    candidates = backend.search_literature(query, before_year, top_k)

    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "before_year": before_year,
        "backend": backend.backend_name,
        "counter_evidence_found": False,
        "confidence": 0.0,
        "rationale": "No semantic counter-evidence judgment was performed.",
        "candidate_papers": candidates,
        "counter_pmids": [],
    }

    if not candidates:
        result["rationale"] = "No pre-cutoff counter-evidence candidates were retrieved."
        return result

    if not deepseek_enabled():
        result["rationale"] = (
            "Counter-evidence candidates were retrieved, but an LLM is required to distinguish "
            "true contradiction from incidental negative wording."
        )
        return result

    judged = chat_json(
        system_prompt=(
            "You are the counter-evidence verifier in a biomedical Literature-Based Discovery "
            "system. Use only the supplied pre-cutoff evidence. Determine whether any paper "
            "actually weakens, contradicts, or reports a null/inverse result for the proposed "
            "A-B relation. Output JSON only with keys counter_evidence_found (boolean), confidence "
            "(0 to 1), rationale (short string), counter_pmids (list of strings)."
        ),
        user_prompt=(
            f"entity_a={entity_a}\n"
            f"entity_b={entity_b}\n"
            f"cutoff_year={before_year}\n"
            f"candidate_evidence={_evidence_payload(candidates)}"
        ),
    )

    valid_pmids = {str(record.get("pmid")) for record in candidates if record.get("pmid")}
    proposed_pmids = judged.get("counter_pmids", [])
    counter_pmids = (
        [str(pmid) for pmid in proposed_pmids if str(pmid) in valid_pmids]
        if isinstance(proposed_pmids, list)
        else []
    )

    result.update(
        {
            "counter_evidence_found": bool(judged.get("counter_evidence_found", False)),
            "confidence": _confidence(judged.get("confidence")),
            "rationale": str(judged.get("rationale", "")),
            "counter_pmids": counter_pmids,
        }
    )
    return result


@tool
def search_counter_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 6,
) -> dict[str, Any]:
    """Search for and semantically judge pre-cutoff evidence that may weaken a relation."""
    return _search_counter_evidence_impl(entity_a, entity_b, before_year, top_k)


def _check_novelty_impl(
    entity_a: str,
    entity_c: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    backend = get_backend()
    evidence = backend.get_pair_evidence(entity_a, entity_c, before_year, top_k)
    mention_count = backend.count_pair_mentions(entity_a, entity_c, before_year)

    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_c": entity_c,
        "before_year": before_year,
        "backend": backend.backend_name,
        "mention_count": mention_count,
        "direct_relation_known": False,
        "novel": True,
        "confidence": 1.0 if mention_count == 0 else 0.0,
        "rationale": "No pre-cutoff pair mentions were found.",
        "evidence": evidence,
        "known_relation_pmids": [],
    }

    if mention_count == 0:
        return result

    if not evidence:
        result.update(
            {
                "direct_relation_known": None,
                "novel": None,
                "confidence": 0.0,
                "rationale": "Pre-cutoff co-mentions exist, but direct relation status is unresolved.",
            }
        )
        return result

    if not deepseek_enabled():
        result.update(
            {
                "direct_relation_known": None,
                "novel": None,
                "confidence": 0.0,
                "rationale": (
                    "Pre-cutoff co-mentions exist. Co-occurrence alone cannot establish whether "
                    "the A-C relation was already known."
                ),
            }
        )
        return result

    judged = chat_json(
        system_prompt=(
            "You are the novelty auditor in a biomedical Literature-Based Discovery system. "
            "Use only the supplied pre-cutoff evidence. Decide whether the literature already "
            "explicitly establishes a direct biomedical relation between A and C. Mere "
            "co-occurrence is not enough. Output JSON only with keys direct_relation_known "
            "(boolean), confidence (0 to 1), rationale (short string), known_relation_pmids "
            "(list of strings)."
        ),
        user_prompt=(
            f"entity_a={entity_a}\n"
            f"entity_c={entity_c}\n"
            f"cutoff_year={before_year}\n"
            f"evidence={_evidence_payload(evidence)}"
        ),
    )

    known = bool(judged.get("direct_relation_known", False))
    valid_pmids = {str(record.get("pmid")) for record in evidence if record.get("pmid")}
    proposed_pmids = judged.get("known_relation_pmids", [])
    known_pmids = (
        [str(pmid) for pmid in proposed_pmids if str(pmid) in valid_pmids]
        if isinstance(proposed_pmids, list)
        else []
    )

    result.update(
        {
            "direct_relation_known": known,
            "novel": not known,
            "confidence": _confidence(judged.get("confidence")),
            "rationale": str(judged.get("rationale", "")),
            "known_relation_pmids": known_pmids,
        }
    )
    return result


@tool
def check_novelty(
    entity_a: str,
    entity_c: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Check whether an A-C relation was already explicitly known before a cutoff."""
    return _check_novelty_impl(entity_a, entity_c, before_year, top_k)


@tool
def aggregate_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Aggregate support and counter-evidence for one temporally grounded relation.

    This is the normalized relation-evidence object intended for later caching, routing,
    progressive disclosure, and MCP exposure.
    """
    support = _verify_relation_impl(entity_a, entity_b, before_year, top_k)
    counter = _search_counter_evidence_impl(
        entity_a,
        entity_b,
        before_year,
        max(3, min(top_k, 8)),
    )

    support_pmids = [str(pmid) for pmid in support.get("supporting_pmids", [])]
    counter_pmids = [str(pmid) for pmid in counter.get("counter_pmids", [])]
    evidence_pmids = list(dict.fromkeys(support_pmids + counter_pmids))

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "before_year": before_year,
        "backend": support.get("backend", get_backend().backend_name),
        "relation_supported": support.get("relation_supported"),
        "relation_type": support.get("relation_type"),
        "support_confidence": support.get("confidence", 0.0),
        "counter_evidence_found": counter.get("counter_evidence_found", False),
        "counter_confidence": counter.get("confidence", 0.0),
        "supporting_pmids": support_pmids,
        "counter_pmids": counter_pmids,
        "evidence_pmids": evidence_pmids,
        "support": support,
        "counter": counter,
    }
