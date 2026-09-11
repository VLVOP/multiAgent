from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from multiagent.llm import chat_json, deepseek_enabled
from multiagent.tools.ncbi import NCBIClient, PubMedArticle


def _articles_to_dict(articles: list[PubMedArticle]) -> list[dict[str, Any]]:
    return [article.to_dict() for article in articles]


def _evidence_payload(articles: list[PubMedArticle], limit_chars: int = 18000) -> str:
    payload = [
        {
            "pmid": article.pmid,
            "year": article.year,
            "title": article.title,
            "abstract": article.abstract,
            "mesh_terms": list(article.mesh_terms),
        }
        for article in articles
    ]
    return json.dumps(payload, ensure_ascii=False)[:limit_chars]


@tool
def get_entity_info(entity: str, before_year: int, top_k: int = 5) -> dict[str, Any]:
    """Get temporally constrained biomedical context for an entity.

    Returns MeSH candidates, the pre-cutoff PubMed mention count, and a small set of
    representative pre-cutoff papers. This tool describes an entity; it does not infer
    a relation to another entity.
    """
    with NCBIClient() as client:
        mesh = [concept.to_dict() for concept in client.search_mesh(entity, top_k)]
        query = f'"{entity}"'
        mention_count = client.count_pubmed(query, before_year)
        papers = client.search_pubmed(query, before_year, min(top_k, 10))

    return {
        "entity": entity,
        "before_year": before_year,
        "mesh_candidates": mesh,
        "mention_count": mention_count,
        "sample_papers": _articles_to_dict(papers),
    }


@tool
def verify_relation(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Verify whether pre-cutoff literature explicitly supports an A-B biomedical relation.

    PubMed co-mentions are retrieved first. When DeepSeek is configured, the abstracts are
    judged for an explicit biomedical relation. Without an LLM key, the tool deliberately
    returns relation_supported=None rather than treating co-occurrence as proof.
    """
    with NCBIClient() as client:
        evidence = client.pair_evidence(entity_a, entity_b, before_year, top_k)
        mention_count = client.pair_mention_count(entity_a, entity_b, before_year)

    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "before_year": before_year,
        "mention_count": mention_count,
        "evidence": _articles_to_dict(evidence),
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
                "rationale": "No pre-cutoff PubMed pair evidence was retrieved.",
            }
        )
        return result

    if not deepseek_enabled():
        return result

    judged = chat_json(
        system_prompt=(
            "You are a biomedical relation-verification component inside a Literature-Based "
            "Discovery system. Use only the supplied pre-cutoff PubMed evidence. Co-occurrence "
            "alone is not sufficient. Decide whether the evidence explicitly supports a meaningful "
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

    valid_pmids = {article.pmid for article in evidence}
    proposed_pmids = judged.get("supporting_pmids", [])
    supporting_pmids = [
        str(pmid) for pmid in proposed_pmids if str(pmid) in valid_pmids
    ] if isinstance(proposed_pmids, list) else []

    result.update(
        {
            "relation_supported": bool(judged.get("relation_supported", False)),
            "relation_type": judged.get("relation_type"),
            "confidence": max(0.0, min(1.0, float(judged.get("confidence", 0.0)))),
            "rationale": str(judged.get("rationale", "")),
            "supporting_pmids": supporting_pmids,
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
    """Search pre-cutoff PubMed for evidence that may weaken or contradict an A-B relation.

    The retrieval query emphasizes negative, null, inconsistent, or inverse findings. Returned
    papers are counter-evidence candidates; when DeepSeek is configured they are semantically
    judged rather than assumed to be contradictory from keywords alone.
    """
    counter_terms = (
        '("no association"[Title/Abstract] OR "not associated"[Title/Abstract] OR '
        '"negative association"[Title/Abstract] OR "no effect"[Title/Abstract] OR '
        '"failed to"[Title/Abstract] OR inconsistent[Title/Abstract] OR '
        'inverse[Title/Abstract] OR contrary[Title/Abstract])'
    )
    query = f'"{entity_a}" AND "{entity_b}" AND {counter_terms}'

    with NCBIClient() as client:
        candidates = client.search_pubmed(query, before_year, top_k)

    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "before_year": before_year,
        "counter_evidence_found": False,
        "confidence": 0.0,
        "rationale": "No semantic counter-evidence judgment was performed.",
        "candidate_papers": _articles_to_dict(candidates),
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
            "system. Use only the supplied pre-cutoff PubMed evidence. Determine whether any paper "
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

    valid_pmids = {article.pmid for article in candidates}
    proposed_pmids = judged.get("counter_pmids", [])
    counter_pmids = [
        str(pmid) for pmid in proposed_pmids if str(pmid) in valid_pmids
    ] if isinstance(proposed_pmids, list) else []

    result.update(
        {
            "counter_evidence_found": bool(judged.get("counter_evidence_found", False)),
            "confidence": max(0.0, min(1.0, float(judged.get("confidence", 0.0)))),
            "rationale": str(judged.get("rationale", "")),
            "counter_pmids": counter_pmids,
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
    """Check whether an A-C biomedical relation was already explicitly known before a cutoff.

    Zero retrieved co-mentions is strong evidence of literature novelty. If pair evidence exists,
    DeepSeek (when configured) judges whether it already establishes the direct A-C relation.
    Without an LLM key, ambiguous co-mention cases are returned as novel=None.
    """
    with NCBIClient() as client:
        evidence = client.pair_evidence(entity_a, entity_c, before_year, top_k)
        mention_count = client.pair_mention_count(entity_a, entity_c, before_year)

    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_c": entity_c,
        "before_year": before_year,
        "mention_count": mention_count,
        "direct_relation_known": False,
        "novel": True,
        "confidence": 1.0 if mention_count == 0 else 0.0,
        "rationale": "No pre-cutoff pair mentions were found.",
        "evidence": _articles_to_dict(evidence),
        "known_relation_pmids": [],
    }

    if mention_count == 0:
        return result

    if not evidence:
        # There are indexed co-mentions, but the top-k evidence fetch did not return a usable record.
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
            "Use only the supplied pre-cutoff PubMed evidence. Decide whether the literature "
            "already explicitly establishes a direct biomedical relation between A and C. Mere "
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
    valid_pmids = {article.pmid for article in evidence}
    proposed_pmids = judged.get("known_relation_pmids", [])
    known_pmids = [
        str(pmid) for pmid in proposed_pmids if str(pmid) in valid_pmids
    ] if isinstance(proposed_pmids, list) else []

    result.update(
        {
            "direct_relation_known": known,
            "novel": not known,
            "confidence": max(0.0, min(1.0, float(judged.get("confidence", 0.0)))),
            "rationale": str(judged.get("rationale", "")),
            "known_relation_pmids": known_pmids,
        }
    )
    return result
