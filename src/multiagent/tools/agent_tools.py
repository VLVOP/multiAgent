from __future__ import annotations

from langchain_core.tools import tool

from multiagent.tools.lbd_tools import (
    aggregate_evidence,
    check_novelty,
    get_entity_info,
    search_counter_evidence,
    verify_relation,
)
from multiagent.tools.runtime import get_backend


@tool
def search_literature(query: str, before_year: int, top_k: int = 5) -> list[dict]:
    """Search temporally constrained biomedical literature using the active backend."""
    return get_backend().search_literature(query, before_year, top_k)


@tool
def resolve_entity(term: str, top_k: int = 5) -> list[dict]:
    """Resolve a biomedical term to normalized entity candidates using the active backend."""
    return get_backend().resolve_entity(term, top_k)


@tool
def expand_entity(
    entity: str,
    before_year: int,
    max_articles: int = 50,
    top_k: int = 15,
) -> list[dict]:
    """Expand an entity into temporally valid candidate bridge entities.

    Returned candidates are discovery neighbors and are not automatically interpreted as
    typed or causal biomedical relations.
    """
    return get_backend().expand_entity(
        entity,
        before_year,
        max_articles=max_articles,
        top_k=top_k,
    )


@tool
def get_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve pre-cutoff literature evidence candidates for a biomedical entity pair."""
    return get_backend().get_pair_evidence(entity_a, entity_b, before_year, top_k)


@tool
def count_pair_mentions(entity_a: str, entity_b: str, before_year: int) -> int:
    """Count pre-cutoff records mentioning both entities as a coarse knownness signal."""
    return get_backend().count_pair_mentions(entity_a, entity_b, before_year)


# Backward-compatible Python aliases. New Agent code should use the stable generic names above.
search_pubmed_literature = search_literature
search_mesh_entity = resolve_entity
expand_biomedical_entity = expand_entity
find_entity_pair_evidence = get_evidence
count_entity_pair_mentions = count_pair_mentions


LBD_TOOLS = [
    resolve_entity,
    get_entity_info,
    search_literature,
    expand_entity,
    get_evidence,
    verify_relation,
    search_counter_evidence,
    check_novelty,
    aggregate_evidence,
    count_pair_mentions,
]
