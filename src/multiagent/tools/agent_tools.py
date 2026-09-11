from __future__ import annotations

from langchain_core.tools import tool

from multiagent.tools.lbd_tools import (
    check_novelty,
    get_entity_info,
    search_counter_evidence,
    verify_relation,
)
from multiagent.tools.ncbi import NCBIClient


@tool
def search_pubmed_literature(query: str, before_year: int, top_k: int = 5) -> list[dict]:
    """Search PubMed before a cutoff year and return article metadata and abstracts."""
    with NCBIClient() as client:
        return [article.to_dict() for article in client.search_pubmed(query, before_year, top_k)]


@tool
def search_mesh_entity(term: str, top_k: int = 5) -> list[dict]:
    """Resolve a biomedical term against the NCBI MeSH database."""
    with NCBIClient() as client:
        return [concept.to_dict() for concept in client.search_mesh(term, top_k)]


@tool
def expand_biomedical_entity(
    entity: str,
    before_year: int,
    max_articles: int = 50,
    top_k: int = 15,
) -> list[dict]:
    """Expand an entity into pre-cutoff MeSH neighbors from retrieved PubMed literature.

    Returned neighbors represent literature co-indexing and are candidate bridge entities,
    not typed or causal biomedical relations.
    """
    with NCBIClient() as client:
        return [
            neighbor.to_dict()
            for neighbor in client.expand_entity_via_mesh(
                entity,
                before_year,
                max_articles=max_articles,
                top_k=top_k,
            )
        ]


@tool
def find_entity_pair_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 5,
) -> list[dict]:
    """Find pre-cutoff PubMed papers mentioning both biomedical entities."""
    with NCBIClient() as client:
        return [
            article.to_dict()
            for article in client.pair_evidence(entity_a, entity_b, before_year, top_k)
        ]


@tool
def count_entity_pair_mentions(entity_a: str, entity_b: str, before_year: int) -> int:
    """Count pre-cutoff PubMed records mentioning both entities.

    This is a literature-overlap signal, not proof that a typed biomedical relation exists.
    """
    with NCBIClient() as client:
        return client.pair_mention_count(entity_a, entity_b, before_year)


LBD_TOOLS = [
    search_pubmed_literature,
    search_mesh_entity,
    expand_biomedical_entity,
    find_entity_pair_evidence,
    count_entity_pair_mentions,
    get_entity_info,
    verify_relation,
    search_counter_evidence,
    check_novelty,
]
