from __future__ import annotations

from langchain_core.tools import tool

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
    find_entity_pair_evidence,
    count_entity_pair_mentions,
]
