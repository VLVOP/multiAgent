from __future__ import annotations

from multiagent.tools.interfaces import Record
from multiagent.tools.ncbi import NCBIClient


class OnlineNCBIBackend:
    """Online PubMed/MeSH backend implementing the stable retrieval contract."""

    backend_name = "online_ncbi"

    def resolve_entity(self, term: str, top_k: int = 5) -> list[Record]:
        with NCBIClient() as client:
            return [concept.to_dict() for concept in client.search_mesh(term, top_k)]

    def search_literature(
        self,
        query: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[Record]:
        with NCBIClient() as client:
            return [
                article.to_dict()
                for article in client.search_pubmed(query, before_year, top_k)
            ]

    def count_literature(self, query: str, before_year: int) -> int:
        with NCBIClient() as client:
            return client.count_pubmed(query, before_year)

    def expand_entity(
        self,
        entity: str,
        before_year: int,
        max_articles: int = 50,
        top_k: int = 15,
    ) -> list[Record]:
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

    def get_pair_evidence(
        self,
        entity_a: str,
        entity_b: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[Record]:
        with NCBIClient() as client:
            return [
                article.to_dict()
                for article in client.pair_evidence(entity_a, entity_b, before_year, top_k)
            ]

    def count_pair_mentions(
        self,
        entity_a: str,
        entity_b: str,
        before_year: int,
    ) -> int:
        with NCBIClient() as client:
            return client.pair_mention_count(entity_a, entity_b, before_year)
