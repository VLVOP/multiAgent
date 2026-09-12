from __future__ import annotations

from typing import Any

from multiagent.tools.agent_tools import (
    count_pair_mentions,
    expand_entity,
    get_evidence,
    resolve_entity,
    search_literature,
)
from multiagent.tools.runtime import reset_backend, set_backend


class FakeBackend:
    backend_name = "fake"

    def resolve_entity(self, term: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [{"name": term, "mesh_ui": "D000001"}][:top_k]

    def search_literature(
        self,
        query: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return [
            {
                "pmid": "1",
                "title": query,
                "abstract": "mock",
                "year": before_year - 1,
                "mesh_terms": [],
            }
        ][:top_k]

    def count_literature(self, query: str, before_year: int) -> int:
        return 1

    def expand_entity(
        self,
        entity: str,
        before_year: int,
        max_articles: int = 50,
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        return [{"entity": f"{entity}-neighbor", "support_count": 3, "supporting_pmids": ["1"]}][
            :top_k
        ]

    def get_pair_evidence(
        self,
        entity_a: str,
        entity_b: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return [
            {
                "pmid": "2",
                "title": f"{entity_a}-{entity_b}",
                "abstract": "pair evidence",
                "year": before_year - 1,
                "mesh_terms": [],
            }
        ][:top_k]

    def count_pair_mentions(self, entity_a: str, entity_b: str, before_year: int) -> int:
        return 7


def test_stable_tools_use_injected_backend():
    set_backend(FakeBackend())
    try:
        assert resolve_entity.invoke({"term": "Migraine", "top_k": 1})[0]["name"] == "Migraine"
        assert search_literature.invoke(
            {"query": "migraine", "before_year": 1985, "top_k": 1}
        )[0]["year"] == 1984
        assert expand_entity.invoke(
            {"entity": "Migraine", "before_year": 1985, "max_articles": 10, "top_k": 1}
        )[0]["entity"] == "Migraine-neighbor"
        assert get_evidence.invoke(
            {
                "entity_a": "Migraine",
                "entity_b": "Magnesium",
                "before_year": 1985,
                "top_k": 1,
            }
        )[0]["pmid"] == "2"
        assert count_pair_mentions.invoke(
            {"entity_a": "Migraine", "entity_b": "Magnesium", "before_year": 1985}
        ) == 7
    finally:
        reset_backend()
