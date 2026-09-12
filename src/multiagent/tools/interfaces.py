from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


Record = dict[str, Any]


@runtime_checkable
class LBDRetrievalBackend(Protocol):
    """Backend contract used by the agent-facing LBD tools.

    Agent code should depend on the stable tool interface, not on PubMed/MeSH-specific
    client methods. Online and frozen/local backends can implement this same contract.
    Every literature-facing method receives an explicit temporal cutoff where relevant.
    """

    backend_name: str

    def resolve_entity(self, term: str, top_k: int = 5) -> list[Record]: ...

    def search_literature(
        self,
        query: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[Record]: ...

    def count_literature(self, query: str, before_year: int) -> int: ...

    def expand_entity(
        self,
        entity: str,
        before_year: int,
        max_articles: int = 50,
        top_k: int = 15,
    ) -> list[Record]: ...

    def get_pair_evidence(
        self,
        entity_a: str,
        entity_b: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[Record]: ...

    def count_pair_mentions(
        self,
        entity_a: str,
        entity_b: str,
        before_year: int,
    ) -> int: ...
