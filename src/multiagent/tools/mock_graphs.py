from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationEdge:
    head: str
    relation: str
    tail: str
    first_year: int
    supporting_pmids: tuple[str, ...]


ENTITY_RELATIONS = [
    # The first candidate is deliberately already known so the MVP exercises BACKTRACK -> EXPLORE.
    RelationEdge("Migraine", "associated_with", "Serotonin", 1975, ("PMID:MOCK3",)),
    RelationEdge("Serotonin", "associated_with", "Calcium", 1980, ("PMID:MOCK4",)),
    RelationEdge("Migraine", "associated_with", "Vascular tone", 1978, ("PMID:MOCK1",)),
    RelationEdge("Vascular tone", "associated_with", "Magnesium", 1981, ("PMID:MOCK2",)),
]

KNOWN_DIRECT_LINKS = {
    ("Migraine", "Calcium", 1985),
}


def expand_entity(entity: str, cutoff_year: int) -> list[RelationEdge]:
    return [
        edge for edge in ENTITY_RELATIONS
        if edge.head == entity and edge.first_year <= cutoff_year
    ]


def get_relation(a: str, b: str, cutoff_year: int) -> RelationEdge | None:
    for edge in ENTITY_RELATIONS:
        if edge.head == a and edge.tail == b and edge.first_year <= cutoff_year:
            return edge
    return None


def get_evidence(a: str, b: str, cutoff_year: int) -> list[str]:
    edge = get_relation(a, b, cutoff_year)
    return list(edge.supporting_pmids) if edge else []


def check_direct_link(a: str, c: str, cutoff_year: int) -> bool:
    return any(
        known_a == a and known_c == c and known_year <= cutoff_year
        for known_a, known_c, known_year in KNOWN_DIRECT_LINKS
    )
