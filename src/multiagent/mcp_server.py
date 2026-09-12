from __future__ import annotations

import json
from typing import Any

from mcp.server import MCPServer

from multiagent.tools.agent_tools import (
    aggregate_evidence as _aggregate_evidence,
    check_novelty as _check_novelty,
    expand_entity as _expand_entity,
    get_entity_info as _get_entity_info,
    get_evidence as _get_evidence,
    resolve_entity as _resolve_entity,
    search_counter_evidence as _search_counter_evidence,
    search_literature as _search_literature,
    verify_relation as _verify_relation,
)
from multiagent.tools.runtime import get_backend


mcp = MCPServer("Biomedical LBD Tool Server")


@mcp.tool()
def resolve_entity(term: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Resolve a biomedical term to normalized entity candidates."""
    return _resolve_entity.invoke({"term": term, "top_k": top_k})


@mcp.tool()
def get_entity_info(
    entity: str,
    before_year: int,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return temporally constrained biomedical context for one entity."""
    return _get_entity_info.invoke(
        {"entity": entity, "before_year": before_year, "top_k": top_k}
    )


@mcp.tool()
def search_literature(
    query: str,
    before_year: int,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search biomedical literature while enforcing the temporal cutoff in the backend."""
    return _search_literature.invoke(
        {"query": query, "before_year": before_year, "top_k": top_k}
    )


@mcp.tool()
def expand_entity(
    entity: str,
    before_year: int,
    max_articles: int = 50,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Expand a biomedical entity into temporally valid bridge candidates."""
    return _expand_entity.invoke(
        {
            "entity": entity,
            "before_year": before_year,
            "max_articles": max_articles,
            "top_k": top_k,
        }
    )


@mcp.tool()
def get_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve pre-cutoff evidence candidates for an entity pair."""
    return _get_evidence.invoke(
        {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "before_year": before_year,
            "top_k": top_k,
        }
    )


@mcp.tool()
def verify_relation(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Semantically verify a pre-cutoff biomedical relation from retrieved evidence."""
    return _verify_relation.invoke(
        {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "before_year": before_year,
            "top_k": top_k,
        }
    )


@mcp.tool()
def search_counter_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 6,
) -> dict[str, Any]:
    """Search and judge evidence that may weaken or contradict a relation."""
    return _search_counter_evidence.invoke(
        {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "before_year": before_year,
            "top_k": top_k,
        }
    )


@mcp.tool()
def check_novelty(
    entity_a: str,
    entity_c: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Audit whether an A-C relation was already explicitly known before the cutoff."""
    return _check_novelty.invoke(
        {
            "entity_a": entity_a,
            "entity_c": entity_c,
            "before_year": before_year,
            "top_k": top_k,
        }
    )


@mcp.tool()
def aggregate_evidence(
    entity_a: str,
    entity_b: str,
    before_year: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return the normalized support/counter-evidence object for one relation."""
    return _aggregate_evidence.invoke(
        {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "before_year": before_year,
            "top_k": top_k,
        }
    )


@mcp.resource("lbd://backend")
def backend_metadata() -> str:
    """Describe the active swappable LBD retrieval backend."""
    backend = get_backend()
    return json.dumps(
        {
            "backend": backend.backend_name,
            "temporal_cutoff_enforced_by_backend": True,
            "tool_interface": "Biomedical LBD Tool Layer v1",
        },
        ensure_ascii=False,
    )


def main() -> None:
    """Run the MCP server over stdio for local hosts and MCP clients."""
    mcp.run()


if __name__ == "__main__":
    main()
