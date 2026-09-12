from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from multiagent.tools.agent_tools import expand_entity, resolve_entity, search_literature
from multiagent.tools.runtime import get_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the biomedical LBD tool layer")
    parser.add_argument("--query", default="migraine magnesium")
    parser.add_argument("--cutoff", type=int, default=1985)
    parser.add_argument("--entity", default="Migraine")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    papers = search_literature.invoke(
        {"query": args.query, "before_year": args.cutoff, "top_k": args.top_k}
    )
    entities = resolve_entity.invoke({"term": args.entity, "top_k": args.top_k})
    neighbors = expand_entity.invoke(
        {
            "entity": args.entity,
            "before_year": args.cutoff,
            "max_articles": max(10, args.top_k * 5),
            "top_k": args.top_k,
        }
    )

    print(
        json.dumps(
            {
                "backend": get_backend().backend_name,
                "literature": papers,
                "resolved_entities": entities,
                "entity_neighbors": neighbors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
