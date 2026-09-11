from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from multiagent.tools.ncbi import NCBIClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the online biomedical tool layer")
    parser.add_argument("--query", default="migraine magnesium")
    parser.add_argument("--cutoff", type=int, default=1985)
    parser.add_argument("--entity", default="Migraine")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    with NCBIClient() as client:
        papers = [
            article.to_dict()
            for article in client.search_pubmed(args.query, args.cutoff, args.top_k)
        ]
        mesh = [concept.to_dict() for concept in client.search_mesh(args.entity, args.top_k)]

    print(json.dumps({"pubmed": papers, "mesh": mesh}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
