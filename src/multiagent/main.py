from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from multiagent.graph import graph
from multiagent.llm import llm_metadata, smoke_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biomedical LBD multi-agent MVP")
    parser.add_argument("--target", default="Migraine")
    parser.add_argument("--cutoff", type=int, default=1985)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-refinement-rounds", type=int, default=3)
    parser.add_argument(
        "--context-mode",
        choices=["hierarchical", "full"],
        default="hierarchical",
        help="Use adaptive hierarchical context or the full-context ablation baseline.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable relation/novelty evidence reuse for cache ablations.",
    )
    parser.add_argument(
        "--no-a2a",
        action="store_true",
        help="Disable structured A2A message emission for communication ablations.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test configured LLM API connectivity and exit.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    if args.smoke_test:
        print(smoke_test())
        return

    initial_state = {
        "target_entity": args.target,
        "cutoff_year": args.cutoff,
        "iteration": 0,
        "max_iterations": args.max_iterations,
        "max_refinement_rounds": args.max_refinement_rounds,
        "context_mode": args.context_mode,
        "cache_enabled": not args.no_cache,
        "a2a_enabled": not args.no_a2a,
        "llm_metadata": llm_metadata(),
        "trace": [],
    }

    recursion_limit = max(20, args.max_iterations * 6)
    result = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
