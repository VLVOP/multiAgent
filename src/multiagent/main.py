from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from multiagent.graph import graph
from multiagent.llm import smoke_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biomedical LBD multi-agent MVP")
    parser.add_argument("--target", default="Migraine")
    parser.add_argument("--cutoff", type=int, default=1985)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test DeepSeek API connectivity and exit.",
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
        "max_iterations": 5,
        "trace": [],
    }

    result = graph.invoke(initial_state, config={"recursion_limit": 20})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
