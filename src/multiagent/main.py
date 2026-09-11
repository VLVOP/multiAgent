from __future__ import annotations

import json

from dotenv import load_dotenv

from multiagent.graph import graph


def main() -> None:
    load_dotenv()

    initial_state = {
        "target_entity": "Migraine",
        "cutoff_year": 1985,
        "iteration": 0,
        "max_iterations": 5,
        "trace": [],
    }

    result = graph.invoke(initial_state, config={"recursion_limit": 20})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
