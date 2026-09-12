from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from multiagent.graph import graph
from multiagent.llm import llm_metadata
from multiagent.state import DiscoveryState


@dataclass(frozen=True)
class ArchitecturePreset:
    name: str
    context_mode: str
    cache_enabled: bool
    a2a_enabled: bool
    description: str


ARCHITECTURE_PRESETS: dict[str, ArchitecturePreset] = {
    "sparse": ArchitecturePreset(
        name="sparse",
        context_mode="hierarchical",
        cache_enabled=True,
        a2a_enabled=True,
        description="Full Sparse Agentic LBD architecture baseline under development.",
    ),
    "full-context": ArchitecturePreset(
        name="full-context",
        context_mode="full",
        cache_enabled=True,
        a2a_enabled=True,
        description="Ablate hierarchical/progressive context while retaining cache and A2A.",
    ),
    "no-cache": ArchitecturePreset(
        name="no-cache",
        context_mode="hierarchical",
        cache_enabled=False,
        a2a_enabled=True,
        description="Ablate evidence reuse while retaining hierarchical context and A2A.",
    ),
    "no-a2a": ArchitecturePreset(
        name="no-a2a",
        context_mode="hierarchical",
        cache_enabled=True,
        a2a_enabled=False,
        description="Ablate structured sparse communication.",
    ),
    "all-off": ArchitecturePreset(
        name="all-off",
        context_mode="full",
        cache_enabled=False,
        a2a_enabled=False,
        description=(
            "Shared-context/no-cache/no-A2A architecture baseline; the common LangGraph loop "
            "is intentionally retained so component effects are isolated."
        ),
    ),
}


def build_initial_state(
    *,
    target: str,
    cutoff: int,
    preset: ArchitecturePreset,
    max_iterations: int = 5,
    max_refinement_rounds: int = 3,
) -> DiscoveryState:
    """Build an explicit, self-describing experiment state."""
    return {
        "target_entity": target,
        "cutoff_year": cutoff,
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_refinement_rounds": max_refinement_rounds,
        "architecture_preset": preset.name,
        "context_mode": preset.context_mode,  # type: ignore[typeddict-item]
        "cache_enabled": preset.cache_enabled,
        "a2a_enabled": preset.a2a_enabled,
        "llm_metadata": llm_metadata(),
        "trace": [],
    }


def summarize_result(result: DiscoveryState) -> dict[str, Any]:
    """Extract architecture-level metrics without requiring a benchmark dataset."""
    access_log = result.get("context_access_log", [])
    levels = [int(item.get("level", 0)) for item in access_log]
    region_accesses = sum(len(item.get("regions", [])) for item in access_log)

    context_summary = {
        "agent_calls": len(access_log),
        "mean_level": mean(levels) if levels else 0.0,
        "max_level": max(levels) if levels else 0,
        "region_accesses": region_accesses,
        "visible_field_total": sum(
            int(item.get("visible_field_count", 0)) for item in access_log
        ),
        "visible_message_total": sum(
            int(item.get("visible_message_count", 0)) for item in access_log
        ),
    }

    hypothesis = result.get("current_hypothesis")
    final_path = None
    if hypothesis:
        final_path = [hypothesis.get("a"), hypothesis.get("b"), hypothesis.get("c")]

    return {
        "target_entity": result.get("target_entity"),
        "cutoff_year": result.get("cutoff_year"),
        "architecture_preset": result.get("architecture_preset", "custom"),
        "architecture": {
            "context_mode": result.get("context_mode"),
            "cache_enabled": result.get("cache_enabled"),
            "a2a_enabled": result.get("a2a_enabled"),
            "max_iterations": result.get("max_iterations"),
            "max_refinement_rounds": result.get("max_refinement_rounds"),
        },
        "llm": result.get("llm_metadata", {}),
        "termination_reason": result.get("termination_reason"),
        "iteration": result.get("iteration", 0),
        "route": result.get("route"),
        "final_path": final_path,
        "trace_length": len(result.get("trace", [])),
        "tool_usage": dict(result.get("tool_usage", {})),
        "cache": dict(result.get("cache_stats", {})),
        "communication": dict(result.get("communication_stats", {})),
        "context": context_summary,
    }


def run_case(
    *,
    target: str,
    cutoff: int,
    preset: ArchitecturePreset,
    max_iterations: int = 5,
    max_refinement_rounds: int = 3,
) -> dict[str, Any]:
    initial_state = build_initial_state(
        target=target,
        cutoff=cutoff,
        preset=preset,
        max_iterations=max_iterations,
        max_refinement_rounds=max_refinement_rounds,
    )
    recursion_limit = max(20, max_iterations * 6)
    result = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
    return {
        "preset": asdict(preset),
        "summary": summarize_result(result),
        "result": result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run architecture-controlled biomedical LBD ablations on one case."
    )
    parser.add_argument("--target", default="Migraine")
    parser.add_argument("--cutoff", type=int, default=1985)
    parser.add_argument(
        "--presets",
        default="sparse",
        help="Comma-separated presets: " + ", ".join(ARCHITECTURE_PRESETS),
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-refinement-rounds", type=int, default=3)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only compact architecture metrics instead of the full DiscoveryState.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always supported.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    preset_names = [item.strip() for item in args.presets.split(",") if item.strip()]
    unknown = [name for name in preset_names if name not in ARCHITECTURE_PRESETS]
    if unknown:
        raise SystemExit(
            f"Unknown architecture preset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(ARCHITECTURE_PRESETS)}"
        )

    runs = [
        run_case(
            target=args.target,
            cutoff=args.cutoff,
            preset=ARCHITECTURE_PRESETS[name],
            max_iterations=args.max_iterations,
            max_refinement_rounds=args.max_refinement_rounds,
        )
        for name in preset_names
    ]

    payload: Any
    if args.summary_only:
        payload = [run["summary"] for run in runs]
    else:
        payload = runs

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
