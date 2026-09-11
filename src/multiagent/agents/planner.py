from __future__ import annotations

from typing import Any

from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState


class PlannerAgent:
    """Lightweight initial planner for biomedical open-LBD exploration."""

    def run(self, state: DiscoveryState) -> dict[str, Any]:
        target = state["target_entity"]
        cutoff = state["cutoff_year"]

        plan: dict[str, Any] = {
            "target": target,
            "cutoff_year": cutoff,
            "objective": "discover novel biomedical entity C through an A-B-C bridge",
            "exploration_focus": [
                "genes/proteins",
                "pathways/biological processes",
                "drugs/compounds",
                "phenotypes/diseases",
            ],
        }

        if deepseek_enabled():
            result = chat_json(
                system_prompt=(
                    "You are the lightweight planning agent of a biomedical Literature-Based "
                    "Discovery system following the Swanson/Arrowsmith ABC paradigm. Do not solve "
                    "the task. Produce a compact exploration plan only. Output JSON with keys "
                    "objective and exploration_focus. exploration_focus must be a short list."
                ),
                user_prompt=(
                    f"Target entity A: {target}\n"
                    f"Temporal cutoff: {cutoff}\n"
                    "Plan which biomedical semantic directions should be explored first."
                ),
            )
            plan["objective"] = str(result.get("objective", plan["objective"]))
            focus = result.get("exploration_focus")
            if isinstance(focus, list) and focus:
                plan["exploration_focus"] = [str(item) for item in focus[:8]]

        return plan
