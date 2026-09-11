from __future__ import annotations

import json
import os
from typing import Any

from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState
from multiagent.tools.agent_tools import expand_biomedical_entity
from multiagent.tools.mock_graphs import expand_entity


def online_tools_enabled() -> bool:
    return os.getenv("ONLINE_TOOLS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


class ExplorerAgent:
    """Explore the biomedical entity graph and propose candidate A-B-C paths."""

    def _mock_paths(self, state: DiscoveryState) -> list[list[str]]:
        target = state["target_entity"]
        cutoff = state["cutoff_year"]
        failed = {tuple(path) for path in state.get("failed_paths", [])}
        paths: list[list[str]] = []
        for ab in expand_entity(target, cutoff):
            for bc in expand_entity(ab.tail, cutoff):
                candidate = [target, ab.tail, bc.tail]
                if tuple(candidate) not in failed:
                    paths.append(candidate)
        return paths

    def _online_paths(self, state: DiscoveryState) -> tuple[list[list[str]], list[dict[str, Any]]]:
        target = state["target_entity"]
        cutoff = state["cutoff_year"]
        failed = {tuple(path) for path in state.get("failed_paths", [])}

        first_hop = expand_biomedical_entity.invoke(
            {"entity": target, "before_year": cutoff, "max_articles": 40, "top_k": 6}
        )
        paths: list[list[str]] = []
        observations: list[dict[str, Any]] = []

        for b_item in first_hop[:4]:
            b = str(b_item.get("entity") or b_item.get("name") or "").strip()
            if not b or b.lower() == target.lower():
                continue
            observations.append({"source": target, "neighbor": b, "metadata": b_item})
            second_hop = expand_biomedical_entity.invoke(
                {"entity": b, "before_year": cutoff, "max_articles": 30, "top_k": 5}
            )
            for c_item in second_hop[:4]:
                c = str(c_item.get("entity") or c_item.get("name") or "").strip()
                if not c or c.lower() in {target.lower(), b.lower()}:
                    continue
                candidate = [target, b, c]
                if tuple(candidate) not in failed:
                    paths.append(candidate)
                    observations.append({"source": b, "neighbor": c, "metadata": c_item})

        unique: list[list[str]] = []
        seen: set[tuple[str, str, str]] = set()
        for path in paths:
            key = tuple(path)
            if key not in seen:
                unique.append(path)
                seen.add(key)
        return unique, observations

    def run(self, state: DiscoveryState) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        paths: list[list[str]] = []

        if online_tools_enabled():
            try:
                paths, observations = self._online_paths(state)
            except Exception as exc:
                observations.append({"tool_error": repr(exc)})

        if not paths:
            paths = self._mock_paths(state)

        if deepseek_enabled() and paths:
            result = chat_json(
                system_prompt=(
                    "You are the exploration agent of a biomedical open-LBD system. Rank only the "
                    "candidate A-B-C entity paths provided. Do not invent paths or evidence. Output "
                    "JSON with key ordered_indices, a list of zero-based candidate indices."
                ),
                user_prompt=(
                    f"cutoff_year={state['cutoff_year']}\n"
                    f"plan={json.dumps(state.get('plan', {}), ensure_ascii=False)}\n"
                    f"candidates={json.dumps(paths[:20], ensure_ascii=False)}"
                ),
            )
            indices = result.get("ordered_indices")
            if isinstance(indices, list):
                valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(paths)]
                if valid:
                    used = set(valid)
                    paths = [paths[i] for i in valid] + [
                        path for i, path in enumerate(paths) if i not in used
                    ]

        return {"entity_paths": paths, "exploration_observations": observations}
