from __future__ import annotations

from copy import deepcopy
from typing import Any


ToolUsage = dict[str, Any]
ToolUsageDelta = dict[str, int]


def empty_tool_usage() -> ToolUsage:
    return {
        "total_calls": 0,
        "by_tool": {},
        "by_agent": {},
    }


def merge_tool_usage(
    current: ToolUsage | None,
    *,
    agent: str,
    delta: ToolUsageDelta | None,
) -> ToolUsage:
    """Return a new cumulative tool-usage object from one agent-local delta."""
    usage = deepcopy(current) if current else empty_tool_usage()
    by_tool = dict(usage.get("by_tool", {}))
    by_agent = dict(usage.get("by_agent", {}))
    agent_tools = dict(by_agent.get(agent, {}))

    for tool_name, count_value in (delta or {}).items():
        count = max(0, int(count_value))
        if count == 0:
            continue
        usage["total_calls"] = int(usage.get("total_calls", 0)) + count
        by_tool[tool_name] = int(by_tool.get(tool_name, 0)) + count
        agent_tools[tool_name] = int(agent_tools.get(tool_name, 0)) + count

    by_agent[agent] = agent_tools
    usage["by_tool"] = by_tool
    usage["by_agent"] = by_agent
    return usage
