from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from multiagent.edges import route_after_critique
from multiagent.nodes import (
    backtrack_node,
    critique_node,
    explore_node,
    hypothesize_node,
    plan_node,
    refine_node,
    verify_node,
)
from multiagent.state import DiscoveryState


def build_graph():
    builder = StateGraph(DiscoveryState)

    builder.add_node("plan", plan_node)
    builder.add_node("explore", explore_node)
    builder.add_node("hypothesize", hypothesize_node)
    builder.add_node("verify", verify_node)
    builder.add_node("critique", critique_node)
    builder.add_node("refine", refine_node)
    builder.add_node("backtrack", backtrack_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "explore")
    builder.add_edge("explore", "hypothesize")
    builder.add_edge("hypothesize", "verify")
    builder.add_edge("verify", "critique")

    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "accept": END,
            "refine": "refine",
            "backtrack": "backtrack",
            "explore": "explore",
            "stop": END,
        },
    )

    builder.add_edge("refine", "verify")
    builder.add_edge("backtrack", "explore")

    return builder.compile()


graph = build_graph()
