import multiagent.nodes as nodes


def test_critic_message_follows_loop_policy_override(monkeypatch):
    monkeypatch.setattr(
        nodes.critic_agent,
        "run",
        lambda state: (
            "refine",
            {
                "issue": "B-C remains unresolved",
                "recommendation": "refine",
                "refinement_target": "bc",
                "_tool_usage_delta": {},
            },
        ),
    )

    result = nodes.critique_node(
        {
            "target_entity": "Migraine",
            "cutoff_year": 1985,
            "iteration": 2,
            "max_iterations": 10,
            "refinement_round": 3,
            "max_refinement_rounds": 3,
            "a2a_enabled": True,
            "current_hypothesis": {
                "a": "Migraine",
                "b": "Vascular tone",
                "c": "Magnesium",
            },
            "verification": {
                "ab_supported": True,
                "bc_supported": False,
                "ac_already_known": False,
                "ac_novelty_resolved": True,
            },
            "agent_messages": [],
            "communication_stats": {},
            "context_access_log": [],
            "trace": [],
        }
    )

    assert result["route"] == "refine"
    assert result["edge_route"] == "backtrack"
    assert result["loop_decision"]["overridden"] is True
    message = result["agent_messages"][-1]
    assert message["dst"] == "explorer"
    assert message["requested_action"] == "backtrack_and_explore"
    assert message["payload"]["semantic_recommendation"] == "refine"
    assert message["payload"]["effective_recommendation"] == "backtrack"
