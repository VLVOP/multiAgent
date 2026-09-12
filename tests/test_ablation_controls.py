from multiagent.context import context_manager
from multiagent.graph import graph


def test_full_context_mode_exposes_discovery_state_but_not_measurement_log():
    state = {
        "target_entity": "Migraine",
        "cutoff_year": 1985,
        "context_mode": "full",
        "entity_paths": [["Migraine", "Vascular tone", "Magnesium"]],
        "verification": {"ab_supported": True},
        "context_access_log": [{"agent": "planner"}],
    }

    view = context_manager.project_state("planner", state)

    assert "entity_paths" in view
    assert "verification" in view
    assert "context_access_log" not in view
    assert context_manager.view_metadata("planner", state)["regions"] == ["all"]


def test_graph_can_disable_sparse_a2a_for_ablation():
    result = graph.invoke(
        {
            "target_entity": "Migraine",
            "cutoff_year": 1985,
            "iteration": 0,
            "max_iterations": 5,
            "a2a_enabled": False,
            "trace": [],
        },
        config={"recursion_limit": 20},
    )

    assert result["a2a_enabled"] is False
    assert result["agent_messages"] == []
    assert result["communication_stats"]["messages"] == 0


def test_graph_records_full_context_ablation_in_access_log():
    result = graph.invoke(
        {
            "target_entity": "Migraine",
            "cutoff_year": 1985,
            "iteration": 0,
            "max_iterations": 5,
            "context_mode": "full",
            "trace": [],
        },
        config={"recursion_limit": 20},
    )

    assert result["context_mode"] == "full"
    assert result["context_access_log"][0]["level"] == 3
    assert result["context_access_log"][0]["regions"] == ["all"]
