from multiagent.edges import route_after_critique
from multiagent.graph import graph


def test_route_after_critique_keeps_loop_branch_under_budget():
    assert route_after_critique(
        {"route": "refine", "iteration": 1, "max_iterations": 3}
    ) == "refine"


def test_route_after_critique_stops_when_budget_is_exhausted():
    assert route_after_critique(
        {"route": "backtrack", "iteration": 3, "max_iterations": 3}
    ) == "stop"


def test_graph_exercises_backtrack_loop_then_accepts():
    result = graph.invoke(
        {
            "target_entity": "Migraine",
            "cutoff_year": 1985,
            "iteration": 0,
            "max_iterations": 5,
            "trace": [],
        },
        config={"recursion_limit": 20},
    )

    assert "BACKTRACK" in result["trace"]
    assert result["trace"].count("EXPLORE") >= 2
    assert result["route"] == "accept"
    assert result["termination_reason"] == "accepted"
    assert result["current_hypothesis"]["c"] == "Magnesium"


def test_graph_ends_cleanly_when_loop_budget_is_exhausted():
    result = graph.invoke(
        {
            "target_entity": "Migraine",
            "cutoff_year": 1985,
            "iteration": 0,
            "max_iterations": 1,
            "trace": [],
        },
        config={"recursion_limit": 20},
    )

    assert result["termination_reason"] == "max_iterations"
    assert result["iteration"] == 1
