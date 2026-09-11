from multiagent.graph import graph


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
    assert result["current_hypothesis"]["c"] == "Magnesium"
