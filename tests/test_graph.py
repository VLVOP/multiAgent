from multiagent.edges import decide_after_critique, route_after_critique
from multiagent.graph import graph
from multiagent.nodes import refine_node


def test_route_after_critique_keeps_loop_branch_under_budget():
    assert route_after_critique(
        {
            "route": "refine",
            "iteration": 1,
            "max_iterations": 3,
            "refinement_round": 1,
            "max_refinement_rounds": 3,
        }
    ) == "refine"


def test_route_after_critique_stops_when_budget_is_exhausted():
    decision = decide_after_critique(
        {"route": "backtrack", "iteration": 3, "max_iterations": 3}
    )

    assert decision.semantic_route == "backtrack"
    assert decision.effective_route == "stop"
    assert decision.overridden is True


def test_route_after_critique_backtracks_after_repeated_refinement():
    decision = decide_after_critique(
        {
            "route": "refine",
            "iteration": 2,
            "max_iterations": 10,
            "refinement_round": 3,
            "max_refinement_rounds": 3,
        }
    )

    assert decision.semantic_route == "refine"
    assert decision.effective_route == "backtrack"
    assert decision.overridden is True
    assert route_after_critique(
        {
            "route": "refine",
            "iteration": 2,
            "max_iterations": 10,
            "refinement_round": 3,
            "max_refinement_rounds": 3,
        }
    ) == "backtrack"


def test_refine_node_translates_reflection_into_targeted_request():
    result = refine_node(
        {
            "reflection": {
                "issue": "B-C bridge evidence is incomplete",
                "recommendation": "refine",
                "refinement_target": "bc",
            },
            "refinement_round": 0,
            "trace": [],
        }
    )

    assert result["refinement_request"]["target"] == "bc"
    assert result["refinement_request"]["top_k"] == 12
    assert result["refinement_round"] == 1
    assert result["trace"][-1] == "REFINE[bc]"


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
    assert result["edge_route"] == "accept"
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
    assert result["edge_route"] == "stop"
    assert result["loop_decision"]["overridden"] is True
    assert result["iteration"] == 1
