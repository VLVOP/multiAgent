from multiagent.context import context_manager


BASE_STATE = {
    "target_entity": "Migraine",
    "cutoff_year": 1985,
    "iteration": 2,
    "max_iterations": 5,
    "plan": {"objective": "test"},
    "frontier": ["Migraine"],
    "entity_paths": [["Migraine", "Vascular tone", "Magnesium"]],
    "exploration_observations": [{"raw": "detail"}],
    "hypotheses": [{"a": "Migraine", "b": "Vascular tone", "c": "Magnesium"}],
    "current_hypothesis": {
        "a": "Migraine",
        "b": "Vascular tone",
        "c": "Magnesium",
    },
    "verification": {
        "ab_supported": True,
        "bc_supported": False,
        "ac_novelty_resolved": True,
    },
    "reflection": {"issue": "B-C weak", "recommendation": "refine"},
    "refinement_request": {"target": "bc", "top_k": 12, "round": 1},
    "refinement_round": 1,
    "failed_paths": [],
    "route": "refine",
    "trace": ["PLAN", "EXPLORE"],
}


def test_planner_receives_l1_without_evidence_state():
    view = context_manager.project_state("planner", BASE_STATE)

    assert view["target_entity"] == "Migraine"
    assert view["cutoff_year"] == 1985
    assert "verification" not in view
    assert "entity_paths" not in view
    assert "exploration_observations" not in view


def test_explorer_receives_structured_l2_but_not_detailed_verification():
    view = context_manager.project_state("explorer", BASE_STATE)

    assert view["plan"] == {"objective": "test"}
    assert view["entity_paths"] == [["Migraine", "Vascular tone", "Magnesium"]]
    assert "verification" not in view
    assert "exploration_observations" not in view


def test_verifier_receives_refinement_and_previous_verification_state():
    view = context_manager.project_state("verifier", BASE_STATE)

    assert view["current_hypothesis"]["c"] == "Magnesium"
    assert view["refinement_request"]["target"] == "bc"
    assert view["verification"]["bc_supported"] is False


def test_critic_receives_evidence_state_without_trace_history():
    view = context_manager.project_state("critic", BASE_STATE)

    assert view["verification"]["ab_supported"] is True
    assert view["current_hypothesis"]["b"] == "Vascular tone"
    assert "trace" not in view


def test_level_three_is_not_disclosed_before_verification_exists():
    state = dict(BASE_STATE)
    state.pop("verification")

    metadata = context_manager.view_metadata("critic", state)

    assert metadata["level"] == 2
