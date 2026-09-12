from multiagent.context import context_manager
from multiagent.graph import graph


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
        "ac_already_known": False,
        "ac_novelty_resolved": True,
        "ab_verification": {"relation_supported": True, "evidence": ["ab-detail"]},
        "bc_verification": {"relation_supported": False, "evidence": ["bc-detail"]},
        "ac_novelty": {"direct_relation_known": False, "evidence": ["ac-detail"]},
    },
    "reflection": {"issue": "B-C weak", "recommendation": "refine"},
    "refinement_request": {"target": "bc", "top_k": 12, "round": 1},
    "refinement_round": 1,
    "failed_paths": [],
    "evidence_cache": {},
    "cache_stats": {"hits": 0, "misses": 0, "writes": 0},
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


def test_verifier_locally_discloses_only_refinement_region_details():
    view = context_manager.project_state("verifier", BASE_STATE)
    verification = view["verification"]

    assert "bc_verification" in verification
    assert "ab_verification" not in verification
    assert "ac_novelty" not in verification
    assert context_manager.view_metadata("verifier", BASE_STATE)["regions"] == ["bc"]


def test_critic_receives_only_disputed_relation_detail():
    view = context_manager.project_state("critic", BASE_STATE)

    assert view["verification"]["ab_supported"] is True
    assert view["current_hypothesis"]["b"] == "Vascular tone"
    assert "bc_verification" in view["verification"]
    assert "ab_verification" not in view["verification"]
    assert context_manager.view_metadata("critic", BASE_STATE)["regions"] == ["bc"]
    assert "trace" not in view


def test_level_three_is_not_disclosed_before_verification_exists():
    state = dict(BASE_STATE)
    state.pop("verification")

    metadata = context_manager.view_metadata("critic", state)

    assert metadata["level"] == 2


def test_graph_records_agent_context_access_levels():
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

    access = result["context_access_log"]
    assert access[0]["agent"] == "planner"
    assert access[0]["level"] == 1
    assert access[0]["regions"] == ["global"]
    assert any(item["agent"] == "explorer" and item["level"] == 2 for item in access)
    assert any(item["agent"] == "critic" and item["level"] == 3 for item in access)
