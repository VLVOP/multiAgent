from multiagent.context_policy import disclosure_policy_for_state


def test_hierarchical_policy_targets_only_bc_during_bc_refinement():
    state = {
        "context_mode": "hierarchical",
        "verification": {"ab_supported": True, "bc_supported": False},
        "refinement_request": {"target": "bc"},
    }

    policy = disclosure_policy_for_state(state)
    decision = policy.decide("verifier", state)

    assert policy.name == "heuristic_lbd"
    assert decision.level == 3
    assert decision.regions == ("bc",)
    assert "bc" in decision.reason


def test_full_context_policy_is_explicit_ablation_baseline():
    state = {"context_mode": "full"}

    policy = disclosure_policy_for_state(state)
    decision = policy.decide("planner", state)

    assert policy.name == "full_context"
    assert decision.level == 3
    assert decision.regions == ("all",)
