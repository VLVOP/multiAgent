from multiagent.agents.critic import CriticAgent


def test_critic_reuses_counter_evidence_for_same_abc_path():
    counter = {
        "ab": {
            "entity_a": "Migraine",
            "entity_b": "Vascular tone",
            "before_year": 1985,
            "counter_evidence_found": False,
        },
        "bc": {
            "entity_a": "Vascular tone",
            "entity_b": "Magnesium",
            "before_year": 1985,
            "counter_evidence_found": False,
        },
    }
    state = {
        "current_hypothesis": {
            "a": "Migraine",
            "b": "Vascular tone",
            "c": "Magnesium",
        },
        "cutoff_year": 1985,
        "cache_enabled": True,
        "reflection": {"counter_evidence": counter},
    }

    result = CriticAgent()._counter_evidence(state)

    assert result["_reused"] is True
    assert result["_tool_calls"] == 0
    assert result["ab"]["entity_b"] == "Vascular tone"


def test_critic_does_not_reuse_counter_evidence_for_different_path():
    state = {
        "current_hypothesis": {
            "a": "Migraine",
            "b": "Serotonin",
            "c": "Calcium",
        },
        "cutoff_year": 1985,
        "cache_enabled": True,
        "reflection": {
            "counter_evidence": {
                "ab": {
                    "entity_a": "Migraine",
                    "entity_b": "Vascular tone",
                    "before_year": 1985,
                },
                "bc": {
                    "entity_a": "Vascular tone",
                    "entity_b": "Magnesium",
                    "before_year": 1985,
                },
            }
        },
    }

    # ONLINE_TOOLS_ENABLED is false in the test environment, so a cache miss makes no tool call.
    result = CriticAgent()._counter_evidence(state)
    assert result == {"_tool_calls": 0}
