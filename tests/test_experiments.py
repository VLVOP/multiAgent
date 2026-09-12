from multiagent.experiments import (
    ARCHITECTURE_PRESETS,
    build_initial_state,
    summarize_result,
)


def test_sparse_preset_builds_self_describing_state(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "generic")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "model-x")

    state = build_initial_state(
        target="Migraine",
        cutoff=1985,
        preset=ARCHITECTURE_PRESETS["sparse"],
    )

    assert state["architecture_preset"] == "sparse"
    assert state["context_mode"] == "hierarchical"
    assert state["cache_enabled"] is True
    assert state["a2a_enabled"] is True
    assert state["llm_metadata"]["model"] == "model-x"
    assert "api_key" not in state["llm_metadata"]


def test_all_off_preset_keeps_common_loop_but_disables_sparse_components():
    preset = ARCHITECTURE_PRESETS["all-off"]

    assert preset.context_mode == "full"
    assert preset.cache_enabled is False
    assert preset.a2a_enabled is False


def test_summary_exposes_architecture_cost_signals():
    result = {
        "target_entity": "Migraine",
        "cutoff_year": 1985,
        "architecture_preset": "sparse",
        "context_mode": "hierarchical",
        "cache_enabled": True,
        "a2a_enabled": True,
        "max_iterations": 5,
        "max_refinement_rounds": 3,
        "llm_metadata": {"provider": "deepseek", "model": "x"},
        "termination_reason": "accepted",
        "iteration": 2,
        "route": "accept",
        "current_hypothesis": {"a": "A", "b": "B", "c": "C"},
        "trace": ["PLAN", "EXPLORE", "VERIFY"],
        "cache_stats": {"hits": 2, "misses": 3, "writes": 3},
        "communication_stats": {"messages": 4, "payload_chars": 100},
        "context_access_log": [
            {
                "level": 1,
                "regions": ["global"],
                "visible_field_count": 3,
                "visible_message_count": 0,
            },
            {
                "level": 3,
                "regions": ["bc"],
                "visible_field_count": 6,
                "visible_message_count": 1,
            },
        ],
    }

    summary = summarize_result(result)

    assert summary["final_path"] == ["A", "B", "C"]
    assert summary["context"]["mean_level"] == 2
    assert summary["context"]["region_accesses"] == 2
    assert summary["cache"]["hits"] == 2
