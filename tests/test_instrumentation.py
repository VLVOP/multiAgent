from multiagent.instrumentation import empty_tool_usage, merge_tool_usage


def test_tool_usage_accumulates_by_tool_and_agent():
    usage = empty_tool_usage()
    usage = merge_tool_usage(
        usage,
        agent="explorer",
        delta={"expand_entity": 3},
    )
    usage = merge_tool_usage(
        usage,
        agent="verifier",
        delta={"verify_relation": 2, "check_novelty": 1},
    )
    usage = merge_tool_usage(
        usage,
        agent="verifier",
        delta={"verify_relation": 1},
    )

    assert usage["total_calls"] == 7
    assert usage["by_tool"]["verify_relation"] == 3
    assert usage["by_agent"]["explorer"]["expand_entity"] == 3
    assert usage["by_agent"]["verifier"]["check_novelty"] == 1


def test_zero_deltas_do_not_inflate_usage():
    usage = merge_tool_usage(
        empty_tool_usage(),
        agent="critic",
        delta={"search_counter_evidence": 0},
    )

    assert usage["total_calls"] == 0
    assert usage["by_tool"] == {}
