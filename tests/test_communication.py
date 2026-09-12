from multiagent.communication import append_message, make_discovery_message, messages_for_agent
from multiagent.graph import graph


def test_messages_for_agent_hides_other_recipients():
    state = {
        "agent_messages": [
            make_discovery_message(
                src="planner",
                dst="explorer",
                kind="plan",
                requested_action="explore",
            ),
            make_discovery_message(
                src="explorer",
                dst="hypothesis",
                kind="paths",
                requested_action="hypothesize",
            ),
        ]
    }

    explorer_messages = messages_for_agent(state, "explorer")

    assert len(explorer_messages) == 1
    assert explorer_messages[0]["src"] == "planner"


def test_append_message_tracks_communication_cost():
    message = make_discovery_message(
        src="verifier",
        dst="critic",
        kind="verification",
        requested_action="critique",
        evidence_refs=["1", "2"],
        cache_references=["cache-a"],
    )

    messages, stats = append_message({}, message)

    assert len(messages) == 1
    assert stats["messages"] == 1
    assert stats["evidence_refs"] == 2
    assert stats["cache_refs"] == 1
    assert stats["payload_chars"] > 0


def test_graph_emits_typed_messages_between_agents():
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

    messages = result["agent_messages"]
    kinds = {message["kind"] for message in messages}

    assert "plan" in kinds
    assert "candidate_paths" in kinds
    assert "hypothesis" in kinds
    assert "verification" in kinds
    assert "reflection" in kinds
    assert result["communication_stats"]["messages"] == len(messages)
