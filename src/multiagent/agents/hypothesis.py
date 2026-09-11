from __future__ import annotations

from multiagent.llm import chat_json, deepseek_enabled
from multiagent.state import DiscoveryState, Hypothesis


class HypothesisAgent:
    """Turn a candidate A-B-C path into an explicit biomedical LBD hypothesis."""

    def run(self, state: DiscoveryState) -> Hypothesis | None:
        paths = state.get("entity_paths", [])
        if not paths:
            return None

        a, b, c = paths[0]
        hypothesis: Hypothesis = {
            "a": a,
            "b": b,
            "c": c,
            "score": 0.5,
            "rationale": f"Candidate ABC bridge: {a} -> {b} -> {c}",
        }

        if deepseek_enabled():
            result = chat_json(
                system_prompt=(
                    "You are the hypothesis agent in a biomedical Literature-Based Discovery "
                    "system. A, B, and C are biomedical entities. Assess the proposed ABC bridge "
                    "without inventing evidence. Output JSON only with keys score and rationale. "
                    "score must be between 0 and 1."
                ),
                user_prompt=(
                    f"A={a}\nB={b}\nC={c}\n"
                    f"cutoff_year={state['cutoff_year']}\n"
                    "Formulate the entity-level LBD hypothesis implied by this bridge."
                ),
            )
            hypothesis["score"] = max(0.0, min(1.0, float(result.get("score", 0.5))))
            hypothesis["rationale"] = str(result.get("rationale", hypothesis["rationale"]))

        return hypothesis
