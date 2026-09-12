from __future__ import annotations

from copy import deepcopy

from multiagent.cache import project_cache_to_abc
from multiagent.communication import messages_for_agent
from multiagent.context_policy import (
    AgentRole,
    ContextLevel,
    disclosure_policy_for_state,
)
from multiagent.state import DiscoveryState


class HierarchicalContextManager:
    """Project global discovery state according to a replaceable disclosure policy.

    The manager owns representation/projection. The policy owns the decision about how
    deeply and where to disclose context. This separation is the insertion point for a
    later learned state-conditioned disclosure policy without changing Agent interfaces.
    """

    _L1_FIELDS = {
        "target_entity",
        "cutoff_year",
        "iteration",
        "max_iterations",
        "max_refinement_rounds",
        "termination_reason",
        "context_mode",
        "cache_enabled",
        "a2a_enabled",
        "architecture_preset",
        "plan",
        "route",
    }

    _L2_FIELDS = _L1_FIELDS | {
        "frontier",
        "entity_paths",
        "current_hypothesis",
        "failed_paths",
        "reflection",
        "refinement_request",
        "refinement_round",
        "cache_stats",
        "communication_stats",
    }

    _L3_FIELDS = _L2_FIELDS | {
        "verification",
        "exploration_observations",
        "hypotheses",
    }

    _REQUIRED_FIELDS: dict[AgentRole, set[str]] = {
        "planner": {"target_entity", "cutoff_year"},
        "explorer": {
            "target_entity",
            "cutoff_year",
            "plan",
            "failed_paths",
            "agent_messages",
        },
        "hypothesis": {"entity_paths", "cutoff_year", "agent_messages"},
        "verifier": {
            "current_hypothesis",
            "cutoff_year",
            "verification",
            "refinement_request",
            "evidence_cache",
            "cache_stats",
            "cache_enabled",
            "iteration",
            "agent_messages",
        },
        "critic": {
            "current_hypothesis",
            "cutoff_year",
            "verification",
            "failed_paths",
            "agent_messages",
        },
    }

    def _decision(self, role: AgentRole, state: DiscoveryState):
        return disclosure_policy_for_state(state).decide(role, state)

    def disclosure_level(self, role: AgentRole, state: DiscoveryState) -> ContextLevel:
        return self._decision(role, state).level

    def disclosure_regions(self, role: AgentRole, state: DiscoveryState) -> list[str]:
        return list(self._decision(role, state).regions)

    def _fields_for_level(self, level: ContextLevel) -> set[str]:
        if level == 1:
            return set(self._L1_FIELDS)
        if level == 2:
            return set(self._L2_FIELDS)
        return set(self._L3_FIELDS)

    @staticmethod
    def _project_verification(
        verification: dict[str, object],
        regions: list[str],
    ) -> dict[str, object]:
        summary_fields = {
            "ab_supported",
            "bc_supported",
            "ac_already_known",
            "ac_novelty_resolved",
            "refinement_target",
            "mode",
            "rationale",
            "tool_error",
        }
        projected = {
            key: deepcopy(value)
            for key, value in verification.items()
            if key in summary_fields
        }

        if "ab" in regions:
            for key in ("ab_verification", "ab_evidence"):
                if key in verification:
                    projected[key] = deepcopy(verification[key])
        if "bc" in regions:
            for key in ("bc_verification", "bc_evidence"):
                if key in verification:
                    projected[key] = deepcopy(verification[key])
        if "ac" in regions:
            for key in ("ac_novelty", "ac_evidence", "ac_mention_count"):
                if key in verification:
                    projected[key] = deepcopy(verification[key])
        return projected

    def project_state(self, role: AgentRole, state: DiscoveryState) -> DiscoveryState:
        """Return the state visible to one agent under the selected context policy."""
        decision = self._decision(role, state)
        policy = disclosure_policy_for_state(state)

        if policy.name == "full_context":
            return {
                key: deepcopy(value)
                for key, value in state.items()
                if key != "context_access_log"
            }  # type: ignore[return-value]

        regions = list(decision.regions)
        allowed = self._fields_for_level(decision.level) | self._REQUIRED_FIELDS[role]
        projected: dict[str, object] = {
            key: deepcopy(value)
            for key, value in state.items()
            if key in allowed
        }

        if "verification" in projected and isinstance(projected["verification"], dict):
            projected["verification"] = self._project_verification(
                projected["verification"],
                regions,
            )

        hypothesis = state.get("current_hypothesis")
        if "evidence_cache" in projected and hypothesis is not None:
            projected["evidence_cache"] = project_cache_to_abc(
                state.get("evidence_cache", {}),
                hypothesis["a"],
                hypothesis["b"],
                hypothesis["c"],
                state["cutoff_year"],
            )

        if "agent_messages" in projected:
            projected["agent_messages"] = messages_for_agent(state, role, limit=4)

        return projected  # type: ignore[return-value]

    def view_metadata(self, role: AgentRole, state: DiscoveryState) -> dict[str, object]:
        decision = self._decision(role, state)
        policy = disclosure_policy_for_state(state)
        projected = self.project_state(role, state)
        return {
            "agent": role,
            "policy": policy.name,
            "level": decision.level,
            "regions": list(decision.regions),
            "reason": decision.reason,
            "visible_fields": sorted(projected.keys()),
            "visible_field_count": len(projected),
            "visible_message_count": len(projected.get("agent_messages", [])),
        }


context_manager = HierarchicalContextManager()

__all__ = ["AgentRole", "ContextLevel", "HierarchicalContextManager", "context_manager"]
