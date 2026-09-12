from __future__ import annotations

from typing import Literal

from multiagent.state import DiscoveryState


AgentRole = Literal["planner", "explorer", "hypothesis", "verifier", "critic"]
ContextLevel = Literal[1, 2, 3]


class HierarchicalContextManager:
    """Project global discovery state into agent-specific hierarchical context views.

    L1: global discovery control state.
    L2: structured ABC/path/relation state.
    L3: detailed verification/evidence state.

    The manager deliberately keeps the global DiscoveryState as the source of truth while
    preventing every agent from receiving the full state by default. A later disclosure
    policy can replace ``disclosure_level`` without changing agent interfaces.
    """

    _L1_FIELDS = {
        "target_entity",
        "cutoff_year",
        "iteration",
        "max_iterations",
        "termination_reason",
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
    }

    _L3_FIELDS = _L2_FIELDS | {
        "verification",
        "exploration_observations",
        "hypotheses",
    }

    _DEFAULT_LEVEL: dict[AgentRole, ContextLevel] = {
        "planner": 1,
        "explorer": 2,
        "hypothesis": 2,
        "verifier": 3,
        "critic": 3,
    }

    # Operational minimums keep each agent functional even if the hierarchy is later
    # reconfigured or learned. They are intentionally small and task-specific.
    _REQUIRED_FIELDS: dict[AgentRole, set[str]] = {
        "planner": {"target_entity", "cutoff_year"},
        "explorer": {"target_entity", "cutoff_year", "plan", "failed_paths"},
        "hypothesis": {"entity_paths", "cutoff_year"},
        "verifier": {
            "current_hypothesis",
            "cutoff_year",
            "verification",
            "refinement_request",
        },
        "critic": {
            "current_hypothesis",
            "cutoff_year",
            "verification",
            "failed_paths",
        },
    }

    def disclosure_level(self, role: AgentRole, state: DiscoveryState) -> ContextLevel:
        """Return the currently disclosed hierarchy depth for an agent.

        This is deliberately deterministic in v1. The method is the future insertion point
        for uncertainty-, budget-, cache-, or learned disclosure policies.
        """
        level = self._DEFAULT_LEVEL[role]

        # Do not expose detailed evidence before it exists. This keeps the view semantics
        # honest while preserving the same agent API.
        if level == 3 and not state.get("verification"):
            return 2
        return level

    def _fields_for_level(self, level: ContextLevel) -> set[str]:
        if level == 1:
            return set(self._L1_FIELDS)
        if level == 2:
            return set(self._L2_FIELDS)
        return set(self._L3_FIELDS)

    def project_state(self, role: AgentRole, state: DiscoveryState) -> DiscoveryState:
        """Return the state subset visible to ``role`` at the current disclosure depth."""
        level = self.disclosure_level(role, state)
        allowed = self._fields_for_level(level) | self._REQUIRED_FIELDS[role]
        return {
            key: value
            for key, value in state.items()
            if key in allowed
        }  # type: ignore[return-value]

    def view_metadata(self, role: AgentRole, state: DiscoveryState) -> dict[str, object]:
        """Expose lightweight diagnostics for evaluation and later disclosure ablations."""
        projected = self.project_state(role, state)
        return {
            "agent": role,
            "level": self.disclosure_level(role, state),
            "visible_fields": sorted(projected.keys()),
            "visible_field_count": len(projected),
        }


context_manager = HierarchicalContextManager()
