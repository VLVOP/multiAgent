from __future__ import annotations

from copy import deepcopy
from typing import Literal

from multiagent.cache import project_cache_to_abc
from multiagent.state import DiscoveryState


AgentRole = Literal["planner", "explorer", "hypothesis", "verifier", "critic"]
ContextLevel = Literal[1, 2, 3]


class HierarchicalContextManager:
    """Project global discovery state into agent-specific hierarchical context views.

    L1: global discovery control state.
    L2: structured ABC/path/relation state.
    L3: detailed verification/evidence state.

    Hierarchy depth and local disclosure region are intentionally separate decisions:
    a level may exist globally while only the disputed A-B, B-C, or A-C region is exposed.
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
        "cache_stats",
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

    _REQUIRED_FIELDS: dict[AgentRole, set[str]] = {
        "planner": {"target_entity", "cutoff_year"},
        "explorer": {"target_entity", "cutoff_year", "plan", "failed_paths"},
        "hypothesis": {"entity_paths", "cutoff_year"},
        "verifier": {
            "current_hypothesis",
            "cutoff_year",
            "verification",
            "refinement_request",
            "evidence_cache",
            "cache_stats",
            "iteration",
        },
        "critic": {
            "current_hypothesis",
            "cutoff_year",
            "verification",
            "failed_paths",
        },
    }

    def disclosure_level(self, role: AgentRole, state: DiscoveryState) -> ContextLevel:
        """Return the currently disclosed hierarchy depth for an agent."""
        level = self._DEFAULT_LEVEL[role]
        if level == 3 and not state.get("verification"):
            return 2
        return level

    def disclosure_regions(self, role: AgentRole, state: DiscoveryState) -> list[str]:
        """Return local semantic regions exposed at the chosen hierarchy depth."""
        if role == "planner":
            return ["global"]
        if role == "explorer":
            return ["frontier", "candidate_paths"]
        if role == "hypothesis":
            return ["candidate_path"]

        if role == "verifier":
            request = state.get("refinement_request") or {}
            target = request.get("target")
            mapping = {
                "ab": ["ab"],
                "bc": ["bc"],
                "both": ["ab", "bc"],
                "ac_novelty": ["ac"],
                "counter": ["ab", "bc"],
            }
            return mapping.get(target, ["ab", "bc", "ac"])

        verification = state.get("verification", {})
        regions: list[str] = []
        if verification.get("ac_already_known") or verification.get("ac_novelty_resolved") is False:
            regions.append("ac")
        if not verification.get("ab_supported", False):
            regions.append("ab")
        if not verification.get("bc_supported", False):
            regions.append("bc")
        return regions or ["ab", "bc", "ac"]

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
        """Keep global judgments while locally disclosing detailed evidence payloads."""
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
        """Return the state subset visible to ``role`` at the current disclosure depth."""
        level = self.disclosure_level(role, state)
        regions = self.disclosure_regions(role, state)
        allowed = self._fields_for_level(level) | self._REQUIRED_FIELDS[role]
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

        # The cache remains global in DiscoveryState but a verifier only receives entries
        # local to its current ABC hypothesis. This is a control-plane projection, not a
        # prompt-time dump of every cached paper.
        hypothesis = state.get("current_hypothesis")
        if "evidence_cache" in projected and hypothesis is not None:
            projected["evidence_cache"] = project_cache_to_abc(
                state.get("evidence_cache", {}),
                hypothesis["a"],
                hypothesis["b"],
                hypothesis["c"],
                state["cutoff_year"],
            )

        return projected  # type: ignore[return-value]

    def view_metadata(self, role: AgentRole, state: DiscoveryState) -> dict[str, object]:
        """Expose diagnostics for context/disclosure ablations."""
        projected = self.project_state(role, state)
        return {
            "agent": role,
            "level": self.disclosure_level(role, state),
            "regions": self.disclosure_regions(role, state),
            "visible_fields": sorted(projected.keys()),
            "visible_field_count": len(projected),
        }


context_manager = HierarchicalContextManager()
