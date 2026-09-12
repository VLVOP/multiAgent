from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from multiagent.state import DiscoveryState


AgentRole = Literal["planner", "explorer", "hypothesis", "verifier", "critic"]
ContextLevel = Literal[1, 2, 3]


@dataclass(frozen=True)
class ContextDecision:
    """One context-access decision made before an agent invocation."""

    level: ContextLevel
    regions: tuple[str, ...]
    reason: str


class ContextDisclosurePolicy(Protocol):
    """Replaceable policy for deciding context depth and local disclosure region."""

    name: str

    def decide(self, role: AgentRole, state: DiscoveryState) -> ContextDecision:
        ...


class HeuristicLBDDisclosurePolicy:
    """LBD-specific deterministic baseline for progressive context disclosure."""

    name = "heuristic_lbd"

    _DEFAULT_LEVEL: dict[AgentRole, ContextLevel] = {
        "planner": 1,
        "explorer": 2,
        "hypothesis": 2,
        "verifier": 3,
        "critic": 3,
    }

    def decide(self, role: AgentRole, state: DiscoveryState) -> ContextDecision:
        level = self._DEFAULT_LEVEL[role]
        if level == 3 and not state.get("verification"):
            level = 2

        if role == "planner":
            return ContextDecision(level, ("global",), "planner needs only global discovery state")
        if role == "explorer":
            return ContextDecision(
                level,
                ("frontier", "candidate_paths"),
                "explorer needs relation-space navigation context",
            )
        if role == "hypothesis":
            return ContextDecision(
                level,
                ("candidate_path",),
                "hypothesis agent reasons over selected ABC candidates",
            )

        if role == "verifier":
            request = state.get("refinement_request") or {}
            target = request.get("target")
            mapping: dict[object, tuple[str, ...]] = {
                "ab": ("ab",),
                "bc": ("bc",),
                "both": ("ab", "bc"),
                "ac_novelty": ("ac",),
                "counter": ("ab", "bc"),
            }
            regions = mapping.get(target, ("ab", "bc", "ac"))
            reason = (
                f"targeted verifier refinement: {target}"
                if target is not None
                else "initial verifier pass covers both bridge edges and A-C novelty"
            )
            return ContextDecision(level, regions, reason)

        verification = state.get("verification", {})
        regions: list[str] = []
        if verification.get("ac_already_known") or verification.get("ac_novelty_resolved") is False:
            regions.append("ac")
        if not verification.get("ab_supported", False):
            regions.append("ab")
        if not verification.get("bc_supported", False):
            regions.append("bc")
        if not regions:
            regions = ["ab", "bc", "ac"]
        return ContextDecision(
            level,
            tuple(regions),
            "critic receives detailed context only for unresolved/disputed ABC regions",
        )


class FullContextDisclosurePolicy:
    """Full shared-context ablation baseline."""

    name = "full_context"

    def decide(self, role: AgentRole, state: DiscoveryState) -> ContextDecision:
        del role, state
        return ContextDecision(3, ("all",), "full-context ablation baseline")


_HEURISTIC_POLICY = HeuristicLBDDisclosurePolicy()
_FULL_POLICY = FullContextDisclosurePolicy()


def disclosure_policy_for_state(state: DiscoveryState) -> ContextDisclosurePolicy:
    """Resolve the active context policy from explicit experiment state."""
    if state.get("context_mode", "hierarchical") == "full":
        return _FULL_POLICY
    return _HEURISTIC_POLICY
