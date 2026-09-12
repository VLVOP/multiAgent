from __future__ import annotations

from typing import Any, Literal, TypedDict


Route = Literal["accept", "refine", "backtrack", "explore"]
RefinementTarget = Literal["ab", "bc", "both", "ac_novelty", "counter"]


class Hypothesis(TypedDict, total=False):
    a: str
    b: str
    c: str
    score: float
    rationale: str


class Verification(TypedDict, total=False):
    ab_supported: bool
    bc_supported: bool
    ac_already_known: bool
    ac_novelty_resolved: bool
    ab_evidence: list[dict[str, Any]]
    bc_evidence: list[dict[str, Any]]
    ac_evidence: list[dict[str, Any]]
    ac_mention_count: int
    ab_verification: dict[str, Any]
    bc_verification: dict[str, Any]
    ac_novelty: dict[str, Any]
    mode: str
    rationale: str
    tool_error: str


class RefinementRequest(TypedDict, total=False):
    target: RefinementTarget
    reason: str
    top_k: int
    round: int


class Reflection(TypedDict, total=False):
    issue: str
    recommendation: Route
    rationale: str
    refinement_target: RefinementTarget
    counter_evidence: dict[str, Any]


class ContextAccess(TypedDict, total=False):
    agent: str
    node: str
    level: int
    visible_fields: list[str]
    visible_field_count: int
    iteration: int


class DiscoveryState(TypedDict, total=False):
    target_entity: str
    cutoff_year: int
    iteration: int
    max_iterations: int
    termination_reason: str | None

    plan: dict[str, Any]
    frontier: list[str]
    entity_paths: list[list[str]]
    exploration_observations: list[dict[str, Any]]
    hypotheses: list[Hypothesis]
    current_hypothesis: Hypothesis | None
    verification: Verification
    reflection: Reflection | None
    refinement_request: RefinementRequest | None
    refinement_round: int
    failed_paths: list[list[str]]

    context_access_log: list[ContextAccess]
    route: Route
    trace: list[str]
