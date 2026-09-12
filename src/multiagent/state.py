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
    refinement_target: str | None
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
    counter_evidence_reused: bool


class ContextAccess(TypedDict, total=False):
    agent: str
    node: str
    level: int
    regions: list[str]
    visible_fields: list[str]
    visible_field_count: int
    visible_message_count: int
    iteration: int


class CacheStats(TypedDict, total=False):
    hits: int
    misses: int
    writes: int


class AgentMessage(TypedDict, total=False):
    src: str
    dst: str
    kind: str
    requested_action: str
    abc_path: list[str]
    relation_under_test: str | None
    evidence_refs: list[str]
    cache_references: list[str]
    uncertainty: float
    payload: dict[str, Any]


class CommunicationStats(TypedDict, total=False):
    messages: int
    payload_chars: int
    evidence_refs: int
    cache_refs: int


class DiscoveryState(TypedDict, total=False):
    target_entity: str
    cutoff_year: int
    iteration: int
    max_iterations: int
    max_refinement_rounds: int
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

    evidence_cache: dict[str, dict[str, Any]]
    cache_stats: CacheStats
    context_access_log: list[ContextAccess]

    agent_messages: list[AgentMessage]
    communication_stats: CommunicationStats

    route: Route
    trace: list[str]
