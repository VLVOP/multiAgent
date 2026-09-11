from __future__ import annotations

from typing import Any, Literal, TypedDict


Route = Literal["accept", "refine", "backtrack", "explore"]


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
    ab_evidence: list[dict[str, Any]]
    bc_evidence: list[dict[str, Any]]
    ac_evidence: list[dict[str, Any]]
    ac_mention_count: int
    mode: str
    rationale: str
    tool_error: str


class Reflection(TypedDict, total=False):
    issue: str
    recommendation: Route
    rationale: str


class DiscoveryState(TypedDict, total=False):
    target_entity: str
    cutoff_year: int
    iteration: int
    max_iterations: int

    plan: dict[str, Any]
    frontier: list[str]
    entity_paths: list[list[str]]
    exploration_observations: list[dict[str, Any]]
    hypotheses: list[Hypothesis]
    current_hypothesis: Hypothesis | None
    verification: Verification
    reflection: Reflection | None
    failed_paths: list[list[str]]

    route: Route
    trace: list[str]
