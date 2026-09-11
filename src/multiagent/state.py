from __future__ import annotations

from typing import Literal, TypedDict


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


class Reflection(TypedDict, total=False):
    issue: str
    recommendation: Route
    rationale: str


class DiscoveryState(TypedDict, total=False):
    target_entity: str
    cutoff_year: int
    iteration: int
    max_iterations: int

    frontier: list[str]
    entity_paths: list[list[str]]
    hypotheses: list[Hypothesis]
    current_hypothesis: Hypothesis | None
    verification: Verification
    reflection: Reflection | None
    failed_paths: list[list[str]]

    route: Route
    trace: list[str]
