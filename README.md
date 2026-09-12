# multiAgent

A LangGraph-first biomedical Literature-Based Discovery (LBD) multi-agent research prototype following the Swanson/Arrowsmith ABC paradigm.

## Current scope

The repository currently contains:

1. a cyclic LangGraph discovery-state graph with explicit loop budgets;
2. five core LBD agents;
3. a stable agent-facing biomedical Tool Interface with a swappable online backend;
4. relation verification, novelty auditing, counter-evidence search, and evidence aggregation;
5. hierarchical agent-specific context views with local relation disclosure;
6. a state-local temporal evidence cache;
7. structured sparse A2A messages and communication diagnostics;
8. a thin MCP v2 adapter exposing the same Tool Interface.

The frozen/local MEDLINE corpus, vector/RAG index, learned Evidence Router, LDA context manager, and benchmark dataset construction are **not implemented yet**.

## Core state graph

```text
START -> PLAN -> EXPLORE -> HYPOTHESIZE -> VERIFY -> CRITIQUE
                                                    |
                    +-------------------------------+-------------------+
                    |                 |                 |               |
                 ACCEPT            REFINE          BACKTRACK         EXPLORE
                    |                 |                 |               |
                   END              VERIFY            EXPLORE          |
                                      ^                                  |
                                      +----------------------------------+
```

Conditional loop policy lives in `src/multiagent/edges.py`; `graph.py` only declares graph topology.

## Agents vs control nodes

Five actual agents live under `src/multiagent/agents/`:

- `PlannerAgent`: lightweight initial exploration planning;
- `ExplorerAgent`: entity-space exploration and candidate A-B-C path ranking;
- `HypothesisAgent`: entity-level hypothesis formulation;
- `VerifierAgent`: pre-cutoff A-B / B-C verification and A-C novelty auditing;
- `CriticAgent`: reflection, counter-evidence inspection, and next-route recommendation.

`REFINE` and `BACKTRACK` are control/state nodes, not separate agents. Routing is implemented by conditional edges.

## Context, cache, and communication

`HierarchicalContextManager` projects the global `DiscoveryState` into agent-specific views:

```text
L1: global discovery/control state
L2: ABC path and structured relation state
L3: detailed verification/evidence state
```

Hierarchy depth is separate from local disclosure. During a B-C refinement, for example, detailed B-C evidence can be disclosed while A-B and A-C details remain hidden.

Verified relation and novelty results are stored in a state-local temporal cache keyed by entity pair and cutoff year. The verifier reuses resolved cache entries instead of repeating unrelated verification work.

Agents also emit compact typed messages containing ABC paths, requested actions, evidence references, cache references, uncertainty, and small structured payloads. Full raw evidence is not copied between agents.

## Tool layer

Stable Agent/MCP-facing tools:

- `resolve_entity`
- `get_entity_info`
- `search_literature`
- `expand_entity`
- `get_evidence`
- `verify_relation`
- `search_counter_evidence`
- `check_novelty`
- `aggregate_evidence`

The current backend is `OnlineNCBIBackend`; the same interface is intended to support a later frozen/local MEDLINE backend without changing Agent code.

Co-occurrence alone is never treated as proof of a typed biomedical relation. Temporal cutoffs are enforced by the backend/tool layer rather than entrusted to the LLM.

## MCP server

The MCP adapter uses the official MCP Python SDK v2 and exposes the same core LBD tools rather than reimplementing them.

Run the stdio server with:

```bash
uv run multiagent-mcp
```

The module-level server object is also available as `multiagent.mcp_server:mcp` for MCP development tooling.

## Project choices

- `uv` for Python/project management
- `LangGraph` for stateful cyclic orchestration
- DeepSeek API for LLM-backed agents and semantic evidence judgment
- NCBI E-utilities for the initial online PubMed/MeSH backend
- MCP v2 for external tool exposure
- `.env` for local API configuration (never committed)
- `einops` and `einx` reserved for later tensor/vector transformations

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
```

Fill your own values in `.env`. Secrets are intentionally blank in the repository.

To let the agents use live biomedical tools, keep:

```bash
ONLINE_TOOLS_ENABLED=true
LBD_BACKEND=online
```

Set `ONLINE_TOOLS_ENABLED=false` to run the deterministic mock graph only.

### Test DeepSeek

```bash
uv run multiagent --smoke-test
```

### Run the LangGraph system

```bash
uv run multiagent --target "Migraine" --cutoff 1985
```

### Test online biomedical retrieval directly

```bash
uv run multiagent-tools --query "migraine magnesium" --cutoff 1985 --entity "Migraine"
```

### Run MCP tool server

```bash
uv run multiagent-mcp
```

### Run tests

```bash
uv run pytest -q
```

Without `DEEPSEEK_API_KEY`, semantic relation judgments remain unresolved rather than being inferred from co-occurrence. The mock graph remains available for deterministic orchestration tests.
