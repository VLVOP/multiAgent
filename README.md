# multiAgent

A LangGraph-first biomedical Literature-Based Discovery (LBD) multi-agent research prototype following the Swanson/Arrowsmith ABC paradigm.

## Current scope

The repository currently contains:

1. a cyclic LangGraph discovery-state graph with explicit loop budgets;
2. five core LBD agents;
3. a stable agent-facing biomedical Tool Interface with a swappable online backend;
4. relation verification, novelty auditing, counter-evidence search, and evidence aggregation;
5. hierarchical agent-specific context views with local relation disclosure;
6. a replaceable context-disclosure policy layer;
7. a state-local temporal evidence cache;
8. structured sparse A2A messages and communication diagnostics;
9. a thin MCP v2 adapter exposing the same Tool Interface;
10. architecture-ablation presets, external tool-call metrics, and provider-independent LLM configuration for cross-model experiments.

The frozen/local MEDLINE corpus, vector/RAG index, learned Evidence Router, learned context policy, LDA context manager, and benchmark dataset construction are **not implemented yet**.

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

Conditional loop policy lives in `src/multiagent/edges.py`; `graph.py` only declares graph topology. Repeated unresolved refinement has its own per-hypothesis budget and is converted into backtracking.

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

The decision itself is separated into `context_policy.py`. The current deterministic `HeuristicLBDDisclosurePolicy` is the baseline insertion point for a later learned state-conditioned policy; the manager remains responsible only for representing/projecting the selected context.

Verified relation and novelty results are stored in a state-local temporal cache keyed by entity pair and cutoff year. The verifier reuses resolved cache entries instead of repeating unrelated verification work. Critic counter-evidence is also reused across repeated refinement of the same ABC path when caching is enabled.

Agents emit compact typed messages containing ABC paths, requested actions, evidence references, cache references, uncertainty, and small structured payloads. Full raw evidence is not copied between agents.

External biomedical tool calls are accumulated by Agent and tool name in `DiscoveryState.tool_usage`, so cache/context policies can later be evaluated against actual tool-call cost rather than only trace length.

## Architecture ablations

The same graph can be run with architecture components enabled or disabled without editing code:

```bash
# Default sparse architecture
uv run multiagent --target "Migraine" --cutoff 1985

# Full shared-context baseline
uv run multiagent --target "Migraine" --cutoff 1985 --context-mode full

# Remove cache reuse
uv run multiagent --target "Migraine" --cutoff 1985 --no-cache

# Remove structured A2A messages
uv run multiagent --target "Migraine" --cutoff 1985 --no-a2a
```

For controlled comparison of the same case across predefined architecture variants:

```bash
uv run multiagent-experiment \
  --target "Migraine" \
  --cutoff 1985 \
  --presets sparse,full-context,no-cache,no-a2a,all-off \
  --summary-only
```

The experiment runner records the architecture preset, non-secret LLM metadata, loop termination, final ABC path, external tool calls, cache statistics, communication cost, and context-access statistics. The `all-off` preset retains the same LangGraph loop but disables hierarchical context, cache reuse, and structured A2A so those components can be isolated without silently changing the task graph.

These controls are stored in `DiscoveryState`, making later Dataset × LLM × Architecture experiments reproducible from the result object itself.

## LLM configuration

DeepSeek remains the default provider and existing `DEEPSEEK_*` variables are fully supported. For cross-model experiments, set generic OpenAI-compatible overrides:

```bash
LLM_PROVIDER=generic
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
```

The result state records non-secret model metadata so runs from different LLM backbones can be compared without changing Agent code.

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

```bash
uv run multiagent-mcp
```

The module-level server object is available as `multiagent.mcp_server:mcp` for MCP development tooling.

## Project choices

- `uv` for Python/project management
- `LangGraph` for stateful cyclic orchestration
- DeepSeek as the default LLM, with an OpenAI-compatible provider abstraction for model ablations
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

### Test configured LLM

```bash
uv run multiagent --smoke-test
```

### Run the LangGraph system

```bash
uv run multiagent --target "Migraine" --cutoff 1985
```

### Run one-case architecture ablations

```bash
uv run multiagent-experiment --presets sparse,all-off --summary-only
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

Without a configured LLM API key, semantic relation judgments remain unresolved rather than being inferred from co-occurrence. The mock graph remains available for deterministic orchestration tests.
