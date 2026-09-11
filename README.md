# multiAgent

A LangGraph-first biomedical Literature-Based Discovery (LBD) multi-agent research prototype.

## Current scope

The repository currently contains:

1. a cyclic LangGraph discovery-state graph;
2. five core LBD agents;
3. an initial online biomedical tool layer backed by NCBI E-utilities.

The local RAG / vector index / full biomedical KG are **not implemented yet**.

## Core state graph

```text
PLAN -> EXPLORE -> HYPOTHESIZE -> VERIFY -> CRITIQUE -> ROUTER
          ^                                    |
          |---- BACKTRACK / EXPLORE -----------|
                         |
                     REFINE -> VERIFY
```

## Agents vs control nodes

Five actual agents live under `src/multiagent/agents/`:

- `PlannerAgent`: lightweight initial exploration planning;
- `ExplorerAgent`: entity-space exploration and candidate A-B-C path ranking;
- `HypothesisAgent`: entity-level hypothesis formulation;
- `VerifierAgent`: pre-cutoff evidence verification and A-C novelty checking;
- `CriticAgent`: reflection/critique and next-route recommendation.

`ROUTER`, `REFINE`, and `BACKTRACK` are LangGraph control/state nodes, not separate agents.

Agents currently communicate through `DiscoveryState`; explicit A2A protocols are intentionally deferred.

## Project choices

- `uv` for Python/project management
- `LangGraph` for stateful cyclic orchestration
- DeepSeek API for LLM-backed agents
- NCBI E-utilities for online PubMed and MeSH access
- `.env` for local API configuration (never committed)
- `einops` and `einx` reserved for later tensor/vector transformations

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
```

Fill your own values in `.env`. Secrets are intentionally blank in the repository.

To let Explorer/Verifier use live biomedical tools, keep:

```bash
ONLINE_TOOLS_ENABLED=true
```

Set it to `false` to run the deterministic mock graph only.

### Test DeepSeek

```bash
uv run multiagent --smoke-test
```

### Run the LangGraph system

```bash
uv run multiagent --target "Migraine" --cutoff 1985
```

### Test online biomedical tools directly

```bash
uv run multiagent-tools --query "migraine magnesium" --cutoff 1985 --entity "Migraine"
```

Current online tools expose:

- temporal PubMed literature search;
- MeSH entity lookup;
- pre-cutoff biomedical entity expansion via MeSH co-indexing;
- PubMed entity-pair evidence retrieval;
- pre-cutoff pair-mention counting.

Entity-pair co-mentions are candidate evidence, not proof of a typed or causal biomedical relation.

### Run tests

```bash
uv run pytest -q
```

Without `DEEPSEEK_API_KEY`, agents fall back to deterministic behavior so the LangGraph loop remains testable offline.
