# multiAgent

A LangGraph-first biomedical Literature-Based Discovery (LBD) multi-agent research prototype.

## Current scope

The repository currently contains:

1. a cyclic LangGraph discovery-state graph;
2. five core LBD agents;
3. an online biomedical tool layer backed mainly by NCBI E-utilities;
4. higher-level LBD tools for relation verification, novelty auditing, and counter-evidence search.

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
- `VerifierAgent`: pre-cutoff A-B / B-C relation verification and A-C novelty auditing;
- `CriticAgent`: reflection/critique, counter-evidence inspection, and next-route recommendation.

`ROUTER`, `REFINE`, and `BACKTRACK` are LangGraph control/state nodes, not separate agents.

Agents currently communicate through `DiscoveryState`; explicit A2A protocols are intentionally deferred.

## Tool layer

Low-level / retrieval tools:

- `search_pubmed_literature`
- `search_mesh_entity`
- `expand_biomedical_entity`
- `find_entity_pair_evidence`
- `count_entity_pair_mentions`

High-level LBD tools:

- `get_entity_info`: temporally constrained entity context and MeSH grounding;
- `verify_relation`: semantic verification of an A-B relation from pre-cutoff evidence;
- `search_counter_evidence`: retrieval and semantic checking of evidence that weakens a relation;
- `check_novelty`: determine whether an A-C relation was already explicitly known before cutoff.

Co-occurrence alone is never treated as proof of a typed biomedical relation. When semantic judgment is unavailable, ambiguous cases remain unresolved instead of being promoted to facts.

## Project choices

- `uv` for Python/project management
- `LangGraph` for stateful cyclic orchestration
- DeepSeek API for LLM-backed agents and semantic evidence judgment
- NCBI E-utilities for online PubMed and MeSH access
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

### Test online biomedical retrieval directly

```bash
uv run multiagent-tools --query "migraine magnesium" --cutoff 1985 --entity "Migraine"
```

### Run tests

```bash
uv run pytest -q
```

Without `DEEPSEEK_API_KEY`, agents fall back to deterministic behavior so the LangGraph loop remains testable offline.
