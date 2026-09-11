# multiAgent

A LangGraph-first biomedical Literature-Based Discovery (LBD) multi-agent research prototype.

## Current scope

The repository currently contains two working layers:

1. a cyclic LangGraph discovery-state MVP;
2. an initial online biomedical tool layer backed by NCBI E-utilities.

The local RAG / vector index / full biomedical KG are **not implemented yet**.

The core discovery loop is:

```text
PLAN -> EXPLORE -> HYPOTHESIZE -> VERIFY -> CRITIQUE -> ROUTER
          ^                                    |
          |---- BACKTRACK / EXPLORE -----------|
                         |
                     REFINE -> VERIFY
```

The project uses:

- `uv` for Python/project management
- `LangGraph` for stateful cyclic orchestration
- DeepSeek API for the current LLM-backed agents
- NCBI E-utilities for online PubMed and MeSH access
- `.env` for local API configuration (never committed)
- `einops` and `einx` reserved for later tensor/vector transformations

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
```

Fill your own values in `.env` as needed. `DEEPSEEK_API_KEY`, `NCBI_EMAIL`, and `NCBI_API_KEY` are intentionally blank in the repository.

### Test DeepSeek

```bash
uv run multiagent --smoke-test
```

### Test the LangGraph loop

```bash
uv run multiagent --target "Migraine" --cutoff 1985
```

### Test online biomedical tools

```bash
uv run multiagent-tools --query "migraine magnesium" --cutoff 1985 --entity "Migraine"
```

Current online tools expose:

- temporal PubMed literature search;
- PubMed entity-pair evidence retrieval;
- pre-cutoff pair-mention counting;
- MeSH entity lookup.

The cutoff is enforced in the PubMed query itself. Entity-pair co-mentions are treated only as candidate evidence, not as proof of a typed or causal biomedical relation.

### Run tests

```bash
uv run pytest -q
```

If `DEEPSEEK_API_KEY` is absent, the graph keeps using the deterministic mock fallback so the LangGraph loop can still be tested offline.
