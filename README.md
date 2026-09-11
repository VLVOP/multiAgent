# multiAgent

A LangGraph-first biomedical Literature-Based Discovery (LBD) multi-agent research prototype.

## Current scope

This repository currently focuses on the **agent/state-graph MVP only**. The retrieval-augmented data layer (entity/relation graph and document/evidence graph) is intentionally mocked for now and will be implemented later.

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
- `.env` for local API configuration (never committed)
- `einops` and `einx` reserved for later tensor/vector transformations

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
```

Edit `.env` and add your DeepSeek key:

```bash
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Test the API first:

```bash
uv run multiagent --smoke-test
```

Expected output:

```text
DEEPSEEK_OK
```

Then run the LangGraph system:

```bash
uv run multiagent --target "Migraine" --cutoff 1985
uv run pytest -q
```

If `DEEPSEEK_API_KEY` is absent, the graph keeps using the deterministic mock fallback so the LangGraph loop can still be tested offline.
