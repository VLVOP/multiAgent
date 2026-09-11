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
- `.env` for local model/API configuration (never committed)
- `einops` and `einx` reserved for later tensor/vector transformations

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
uv run python -m multiagent.main
uv run pytest -q
```

On Git Bash, `cp .env.example .env` creates the local `.env` file.
