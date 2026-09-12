# AGENTS.md

## Project identity

This repository implements **Sparse Agentic LBD**, a biomedical Literature-Based Discovery (LBD) research system following the classical Swanson/Arrowsmith ABC paradigm.

The goal is not to build a generic scientific-discovery chatbot. The system should discover biomedical entity-level hypotheses under temporal constraints while making the architecture, tool use, context use, communication, and compute cost measurable.

## Non-negotiable research semantics

1. **A, B, and C are normalized biomedical entities/concepts, not papers.** Papers are evidence/provenance.
2. **Co-occurrence is not a typed or causal biomedical relation.** Never promote co-mention counts, MeSH co-indexing, or lexical overlap into relation support unless a semantic verification component explicitly supports the relation.
3. **Temporal cutoff is enforced by tools/backends.** Do not rely on an LLM prompt to respect the cutoff.
4. Keep the two retrieval roles conceptually distinct:
   - entity/relation discovery retrieval finds candidate entities/relations;
   - document/evidence retrieval finds evidence for a specific relation.
5. The entity/relation graph organizes biomedical relations; the document/evidence store organizes supporting provenance.
6. Natural-language LLM knowledge must not silently substitute for missing evidence.

## Fixed orchestration architecture

The base LangGraph topology is intentionally stable:

```text
START -> PLAN -> EXPLORE -> HYPOTHESIZE -> VERIFY -> CRITIQUE
                                                    |
                    +-------------------------------+-------------------+
                    |                 |                 |               |
                 ACCEPT            REFINE          BACKTRACK         EXPLORE
                    |                 |                 |               |
                   END              VERIFY            EXPLORE          |
```

There are exactly five research Agents unless the research design is explicitly changed:

1. `PlannerAgent`
2. `ExplorerAgent`
3. `HypothesisAgent`
4. `VerifierAgent`
5. `CriticAgent`

`REFINE`, `BACKTRACK`, routers, guards, and policy functions are control/state mechanisms, **not Agents**.

### File responsibilities

- `src/multiagent/graph.py`: graph topology only.
- `src/multiagent/edges.py`: conditional-edge routing, loop policy, and budget guards.
- `src/multiagent/nodes.py`: state transitions and node orchestration.
- `src/multiagent/agents/`: Agent reasoning behavior.
- `src/multiagent/context.py` and `context_policy.py`: context projection/disclosure.
- `src/multiagent/cache.py`: reusable evidence/cache structures.
- `src/multiagent/communication.py`: structured sparse A2A messages.
- `src/multiagent/tools/`: stable tool interface and swappable backends.

Do not move loop policy back into `graph.py`.

## Current research components

The system currently develops three architecture-level ideas:

1. **ABC/state-conditioned evidence routing**: future learned Router should select evidence based on the current ABC state and action, not merely generic query relevance.
2. **Reflection-guided cache-aware discovery**: use critique/reflection to decide what computation to reuse, refine, or abandon.
3. **Adaptive hierarchical context for Agentic LBD**: L1/L2/L3-style global context is acceptable system infrastructure; the methodological focus is LBD-specific, state-conditioned, agent-specific access/disclosure.

Supporting mechanisms include structured sparse A2A, temporal retrieval, MCP exposure, and experiment instrumentation. Do not overclaim infrastructure as standalone novelty.

## Router generalization requirement

The future evidence Router is **task-conditioned/state-conditioned, not dataset-conditioned**.

Target input semantics are conceptually:

```text
(A, B, C, discovery_state, current_action, candidate_document)
```

Typical actions include:

```text
verify_AB
verify_BC
check_AC_novelty
find_bridge
find_counter_evidence
```

Primary generalization experiments should support cross-dataset zero-shot and leave-one-dataset-out evaluation. Dataset-specific calibration, if ever used, must be a thin optional layer rather than a separately trained full Router per dataset.

## Architecture-ablation requirement

The paper must be able to argue that gains come from architecture rather than a particular LLM. Preserve independent replaceability of:

```text
Dataset × LLM × Architecture Variant
```

Do not hard-wire Agents to one model provider. DeepSeek is the default, but OpenAI-compatible model configuration is intentionally supported.

Preserve ablation switches/presets for at least:

- hierarchical vs full shared context;
- cache on/off;
- structured A2A on/off;
- future Router variants;
- future loop/disclosure policy variants.

Keep cost/behavior instrumentation intact: tool calls, cache hits/misses, context depth/regions, communication cost, loop decisions, and model metadata.

## Tool/backend design

Agents should depend on the stable Tool Interface, not directly on PubMed/NCBI implementation details.

Current stable tools include:

```text
resolve_entity
get_entity_info
search_literature
expand_entity
get_evidence
verify_relation
search_counter_evidence
check_novelty
aggregate_evidence
```

Architecture:

```text
Agents / LangGraph -> Tool Interface -> Backend
```

Keep the interface compatible with both:

- current online NCBI-backed operation;
- future frozen/local MEDLINE operation.

MCP is a thin adapter around the same tools. Do not move core biomedical/tool logic into MCP.

## Hard stop: data and retrieval-augmentation boundary

**Do not autonomously start the data/retrieval-augmentation phase.**

Stop and ask for research-design approval before implementing any of the following:

- downloading/building a frozen MEDLINE/PubMed corpus;
- benchmark dataset construction or train/dev/test splits;
- vector database or dense RAG index;
- biomedical KG construction beyond existing lightweight/mock structures;
- LDA/topic indexing for the context manager;
- Evidence Router training data;
- Temporal Agentic LBD episode generation;
- hard-negative mining;
- learned Router training;
- learned context-disclosure policy training;
- future-literature validation dataset construction.

At that point, report that the work has reached **Retrieval Augmentation / Data Design** and wait for explicit approval. Do not infer the dataset design yourself.

## Safe autonomous work before the hard stop

Codex may continue engineering work that does not cross the data boundary, including:

- tests, typing, linting, and refactoring;
- loop corner cases and route/state consistency;
- deterministic context/disclosure policy improvements;
- cache and reuse semantics;
- structured A2A message plumbing and metrics;
- tool-call/token/context instrumentation;
- architecture presets and experiment runners;
- MCP adapter tests;
- provider-independent LLM configuration;
- CI and `uv` developer workflow;
- documentation synchronization;
- deterministic/mock benchmark harnesses that do not create the real research dataset.

When uncertain whether a change crosses the data/research-design boundary, stop rather than guessing.

## Engineering conventions

- Use **Python 3.11+**.
- Use **`uv`** for dependency and command workflow.
- Prefer `einops` / `einx` for later tensor reshaping/vector operations where appropriate.
- Never commit `.env`, API keys, credentials, downloaded corpora, or secrets.
- Keep `.env.example` secret-free.
- Preserve deterministic behavior when no LLM key is configured.
- Do not silently change the scientific meaning of a state field, tool, route, or metric.
- Prefer small, reviewable commits and update tests with behavior changes.

## Validation commands

Before declaring an engineering task complete, run when the environment permits:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

For relevant changes also exercise:

```bash
uv run multiagent --target "Migraine" --cutoff 1985
uv run multiagent-experiment --target "Migraine" --cutoff 1985 --presets sparse,full-context,no-cache,no-a2a,all-off --summary-only
```

If the environment prevents a command from running, say exactly which validation was not executed. Never claim tests passed when they were not run.

## Research-change protocol

A change is a **research-design change**, not a routine refactor, if it changes any of the following:

- ABC/LBD semantics;
- number or roles of Agents;
- base graph topology;
- temporal leakage policy;
- Router training target/schema;
- dataset definition;
- evidence-label meaning;
- benchmark/evaluation protocol;
- publication claims.

For such changes, stop and request approval before implementation.
