# Codex Engineering Handoff

This file is the practical continuation plan for Codex. Repository-level constraints in `/AGENTS.md` are authoritative and must be read first.

## Current implementation status

The repository already contains the following working design layers:

- five-Agent LangGraph LBD architecture;
- conditional loop policy separated into `edges.py`;
- reflection-guided targeted refinement;
- explicit global and per-hypothesis loop budgets;
- stable Agent-facing LBD Tool Interface;
- online NCBI/PubMed/MeSH backend abstraction;
- temporal relation verification, A-C novelty auditing, and counter-evidence checking;
- hierarchical context projection with local ABC-region disclosure;
- state-local evidence cache and reuse statistics;
- structured sparse A2A message schema and communication diagnostics;
- architecture ablation switches/presets;
- tool-call instrumentation;
- provider-independent OpenAI-compatible LLM configuration with DeepSeek as default;
- thin MCP adapter around the stable tools;
- single-case architecture experiment runner.

The repository is still a research prototype. Do not interpret the presence of interfaces as evidence that all planned methods are implemented.

## Explicitly not implemented yet

The following are intentionally pending research-design approval:

- frozen/local MEDLINE backend and corpus snapshot;
- benchmark dataset construction;
- vector/dense RAG retrieval;
- biomedical KG beyond current lightweight structures;
- learned ABC-aware Evidence Router;
- Router training pipeline and hard negatives;
- learned cache-aware discovery policy;
- LDA/topic organizer;
- learned hierarchical disclosure policy;
- real Temporal Agentic LBD episode dataset;
- future-literature retrospective validation pipeline.

Do not implement these until approval is given.

## Near-term autonomous engineering queue

Work in small steps. Prefer completing and validating one item before starting the next.

### P0 — repository health

1. Run the full local validation suite:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

2. Fix only genuine engineering errors without changing research semantics.
3. Add missing tests for recently added modules, especially:
   - context disclosure policy;
   - cache hit/miss behavior;
   - effective route vs semantic route;
   - architecture presets;
   - tool instrumentation;
   - MCP tool registration.
4. Keep deterministic/mock tests independent of network access and API keys.

### P1 — loop-engineering hardening

Audit the following invariants:

- `graph.py` declares topology but contains no policy logic;
- `edges.py` is the sole graph-routing/budget authority;
- `semantic_route` records the Critic recommendation;
- `effective_route` records the actual Edge-policy decision;
- A2A messages follow the effective route, not a stale Critic recommendation;
- ACCEPT terminates cleanly even at the global budget boundary;
- unresolved repeated REFINE eventually BACKTRACKs;
- BACKTRACK marks failed ABC paths without discarding reusable evidence cache entries;
- EXPLORE does not repeatedly propose known failed paths.

Add regression tests for any discovered corner case.

### P2 — context/disclosure engineering

Keep the current L1/L2/L3 hierarchy, but strengthen separation between:

```text
global stored context
!=
agent-visible context
```

Recommended work:

- ensure each Agent receives the smallest operational view required for its current action;
- preserve local disclosure by ABC region (`ab`, `bc`, `ac`);
- add disclosure diagnostics such as reason, policy name, depth, and region count;
- keep the policy interface replaceable so a learned policy can be added later;
- support the `full-context` baseline through the same code path.

Do not train a disclosure model yet.

### P3 — cache/reuse hardening

Strengthen deterministic reuse behavior before any learned policy exists:

- relation cache keys must include temporal cutoff;
- novelty decisions must be temporally keyed;
- cache entries should record the retrieval depth/evidence depth used;
- deeper refinement may refresh a shallower cached result;
- unrelated ABC paths must not receive irrelevant cached evidence through context projection;
- cache-off ablation must truly avoid reuse rather than merely hiding metrics;
- cache statistics should distinguish hit/miss/write consistently.

Do not add a persistent database yet unless explicitly approved; state-local/serializable cache is preferred for reproducible experiments at this stage.

### P4 — sparse A2A hardening

Keep A2A messages compact and typed. They should carry references and decisions, not raw evidence dumps.

Audit/extend metrics for:

- message count;
- serialized payload size;
- evidence-reference count;
- cache-reference count;
- duplicated references;
- messages received per Agent.

Preserve A2A-off ablation.

### P5 — instrumentation and experiment runner

The main publication story requires measuring quality-vs-cost later, so engineering should make cost signals reliable now.

Continue instrumentation for:

- actual tool calls by Agent/tool;
- cache reuse;
- loop iterations and overrides;
- context levels and disclosed regions;
- communication payload size;
- LLM provider/model metadata;
- if available from API responses, prompt/completion/total token usage without exposing secrets.

Do not fabricate token counts when the provider does not return them.

The experiment runner should make runs self-describing and easy to compare across:

```text
Dataset × LLM × Architecture Variant
```

For now, use mock/single-case engineering runs only. Do not create the real benchmark dataset.

### P6 — LLM-provider abstraction cleanup

DeepSeek remains the default. Preserve existing `DEEPSEEK_*` variables while supporting generic OpenAI-compatible overrides.

Requirements:

- Agents should not contain provider-specific branching where avoidable;
- provider/model metadata should be recorded without API keys;
- provider-specific request options should be guarded so generic providers do not receive unsupported DeepSeek-only fields;
- deterministic no-key behavior must remain available for tests.

### P7 — MCP adapter validation

MCP is infrastructure, not research novelty.

Validate that:

- MCP exposes the same stable Tool Interface rather than duplicate biomedical logic;
- direct in-process Python tool calls remain available for benchmark execution;
- adapter errors do not change core tool semantics;
- MCP-specific code stays thin.

## Stop condition

When the next meaningful task would require any of the following:

```text
real dataset
frozen corpus
retrieval index
RAG
hard negatives
Router training examples
benchmark split
learned retrieval/routing model
```

**STOP.**

Report exactly:

> Reached Retrieval Augmentation / Data Design boundary. Architecture engineering is ready for the next research-design decision.

Then summarize:

1. current tests/status;
2. remaining architecture-only technical debt;
3. the concrete data/retrieval design questions that must be answered next.

Do not proceed into data collection or retrieval-index construction without explicit approval.

## Suggested commit discipline

Use small commits with intent-revealing messages, for example:

```text
fix: align effective loop route with A2A message

test: cover local evidence disclosure

refactor: isolate context disclosure policy

feat: record provider token usage when available
```

Avoid broad rewrites unless a test-backed reason exists.

## Useful reference docs

Read these before making architectural changes:

- `README.md`
- `docs/method_ideas.md`
- `docs/retrieval_design.md`
- `docs/context_hierarchy.md`
- `docs/router_generalization.md`
- `/AGENTS.md`

If repository code and documentation disagree, do not silently choose one. Preserve the code state, identify the conflict, and request a research-design decision when the discrepancy affects scientific meaning.
