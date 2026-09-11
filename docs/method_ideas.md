# Method Ideas

## Sparse Agentic LBD

Working method direction for the paper:

### 1. ABC-state-conditioned Evidence Router

Instead of generic query-document reranking, score candidate literature conditioned on the current biomedical LBD state:

- entities: A, B, C;
- current discovery state S_t;
- current action a_t, e.g. verify A-B, verify B-C, check A-C novelty, critique;
- candidate document D_i.

Conceptually:

```text
P(D_i | A, B, C, S_t, a_t)
```

The router can predict multiple evidence dimensions, for example support, contradiction, bridge relevance, and novelty sensitivity, then select Top-K evidence for expensive LLM reasoning.

### 2. Cache-aware Sparse Discovery Policy

Introduce an LBD-specific evidence cache keyed by temporally grounded entity relations, for example:

```text
(A, B, cutoff_year) -> {
  supporting_pmids,
  counter_pmids,
  evidence_embeddings,
  relation_judgment,
  confidence,
  verification_state
}
```

The discovery policy should decide whether to retrieve, reuse cached evidence, refine an existing relation, or explore a new branch.

A useful objective is to trade expected scientific discovery gain against computation cost:

```text
a_t = argmax_a ExpectedDiscoveryGain(a | S_t) - lambda * Cost(a)
```

### 3. Infrastructure-inspired Sparse Computation

Borrow sparse-compute ideas from LLM infrastructure rather than treating all tools, agents, documents, and token budgets uniformly. The current ABC state may route:

- which evidence to inspect;
- which tool to invoke;
- which agent should act next;
- how much retrieval/LLM compute to allocate;
- whether cached evidence can be reused.

The intended contribution is not generic caching or generic routing by itself, but **ABC-aware routing + evidence caching + adaptive compute** inside iterative biomedical Literature-Based Discovery.

### Current framing

A possible umbrella term is:

**Sparse Agentic LBD**

with two tightly connected method contributions:

1. **ABC-State-Conditioned Evidence Routing**: decide which evidence is worth expensive reasoning.
2. **Cache-Aware Sparse Discovery Policy**: decide what to retrieve, reuse, verify, explore, and how much compute to spend.

These ideas should be evaluated against full-retrieval/all-agent baselines, fixed Top-K, generic rerankers, no-cache, LRU/semantic-cache baselines, and generic learned routers, while reporting both discovery quality and compute/tool/token cost.
