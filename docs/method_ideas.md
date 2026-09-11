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

### 4. Topic-aware Context Management with LDA

Use Latent Dirichlet Allocation (LDA) as a lightweight context organizer rather than as the primary semantic retriever.

Each document, tool observation, reflection, or cached evidence item may carry a topic distribution:

```text
theta_d = P(z | d)
```

The current discovery state can maintain a corresponding topic profile:

```text
theta_S = P(z | S_t)
```

The context manager can then use topic information for:

- retaining context relevant to the current ABC state;
- maintaining topical diversity in the selected evidence;
- reducing redundant historical observations;
- deciding which older context can be compressed, cached, or discarded;
- organizing the evidence cache by topic coverage.

The learned Evidence Router should primarily model task-specific relevance, while LDA contributes topic coverage and diversity. A conceptual combined score is:

```text
Score(d) = alpha * RouterRelevance(d)
         + beta  * TopicCoverage(d)
         - gamma * Redundancy(d)
```

This can produce a **topic-aware evidence cache**, where cached entries store not only relation evidence but also topic distributions and topical coverage metadata.

LDA itself is not intended as a standalone contribution. The intended role is a lightweight context-management component inside **ABC-aware routing + evidence cache + adaptive compute**.

### 5. Structured Sparse A2A Communication

Do not treat A2A as a standalone protocol contribution. Instead, use **LBD-specific structured communication** between agents so that they exchange typed discovery state rather than long free-form histories.

A candidate message schema is:

```text
DiscoveryMessage {
  abc_path,
  relation_under_test,
  evidence_ids,
  support_score,
  contradiction_score,
  novelty_status,
  uncertainty,
  requested_action,
  provenance,
  cache_references
}
```

Examples:

```text
Explorer -> Verifier:
(B, C, evidence_ids, uncertainty, request=verify)

Verifier -> Critic:
(support, contradiction, novelty, confidence, provenance)
```

A lightweight message router may decide:

- which observations are worth communicating;
- whether to send raw evidence, a compact summary, or a cache reference;
- whether another agent should be triggered at all;
- which parts of the current ABC state are relevant to the receiving agent.

Conceptually:

```text
P(m_i | S_t, agent_src, agent_dst)
```

This turns communication into another sparse resource allocation problem. The intended framing is therefore not generic A2A, but **ABC-aware structured A2A + selective communication + evidence/cache references**.

This yields a unified sparse-system view:

```text
Sparse Evidence
+ Sparse Compute
+ Sparse Communication
```

Useful communication-efficiency metrics include:

- communication tokens;
- number of inter-agent messages;
- duplicated evidence transmitted;
- cache-reference hit rate;
- discovery quality under fixed communication budgets.

Natural baselines include full shared context, free-form A2A, structured A2A without routing, and structured sparse A2A.

### Current framing

A possible umbrella term is:

**Sparse Agentic LBD**

with two primary method contributions:

1. **ABC-State-Conditioned Evidence Routing**: decide which evidence is worth expensive reasoning.
2. **Cache-Aware Sparse Discovery Policy**: decide what to retrieve, reuse, verify, explore, and how much compute to spend.

Topic-aware context management and structured sparse A2A are treated as supporting mechanisms that strengthen the same sparse-discovery story rather than as disconnected standalone claims.

These ideas should be evaluated against full-retrieval/all-agent baselines, fixed Top-K, generic rerankers, no-cache, LRU/semantic-cache baselines, generic learned routers, and communication-routing baselines, while reporting both discovery quality and compute/tool/token/communication cost.
